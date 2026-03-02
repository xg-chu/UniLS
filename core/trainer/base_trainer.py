#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os

import ipdb
import numpy as np
import torch
from accelerate import Accelerator

from core.libs.utils import FileLogger, WandbLogger, calc_params, run_bar
from core.libs.utils_ema import EMA


class BaseTrainer:
    def __init__(
        self, meta_cfg, model, optimizer, scheduler, train_dataset, val_dataset, test_dataset, devices, debug=False
    ):
        self._debug = debug
        self._meta_cfg = meta_cfg
        self._log_interval = 100
        self._total_iters = meta_cfg.TRAINER.TRAIN_ITER
        self._check_interval = meta_cfg.TRAINER.CHECK_INTERVAL if not debug else 150
        op_params, all_params = calc_params(model)

        # setup accelerator for multi-GPU
        if len(devices) > 1:
            print("Using multi-GPUs[{}]: training...".format(len(devices)))
        else:
            print("Using single-GPU training...")
        self.accelerator = Accelerator()

        # create dataloaders
        batch_size = 4 if debug else meta_cfg.TRAINER.BATCH_SIZE
        num_workers = 0 if debug else 8
        persistent_workers = None if debug else True
        prefetch_factor = None if debug else 2
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=0 if debug else 1, shuffle=False, drop_last=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, num_workers=0 if debug else 1, shuffle=False, drop_last=True
        )

        # setup training materials
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(model, optimizer, scheduler)
        self.train_dataloader = self.accelerator.prepare(train_dataloader)
        self.val_dataloader = self.accelerator.prepare(val_dataloader)
        self.test_dataloader = self.accelerator.prepare(test_dataloader)

        if meta_cfg.TRAINER.USING_EMA:
            self.ema_model = EMA(
                self.model, decay=meta_cfg.TRAINER.EMA_DECAY, update_freq=meta_cfg.TRAINER.EMA_UPDATE_FREQ
            )
        # logger init only on main process
        if self.accelerator.is_main_process:
            self._dump_dir = (
                os.path.join(meta_cfg.DUMP_DIR, "debug")
                if debug
                else os.path.join(meta_cfg.DUMP_DIR, meta_cfg.EXP_STR, meta_cfg.TIME_STR)
            )
            os.makedirs(os.path.join(self._dump_dir, "examples"), exist_ok=True)
            os.makedirs(os.path.join(self._dump_dir, "checkpoints"), exist_ok=True)
            self.wandb_logger = WandbLogger(
                entity=meta_cfg.WANDB_ENTITY,
                project=meta_cfg.WANDB_PROJECT,
                name=f"{meta_cfg.EXP_STR}_{meta_cfg.TIME_STR}",
                config=meta_cfg,
                debug=debug,
            )
            self.file_logger = FileLogger(os.path.join(self._dump_dir, "train_log.txt"))
            self.file_logger.info(meta_cfg._raw_string, print_string=True)
            self.file_logger.info(f"Training on devices: {devices}", print_string=True)
            self.file_logger.info(
                "Parameters: {:.2f}M / {:.2f}M.".format(op_params / 1e6, all_params / 1e6), print_string=True
            )
            self.file_logger.info(f"Train Data: {len(train_dataset)}, Val Data: {len(val_dataset)}.", print_string=True)
            self.file_logger.info(f"Data_path: {train_dataset._data_path}", print_string=True)

    def run_fit(self):
        fit_bar = run_bar(
            range(1, self._total_iters + 1),
            disable=not self.accelerator.is_main_process,
            debug=self._debug,
        )

        train_iter = iter(self.train_dataloader)
        self.model.train()

        for iter_idx in fit_bar:
            # get data
            try:
                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch_data = next(train_iter)
            # forward pass
            train_results = self.model(batch_data, training=True)
            loss_metrics = self._calc_losses(train_results, self._meta_cfg.LOSS_KWARGS)
            loss = sum(loss_metrics.values())

            # backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            if self._meta_cfg.TRAINER.USING_EMA:
                self.ema_model.update(self.model)

            # logging
            log_metrics = {k: v.item() for k, v in loss_metrics.items()}
            del loss, loss_metrics, train_results
            self._log_metrics(iter_idx, log_metrics)

            # validation and checkpointing
            if iter_idx % self._check_interval == 0 or iter_idx == self._total_iters:
                self.accelerator.wait_for_everyone()
                self._save_checkpoint(f"iter_{iter_idx}.pt")
                self.run_validation(iter_idx, stage="val")
                torch.cuda.empty_cache()
                self.accelerator.wait_for_everyone()
        self.run_validation(iter_idx, stage="test")

    def __del__(self):
        self.train_dataloader.dataset.close()
        self.val_dataloader.dataset.close()

    @torch.inference_mode()
    def run_validation(self, iter_idx, stage="val"):
        self.model.eval()
        if self._meta_cfg.TRAINER.USING_EMA:
            inf_model = self.ema_model.get_model()
            inf_model.eval()
        else:
            inf_model = self.model
        val_metrics = []
        data_loader = self.val_dataloader if stage == "val" else self.test_dataloader
        val_bar = run_bar(data_loader, disable=not self.accelerator.is_main_process, debug=self._debug, leave=False)
        for idx, batch_data in enumerate(val_bar):
            if self.accelerator.is_main_process and stage == "val" and idx == 0:
                visualize = {"iter_idx": iter_idx, "render_length": 500}
            else:
                visualize = None
            infer_results = self.accelerator.unwrap_model(inf_model).inference(**batch_data)
            one_val_metrics = self._calc_metrics(infer_results, visualize=visualize)
            one_val_metrics = self.accelerator.gather_for_metrics(one_val_metrics)
            one_val_metrics = {k: v.mean().item() for k, v in one_val_metrics.items()}
            val_metrics.append(one_val_metrics)
        if self.accelerator.is_main_process:
            mean_metrics = {k: np.mean([r[k] for r in val_metrics]) for k in val_metrics[0]}
            self.file_logger.info(
                f"Step {iter_idx}, metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in mean_metrics.items()]),
                print_string=True,
            )
            for key in mean_metrics:
                self.wandb_logger.log({f"{stage}/{key}": mean_metrics[key]}, step=iter_idx)
        self.model.train()

    def _save_checkpoint(self, name="latest.pt"):
        if not self.accelerator.is_main_process or self._debug:
            return
        saving_path = os.path.join(self._dump_dir, "checkpoints")
        if self._meta_cfg.TRAINER.USING_EMA:
            base_model = self.ema_model.get_model()
        else:
            base_model = self.model
        if hasattr(base_model, "module"):
            base_model = base_model.module
        state = {"model": base_model.state_dict(), "meta_cfg": self._meta_cfg._dump}
        torch.save(state, os.path.join(saving_path, name))

    def _log_metrics(self, iter_idx, log_metrics):
        if not self.accelerator.is_main_process:
            return
        # periodic logging
        if iter_idx % self._log_interval == 0:
            # simple loss calculation
            learning_rate = self.optimizer.param_groups[0]["lr"]
            current_loss = sum(v for v in log_metrics.values())
            self.file_logger.info(
                f"Step {iter_idx}/{self._total_iters}: loss={current_loss:.4f}, lr={learning_rate:.5f} | "
                + ", ".join([f"{k}={v:.4f}" for k, v in log_metrics.items()]),
                print_string=True,
            )
            self.wandb_logger.log({"train/lr": learning_rate, "train/loss": current_loss}, step=iter_idx)
            for k, v in log_metrics.items():
                self.wandb_logger.log({f"train/{k}": v}, step=iter_idx)
