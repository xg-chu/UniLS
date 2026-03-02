#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
from copy import deepcopy

import ipdb
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

from core.libs.flame_model import FLAMEModel, RenderMesh
from core.libs.utils import FileLogger, WandbLogger, calc_params, run_bar
from core.libs.utils_ema import EMA
from core.libs.utils_videos import write_video
from core.models.modules import calc_val_metrics

from .base_trainer import BaseTrainer


class ARLoRATrainer(BaseTrainer):
    def __init__(
        self, meta_cfg, model, optimizer, scheduler, train_dataset, val_dataset, test_dataset, devices, debug=False
    ):
        self._debug = debug
        self._meta_cfg = meta_cfg
        self._log_interval = 100
        self._total_iters = meta_cfg.TRAINER.TRAIN_ITER
        self._check_interval = meta_cfg.TRAINER.CHECK_INTERVAL if not debug else 150

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

        filtered_target_modules = select_target_modules(
            model, include_keys=["to_qkv", "to_out", "ffn"], exclude_keys=["audio_encoder", "base_codec"]
        )
        # setup training materials
        lora_config = LoraConfig(
            r=8,  # Rank of the LoRA layers
            lora_alpha=32,  # Scaling factor for the LoRA layers
            target_modules=filtered_target_modules,  # Target modules to apply LoRA
            lora_dropout=0.1,  # Dropout for the LoRA layers
            bias="none",  # No bias in the LoRA layers
            task_type=None,
            modules_to_save=["self_audio_attn", "other_audio_attn"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        normal_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            else:
                # print(name)
                normal_params.append(param)
        optimizer = torch.optim.AdamW(normal_params, lr=self._meta_cfg.TRAINER.LEARNING_RATE)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=self._meta_cfg.TRAINER.WARMUP_ITER
        )
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self._meta_cfg.TRAINER.LR_DECAY_RATE,
            total_iters=(self._meta_cfg.TRAINER.LR_DECAY_ITER - self._meta_cfg.TRAINER.WARMUP_ITER),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self._meta_cfg.TRAINER.WARMUP_ITER],
        )
        self.scheduler = scheduler
        op_params, all_params = calc_params(model)
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
        self._calc_losses = self.accelerator.unwrap_model(self.model)._calc_losses

    def _init_face_decoder(self, device):
        self.face_decoder = FLAMEModel(n_shape=300, n_exp=100)
        self.face_decoder.eval()
        self.face_decoder.to(device)
        self.face_renderer = RenderMesh(image_size=512, faces=self.face_decoder.get_faces())
        self._all_visualize_frames = []
        self._all_visualize_audios = []

    def _calc_metrics(self, infer_results, visualize=None):
        if not hasattr(self, "face_decoder"):
            self._init_face_decoder(infer_results["pred_motion_code"].device)
        gt_motion_code = infer_results["gt_motion_code"]
        pred_motion_code = infer_results["pred_motion_code"]
        # val_metrics
        gt_verts = self.face_decoder.get_flame_verts(gt_motion_code, with_headpose=False)
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code, with_headpose=False)
        lve, avd, fdd = calc_val_metrics(pred_verts, gt_verts)
        if visualize is not None:
            iter_idx = visualize["iter_idx"]
            render_length = visualize["render_length"]
            curr_audio = infer_results["audio"]
            gt_motion_code = gt_motion_code[:1, :render_length]
            pred_motion_code = pred_motion_code[:1, :render_length]
            gt_verts = self.face_decoder.get_flame_verts(gt_motion_code)[0]
            pred_verts = self.face_decoder.get_flame_verts(pred_motion_code)[0]
            gt_images, _ = self.face_renderer(gt_verts, colors=self.face_decoder.get_colors())
            pred_images, _ = self.face_renderer(pred_verts, colors=self.face_decoder.get_colors())
            vis_frames = torch.cat([gt_images, pred_images], dim=-1).to(torch.uint8)
            # dump to video
            fps = int(self._meta_cfg.DATASET.MOTION_FPS)
            sample_rate = int(self._meta_cfg.DATASET.AUDIO_SAMPLE_RATE)
            vis_path = os.path.join(self._dump_dir, "examples", f"{iter_idx}.mp4")
            vis_audios = curr_audio[0, : int(render_length * sample_rate / fps)]
            write_video(vis_frames, vis_path, fps, vis_audios, sample_rate, "aac")
            self.wandb_logger.log_video(vis_path, "examples", step=iter_idx)
        return {"LVE": lve, "AVD": avd, "FDD": fdd}

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
        base_model = deepcopy(base_model).to("cpu")
        base_model.eval()
        merged_model = base_model.merge_and_unload()
        state = {"model": merged_model.state_dict(), "meta_cfg": self._meta_cfg._dump}
        torch.save(state, os.path.join(saving_path, name))


def select_target_modules(model, include_keys, exclude_keys):
    target_module_names = []
    for name, module in model.named_modules():
        for key in include_keys:
            if key in name and isinstance(module, torch.nn.Linear):
                should_exclude = False
                for excluded_path in exclude_keys:
                    if excluded_path in name:
                        should_exclude = True
                        break
                if not should_exclude:
                    target_module_names.append(name)
    return list(set(target_module_names))
