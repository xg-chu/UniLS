#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import argparse
import os
import warnings

import accelerate
import ipdb
import torch
import transformers

from core.data import build_dataset
from core.libs.utils import ConfigDict
from core.models import build_model
from core.trainer import build_trainer

os.environ["OMP_NUM_THREADS"] = "2"


def train(config, base_model, debug=False, cli_args=[]):
    # build config
    devices = list(range(int(os.environ.get("WORLD_SIZE", 1))))
    config_path = os.path.join("./configs", f"{config}.yaml")
    meta_cfg = ConfigDict(config_path=config_path, gpus=len(devices), cli_args=cli_args)
    accelerate.utils.set_seed(42)

    # setup model and optimizer
    model = build_model(meta_cfg.MODEL)
    optimizer, scheduler = model.configure_optimizers(meta_cfg.TRAINER)

    if base_model is not None:
        assert os.path.exists(base_model), f"Base model not found: {base_model}."
        model.load_state_dict(
            torch.load(base_model, map_location="cpu", weights_only=True)["model"],
            strict=False,
        )
        print("Load base model from: {}.".format(base_model))

    # load dataset
    train_dataset = build_dataset(meta_cfg.DATASET, split="train", debug=debug)
    val_dataset = build_dataset(meta_cfg.DATASET, split="val", debug=debug)
    val_dataset.slice(32)
    # test_dataset = build_dataset(meta_cfg.DATASET, split="test", debug=debug)

    Trainer = build_trainer(meta_cfg)
    trainer = Trainer(
        meta_cfg=meta_cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        devices=devices,
        debug=debug,
    )
    trainer.run_fit()
    trainer.cleanup()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    warnings.filterwarnings("ignore", message=".*using the device under current context.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, type=str)
    parser.add_argument("--basemodel", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    args, unknown = parser.parse_known_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision("high")
    transformers.logging.set_verbosity_error()
    train(args.config, args.basemodel, args.debug, cli_args=unknown)
