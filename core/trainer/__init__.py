#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import importlib


def build_trainer(meta_cfg, init_submodule=True, **kwargs):
    trainer_class = meta_cfg.PIPELINE
    trainer_module = importlib.import_module(f"core.trainer.{trainer_class.split('.')[0].lower()}")
    trainer_class = getattr(trainer_module, trainer_class.split(".")[1])
    return trainer_class
