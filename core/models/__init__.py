#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import importlib


def build_model(model_cfg, init_submodule=True, **kwargs):
    model_class = model_cfg.NAME
    module_path = f"core.models.{model_class.split(".")[0].lower()}"
    model_module = importlib.import_module(module_path)
    model_class = getattr(model_module, model_class.split(".")[1])
    return model_class(model_cfg, init_submodule, **kwargs)
