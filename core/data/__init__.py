#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import importlib


def build_dataset(data_cfg, split, debug=False):
    data_class = data_cfg.LOADER
    module_name = data_class.split(".")[0].lower()
    data_module = importlib.import_module(f"core.data.{module_name}")
    data_class = getattr(data_module, data_class.split(".")[1])
    return data_class(data_cfg, split, debug)
