#!/usr/bin/env python3
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import copy

import torch


class EMA:
    def __init__(self, model, decay, update_freq=1, skip_keys=[]):
        self.decay = decay
        self.update_freq = update_freq
        self.skip_keys = skip_keys
        self.update_freq_counter = 0
        # Create shadow model
        self.shadow_model = copy.deepcopy(model)
        self.shadow_model.requires_grad_(False)
        self.shadow_model.eval()

    def update(self, new_model):
        if self.update_freq > 1:
            self.update_freq_counter += 1
            if self.update_freq_counter >= self.update_freq:
                self._step_internal(new_model)
                self.update_freq_counter = 0
        else:
            self._step_internal(new_model)

    def _step_internal(self, new_model):
        # Work fine with ddp and lightning fabric.
        ema_state_dict = {}
        shadow_params = self.shadow_model.state_dict()
        new_model_params = new_model.state_dict()
        for key, param in new_model_params.items():
            ema_param = shadow_params[key]
            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )
            if param.dtype == torch.int64 or param.dtype == torch.int32:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                if key in self.skip_keys:
                    ema_param = param.to(dtype=ema_param.dtype).clone()
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.shadow_model.load_state_dict(ema_state_dict, strict=False)

    def get_model(self):
        return self.shadow_model
