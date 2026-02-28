#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import os
import random
import time

import ipdb
import torch
import torch.nn.functional as F

from core.libs.utils_lmdb import LMDBEngine


class TalkMotionOnlyData(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split, debug=False):
        super().__init__()
        # build path
        self._split = split
        assert self._split in ["train", "val", "test"], f"Invalid split: {self._split}"
        # meta data
        self._fixed_style = False
        self._data_path = data_cfg.DATA_PATH
        self._clip_length = data_cfg.CLIP_LENGTH
        self._prev_length = data_cfg.PREV_LENGTH
        self._style_length = data_cfg.STYLE_LENGTH
        self._motion_fps = data_cfg.MOTION_FPS
        # build records
        # build records
        _metadata = json.load(open(data_cfg.META_PATH, "r"))[split]
        _metadata = [i for i in _metadata if i[1] > (max(100, self._clip_length) + 1)]
        if self._split == "train":
            self._all_data = []
            for _data in _metadata:
                for i in range(0, _data[1] - self._clip_length, 50):
                    self._all_data.append([_data[0], _data[1], i])
        else:
            self._fixed_style = True
            self._all_data = [[i[0], i[1], 0] for i in _metadata]

    def slice(self, slice, shuffle=False):
        if shuffle:
            random.shuffle(self._all_data)
        self._all_data = self._all_data[:slice]

    def sample(self, data_key=None):
        if data_key is None:
            random.seed(time.time())
            _all_data = random.sample(self._all_data, 1)
            random.seed(42)
            return self._load_one_record(_all_data[0])
        else:
            _all_data = [i for i in self._all_data if i[0] == data_key]
            if len(_all_data) == 0:
                raise ValueError(f"Data key {data_key} not found.")
            return self._load_one_record(_all_data[0])

    def __getitem__(self, index):
        motion_data = self._all_data[index]
        return self._load_one_record(motion_data)

    def __len__(self):
        return len(self._all_data)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(self._data_path, write=False)

    def _load_one_record(self, motion_data):
        if not hasattr(self, "_lmdb_engine"):
            self._init_lmdb_database()
        motion_key, seq_len, start_frame = motion_data
        if self._split == "train":
            start_frame = random.randint(start_frame, start_frame + 50)
            start_frame = min(start_frame, seq_len - self._clip_length)
            prev_start_frame = max(start_frame - self._prev_length, 0)
        this_records = self._lmdb_engine[motion_key]
        motion_tensor = torch.from_numpy(this_records["motioncode"])
        motion_tensor = torch.cat(
            [motion_tensor[:, :104], motion_tensor[:, 106:108], motion_tensor[:, 109:111]],
            dim=1,
        )
        assert motion_tensor.dim() == 2, "Motion tensor should be 2D"
        # style motion
        if self._fixed_style:
            style_frame = min(seq_len // 2, seq_len - self._style_length)
            style_motion = motion_tensor[style_frame : style_frame + self._style_length]
        else:
            style_frame = random.randint(0, seq_len - self._style_length)
            style_motion = motion_tensor[style_frame : style_frame + self._style_length]
        # clip strategy
        one_record = {
            "style_motion_code": style_motion.float(),
            "infos": {"motion_key": motion_key},
        }
        if self._split == "train":
            curr_motion = motion_tensor[start_frame : start_frame + self._clip_length]
            prev_motion = motion_tensor[prev_start_frame:start_frame]
            if prev_motion.shape[0] < self._prev_length:
                pad_len = self._prev_length - prev_motion.shape[0]
                prev_motion = F.pad(prev_motion, (0, 0, pad_len, 0), value=0)
            assert curr_motion.shape[0] == self._clip_length
            assert prev_motion.shape[0] == self._prev_length
            one_record["motion_code"] = curr_motion.float()
            one_record["prev_motion_code"] = prev_motion.float()
        else:
            one_record["motion_code"] = motion_tensor.float()
        return one_record

    def close(self):
        if hasattr(self, "_lmdb_engine"):
            self._lmdb_engine.close()
