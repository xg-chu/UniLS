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


class SeamlessInteractionTalkData(torch.utils.data.Dataset):
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
        self._audio_sample_rate = data_cfg.AUDIO_SAMPLE_RATE
        self._audio_frames_rate = self._audio_sample_rate / self._motion_fps
        self._clip_audio_length = int(self._clip_length * self._audio_frames_rate)
        # build records
        _metadata = json.load(open(data_cfg.META_PATH, "r"))[split]
        _metadata = [i for i in _metadata.values()]
        _metadata = [i for i in _metadata if i[2] > (max(100, self._clip_length) + 1)]
        if self._split == "train":
            self._all_data = []
            for _data in _metadata:
                base_name_0 = os.path.basename(_data[0])
                base_name_1 = os.path.basename(_data[1])
                for i in range(0, _data[2] - self._clip_length, 50):
                    self._all_data.append([base_name_0, base_name_1, _data[2], i])
        else:
            self._fixed_style = True
            self._all_data = [[os.path.basename(i[0]), os.path.basename(i[1]), i[2], 0] for i in _metadata]

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
        audio_data = self._all_data[index]
        return self._load_one_record(audio_data)

    def __len__(self):
        return len(self._all_data)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(self._data_path, write=False)

    def _load_one_record(self, audio_data):
        if not hasattr(self, "_lmdb_engine"):
            self._init_lmdb_database()
        audio_key1, audio_key2, seq_len, start_frame = audio_data
        double_records = []
        if self._split == "train":
            start_frame = random.randint(start_frame, start_frame + 50)
            start_frame = min(start_frame, seq_len - self._clip_length)
            start_wave = int(start_frame * self._audio_frames_rate)
            prev_start_frame = max(start_frame - self._prev_length, 0)
        for audio_key in [audio_key1, audio_key2]:
            this_records = self._lmdb_engine[audio_key]
            audio_tensor = torch.from_numpy(this_records["audio"])
            audio_len = int(seq_len * self._audio_frames_rate)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, audio_len - audio_tensor.shape[0]), value=0)
            audio_tensor = audio_tensor[:audio_len]
            motion_tensor = torch.from_numpy(this_records["motioncode"])
            motion_tensor = torch.cat(
                [motion_tensor[:, :104], motion_tensor[:, 106:108], motion_tensor[:, 109:111]],
                dim=1,
            )
            assert audio_tensor.dim() == 1, "Audio tensor should be 1D"
            assert motion_tensor.dim() == 2, "Motion tensor should be 2D"
            # style motion
            if self._fixed_style:
                style_frame = min(seq_len // 2, seq_len - self._style_length)
                style_motion = motion_tensor[style_frame : style_frame + self._style_length]
            else:
                style_frame = random.randint(0, seq_len - self._style_length)
                style_motion = motion_tensor[style_frame : style_frame + self._style_length]
            if style_motion.shape[0] < self._style_length:
                pad_len = self._style_length - style_motion.shape[0]
                style_motion = F.pad(style_motion, (0, 0, pad_len, 0), value=0)
            assert style_motion.shape[0] == self._style_length
            # clip strategy
            one_record = {
                "style_motion_code": style_motion.float(),
                "infos": {"audio_key": audio_key},
            }
            if self._split == "train":
                curr_motion = motion_tensor[start_frame : start_frame + self._clip_length]
                curr_audio = audio_tensor[start_wave : start_wave + self._clip_audio_length]
                prev_motion = motion_tensor[prev_start_frame:start_frame]
                if prev_motion.shape[0] < self._prev_length:
                    pad_len = self._prev_length - prev_motion.shape[0]
                    prev_motion = F.pad(prev_motion, (0, 0, pad_len, 0), value=0)
                assert curr_audio.shape[0] == self._clip_audio_length
                assert curr_motion.shape[0] == self._clip_length
                assert prev_motion.shape[0] == self._prev_length
                one_record["audio"] = curr_audio.float()
                one_record["motion_code"] = curr_motion.float()
                one_record["prev_motion_code"] = prev_motion.float()
            else:
                curr_audio = audio_tensor[: int(seq_len * self._audio_frames_rate)]
                speech_mask = torch.from_numpy(this_records["speech_mask"])
                one_record["audio"] = curr_audio.float()
                one_record["motion_code"] = motion_tensor.float()
                one_record["speech_mask"] = speech_mask.float()
            double_records.append(one_record)

        si_record = {}
        for key in double_records[0].keys():
            if key == "infos":
                si_record[key] = {
                    "audio_key": [
                        double_records[0][key]["audio_key"],
                        double_records[1][key]["audio_key"],
                    ]
                }
            else:
                si_record[key] = torch.cat(
                    [double_records[0][key].unsqueeze(0), double_records[1][key].unsqueeze(0)],
                    dim=0,
                )
        return si_record

    def close(self):
        if hasattr(self, "_lmdb_engine"):
            self._lmdb_engine.close()
