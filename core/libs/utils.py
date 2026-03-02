#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os

os.environ["WANDB_SILENT"] = "true"

import random
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import colored
import wandb
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm.rich import tqdm as rtqdm
from tqdm.std import TqdmExperimentalWarning


def run_bar(bar_range, disable=False, debug=False, leave=True):
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    if debug:
        return tqdm(bar_range, disable=disable, leave=leave)
    else:
        return rtqdm(bar_range, disable=disable, leave=leave)


def device_parser(str_device):
    def parser_dash(str_device):
        device_id = str_device.split("-")
        device_id = [i for i in range(int(device_id[0]), int(device_id[-1]) + 1)]
        return device_id

    if "cpu" in str_device:
        device_id = ["cpu"]
    else:
        device_id = str_device.split(",")
        device_id = [parser_dash(i) for i in device_id]
    res = []
    for i in device_id:
        res += i
    return res


def pretty_dict(input_dict, indent=0, highlight_keys=[]):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        if key in highlight_keys:
            out_line += tab * indent + colored.stylize(str(key), colored.fg(1))
        else:
            out_line += tab * indent + colored.stylize(str(key), colored.fg(2))
        if isinstance(value, dict):
            out_line += ":\n"
            out_line += pretty_dict(value, indent + 1, highlight_keys)
        else:
            if key in highlight_keys:
                out_line += ":" + "\t" + colored.stylize(str(value), colored.fg(1)) + "\n"
            else:
                out_line += ":" + "\t" + colored.stylize(str(value), colored.fg(2)) + "\n"
    if indent == 0:
        max_length = 0
        for line in out_line.split("\n"):
            max_length = max(max_length, len(line.split("\t")[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split("\n"):
            if "\t" in line:
                aligned_number = max_length - len(line.split("\t")[0])
                line = line.replace("\t", aligned_number * " ")
            aligned_line += line + "\n"
        return aligned_line[:-2]
    return out_line


def calc_params(model):
    op_para_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_para_num = sum(p.numel() for p in model.parameters())
    return op_para_num, all_para_num


def find_latest_model(checkpoints_dir):
    all_models = os.listdir(checkpoints_dir)
    if len(all_models) == 0:
        return None
    all_iterations = [int(i.split("_")[-1].split(".")[0]) for i in all_models]
    latest_iteration = max(all_iterations)
    return os.path.join(checkpoints_dir, f"iter_{latest_iteration}.pt")


### CONFIG ###
def read_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(b[k], dict), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


class WandbLogger:
    def __init__(self, entity, project, name, config, debug=False):
        self.debug = debug
        if not debug:
            self.wandb_logger = wandb.init(
                entity=entity,
                project=project,
                name=name,
                config=config,
                dir="/tmp/wandb",
                settings=wandb.Settings(_disable_stats=True),
            )

    def log(self, metrics, step):
        if self.debug:
            return
        self.wandb_logger.log(metrics, step=step)

    def log_video(self, video_path, video_key, step):
        if self.debug:
            return
        self.wandb_logger.log({video_key: wandb.Video(video_path, format="mp4")}, step=step)

    def close(self):
        if self.debug:
            return
        self.wandb_logger.finish()


class TBLogger:
    def __init__(self, tb_path, debug=False):
        self.debug = debug
        if not debug:
            self.tb_logger = SummaryWriter(tb_path)

    def add_scalar(self, scalar_key, scalar_value, step):
        if self.debug:
            return
        self.tb_logger.add_scalar(scalar_key, scalar_value, step)


class FileLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_handler = open(self.file_path, "a")

    def info(self, log_string, print_string=False):
        self.file_handler.write(log_string + "\n")
        self.file_handler.flush()
        if print_string:
            print(log_string)

    def close(self):
        self.file_handler.close()

    def __del__(self):
        self.close()


class ConfigDict(dict):
    def __init__(self, config_path=None, data_config_path=None, gpus=1, cli_args=[]):
        if isinstance(config_path, str):
            # build new config
            config_dict = read_config(config_path)
            if data_config_path is not None:
                dataset_dict = read_config(data_config_path)
                merge_a_into_b(dataset_dict, config_dict)
            # set output path
            experiment_string = "{}_{}".format(config_dict["MODEL"]["NAME"], config_dict["DATASET"]["NAME"])
            timeInTokyo = datetime.now()
            timeInTokyo = timeInTokyo.astimezone(ZoneInfo("Asia/Tokyo"))
            time_string = timeInTokyo.strftime("%b%d_%H%M_") + "".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 5))
            config_dict["EXP_STR"] = experiment_string
            config_dict["TIME_STR"] = time_string
        elif isinstance(config_path, dict):
            config_dict = config_path
        else:
            raise ValueError("config_path must be a string or a dict")
        config_dict["TRAINER"]["TRAIN_ITER"] //= gpus
        config_dict["TRAINER"]["CHECK_INTERVAL"] //= gpus
        config_dict["TRAINER"]["LEARNING_RATE"] *= gpus
        config_dict["TRAINER"]["WARMUP_ITER"] //= gpus
        config_dict["TRAINER"]["LR_DECAY_ITER"] //= gpus
        config_dict["TRAINER"]["EMA_DECAY"] **= gpus

        _dot_config = OmegaConf.create(config_dict)
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        _dot_config.merge_with(cli_cfg)
        super().__init__(OmegaConf.to_container(_dot_config, resolve=True))
        self._dot_config = _dot_config
        OmegaConf.set_readonly(self._dot_config, True)

    def __getattr__(self, name):
        if name == "_dump":
            return dict(self)
        if name == "_raw_string":
            import re

            ansi_escape = re.compile(
                r"""
                \x1B  # ESC
                (?:   # 7-bit C1 Fe (except CSI)
                    [@-Z\\-_]
                |     # or [ for CSI, followed by a control sequence
                    \[
                    [0-?]*  # Parameter bytes
                    [ -/]*  # Intermediate bytes
                    [@-~]   # Final byte
                )
            """,
                re.VERBOSE,
            )
            result = "\n" + ansi_escape.sub("", pretty_dict(self))
            return result
        return getattr(self._dot_config, name)

    def __str__(self):
        return pretty_dict(self)

    def update(self, key, value):
        OmegaConf.set_readonly(self._dot_config, False)
        self._dot_config[key] = value
        self[key] = value
        OmegaConf.set_readonly(self._dot_config, True)


class ConfigDictWrapper(dict):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self._dot_config = OmegaConf.create(dict(self))
        OmegaConf.set_readonly(self._dot_config, True)

    def __getattr__(self, name):
        return getattr(self._dot_config, name)

    def __str__(self):
        return pretty_dict(self)
