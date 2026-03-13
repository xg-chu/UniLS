#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import argparse
import warnings

import accelerate
import numpy as np
import torch
import transformers
from tqdm import tqdm

from core.data import build_dataset
from core.libs.utils import run_bar
from core.trainer.inferencer import InferEngine


@torch.inference_mode()
def eval(resume_path, dataset):
    # build config
    accelerator = accelerate.Accelerator()
    accelerate.utils.set_seed(42)

    infer_engine = InferEngine(resume_path, device=accelerator.device)
    if accelerator.is_main_process:
        print(f"Evaluation start, loading model from {resume_path}")

    # dataset inference
    data_split = dataset if dataset in {"train", "val", "test"} else "test"
    test_dataset = build_dataset(data_cfg=infer_engine.meta_cfg.DATASET, split=data_split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
    test_dataloader = accelerator.prepare(test_dataloader)

    # run evaluation
    eval_metrics = []
    tqdm_bar = run_bar(test_dataloader, disable=not accelerator.is_main_process)
    for batch_data in tqdm_bar:
        infer_results = infer_engine.inference(batch_data)
        metrics = infer_engine._calc_metrics(infer_results, batch_data)
        metrics = accelerator.gather_for_metrics(metrics)
        metrics = {k: v.nanmean().item() for k, v in metrics.items()}
        eval_metrics.append(metrics)
        if accelerator.is_main_process:
            tqdm_bar.set_postfix(metrics | {"len": infer_results["pred_motion_code"].shape[1]})

    if accelerator.is_main_process:
        mean_metrics = {k: np.nanmean([r[k] for r in eval_metrics]) for k in eval_metrics[0]}
        print(", ".join([f"{k}={v:.4f}" for k, v in mean_metrics.items()]))
        print("Evaluation done.")

    accelerator.wait_for_everyone()
    test_dataset.close()
    accelerator.end_training()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*The `srun` command is available.*")
    warnings.filterwarnings("ignore", message=".*using the device under current context.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", "-r", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision("high")
    transformers.logging.set_verbosity_error()
    eval(args.resume_path, args.dataset)
