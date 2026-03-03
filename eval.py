#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import argparse
import warnings

import lightning
import numpy as np
import torch
from tqdm import tqdm

from core.data import build_dataset
from core.trainer.inferencer import InferEngine


@torch.inference_mode()
def eval(resume_path, dataset):
    # build config
    lightning.fabric.seed_everything(42)
    infer_engine = InferEngine(resume_path)
    print(f"Evaluation start, loading model from {resume_path}")
    # dataset inference
    data_split = dataset if dataset in {"train", "val", "test"} else "test"
    test_dataset = build_dataset(data_cfg=infer_engine.meta_cfg.DATASET, split=data_split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    # run evaluation
    eval_metrics = []
    tqdm_bar = tqdm(test_dataloader)
    for batch_data in tqdm_bar:
        infer_results = infer_engine.inference(batch_data)
        metrics = infer_engine._calc_metrics(infer_results, batch_data)
        metrics = {k: v.item() for k, v in metrics.items()}
        eval_metrics.append(metrics)
        tqdm_bar.set_postfix(metrics | {"len": infer_results["pred_motion_code"].shape[1]})
    mean_metrics = {k: np.mean([r[k] for r in eval_metrics]) for k in eval_metrics[0]}
    print(", ".join([f"{k}={v:.4f}" for k, v in mean_metrics.items()]))
    print("Evaluation done.")
    test_dataset.close()


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*The `srun` command is available.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", "-r", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision("high")
    eval(args.resume_path, args.dataset)
