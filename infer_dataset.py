#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import argparse
import os
import warnings

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from core.data import build_dataset
from core.libs.utils_videos import write_video
from core.trainer.inferencer import InferEngine


@torch.inference_mode()
def infer(resume_path, dump_dir, dataset, clip_length=20, tau=1.0, cfg=1.5, num_samples=32):
    infer_engine = InferEngine(resume_path)
    print(f"Inference start, loading model from {resume_path}")

    # create output directory
    try:
        resume_name = f"{infer_engine.meta_cfg.EXP_STR}_{infer_engine.meta_cfg.TIME_STR}_tau{tau}_cfg{cfg}"
    except Exception:
        resume_name = f"{os.path.splitext(os.path.basename(resume_path))[0]}_tau{tau}_cfg{cfg}"
    output_dir = os.path.join(dump_dir, resume_name)
    os.makedirs(output_dir, exist_ok=True)

    # build dataset
    if dataset is not None:
        dataset_cfg = OmegaConf.load(dataset).DATASET
        print(f"Using external dataset config: {dataset}")
    else:
        dataset_cfg = infer_engine.meta_cfg.DATASET
    test_dataset = build_dataset(data_cfg=dataset_cfg, split="test")
    test_dataset.slice(num_samples, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    fps = int(dataset_cfg.MOTION_FPS)
    sample_rate = int(dataset_cfg.AUDIO_SAMPLE_RATE)
    frame_length = int(clip_length * fps)
    audio_length = int(clip_length * sample_rate)

    tqdm_bar = tqdm(test_dataloader)
    for data_idx, batch_data in enumerate(tqdm_bar):
        # move tensors to device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(infer_engine.device)

        # build clean inference batch (no GT motion needed)
        infer_batch = {k: v for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
        infer_batch.pop("motion_code", None)
        infer_batch.pop("prev_motion_code", None)
        infer_batch["style_motion_code"][..., 100:103] *= 0.1

        # infer speaker 0: audio[:, 0] is self-audio, audio[:, 1] is other-audio
        paired_audio = infer_batch["audio"][:, [0, 1], :]
        prev_motion_code = infer_batch["style_motion_code"][:, 0, :]
        results_spk0 = infer_engine.model.inference(
            audio=paired_audio, prev_motion_code=prev_motion_code, tau=tau, cfg=cfg
        )
        pred_code_0 = infer_engine.smooth_motion_savgol(results_spk0["pred_motion_code"][0])

        # infer speaker 1: audio[:, 1] is self-audio, audio[:, 0] is other-audio
        paired_audio = infer_batch["audio"][:, [1, 0], :]
        prev_motion_code = infer_batch["style_motion_code"][:, 1, :]
        results_spk1 = infer_engine.model.inference(
            audio=paired_audio, prev_motion_code=prev_motion_code, tau=tau, cfg=cfg
        )
        pred_code_1 = infer_engine.smooth_motion_savgol(results_spk1["pred_motion_code"][0])

        # truncate to clip_length
        pred_code_0 = pred_code_0[:frame_length][None]
        pred_code_1 = pred_code_1[:frame_length][None]

        # render face meshes for both speakers
        if not hasattr(infer_engine, "face_decoder"):
            infer_engine._init_face_decoder()
        colors = infer_engine.face_decoder.get_colors()

        verts_0 = infer_engine.face_decoder.get_flame_verts(pred_code_0)[0]
        frames_0 = torch.cat([infer_engine.face_renderer(v[None], colors=colors)[0] for v in verts_0], dim=0).cpu()

        verts_1 = infer_engine.face_decoder.get_flame_verts(pred_code_1)[0]
        frames_1 = torch.cat([infer_engine.face_renderer(v[None], colors=colors)[0] for v in verts_1], dim=0).cpu()

        # speaker 0 on left, speaker 1 on right
        vis_frames = torch.cat([frames_0, frames_1], dim=-1)

        # mixed audio from both speakers, peak-normalize to avoid low volume
        vis_audio = results_spk0["audio"][0][:audio_length]
        peak = vis_audio.abs().max()
        if peak > 1e-6:
            vis_audio = vis_audio / peak * 0.9

        # determine save name from data infos
        try:
            audio_keys = batch_data["infos"]["audio_key"]
            save_name = f"{audio_keys[0][0]}_{audio_keys[1][0]}.mp4"
        except Exception:
            save_name = f"{data_idx:04d}.mp4"
        dump_path = os.path.join(output_dir, save_name)

        write_video(vis_frames, dump_path, fps, vis_audio, sample_rate, "aac")
        tqdm_bar.set_postfix({"frames": vis_frames.shape[0], "file": save_name})

    test_dataset.close()
    print(f"Inference done. Results saved to {output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*The `srun` command is available.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", "-r", type=str, required=True)
    parser.add_argument("--dump_dir", "-d", type=str, default="./render_results")
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--clip_length", default=20, type=int)
    parser.add_argument("--num_samples", "-n", default=32, type=int)
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--cfg", default=1.5, type=float)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision("high")
    infer(args.resume_path, args.dump_dir, args.dataset, args.clip_length, args.tau, args.cfg, args.num_samples)
