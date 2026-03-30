#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import argparse
import os
import warnings

import torch

from core.libs.utils_videos import read_audio, write_video
from core.trainer.inferencer import InferEngine


@torch.inference_mode()
def infer_audio(resume_path, audio_path, audio_path_2=None, dump_dir="./render_results", tau=1.0, cfg=1.5):
    infer_engine = InferEngine(resume_path)
    print(f"Inference start, loading model from {resume_path}")

    fps = int(infer_engine.meta_cfg.DATASET.MOTION_FPS)
    sample_rate = int(infer_engine.meta_cfg.DATASET.AUDIO_SAMPLE_RATE)
    device = infer_engine.device
    two_speaker = audio_path_2 is not None

    # load audio (read_audio returns mono 1D tensor, already resampled)
    audio_0, _ = read_audio(audio_path, target_sr=sample_rate)

    if two_speaker:
        audio_1, _ = read_audio(audio_path_2, target_sr=sample_rate)
        # align length
        max_len = max(audio_0.shape[0], audio_1.shape[0])
        audio_0 = torch.nn.functional.pad(audio_0, (0, max_len - audio_0.shape[0]))
        audio_1 = torch.nn.functional.pad(audio_1, (0, max_len - audio_1.shape[0]))
    else:
        audio_1 = torch.zeros_like(audio_0)

    # paired audio: [batch=1, 2, audio_samples]
    paired_audio = torch.stack([audio_0, audio_1], dim=0).unsqueeze(0).to(device)

    # prev_motion_code: zeros (no style reference)
    motion_dim = infer_engine.model.motion_dim
    patch_len = max(infer_engine.model.patch_nums)
    prev_motion_code = torch.zeros(1, patch_len, motion_dim, device=device)

    # infer speaker 0
    results_spk0 = infer_engine.model.inference(audio=paired_audio, prev_motion_code=prev_motion_code, tau=tau, cfg=cfg)
    pred_code_0 = infer_engine.smooth_motion_savgol(results_spk0["pred_motion_code"][0])

    if two_speaker:
        # infer speaker 1: swap audio channels
        results_spk1 = infer_engine.model.inference(
            audio=paired_audio[:, [1, 0], :], prev_motion_code=prev_motion_code, tau=tau, cfg=cfg
        )
        pred_code_1 = infer_engine.smooth_motion_savgol(results_spk1["pred_motion_code"][0])
    else:
        pred_code_1 = torch.zeros_like(pred_code_0)

    # render
    if not hasattr(infer_engine, "face_decoder"):
        infer_engine._init_face_decoder()
    colors = infer_engine.face_decoder.get_colors()

    verts_0 = infer_engine.face_decoder.get_flame_verts(pred_code_0[None])[0]
    frames_0 = torch.cat([infer_engine.face_renderer(v[None], colors=colors)[0] for v in verts_0], dim=0).cpu()

    if two_speaker:
        verts_1 = infer_engine.face_decoder.get_flame_verts(pred_code_1[None])[0]
        frames_1 = torch.cat([infer_engine.face_renderer(v[None], colors=colors)[0] for v in verts_1], dim=0).cpu()
        vis_frames = torch.cat([frames_0, frames_1], dim=-1)
    else:
        vis_frames = frames_0

    # audio for video
    audio_len = vis_frames.shape[0] * sample_rate // fps
    if two_speaker:
        vis_audio = (audio_0[:audio_len] + audio_1[:audio_len]) / 2
    else:
        vis_audio = audio_0[:audio_len]
    peak = vis_audio.abs().max()
    if peak > 1e-6:
        vis_audio = vis_audio / peak * 0.9

    # save
    os.makedirs(dump_dir, exist_ok=True)
    save_name = os.path.splitext(os.path.basename(audio_path))[0]
    if two_speaker:
        save_name += f"_x_{os.path.splitext(os.path.basename(audio_path_2))[0]}"
    dump_path = os.path.join(dump_dir, f"{save_name}_tau{tau}_cfg{cfg}.mp4")

    write_video(vis_frames, dump_path, fps, vis_audio, sample_rate, "aac")
    print(f"Inference done. Saved to {dump_path} ({vis_frames.shape[0]} frames)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*The `srun` command is available.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", "-r", type=str, required=True)
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to speaker 0 audio file")
    parser.add_argument("--audio2", type=str, default=None, help="Path to speaker 1 audio file (optional)")
    parser.add_argument("--dump_dir", "-d", type=str, default="./render_results")
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--cfg", default=1.5, type=float)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    torch.set_float32_matmul_precision("high")
    infer_audio(args.resume_path, args.audio, args.audio2, args.dump_dir, args.tau, args.cfg)
