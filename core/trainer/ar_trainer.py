#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os

import torch

from core.libs.flame_model import FLAMEModel, RenderMesh
from core.libs.utils_videos import write_video
from core.models.modules import calc_val_metrics

from .base_trainer import BaseTrainer


class ARTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calc_losses = self.accelerator.unwrap_model(self.model)._calc_losses

    def _init_face_decoder(self, device):
        self.face_decoder = FLAMEModel(n_shape=300, n_exp=100)
        self.face_decoder.eval()
        self.face_decoder.to(device)
        self.face_renderer = RenderMesh(image_size=512, faces=self.face_decoder.get_faces())
        self._all_visualize_frames = []
        self._all_visualize_audios = []

    def _calc_metrics(self, infer_results, visualize=None):
        if not hasattr(self, "face_decoder"):
            self._init_face_decoder(infer_results["pred_motion_code"].device)
        gt_motion_code = infer_results["gt_motion_code"]
        pred_motion_code = infer_results["pred_motion_code"]
        # val_metrics
        gt_verts = self.face_decoder.get_flame_verts(gt_motion_code, with_headpose=False)
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code, with_headpose=False)
        lve, avd, fdd = calc_val_metrics(pred_verts, gt_verts)
        if visualize is not None:
            iter_idx = visualize["iter_idx"]
            render_length = visualize["render_length"]
            curr_audio = infer_results["audio"]
            gt_motion_code = gt_motion_code[:1, :render_length]
            pred_motion_code = pred_motion_code[:1, :render_length]
            gt_verts = self.face_decoder.get_flame_verts(gt_motion_code)[0]
            pred_verts = self.face_decoder.get_flame_verts(pred_motion_code)[0]
            gt_images, _ = self.face_renderer(gt_verts, colors=self.face_decoder.get_colors())
            pred_images, _ = self.face_renderer(pred_verts, colors=self.face_decoder.get_colors())
            vis_frames = torch.cat([gt_images, pred_images], dim=-1).to(torch.uint8)
            # dump to video
            fps = int(self._meta_cfg.DATASET.MOTION_FPS)
            sample_rate = int(self._meta_cfg.DATASET.AUDIO_SAMPLE_RATE)
            vis_path = os.path.join(self._dump_dir, "examples", f"{iter_idx}.mp4")
            vis_audios = curr_audio[0, : int(render_length * sample_rate / fps)]
            write_video(vis_frames, vis_path, fps, vis_audios, sample_rate, "aac")
            self.wandb_logger.log_video(vis_path, "examples", step=iter_idx)
        return {"LVE": lve, "AVD": avd, "FDD": fdd}
