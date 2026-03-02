#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os

import torch
from scipy.signal import savgol_filter

from core.libs.flame_model import FLAMEModel, RenderMesh
from core.libs.utils import ConfigDictWrapper, find_latest_model
from core.libs.utils_videos import write_video
from core.models import build_model
from core.models.modules import calc_val_metrics


class InferEngine:
    def __init__(self, resume_path, device="cuda:0", meta_cfg=None):
        # build config
        if os.path.isdir(resume_path):
            resume_path = find_latest_model(os.path.join(resume_path, "checkpoints"))
        print(f"Loading model from {resume_path}...")
        self.device = device

        # load checkpoint
        full_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if meta_cfg is None:
            meta_cfg = ConfigDictWrapper(full_checkpoint["meta_cfg"])
        self.meta_cfg = meta_cfg

        # build model
        model = build_model(meta_cfg.MODEL, init_submodule=False).to(device)
        missing, unexpected = model.load_state_dict(full_checkpoint["model"], strict=False)
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
        if len(missing) > 0:
            missing = list(set([m.split(".")[0] for m in missing]))
            print(f"Missing keys: {missing}")
        model.eval()
        self.model = model

    def _init_face_decoder(self):
        # build face decoder
        self.face_decoder = FLAMEModel(n_shape=300, n_exp=100)
        self.face_decoder.eval()
        self.face_decoder.to(self.device)
        self.face_renderer = RenderMesh(image_size=512, faces=self.face_decoder.get_faces())

    @torch.inference_mode()
    def inference(self, batch_data, dump_path=None, clip_length=20, tau=0.01, cfg=1.0):
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(self.device)
        # inference
        infer_results = self.model.inference(**batch_data, tau=tau, cfg=cfg)
        # print("pred motion code shape: ", infer_results["pred_motion_code"][0].shape)
        if dump_path is not None:
            pred_motion_code = infer_results["pred_motion_code"][0]
            pred_motion_code = self.smooth_motion_savgol(pred_motion_code)
            if "gt_motion_code" in infer_results.keys():
                gt_motion_code = infer_results["gt_motion_code"][0]
            else:
                gt_motion_code = None
            vis_audios = infer_results["audio"][0]
            self.visualize(
                pred_motion_code,
                dump_path=dump_path,
                vis_audios=vis_audios,
                gt_motion_code=gt_motion_code,
                render_length=clip_length,
                fps=25,
                sample_rate=16000,
            )
        return infer_results

    @torch.inference_mode()
    def visualize(
        self,
        pred_motion_code,
        dump_path,
        vis_audios=None,
        gt_motion_code=None,
        render_length=20,
        fps=30,
        sample_rate=16000,
    ):
        if not hasattr(self, "face_decoder"):
            self._init_face_decoder()
        assert pred_motion_code.dim() == 2, "pred_motion_code should be 2D"
        frame_length = int(render_length * fps)
        audio_length = int(render_length * sample_rate)
        pred_motion_code = pred_motion_code[:frame_length][None]
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code)[0]
        vis_frames = []
        for pidx, pred_vert in enumerate(pred_verts):
            vis_frame, _ = self.face_renderer(pred_vert[None], colors=self.face_decoder.get_colors())
            vis_frames.append(vis_frame)
        vis_frames = torch.cat(vis_frames, dim=0).cpu()
        if gt_motion_code is not None:
            gt_verts = self.face_decoder.get_flame_verts(gt_motion_code[None])[0]
            gt_frames = []
            for gidx, gt_vert in enumerate(gt_verts):
                gt_frame, _ = self.face_renderer(gt_vert[None], colors=self.face_decoder.get_colors())
                gt_frames.append(gt_frame)
            gt_frames = torch.cat(gt_frames, dim=0).cpu()
            vis_frames = torch.cat([gt_frames, vis_frames], dim=-1)
        if vis_audios is not None:
            assert vis_audios.dim() == 1, "vis_audios should be 1D"
            vis_audios = vis_audios[:audio_length]
            write_video(vis_frames, dump_path, fps, vis_audios, sample_rate, "aac")
        else:
            write_video(vis_frames, dump_path, fps)

    @staticmethod
    def smooth_motion_savgol(motion_code, window_length=9, polyorder=3):
        motion_np = motion_code.clone().detach().cpu().numpy()
        motion_smoothed = savgol_filter(motion_np, window_length=window_length, polyorder=polyorder, axis=0)
        motion_smoothed = torch.from_numpy(motion_smoothed).type_as(motion_code)
        return motion_smoothed

    @torch.inference_mode()
    def _calc_metrics(self, infer_results, batch_data):
        if not hasattr(self, "face_decoder"):
            self._init_face_decoder()
        gt_motion_code = batch_data["motion_code"]
        pred_motion_code = infer_results["pred_motion_code"]
        # val_metrics
        gt_verts = self.face_decoder.get_flame_verts(gt_motion_code, with_headpose=False)
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code, with_headpose=False)
        lve, avd, fdd = calc_val_metrics(pred_verts, gt_verts)
        return {"LVE": lve, "AVD": avd, "FDD": fdd}
