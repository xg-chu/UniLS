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
            meta_cfg = self._extract_meta_cfg(full_checkpoint)
            if meta_cfg is None:
                raise KeyError(
                    "Cannot find model config in checkpoint. "
                    "Please pass `meta_cfg` to InferEngine or save `meta_cfg` in checkpoint."
                )
        self.meta_cfg = meta_cfg

        # build model
        model = build_model(meta_cfg.MODEL, init_submodule=False).to(device)
        model_state = self._extract_model_state(full_checkpoint)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
        if len(missing) > 0:
            missing = list(set([m.split(".")[0] for m in missing]))
            print(f"Missing keys: {missing}")
        model.eval()
        self.model = model

    @staticmethod
    def _extract_meta_cfg(full_checkpoint):
        if not isinstance(full_checkpoint, dict):
            return None
        for key in ("meta_cfg", "config", "cfg"):
            if key in full_checkpoint and isinstance(full_checkpoint[key], dict):
                return ConfigDictWrapper(full_checkpoint[key])
        # some pipelines keep config under hyper_parameters
        hparams = full_checkpoint.get("hyper_parameters", {})
        if isinstance(hparams, dict):
            for key in ("meta_cfg", "config", "cfg"):
                if key in hparams and isinstance(hparams[key], dict):
                    return ConfigDictWrapper(hparams[key])
        return None

    @staticmethod
    def _extract_model_state(full_checkpoint):
        if not isinstance(full_checkpoint, dict):
            raise TypeError("Checkpoint must be a dict-like object.")

        state = None
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in full_checkpoint and isinstance(full_checkpoint[key], dict):
                state = full_checkpoint[key]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in full_checkpoint.values()):
            state = full_checkpoint
        if state is None:
            raise KeyError(
                "Cannot find model weights in checkpoint. Expected keys: model/state_dict/model_state_dict/net."
            )

        # Try to normalize common training wrappers: module.*, model.*, net.*
        strip_prefixes = ("module.", "model.", "net.")
        normalized_state = {}
        for key, value in state.items():
            new_key = key
            for prefix in strip_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            normalized_state[new_key] = value
        return normalized_state

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
        assert gt_motion_code.shape[0] == 1 and gt_motion_code.shape[1] == 2, "gt_motion_code [1, 2, len, 108]"
        assert pred_motion_code.shape[0] == 1, "pred_motion_code [1, len, 108]"
        gt_motion_code = gt_motion_code[:, 0]
        # val_metrics
        gt_verts = self.face_decoder.get_flame_verts(gt_motion_code, with_headpose=False)
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code, with_headpose=False)
        lve, mhd, fdd = calc_val_metrics(pred_verts, gt_verts)
        _, gt_gpose_code, gt_jaw_code, _ = gt_motion_code.split([100, 3, 1, 4], dim=-1)
        _, pred_gpose_code, pred_jaw_code, _ = pred_motion_code.split([100, 3, 1, 4], dim=-1)
        pdd = self._calc_code_dispersion_delta(pred_gpose_code, gt_gpose_code)
        jdd = self._calc_code_dispersion_delta(pred_jaw_code, gt_jaw_code)
        return {"LVE": lve, "MHD": mhd, "FDD": fdd, "PDD": pdd, "JDD": jdd}

    @staticmethod
    def _calc_code_dispersion_delta(pred_code, gt_code):
        # Use the same formula as FDD, applied on selected motion-code channels.
        std_pred = pred_code.std(dim=1)
        std_gt = gt_code.std(dim=1)
        return (std_pred - std_gt).abs().mean().detach() * 100
