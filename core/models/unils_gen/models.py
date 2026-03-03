#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import random

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from core.models.modules import MimiModelWrapper, Wav2Vec2Model
from core.models.unils_codec import UniLSCodec

from .transformer import MixedARTalkDecoder


class UniLSGen(nn.Module):
    def __init__(self, model_cfg=None, init_submodule=True, **kwargs):
        super().__init__()
        self._motion_fps = 25
        self._sample_rate = 16000

        # build basic vae
        base_codec = UniLSCodec(model_cfg.VAE_CONFIG, init_submodule=False)
        if init_submodule:
            vae_ckpt = torch.load(model_cfg.VAE_CONFIG.VAE_PATH, map_location="cpu", weights_only=True)
            print("Loading vae from {}...".format(model_cfg.VAE_CONFIG.VAE_PATH))
            base_codec.load_state_dict(vae_ckpt["model"], strict=True)
        base_codec.eval()
        for param in base_codec.parameters():
            param.requires_grad = False

        # build audio encoder
        if model_cfg.AUDIO_ENCODER == "wav2vec":
            audio_dim = 1024
            audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        elif model_cfg.AUDIO_ENCODER == "mimi":
            audio_dim = 512
            audio_encoder = MimiModelWrapper("kyutai/mimi", default_sample_rate=self._sample_rate)
        else:
            raise ValueError(f"Invalid audio encoder: {model_cfg.AUDIO_ENCODER}")
        audio_encoder.eval()
        for param in audio_encoder.parameters():
            param.requires_grad = False

        # autoregressive generator
        self.base_codec = base_codec
        self.audio_encoder = audio_encoder
        self.patch_nums = self.base_codec.patch_nums
        self.motion_dim = self.base_codec.motion_dim
        self.attn_dim = model_cfg.T_EMBED_DIM
        self.attn_head = model_cfg.T_NUM_HEAD
        self.attn_depth = model_cfg.T_DEPTH

        self.code_token_embed = nn.Linear(self.base_codec.code_dim, self.attn_dim)
        self.attn_blocks = MixedARTalkDecoder(
            embed_dim=self.attn_dim,
            audio_dim=audio_dim,
            num_heads=self.attn_head,
            depth=self.attn_depth,
            patch_nums=self.patch_nums,
        )
        self.logits_head = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, self.base_codec.code_dim * 2),
        )

        # classifier-free training
        self.flip_quant_ratio = model_cfg.FLIP_QUANT
        self.prev_free_ratio = model_cfg.PREV_FREE
        self.style_free_ratio = model_cfg.STYLE_FREE
        self.audio_free_ratio = model_cfg.AUDIO_FREE

        # learnable embeddings
        self.sos_embed = nn.Parameter(torch.zeros(1, 1, self.attn_dim))
        nn.init.trunc_normal_(self.sos_embed, mean=0, std=math.sqrt(1 / self.attn_dim / 3))

        # load pretrained model
        if model_cfg.PRETRAIN_PATH is not None:
            ckpt = torch.load(model_cfg.PRETRAIN_PATH, map_location="cpu", weights_only=True)
            print("Loading pretrain from {}...".format(model_cfg.PRETRAIN_PATH))
            missing, unexpected = self.load_state_dict(ckpt["model"], strict=False)
            for name, param in self.named_parameters():
                if name in ckpt["model"]:
                    param.requires_grad = False
            if len(missing) > 0:
                missing = [i for i in missing if "audio_encoder" not in i]
                missing = [i for i in missing if "other_audio_attn" not in i]
                missing = [i for i in missing if "self_audio_attn" not in i]
                missing = [i for i in missing if "self_pos" not in i]
                missing = [i for i in missing if "other_pos" not in i]
                assert len(missing) == 0, f"Missing keys: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
            print("Pretrained model loaded successfully.")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch, training=True):
        # audio / motion
        batch_size = batch["audio"].shape[0] * 2
        audio_chunk = batch["audio"]
        motion_code = batch["motion_code"]
        prev_motion_code = batch["prev_motion_code"]
        style_motion_code = batch["style_motion_code"]
        assert audio_chunk.dim() == 3 and motion_code.dim() == 4

        audio_chunk_0, audio_chunk_1 = audio_chunk.unbind(dim=1)
        raw_audio_feat_0 = self.audio_encoder(audio_chunk_0, self._sample_rate)
        raw_audio_feat_1 = self.audio_encoder(audio_chunk_1, self._sample_rate)
        audio_feat_0 = torch.cat([raw_audio_feat_0, raw_audio_feat_1], dim=0)
        audio_feat_1 = torch.cat([raw_audio_feat_1, raw_audio_feat_0], dim=0)
        motion_code = torch.cat(motion_code.unbind(dim=1), dim=0)
        prev_motion_code = torch.cat(prev_motion_code.unbind(dim=1), dim=0)
        style_motion_code = torch.cat(style_motion_code.unbind(dim=1), dim=0)

        # mask conditions for classifier-free training
        m_audio_0 = torch.rand(batch_size, 1, 1).to(self.device) < self.audio_free_ratio
        m_audio_1 = torch.rand(batch_size, 1, 1).to(self.device) < self.audio_free_ratio
        m_style = torch.rand(batch_size, 1, 1).to(self.device) < self.style_free_ratio
        m_prev = torch.rand(batch_size, 1, 1).to(self.device) < self.prev_free_ratio
        zeros_audio_feat = torch.zeros_like(audio_feat_0)
        zeros_style_code = torch.zeros_like(style_motion_code)
        zeros_prev_code = torch.zeros_like(prev_motion_code)
        audio_feat_0 = torch.where(m_audio_0, zeros_audio_feat, audio_feat_0)
        audio_feat_1 = torch.where(m_audio_1, zeros_audio_feat, audio_feat_1)
        style_motion_code = torch.where(m_style, zeros_style_code, style_motion_code)
        prev_motion_code = torch.where(m_prev, zeros_prev_code, prev_motion_code)

        # gather prev and style features
        prev_motion_feat = self.get_motion_feat(prev_motion_code)
        style_motion_feat = self.get_motion_feat(style_motion_code)
        prev_feat = self.code_token_embed(prev_motion_feat)
        style_feat = self.code_token_embed(style_motion_feat)

        # generate
        sos_token = self.sos_embed.expand(batch_size, 1, -1)
        next_vqfeat, this_code_bits = self.base_codec.flip_quant_to_feat(
            motion_code, self.flip_quant_ratio, "accum_next"
        )
        attn_feat = torch.cat([sos_token, self.code_token_embed(next_vqfeat)], dim=1)
        attn_feat = self.attn_blocks(attn_feat, audio_feat_0, audio_feat_1, prev_feat, style_feat)
        pred_motion_logits = self.logits_head(attn_feat)

        return {
            "pred_motion_logits": pred_motion_logits,
            "gt_motion_bits": this_code_bits,
        }

    @torch.inference_mode()
    def inference(self, audio, style_motion_code=None, prev_motion_code=None, tau=1.0, cfg=2.0, **kwargs):
        batch_size = audio.shape[0]
        patch_len = max(self.patch_nums)
        assert batch_size == 1, "Only support batch size 1 for inference."
        if style_motion_code is not None:
            style_motion_code = style_motion_code.unbind(dim=1)[0]
        else:
            style_motion_code = audio.new_zeros(batch_size, patch_len, self.motion_dim)
            # prev_motion_code[..., 100:103] *= 0.1
        if prev_motion_code is None:
            prev_motion_code = audio.new_zeros(batch_size, patch_len, self.motion_dim)
        # print("Inference with tau {} and cfg {}".format(tau, cfg))

        # prepare audio and other inputs
        audio_frame_len = self._sample_rate / self._motion_fps
        frame_length = math.ceil(audio.shape[-1] / audio_frame_len)
        frame_chunk_length = math.ceil(frame_length / patch_len) * patch_len
        audio_chunk_size = int(patch_len * audio_frame_len)
        audio_chunk_length = int(frame_chunk_length * audio_frame_len)
        audio_0, audio_1 = audio.unbind(dim=1)
        audio_chunks_0 = torch.cat(
            [audio_0, audio_0.new_zeros(batch_size, audio_chunk_length - audio_0.shape[-1])], dim=-1
        ).split(audio_chunk_size, dim=-1)
        audio_chunks_1 = torch.cat(
            [audio_1, audio_1.new_zeros(batch_size, audio_chunk_length - audio_1.shape[-1])], dim=-1
        ).split(audio_chunk_size, dim=-1)
        audio_feats_0 = [self.audio_encoder(ac) for ac in audio_chunks_0]
        audio_feats_1 = [self.audio_encoder(ac) for ac in audio_chunks_1]

        # cfg-inf
        sos_token = self.sos_embed.expand(batch_size * 2, 1, -1)
        audio_uncond = audio_feats_0[0].new_zeros(audio_feats_0[0].shape)
        style_uncond = style_motion_code.new_zeros(style_motion_code.shape)
        prev_uncond = audio.new_zeros(batch_size, patch_len, self.motion_dim)
        prev_motion_code = torch.cat([prev_motion_code, prev_uncond], dim=0)
        audio_feats_0 = [torch.cat([ac, audio_uncond], dim=0) for ac in audio_feats_0]
        audio_feats_1 = [torch.cat([ac, audio_uncond], dim=0) for ac in audio_feats_1]
        style_motion_code = torch.cat([style_motion_code, style_uncond], dim=0)
        style_feat = self.code_token_embed(self.get_motion_feat(style_motion_code))
        # run audio chunks
        pred_motion_codes = []
        for chunk_idx in range(len(audio_feats_0)):
            # audio & style feature
            next_ar_vqfeat = sos_token
            curr_audio_feat_0 = audio_feats_0[chunk_idx]
            curr_audio_feat_1 = audio_feats_1[chunk_idx]
            prev_feat = self.code_token_embed(self.get_motion_feat(prev_motion_code))

            patch_motion_bits = []
            for pidx, pn in enumerate(self.patch_nums):
                attn_feat = self.attn_blocks(
                    next_ar_vqfeat, curr_audio_feat_0, curr_audio_feat_1, prev_feat, style_feat
                )
                motion_logits = self.logits_head(attn_feat)
                motion_logits = motion_logits[:, sum(self.patch_nums[:pidx]) :]
                motion_logits = motion_logits.mul(1 / tau)
                motion_logits = motion_logits.view(motion_logits.shape[0], motion_logits.shape[1], -1, 2)
                if cfg > 1.0:
                    motion_logits = cfg * motion_logits[:batch_size] + (1 - cfg) * motion_logits[batch_size:]
                else:
                    motion_logits = motion_logits[:batch_size]
                motion_bits = sample_idx_with_top_p_(motion_logits)
                patch_motion_bits.append(motion_bits)
                # gready_motion_bits = pred_motion_logits.argmax(dim=-1)
                # flip_ratio = (
                #     gready_motion_bits != patch_motion_bits
                # ).sum() / patch_motion_bits.numel()
                # print(f"{pidx} flip_ratio: {flip_ratio}")
                if pidx < len(self.patch_nums) - 1:
                    next_ar_vqfeat = self.base_codec.vqidx_to_next_feat(
                        torch.cat(patch_motion_bits, dim=1), pidx, "accum_next"
                    )
                    next_ar_vqfeat = self.code_token_embed(next_ar_vqfeat)
                    next_ar_vqfeat = torch.cat([next_ar_vqfeat, next_ar_vqfeat], dim=0)
                    next_ar_vqfeat = torch.cat([sos_token, next_ar_vqfeat], dim=1)
            patch_motion_bits = torch.cat(patch_motion_bits, dim=1)
            pred_motion_code = self.base_codec.vqidx_to_motion(patch_motion_bits)
            pred_motion_codes.append(pred_motion_code)
            prev_motion_code = torch.cat([pred_motion_code, pred_motion_code], dim=0)
        pred_motion_codes = torch.cat(pred_motion_codes, dim=1)[:, :frame_length]
        results = {"audio": audio_0 + audio_1, "pred_motion_code": pred_motion_codes}
        if "motion_code" in kwargs:
            gt_motion_code = kwargs["motion_code"][:, 0]
            assert pred_motion_codes.shape[1] == gt_motion_code.shape[1]
            results["gt_motion_code"] = gt_motion_code
        return results

    def _calc_losses(self, train_results, _loss_kwargs):
        if train_results["pred_motion_logits"].dim() == 3:
            B, L, _ = train_results["pred_motion_logits"].shape
            gt_motion_bits = train_results["gt_motion_bits"]
            pred_motion_logits = train_results["pred_motion_logits"]
            pred_motion_logits = pred_motion_logits.view(B, L, -1, 2)
            ce_loss = torch.nn.functional.cross_entropy(
                pred_motion_logits.permute(0, 3, 1, 2), gt_motion_bits.type(torch.long)
            )
        elif train_results["pred_motion_logits"].dim() == 4:
            b, n, lv, _ = train_results["pred_motion_logits"].shape
            gt_motion_bits = train_results["gt_motion_bits"]
            pred_motion_logits = train_results["pred_motion_logits"]
            pred_motion_logits = pred_motion_logits.view(b, n, lv, -1, 2)
            ce_loss = torch.nn.functional.cross_entropy(
                pred_motion_logits.permute(0, 4, 1, 2, 3),
                gt_motion_bits.type(torch.long),
            )
        loss = {
            "ce_loss": ce_loss * _loss_kwargs.CE_WEIGHT,
        }
        # print({k: "{:.4f}".format(v.item()) for k, v in loss.items()})
        return loss

    def configure_optimizers(self, config):
        learning_rate = config.LEARNING_RATE
        # print("Learning rate: {}".format(learning_rate))
        # params
        normal_params, decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)
        # optimizer
        optimizer = torch.optim.AdamW(
            [
                {"params": normal_params, "lr": learning_rate},
                {"params": decay_params, "lr": learning_rate * 0.1},
            ],
            lr=learning_rate,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_ITER,
        )
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.LR_DECAY_RATE,
            total_iters=(config.LR_DECAY_ITER - config.WARMUP_ITER),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.WARMUP_ITER],
        )
        return optimizer, scheduler

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.base_codec.eval()
        self.audio_encoder.eval()
        return self

    @torch.no_grad()
    def get_motion_cond_feat(self, motion_code, flip_cond_ratio=0.0):
        assert motion_code.shape[1] % self.patch_nums[-1] == 0, "motion_code length must be divisible by patch_nums[-1]"
        motion_codes = motion_code.split(self.patch_nums[-1], dim=1)
        curr_vqfeats = []
        for motion_code in motion_codes:
            curr_vqfeat, _ = self.base_codec.flip_quant_to_feat(motion_code, flip_cond_ratio, "accum_curr")
            curr_vqfeats.append(curr_vqfeat)
        curr_vqfeats = torch.cat(curr_vqfeats, dim=1)
        return curr_vqfeats.detach()

    @torch.no_grad()
    def get_motion_feat(self, motion_code):
        assert motion_code.shape[1] % self.patch_nums[-1] == 0, "motion_code length must be divisible by patch_nums[-1]"
        motion_codes = motion_code.split(self.patch_nums[-1], dim=1)
        curr_vqfeats = []
        for motion_code in motion_codes:
            curr_vqfeat = self.base_codec.quant_to_sum_feat(motion_code)
            curr_vqfeats.append(curr_vqfeat)
        curr_vqfeats = torch.cat(curr_vqfeats, dim=1)
        return curr_vqfeats.detach()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state.keys()):
            if key.startswith("audio_encoder"):
                state.pop(key)
        return state


def sample_idx_with_top_p_(logits_BLCV, top_p=0.97):
    B, L, C, V = logits_BLCV.shape
    logits_BLV = logits_BLCV.view(B, -1, V)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BLV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BLV.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove),
            -torch.inf,
        )
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    sampled_idx = torch.multinomial(logits_BLV.softmax(dim=-1).view(-1, V), num_samples=1, replacement=True)
    sampled_idx = sampled_idx.reshape(B, L, C, 1)
    return sampled_idx[..., 0]
