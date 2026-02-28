#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from core.models.modules.rope import RotaryPositionalEmbeddings


class MixedARTalkDecoder(nn.Module):
    def __init__(self, embed_dim, audio_dim, num_heads, depth, patch_nums):
        super(MixedARTalkDecoder, self).__init__()
        self.attn_depth = depth
        self.patch_nums = patch_nums
        self.msblock_len = sum(patch_nums)
        self.one_patch_len = max(patch_nums)
        drop_rate = [x.item() for x in torch.linspace(0, 0.1 * depth / 24, depth)]
        self.attn_blocks = nn.ModuleList(
            [
                SelfCrossAttn(
                    embed_dim=embed_dim,
                    audio_dim=audio_dim,
                    num_heads=num_heads,
                    drop_path=drop_rate[depth_idx],
                )
                for depth_idx in range(depth)
            ]
        )
        self.lvl_embed = nn.Embedding(len(patch_nums), embed_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / embed_dim / 3))
        self.style_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.style_pos, mean=0, std=math.sqrt(1 / embed_dim / 3))
        self.prev_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.prev_pos, mean=0, std=math.sqrt(1 / embed_dim / 3))
        attn_masking, attn_rope_pos, lvl_idx = self.build_attn_mask(patch_nums)
        self.self_pos = nn.Parameter(torch.zeros(1, 1, audio_dim))
        nn.init.trunc_normal_(self.self_pos, mean=0, std=math.sqrt(1 / audio_dim / 3))
        self.other_pos = nn.Parameter(torch.zeros(1, 1, audio_dim))
        nn.init.trunc_normal_(self.other_pos, mean=0, std=math.sqrt(1 / audio_dim / 3))
        self.register_buffer("attn_masking", attn_masking, persistent=False)
        self.register_buffer("attn_rope_pos", attn_rope_pos, persistent=False)
        self.register_buffer("lvl_idx", lvl_idx, persistent=False)

    def forward(self, feat, audio_feat_0, audio_feat_1, prev_feat, style_feat):
        batch, curr_seq_len, _ = feat.shape
        assert prev_feat.shape[1] % self.one_patch_len == 0
        assert style_feat.shape[1] % self.one_patch_len == 0
        prev_num = prev_feat.shape[1] // self.one_patch_len
        style_num = style_feat.shape[1] // self.one_patch_len
        cond_num = prev_num + style_num
        full_curr_len = cond_num * self.one_patch_len + curr_seq_len
        self_attn_bias, attn_rope_pos = self.expand_attn_mask(cond_num)
        audio_rope_pos = [max(self.patch_nums), cond_num * self.one_patch_len]

        feat = feat + self.lvl_embed(self.lvl_idx)[:, :curr_seq_len]
        prev_feat = prev_feat + self.prev_pos
        style_feat = style_feat + self.style_pos
        attn_feat = torch.cat([style_feat, prev_feat, feat], dim=1)
        attn_rope_pos = attn_rope_pos[:full_curr_len]
        self_attn_bias = self_attn_bias[:, :, :full_curr_len, :full_curr_len]
        audio_feat_0 = audio_feat_0 + self.self_pos
        audio_feat_1 = audio_feat_1 + self.other_pos
        for bidx in range(self.attn_depth):
            attn_feat = self.attn_blocks[bidx](
                attn_feat,
                audio_feat_0,
                audio_feat_1,
                attn_rope_pos=attn_rope_pos,
                audio_rope_pos=audio_rope_pos,
                self_attn_bias=self_attn_bias,
            )
        curr_feat = attn_feat[:, -curr_seq_len:, :]
        return curr_feat

    @torch.no_grad()
    def build_attn_mask(self, patch_nums):
        L = sum(patch_nums)
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(patch_nums)]).view(1, L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_idx = dT[:, 0].contiguous()
        attn_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(1, 1, L, L).contiguous()
        patch_len = max(patch_nums)
        patchs_ids = [torch.linspace(0, patch_len - 1, pn * 2 + 1)[1::2].round().long() for pn in patch_nums]
        rope_pos = torch.cat(patchs_ids, dim=0)
        return attn_masking, rope_pos, lvl_idx

    def expand_attn_mask(self, num_style_prev=2):
        attn_masking = self.attn_masking
        rope_pos = self.attn_rope_pos

        L = self.one_patch_len * num_style_prev + self.msblock_len
        prev_L = self.one_patch_len * num_style_prev
        idx = torch.arange(L)
        idx = idx.view(1, L, 1)
        idxT = idx.transpose(1, 2).clone()
        idx[:, :prev_L] = prev_L - 1
        exp_attn_mask = torch.where(idx >= idxT, 0.0, -torch.inf).reshape(1, 1, L, L)
        exp_attn_mask = exp_attn_mask.to(attn_masking.device)
        exp_attn_mask[:, :, -self.msblock_len :, -self.msblock_len :] = attn_masking

        prev_rope_pos = torch.arange(0, self.one_patch_len * num_style_prev)
        prev_rope_pos = prev_rope_pos.to(rope_pos.device)
        rope_pos = torch.cat([prev_rope_pos, rope_pos + self.one_patch_len * num_style_prev])
        return exp_attn_mask, rope_pos


class SelfCrossAttn(nn.Module):
    def __init__(self, embed_dim, audio_dim, num_heads, drop_path=0.0):
        super(SelfCrossAttn, self).__init__()
        hidden_features = round(embed_dim * 4.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.self_attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.self_audio_attn = CrossAttention(embed_dim=embed_dim, context_embed_dim=audio_dim, num_heads=num_heads)
        self.other_audio_attn = CrossAttention(embed_dim=embed_dim, context_embed_dim=audio_dim, num_heads=num_heads)
        self.ffn = torch.nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=True, eps=1e-6),
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_features, embed_dim),
        )

    def forward(self, feat, audio_feat_0, audio_feat_1, attn_rope_pos, audio_rope_pos, self_attn_bias):
        feat = feat + self.drop_path(self.self_attn(feat, attn_rope_pos, self_attn_bias))
        feat = feat + self.drop_path(self.self_audio_attn(feat, audio_feat_0, attn_rope_pos, audio_rope_pos))
        feat = feat + self.drop_path(self.other_audio_attn(feat, audio_feat_1, attn_rope_pos, audio_rope_pos))
        feat = feat + self.drop_path(self.ffn(feat))
        return feat


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.rearrange_qkv = Rearrange("b n (qkv h d) -> qkv b n h d", qkv=3, h=self.num_heads)
        self.rearrange_rope = Rearrange("b n h d -> b h n d")
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        # rotary positional embeddings
        self.rope = RotaryPositionalEmbeddings(dim=embed_dim // num_heads, max_seq_len=5000)
        # layer norm
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, x, rope_pos, attn_bias):
        _, curr_len, _ = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = self.rearrange_qkv(qkv).unbind(0)  # b n h d
        q = self.rearrange_rope(self.rope(q, input_pos=rope_pos))
        k = self.rearrange_rope(self.rope(k, input_pos=rope_pos))
        v = self.rearrange_rope(v)
        # compute attention
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_bias)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, context_embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.rearrange_qkv = Rearrange("b n (h d) -> b n h d", h=self.num_heads)
        self.rearrange_rope = Rearrange("b n h d -> b h n d")
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(context_embed_dim, embed_dim)
        self.v_proj = nn.Linear(context_embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # rotary positional embeddings
        self.rope = RotaryPositionalEmbeddings(dim=embed_dim // num_heads, max_seq_len=5000)
        # layer norm
        self.self_norm = nn.LayerNorm(embed_dim, elementwise_affine=True, eps=1e-6)
        self.context_norm = nn.LayerNorm(context_embed_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, x, context, rope_pos, context_rope_pos):
        _, curr_len, _ = x.shape
        x, context = self.self_norm(x), self.context_norm(context)
        patch_len, patch_offsets = context_rope_pos
        context_rope_pos = torch.linspace(0, patch_len, steps=(context.shape[1] + 1))[:-1].long() + patch_offsets
        q = self.rearrange_qkv(self.q_proj(x))
        k = self.rearrange_qkv(self.k_proj(context))
        v = self.rearrange_qkv(self.v_proj(context))
        q = self.rearrange_rope(self.rope(q, input_pos=rope_pos))
        k = self.rearrange_rope(self.rope(k, input_pos=context_rope_pos))
        v = self.rearrange_rope(v)

        # scaled_dot_product_attention inputs shape (B, Hq, L, c)
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v)
        out = self.rearrange_out(out)
        out = self.out_proj(out)
        return out


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
