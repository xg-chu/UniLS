#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from core.models.modules.rope import RotaryPositionalEmbeddings


class TransformerEncoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, code_dim, depth=6, n_heads=8):
        super().__init__()
        self.inp_mapping = nn.Sequential(nn.Linear(inp_dim, hidden_dim), nn.LeakyReLU(0.2, True))
        self.code_mapping = nn.Linear(hidden_dim, code_dim)
        # transformer
        blocks = []
        for i in range(depth):
            blocks += [
                SimpleSelfAttention(hidden_dim, n_heads=n_heads),
                torch.nn.Sequential(
                    nn.Linear(hidden_dim, int(1.5 * hidden_dim)),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(int(1.5 * hidden_dim), hidden_dim),
                ),
            ]
        self.encoder_transformer = nn.ModuleList(blocks)

    def forward(self, inp_BLC, attn_mask=None):
        feat = self.inp_mapping(inp_BLC)
        for block in self.encoder_transformer:
            if isinstance(block, SimpleSelfAttention):
                feat = feat + block(feat, attn_mask)
            else:
                feat = feat + block(feat)
        out = self.code_mapping(feat)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, code_dim, hidden_dim, out_dim, depth=6, n_heads=8):
        super().__init__()
        self.inp_mapping = nn.Sequential(nn.Linear(code_dim, hidden_dim), nn.LeakyReLU(0.2, True))
        self.out_mapping = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.out_mapping.weight, gain=0.05)
        nn.init.constant_(self.out_mapping.bias, 0)
        # transformer
        blocks = []
        for i in range(depth):
            blocks += [
                SimpleSelfAttention(hidden_dim, n_heads=n_heads),
                torch.nn.Sequential(
                    nn.Linear(hidden_dim, int(1.5 * hidden_dim)),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(int(1.5 * hidden_dim), hidden_dim),
                ),
            ]
        self.decoder_transformer = nn.ModuleList(blocks)

    def forward(self, inp_BLC, attn_mask=None):
        feat = self.inp_mapping(inp_BLC)
        for block in self.decoder_transformer:
            if isinstance(block, SimpleSelfAttention):
                feat = feat + block(feat, attn_mask)
            else:
                feat = feat + block(feat)
        out = self.out_mapping(feat)
        return out


class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.rearrange_qkv = Rearrange("b n (qkv h d) -> qkv b n h d", qkv=3, h=self.n_heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        # rotary positional embeddings
        self.rope = RotaryPositionalEmbeddings(dim=hidden_dim // n_heads, max_seq_len=300)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        qkv = self.to_qkv(self.norm(x))  # [B, L, C]
        q, k, v = self.rearrange_qkv(qkv).unbind(0)  # [B, L, C] -> [B, L, H, c]
        q = self.rope(q).permute(0, 2, 1, 3)  # rope inputs shape (B, L, Hq, c)
        k = self.rope(k).permute(0, 2, 1, 3)  # rope inputs shape (B, L, Hk, c)
        v = v.permute(0, 2, 1, 3)
        # compute attention
        # scaled_dot_product_attention inputs shape (B, Hq, L, c)
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, dropout_p=0.0
        )
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return out
