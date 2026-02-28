"""
Binary Spherical Quantization
Proposed in https://arxiv.org/abs/2406.07548

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBSQ(nn.Module):
    def __init__(self, codebook_dim=32, scale_schedule=None):
        super().__init__()
        # codebook size -> 2 ** codebook_dim
        self.codebook_dim = codebook_dim
        self.scale_lvls = len(scale_schedule)
        self.scale_schedule = scale_schedule
        self.bsq_quant = BSQ(codebook_dim=codebook_dim)

    def forward(self, f_BTC):
        B, T, C = f_BTC.size()
        quantized_out, residual = 0.0, f_BTC
        all_losses, all_bit_indices = [], []
        for lvl_idx, pt in enumerate(self.scale_schedule):
            interpolate_residual = (
                F.interpolate(residual.permute(0, 2, 1), size=(pt), mode="area").permute(0, 2, 1)
                if pt != T
                else residual
            )
            quantized, bit_indices, loss = self.bsq_quant(interpolate_residual)
            quantized = (
                F.interpolate(quantized.permute(0, 2, 1), size=(T), mode="linear").permute(0, 2, 1)
                if pt != T
                else quantized
            )
            residual = residual - quantized.detach()  # remove_residual_detach = False
            quantized_out = quantized_out + quantized
            all_bit_indices.append(bit_indices)
            all_losses.append(loss)
        # stack all losses and indices
        all_losses = torch.stack(all_losses, dim=-1)
        all_bit_indices = torch.cat(all_bit_indices, dim=1)
        return quantized_out, all_bit_indices, all_losses

    @torch.no_grad()
    def feat_to_vqidx(self, f_BTC):
        B, T, C = f_BTC.size()
        residual = f_BTC
        all_bit_indices = []
        for lvl_idx, pt in enumerate(self.scale_schedule):
            interpolate_residual = (
                F.interpolate(residual.permute(0, 2, 1), size=(pt), mode="area").permute(0, 2, 1)
                if pt != T
                else residual
            )
            quantized, bit_indices, _ = self.bsq_quant(interpolate_residual)
            quantized = (
                F.interpolate(quantized.permute(0, 2, 1), size=(T), mode="linear").permute(0, 2, 1)
                if pt != T
                else quantized
            )
            residual = residual - quantized.detach()  # remove_residual_detach = False
            all_bit_indices.append(bit_indices)
        all_bit_indices = torch.cat(all_bit_indices, dim=1)
        return all_bit_indices

    @torch.no_grad()
    def vqidx_to_feat(self, bit_indices):
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        curr_feat_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)
        accum_feat_BTC = bit_indices.new_zeros(B, T, C, dtype=torch.float32)
        pn_start, pn_next = 0, self.scale_schedule[0]
        for pidx in range(self.scale_lvls - 1):
            residual = F.interpolate(
                curr_feat_BTC[:, pn_start:pn_next].permute(0, 2, 1),
                size=(T),
                mode="linear",
            ).permute(0, 2, 1)
            accum_feat_BTC.add_(residual)
            pn_start = pn_next
            pn_next = pn_next + self.scale_schedule[pidx + 1]
        accum_feat_BTC.add_(curr_feat_BTC[:, pn_start:])
        return accum_feat_BTC

    # for training of ms-gen model
    @torch.no_grad()
    def flip_feat_to_vqidx(self, f_BTC, flip_ratio=0.3):
        B, T, C = f_BTC.size()
        residual = f_BTC
        flip_vqidx, real_vqidx = [], []
        for lvl_idx, pt in enumerate(self.scale_schedule):
            interpolate_residual = (
                F.interpolate(residual.permute(0, 2, 1), size=(pt), mode="area").permute(0, 2, 1)
                if pt != T
                else residual
            )
            _, bit_indices, _ = self.bsq_quant(interpolate_residual)
            real_vqidx.append(bit_indices)

            # run flip and get new quantized
            if lvl_idx < self.scale_lvls - 1 and flip_ratio > 0.0:
                real_flip_ratio = np.random.randint(0, 100 * flip_ratio + 1) * 0.01
                mask_flip = torch.rand(bit_indices.shape) < real_flip_ratio
                mask_flip = mask_flip.to(bit_indices.device)
                flip_bit_indices = bit_indices.clone()
                flip_bit_indices[mask_flip] = 1 - flip_bit_indices[mask_flip]
            else:
                flip_bit_indices = bit_indices.clone()
            flip_vqidx.append(flip_bit_indices)
            quantized = (flip_bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)

            quantized = (
                F.interpolate(quantized.permute(0, 2, 1), size=(T), mode="linear").permute(0, 2, 1)
                if pt != T
                else quantized
            )
            residual = residual - quantized.detach()  # remove_residual_detach = False
        real_vqidx = torch.cat(real_vqidx, dim=1)
        flip_vqidx = torch.cat(flip_vqidx, dim=1)
        return real_vqidx, flip_vqidx

    # for training of ms-gen model
    @torch.no_grad()
    def vqidx_to_local_next_feat(self, bit_indices, pidx=None):
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        curr_feat_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)
        next_feat_BTC = []
        pn_start, pn_next = 0, self.scale_schedule[0]
        end_pidx = pidx + 1 if pidx is not None else self.scale_lvls - 1
        for pidx in range(end_pidx):
            residual = F.interpolate(
                curr_feat_BTC[:, pn_start:pn_next].permute(0, 2, 1),
                size=(self.scale_schedule[pidx + 1]),
                mode="linear",
            ).permute(0, 2, 1)
            next_feat_BTC.append(residual)
            pn_start = pn_next
            pn_next = pn_next + self.scale_schedule[pidx + 1]
        next_feat_BTC = torch.cat(next_feat_BTC, dim=1).contiguous()
        return next_feat_BTC

    # for training of ms-gen model
    @torch.no_grad()
    def vqidx_to_accum_next_feat(self, bit_indices, pidx=None):
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        curr_feat_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)
        accum_feat_BTC = bit_indices.new_zeros(B, T, C, dtype=torch.float32)
        next_feat_BTC = []
        pn_start, pn_next = 0, self.scale_schedule[0]
        end_pidx = pidx + 1 if pidx is not None else self.scale_lvls - 1
        for pidx in range(end_pidx):
            if pn_start == curr_feat_BTC.shape[1]:
                break
            residual = F.interpolate(
                curr_feat_BTC[:, pn_start:pn_next].permute(0, 2, 1),
                size=(T),
                mode="linear",
            ).permute(0, 2, 1)
            accum_feat_BTC.add_(residual)
            next_feat_BTC.append(
                F.interpolate(
                    accum_feat_BTC.permute(0, 2, 1),
                    size=(self.scale_schedule[pidx + 1]),
                    mode="area",
                ).permute(0, 2, 1)
            )
            pn_start = pn_next
            pn_next = pn_next + self.scale_schedule[pidx + 1]
        next_feat_BTC = torch.cat(next_feat_BTC, dim=1).contiguous()
        return next_feat_BTC

    # for training of ms-gen model
    @torch.no_grad()
    def vqidx_to_local_curr_feat(self, bit_indices):
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        curr_feat_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)
        return curr_feat_BTC

    # for training of ms-gen model
    @torch.no_grad()
    def vqidx_to_accum_curr_feat(self, bit_indices):
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        curr_feat_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)
        accum_feat_BTC = bit_indices.new_zeros(B, T, C, dtype=torch.float32)
        accum_curr_feat_BTC = []
        pn_start, pn_next = 0, self.scale_schedule[0]
        for pidx in range(self.scale_lvls - 1):
            residual = F.interpolate(
                curr_feat_BTC[:, pn_start:pn_next].permute(0, 2, 1),
                size=(T),
                mode="linear",
            ).permute(0, 2, 1)
            accum_feat_BTC.add_(residual)
            accum_curr_feat_BTC.append(
                F.interpolate(
                    accum_feat_BTC.permute(0, 2, 1),
                    size=(self.scale_schedule[pidx]),
                    mode="area",
                ).permute(0, 2, 1)
            )
            pn_start = pn_next
            pn_next = pn_next + self.scale_schedule[pidx + 1]
        accum_feat_BTC.add_(curr_feat_BTC[:, pn_start:])
        accum_curr_feat_BTC.append(accum_feat_BTC)
        accum_curr_feat_BTC = torch.cat(accum_curr_feat_BTC, dim=1).contiguous()
        return accum_curr_feat_BTC

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    @torch.no_grad()
    def vqidx_to_ar_vqfeat(self, this_pidx, bit_indices):  # only used in VAR inference
        B, T, C = bit_indices.shape[0], self.scale_schedule[-1], self.codebook_dim
        f_hat = bit_indices.new_zeros(B, C, T, dtype=torch.float32)
        ori_h_BTC = (bit_indices.float() * 2 - 1.0) / (self.codebook_dim**0.5)
        ori_h_BCT = ori_h_BTC.permute(0, 2, 1).contiguous()
        pn_start, pn_next = 0, self.scale_schedule[0]
        next_scales = []
        for pidx in range(this_pidx + 1):
            h_BCL = F.interpolate(ori_h_BCT[..., pn_start:pn_next], size=(T), mode="linear").contiguous()
            f_hat.add_(h_BCL)
            pn_start = pn_next
            pn_next = pn_next + self.scale_schedule[pidx + 1]
            next_scales.append(
                F.interpolate(f_hat.clone(), size=(self.scale_schedule[pidx + 1]), mode="area").contiguous()
            )
        return torch.cat(next_scales, dim=-1).permute(0, 2, 1).contiguous()


class BSQ(nn.Module):
    def __init__(self, codebook_dim=32):
        super().__init__()
        self.inv_temperature = 100.0
        self.commit_loss_weight = 0.2
        self.entropy_loss_weight = 0.1
        self.codebook_dim = codebook_dim

    def forward(self, f_BTC):
        f_BTC = F.normalize(f_BTC, dim=-1)
        # use straight-through gradients (optionally with custom activation fn) if training
        quantized = self.quantize(f_BTC)  # B, T, C
        # calculate loss
        persample_entropy, cb_entropy = self.soft_entropy_loss(f_BTC)
        entropy_penalty = (persample_entropy - cb_entropy) / self.inv_temperature
        commit_loss = torch.mean(((quantized.detach() - f_BTC) ** 2).sum(dim=-1))
        aux_loss = entropy_penalty * self.entropy_loss_weight + commit_loss * self.commit_loss_weight
        # gather the indices
        bit_indices = (quantized > 0).int()  # B, T, C
        return quantized, bit_indices, aux_loss

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dim, f"Expected {self.codebook_dim} dimensions, got {z.shape[-1]}"
        q_scale = 1.0 / (self.codebook_dim**0.5)
        zhat = torch.where(z > 0, torch.tensor(1).type_as(z), torch.tensor(-1).type_as(z))
        zhat = q_scale * zhat  # on unit sphere
        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        def get_entropy(count, dim=-1):
            H = -(count * torch.log(count + 1e-8)).sum(dim=dim)
            return H

        p = torch.sigmoid(-4 * z / (self.codebook_dim**0.5) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)  # (b, l, codebook_dim, 2)
        per_sample_entropy = get_entropy(prob, dim=-1).sum(dim=-1).mean()  # (b,l, codebook_dim)->(b,l)->scalar
        # macro average of the probability of each subgroup
        avg_prob = prob.mean(dim=[0, 1])  # (codebook_dim, 2)
        codebook_entropy = get_entropy(avg_prob, dim=-1)
        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum()


if __name__ == "__main__":
    model = MultiScaleBSQ(codebook_dim=32, scale_schedule=[1, 5, 25, 50, 100])
    inp = torch.randn(2, 100, 32)
    out = model(inp)
    import ipdb

    ipdb.set_trace()
