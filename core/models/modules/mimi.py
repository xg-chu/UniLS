import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from transformers import MimiModel


class MimiModelWrapper(nn.Module):
    def __init__(self, model_name="kyutai/mimi", feature_type="acoustic", default_sample_rate=24000):
        super().__init__()
        self.latent_dim = 256 if "half" in feature_type else 512
        self.feature_type = feature_type
        self.default_sample_rate = default_sample_rate
        # model
        model = MimiModel.from_pretrained(model_name).eval()
        self.encoder = model.encoder
        self.encoder_transformer = model.encoder_transformer
        self.downsample = model.downsample
        self.semantic_quant = model.quantizer.semantic_residual_vector_quantizer
        self.accoustic_quant = model.quantizer.acoustic_residual_vector_quantizer
        for p in model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, audio, ori_sample_rate=None):
        if ori_sample_rate is None:
            ori_sample_rate = self.default_sample_rate
        ori_shape = audio.shape
        if len(ori_shape) == 3:
            audio = rearrange(audio, "b t c -> (b t) c")
        assert audio.dim() == 2, f"Invalid input shape: {audio.shape}"

        if ori_sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, orig_freq=ori_sample_rate, new_freq=24000)
        audio = audio.unsqueeze(1)
        # audio feature
        embeddings = self.encoder(audio)  # B, 512, L (40ms per frame)
        encoder_outputs = self.encoder_transformer(embeddings.transpose(1, 2))
        embeddings = encoder_outputs.last_hidden_state.transpose(1, 2)

        # project to semantic and accoustic space
        if self.feature_type == "acoustic":
            # (B, 512, L) -> (B, 256, L) -> (B, 512, L)
            embeddings = self.downsample(embeddings)
            acoustic_emb = self.accoustic_quant.input_proj(embeddings)
            acoustic_emb_proj = self.accoustic_quant.output_proj(acoustic_emb)
            embeddings = acoustic_emb_proj
            embeddings = torch.nn.functional.interpolate(embeddings, scale_factor=2, mode="nearest")
        elif self.feature_type == "semantic":
            embeddings = self.downsample(embeddings)
            semantic_emb = self.semantic_quant.input_proj(embeddings)
            semantic_emb_proj = self.semantic_quant.output_proj(semantic_emb)
            embeddings = semantic_emb_proj
            embeddings = torch.nn.functional.interpolate(embeddings, scale_factor=2, mode="nearest")
        elif self.feature_type == "both":
            embeddings = self.downsample(embeddings)
            acoustic_emb = self.accoustic_quant.input_proj(embeddings)
            acoustic_emb_proj = self.accoustic_quant.output_proj(acoustic_emb)
            semantic_emb = self.semantic_quant.input_proj(embeddings)
            semantic_emb_proj = self.semantic_quant.output_proj(semantic_emb)
            embeddings = semantic_emb_proj + acoustic_emb_proj
            embeddings = torch.nn.functional.interpolate(embeddings, scale_factor=2, mode="nearest")
        elif self.feature_type == "no_proj":
            pass
        else:
            raise ValueError(f"Invalid feature type: {self.feature_type}.")
        embeddings = torch.transpose(embeddings, 1, 2)
        if len(ori_shape) == 3:
            embeddings = rearrange(embeddings, "(b t) c d-> b t c d", b=ori_shape[0], t=ori_shape[1])
        return embeddings
