import torch
import torchaudio
from transformers import Wav2Vec2Model


class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def forward(self, input_values, ori_sample_rate=16000):
        if ori_sample_rate != 16000:
            input_values = torchaudio.functional.resample(input_values, orig_freq=ori_sample_rate, new_freq=16000)
        input_values = self.normalize_audio(input_values)
        hidden_states = self.feature_extractor(input_values).transpose(1, 2)
        hidden_states = self.feature_projection(hidden_states)[0]
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = encoder_outputs[0]
        return hidden_states

    @staticmethod
    def normalize_audio(audio_tensor, dim=-1):
        audio_mean = audio_tensor.mean(dim=dim, keepdim=True)
        audio_std = audio_tensor.std(dim=dim, keepdim=True)
        audio_tensor = (audio_tensor - audio_mean) / (audio_std + 1e-6)
        return audio_tensor
