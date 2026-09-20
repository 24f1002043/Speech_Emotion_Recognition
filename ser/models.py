"""MFCC-CNN baseline and a self-supervised (WavLM / wav2vec2) classifier."""
from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SR

# ---------------------------------------------------------------- MFCC baseline
N_MFCC = 40
HOP = 160  # 10 ms


def mfcc_features(y: np.ndarray) -> np.ndarray:
    """(3, 40, T): MFCCs plus first and second deltas, 25 ms window, 10 ms hop."""
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=512, win_length=400, hop_length=HOP, n_mels=64)
    if m.shape[1] < 9:  # delta needs at least 9 frames
        m = np.pad(m, ((0, 0), (0, 9 - m.shape[1])), mode="edge")
    return np.stack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]).astype(np.float32)


class MFCCCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        chans, layers, c = [width, width * 2, width * 4, width * 4], [], 3
        for o in chans:
            layers += [nn.Conv2d(c, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2), nn.Dropout2d(0.1)]
            c = o
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 3, 40, T)
        h = self.features(x).mean(2)  # average over frequency -> (B, C, T')
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


# ---------------------------------------------------------------- SSL model
SSL_MODEL = "microsoft/wavlm-base-plus"


def conv_output_lengths(config, lengths: torch.Tensor) -> torch.Tensor:
    for k, s in zip(config.conv_kernel, config.conv_stride):
        lengths = torch.div(lengths - k, s, rounding_mode="floor") + 1
    return lengths


def masked_layer_means(hidden_states, frame_mask: torch.Tensor | None) -> torch.Tensor:
    """Tuple of L (B, T, H) tensors -> (B, L, H) time-averaged over valid frames."""
    h = torch.stack(hidden_states, dim=1).float()  # (B, L, T, H)
    if frame_mask is None:
        return h.mean(2)
    m = frame_mask[:, None, :, None].to(h.dtype)
    return (h * m).sum(2) / m.sum(2).clamp(min=1)


class LayerWeightedHead(nn.Module):
    """SUPERB-style learnable softmax weighting of all layers + small MLP.

    Operates on per-layer, time-averaged vectors. Because averaging is linear,
    this equals weighting frame-level layers and then averaging, which lets the
    frozen model be trained on cached features in seconds.
    """

    def __init__(self, n_layers: int, hidden: int, n_classes: int, proj: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(nn.Linear(hidden, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, n_classes))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:  # (B, L, H)
        assert pooled.shape[1] == len(self.layer_logits), (
            f"got {pooled.shape[1]} layers, expected {len(self.layer_logits)} (is LayerDrop on?)")
        pooled = F.layer_norm(pooled, pooled.shape[-1:])
        w = torch.softmax(self.layer_logits, 0)
        return self.mlp((w[None, :, None] * pooled).sum(1))

    def layer_weights(self) -> list[float]:
        return torch.softmax(self.layer_logits.detach().cpu(), 0).tolist()


class SSLClassifier(nn.Module):
    def __init__(self, n_classes: int, model_name: str = SSL_MODEL, encoder: nn.Module | None = None):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(model_name)
        # LayerDrop would randomly skip transformer layers while training, so
        # output_hidden_states would return fewer than num_hidden_layers + 1 entries
        # and the fixed layer-weight vector would no longer line up.
        encoder.config.layerdrop = 0.0
        self.encoder = encoder
        cfg = encoder.config
        self.head = LayerWeightedHead(cfg.num_hidden_layers + 1, cfg.hidden_size, n_classes)

    def freeze_feature_encoder(self) -> None:
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False

    def pooled(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.encoder(wav, attention_mask=attention_mask, output_hidden_states=True)
        frame_mask = None
        if attention_mask is not None:
            n = conv_output_lengths(self.encoder.config, attention_mask.sum(-1))
            T = out.hidden_states[0].shape[1]
            frame_mask = torch.arange(T, device=wav.device)[None, :] < n[:, None]
        return masked_layer_means(out.hidden_states, frame_mask)

    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.pooled(wav, attention_mask))
