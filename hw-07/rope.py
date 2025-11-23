import torch
import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
    def __init__(self, seq_len: int, d_model: int, theta: float = 10000.0) -> None:
        super().__init__()

        half = d_model // 2
        freq_seq = torch.arange(half, dtype=torch.float)
        inv_freq = 1.0 / (theta ** (freq_seq / half))

        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)

        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.size(1)
        sin = self.sin[None, offset : seq_len + offset, :]
        cos = self.cos[None, offset : seq_len + offset, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x
