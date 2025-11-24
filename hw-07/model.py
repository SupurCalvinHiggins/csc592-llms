import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        print(x.shape, self.pe.shape)
        x = x + self.pe[:, :x.size(0), :]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, num_heads: int) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "attn_mask",
            torch.full((seq_len, seq_len), -torch.inf).triu_(diagonal=1),
        )
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.fc_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        q = k = v = self.attn_norm(x)
        attn_out, _ = self.attn(q, k, v, attn_mask=self.attn_mask)
        x = x + attn_out
        x = x + self.fc(self.fc_norm(x))
        return x


class PerceiverAR(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        n_vocab: int,
        seq_len: int,
        lat_len: int,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.lat_len = lat_len
        hst_len = seq_len - lat_len
        self.hst_len = hst_len

        self.embed = nn.Embedding(n_vocab, d_model)
        self.pe = PositionalEncoding(seq_len, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "attn_mask",
            torch.cat(
                [
                    torch.zeros(lat_len, hst_len),
                    torch.full((lat_len, lat_len), -torch.inf).triu_(diagonal=1),
                ],
                dim=1,
            ),
        )
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(lat_len, d_model, num_heads)
                for _ in range(num_layers)
            )
        )
        self.fc_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_vocab)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pe(self.embed(x))
        norm = self.attn_norm(x)
        k = v = norm
        q = norm[..., -self.lat_len :, :]
        attn_out, _ = self.attn(q, k, v, attn_mask=self.attn_mask)
        x = x[..., -self.lat_len :, :] + attn_out
        x = self.blocks(x)
        return self.fc(self.fc_norm(x))
