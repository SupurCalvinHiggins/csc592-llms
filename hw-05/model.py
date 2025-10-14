import einops
import torch
import torch.nn as nn
from torch import Tensor

device = torch.device("cuda")


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.enc = nn.Buffer(torch.zeros(seq_len, d_model, device=device))
        num = torch.arange(seq_len, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, device=device)
            * -(torch.log(torch.tensor(10000.0, device=device)) / d_model)
        ).unsqueeze(0)
        self.enc[:, 0::2] = torch.sin(num * div)
        self.enc[:, 1::2] = torch.cos(num * div)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.enc


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.mask = nn.Buffer(
            torch.triu(
                torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=1
            )
        )

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

    def forward(self, x: Tensor) -> Tensor:
        Q, K, V = einops.rearrange(
            self.qkv(x), "b t (k h dh) -> k b h t dh", h=self.n_heads, k=3
        )
        attn = torch.softmax(
            einops.einsum(Q, K, "b h tq dh, b h tk dh -> b h tq tk").div_(
                self.d_head**0.5
            )
            + self.mask,
            dim=-1,
        )
        O = self.o(
            einops.rearrange(
                einops.einsum(attn, V, "b h tq t, b h t dh -> b h tq dh"),
                "b h t dh -> b t (h dh)",
            )
        )
        return O


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, seq_len: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            Attention(d_model, n_heads, seq_len),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        seq_len: int,
        n_vocab: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_vocab, d_model)
        self.pos = PositionalEncoding(seq_len, d_model)
        self.backbone = nn.Sequential(
            *(
                TransformerBlock(d_model, n_heads, seq_len, dropout=dropout)
                for _ in range(n_layers)
            )
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_vocab))

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = x + self.pos(x)
        x = self.backbone(x)
        x = self.head(x)
        return x
