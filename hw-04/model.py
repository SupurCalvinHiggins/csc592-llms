import einops
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        ).unsqueeze(0)
        # pe is [t, dm]
        pe = torch.zeros((seq_len, d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = nn.Buffer(pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x is [b, t, dm]
        x = x * self.d_model**0.5
        x = x + self.pe
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, n_heads: int) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.mask = nn.Buffer(
            torch.ones((seq_len, seq_len), dtype=torch.bool).triu_(diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x is [b, t, dm]
        qkv = self.qkv(x)
        # qkv is [b, t, 3 * dm]
        q, k, v = einops.rearrange(
            qkv, "b t (k h dh) -> k b h t dh", h=self.n_heads, dh=self.d_head, k=3
        )
        # q, k, v are [b, h, t, dh]
        qk = einops.einsum(q, k, "b h qt i, b h kt i -> b h qt kt")
        qk.div_(self.d_head**0.5)
        qk.masked_fill_(self.mask, -torch.inf)
        # qk is [b, h, t, t]
        attn = torch.softmax(qk, dim=-1)
        o = einops.einsum(attn, v, "b h qt t, b h t dh -> b qt h dh")
        o = einops.rearrange(o, "b t h dh -> b t (h dh)")
        # o is [b, t, dm]
        return self.o(o)


class TransformerBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, n_heads: int, dropout=0.1) -> None:
        super().__init__()

        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            Attention(seq_len, d_model, n_heads),
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
        # x is [b, t, dm]
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vocab: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        seq_len: int = 1024,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(n_vocab, d_model)
        self.pos = PositionalEncoding(seq_len, d_model)
        self.backbone = nn.Sequential(
            *(TransformerBlock(seq_len, d_model, n_heads) for _ in range(n_layers))
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_vocab),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x is [b, t]
        x = self.embed(x.int())
        x = self.pos(x)
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = Transformer(512)
    model(torch.zeros((1, 1024), dtype=torch.int))
