import einops
import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln(x)
        return x


class CustomLayerNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(4, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(4, requires_grad=True))
        self.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        mean = einops.reduce(x, "b d -> b", "mean")
        var = einops.reduce(x**2, "b d -> b", "mean") - mean**2
        z = (x - mean) / (var + self.eps).sqrt()
        return z * self.gamma + self.beta


def main() -> None:
    x = torch.arange(4).float().view(1, -1)
    print(x)

    ln = LayerNorm()
    print(ln(x.clone()))
    cln = CustomLayerNorm()
    print(cln(x.clone()))


if __name__ == "__main__":
    main()
