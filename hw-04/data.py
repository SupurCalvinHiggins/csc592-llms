from __future__ import annotations

import gzip
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda")


class EnWik8Dataset(Dataset):
    def __init__(self, data: Tensor, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len
        self.offsets = torch.arange(self.seq_len + 1, device="cuda")

    @staticmethod
    def load(
        path: Path,
        seq_len: int,
        is_train: bool,
    ) -> EnWik8Dataset:
        train_size = 90 * 10**6
        val_size = 5 * 10**6
        with gzip.open(path) as f:
            if not is_train:
                f.seek(train_size)
            data = torch.frombuffer(
                f.read(train_size if is_train else val_size),
                dtype=torch.uint8,
            ).to(device=device, non_blocking=True)
        return EnWik8Dataset(data, seq_len)

    def __getitems__(self, indices: list[int] | Tensor) -> Tensor:
        if not isinstance(indices, Tensor):
            indices = torch.tensor(indices, device=device)
        idx = indices[:, None] + self.offsets
        return self.data[idx]

    def __len__(self) -> int:
        return self.data.size(0) - self.seq_len - 1


def get_loaders(
    path: Path, seq_len: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    train_dataset = EnWik8Dataset.load(path, seq_len, is_train=True)
    val_dataset = EnWik8Dataset.load(path, seq_len, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: x)

    return train_loader, val_loader


def cycle(loader: DataLoader):
    while True:
        for x in loader:
            yield x


if __name__ == "__main__":
    train_loader, val_loader = get_loaders(
        Path("../datasets/enwik8.gz"), seq_len=1024, batch_size=32
    )

    print(next(iter(train_loader)))
    print(next(iter(val_loader)))
