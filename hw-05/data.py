from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

device = torch.device("cuda")


class WikiText103Split(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class WikiText103(Dataset):
    def __init__(self, data: Tensor, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len
        self.offset = torch.arange(seq_len + 1, device=device)

    @staticmethod
    def load(
        path: Path, tokenizer: BertTokenizerFast, seq_len: int, split: WikiText103Split
    ) -> WikiText103:
        ext = {
            WikiText103Split.TRAIN: "train",
            WikiText103Split.VALIDATION: "valid",
            WikiText103Split.TEST: "test",
        }[split]

        cache_path = path / f"wiki.{ext}.dat"
        raw_path = path / f"wiki.{ext}.tokens"

        if not cache_path.exists():
            text = raw_path.read_text().replace("\n", "[SEP]").replace("<unk>", "[UNK]")
            data = tokenizer.encode(
                text, return_tensors="pt", add_special_tokens=False
            ).view(-1)
            torch.save(data, cache_path)
        else:
            data = torch.load(cache_path)

        return WikiText103(data.to(device), seq_len)

    def __getitems__(self, indices: list[int] | Tensor) -> Tensor:
        indices = torch.tensor(indices, device=device)
        indices = indices[:, None] + self.offset
        return self.data[indices]

    def __len__(self) -> int:
        return self.data.size(0) - self.seq_len + 1


def cycle(loader: DataLoader):
    while True:
        for x in loader:
            yield x


def get_loaders(
    path: Path, tokenizer: BertTokenizerFast, seq_len: int, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = WikiText103.load(path, tokenizer, seq_len, WikiText103Split.TRAIN)
    val_dataset = WikiText103.load(
        path, tokenizer, seq_len, WikiText103Split.VALIDATION
    )
    test_dataset = WikiText103.load(path, tokenizer, seq_len, WikiText103Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = WikiText103.load(
        Path("../datasets/wikitext-103/"), 256, WikiText103Split.TEST
    )
