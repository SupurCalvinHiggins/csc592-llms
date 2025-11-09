from typing import Iterable

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class CudaTensorDataset(Dataset):
    def __init__(self, *tensors: Iterable[Tensor]) -> None:
        super().__init__()

        self.tensors = [tensor.cuda() for tensor in tensors]
        assert len(self.tensors) >= 1

        length = len(self)
        assert all(tensor.size(0) == length for tensor in tensors)

    def __getitems__(self, indices: list[int]) -> list[Tensor]:
        return [tensor[indices] for tensor in self.tensors]

    def __len__(self) -> int:
        return self.tensors[0].size(0)


def id_and_mask(
    documents: list[str], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
) -> tuple[Tensor, Tensor]:
    encoded_dict = tokenizer.batch_encode_plus(
        documents,
        max_length=128,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt",
    )
    return encoded_dict["input_ids"], encoded_dict["attention_mask"]


def get_loaders(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int = 16
) -> tuple[DataLoader, DataLoader, DataLoader]:
    assert batch_size >= 1

    train_dataset = load_dataset("emotion", split="train").select(range(500))
    val_dataset = load_dataset("emotion", split="validation").select(range(80))
    test_dataset = load_dataset("emotion", split="test").select(range(65))

    train_input_ids, train_attention_mask = id_and_mask(
        train_dataset["text"], tokenizer
    )
    val_input_ids, val_attention_mask = id_and_mask(val_dataset["text"], tokenizer)
    test_input_ids, test_attention_mask = id_and_mask(test_dataset["text"], tokenizer)

    train_labels = torch.tensor(train_dataset["labels"], dtype=torch.int)
    val_labels = torch.tensor(val_dataset["labels"], dtype=torch.int)
    test_labels = torch.tensor(test_dataset["labels"], dtype=torch.int)

    train_dataset = CudaTensorDataset(
        train_input_ids, train_attention_mask, train_labels
    )
    val_dataset = CudaTensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = CudaTensorDataset(test_input_ids, test_attention_mask, test_labels)

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
