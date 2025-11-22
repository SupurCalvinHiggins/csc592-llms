from typing import Iterable

import torch
from datasets import concatenate_datasets, load_dataset
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


def get_emotion_loaders(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int = 16
) -> tuple[DataLoader, DataLoader, DataLoader]:
    assert batch_size >= 1

    train_dataset = (
        load_dataset("emotion", split="train")
        .to_pandas()
        .groupby("label")
        .apply(lambda x: x.sample(512))
        .reset_index(drop=True)
    )
    val_dataset = (
        load_dataset("emotion", split="validation")
        .to_pandas()
        .groupby("label")
        .apply(lambda x: x.sample(64))
        .reset_index(drop=True)
    )
    test_dataset = (
        load_dataset("emotion", split="test")
        .to_pandas()
        .groupby("label")
        .apply(lambda x: x.sample(64))
        .reset_index(drop=True)
    )

    train_input_ids, train_attention_mask = id_and_mask(
        list(train_dataset["text"]), tokenizer
    )
    val_input_ids, val_attention_mask = id_and_mask(
        list(val_dataset["text"]), tokenizer
    )
    test_input_ids, test_attention_mask = id_and_mask(
        list(test_dataset["text"]), tokenizer
    )

    train_labels = torch.tensor(train_dataset["label"], dtype=torch.long)
    val_labels = torch.tensor(val_dataset["label"], dtype=torch.long)
    test_labels = torch.tensor(test_dataset["label"], dtype=torch.long)

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


def get_poj104_loaders(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int = 16
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = (
        concatenate_datasets(
            [
                load_dataset("semeru/Code-Code-CloneDetection-POJ104", split="train"),
                load_dataset(
                    "semeru/Code-Code-CloneDetection-POJ104", split="validation"
                ),
                load_dataset("semeru/Code-Code-CloneDetection-POJ104", split="test"),
            ]
        )
        .to_pandas()
        .dropna()
        .groupby("label")
        .apply(lambda x: x.sample(frac=1, random_state=0))
        .reset_index(drop=True)
    )

    train_dataset = (
        dataset.groupby("label")
        .apply(lambda x: x[:400].sample(32, random_state=0))
        .reset_index(drop=True)
    )
    val_dataset = (
        dataset.groupby("label")
        .apply(lambda x: x[400:450].sample(2, random_state=0))
        .reset_index(drop=True)
    )
    test_dataset = (
        dataset.groupby("label")
        .apply(lambda x: x[450:].sample(2, random_state=0))
        .reset_index(drop=True)
    )

    train_input_ids, train_attention_mask = id_and_mask(
        list(train_dataset["code"]), tokenizer
    )
    val_input_ids, val_attention_mask = id_and_mask(
        list(val_dataset["code"]), tokenizer
    )
    test_input_ids, test_attention_mask = id_and_mask(
        list(test_dataset["code"]), tokenizer
    )

    train_labels = torch.tensor(
        train_dataset["label"].apply(lambda x: int(x) - 1), dtype=torch.long
    )
    val_labels = torch.tensor(
        val_dataset["label"].apply(lambda x: int(x) - 1), dtype=torch.long
    )
    test_labels = torch.tensor(
        test_dataset["label"].apply(lambda x: int(x) - 1), dtype=torch.long
    )

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
