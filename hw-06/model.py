import torch.nn as nn
from torch import Tensor
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(
        self, hidden_size: int, num_classes: int, dropout_p: float = 0.1
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return x


class BertClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze: bool = False) -> None:
        self.model = AutoModel.from_pretrained("bert-base-cased")
        if freeze:
            self.freeze()

        hidden_size = self.model.config.hidden_size
        self.head = Classifier(hidden_size, num_classes)

    def freeze(self) -> None:
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        for layer in self.model.encoder.layer[:5]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.head(pooled_output)


class RobertaClassifier(nn.Module):
    pass


class DistilBertClassifier(nn.Module):
    pass
