import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp.utils import GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import get_loaders
from model import BertClassifier


def main(model_name: str, debug: bool = False) -> None:
    epochs = 15 if not debug else 1
    batch_size = 16
    learning_rate = 3e-6

    name_to_cls = {"bert-base-cased": BertClassifier}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, _ = get_loaders(tokenizer, batch_size=batch_size)

    model = name_to_cls[model_name](6)
    if not debug:
        model.compile(mode="max-autotune")

    opt = get_linear_schedule_with_warmup(
        optim.AdamW(model.parameters(), lr=learning_rate, fused=True),
        0,
        epochs * len(train_loader),
    )
    criteron = nn.CrossEntropyLoss()
    scaler = GradScaler()

    val_pred = torch.zeros(len(val_loader) * batch_size, dtype=torch.int)
    val_labels = torch.zeros(len(val_loader) * batch_size, dtype=torch.int)

    for epoch in range(epochs):
        total_train_loss = torch.zeros((1,), device="cuda")
        for input_ids, attention_mask, labels in train_loader:
            with torch.autocast(device_type="cuda"):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criteron(output, labels)
            total_train_loss += loss.detach()
            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(opt)
            scaler.update()

        train_loss = (total_train_loss / len(train_loader)).item()
        print(f"Epoch [{epoch}/{epochs}]: train_loss = {train_loss}")

        with torch.no_grad():
            total_val_loss = torch.zeros((1,), device="cuda")
            for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                with torch.autocast(device_type="cuda"):
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criteron(output, labels)
                total_val_loss += loss
                val_pred[i * batch_size : (i + 1) * batch_size] = output.argmax(dim=-1)
                val_labels[i * batch_size : (i + 1) * batch_size] = labels

            val_loss = (total_val_loss / len(val_loader)).item()
            print(f"Epoch [{epoch}/{epochs}]: val_loss = {val_loss}")

    val_pred, val_labels = val_pred.cpu(), val_labels.cpu()
    cm = confusion_matrix(val_labels, val_pred)
    print(cm)

    target_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    cr = classification_report(val_labels, val_pred, target_names=target_names)
    print(cr)


if __name__ == "__main__":
    main("bert-base-cased")
