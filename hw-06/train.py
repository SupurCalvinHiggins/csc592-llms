import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import get_emotion_loaders, get_poj104_loaders
from model import BertClassifier, DistilBertClassifier, RobertaClassifier


def main(
    model_name: str, dataset_name: str, freeze: bool = False, debug: bool = False
) -> None:
    torch.cuda.empty_cache()

    epochs = 15 if not debug else 1
    batch_size = 16
    learning_rate = 3e-6

    name_to_cls = {
        "bert-base-cased": BertClassifier,
        "distilbert/distilbert-base-cased": DistilBertClassifier,
        "FacebookAI/roberta-base": RobertaClassifier,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    name_to_num_classes = {"emotion": 6, "poj104": 104}
    name_to_loaders = {"emotion": get_emotion_loaders, "poj104": get_poj104_loaders}
    train_loader, val_loader, _ = name_to_loaders[dataset_name](
        tokenizer, batch_size=batch_size
    )

    model = name_to_cls[model_name](
        name_to_num_classes[dataset_name], freeze=freeze
    ).to("cuda")
    # NOTE: `transformers` does not support AMP + compile.
    # if not debug:
    #    model.compile(mode="max-autotune")

    opt = optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
    # NOTE: `transformers` does not support AMP.
    # opt = get_linear_schedule_with_warmup(opt, 0, epochs * len(train_loader))

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    val_pred = torch.zeros(len(val_loader) * batch_size, dtype=torch.int, device="cuda")
    val_labels = torch.zeros(
        len(val_loader) * batch_size, dtype=torch.int, device="cuda"
    )

    for epoch in range(epochs):
        total_train_loss = torch.zeros((1,), device="cuda")
        for input_ids, attention_mask, labels in train_loader:
            with torch.autocast(device_type="cuda"):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(output, labels)
            total_train_loss += loss.detach()
            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(opt)
            scaler.update()

        train_loss = (total_train_loss / len(train_loader)).item()
        print(f"Epoch [{epoch + 1}/{epochs}]: train_loss = {train_loss}")

        with torch.no_grad():
            total_val_loss = torch.zeros((1,), device="cuda")
            for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
                with torch.autocast(device_type="cuda"):
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(output, labels)
                total_val_loss += loss
                val_pred[i * batch_size : (i + 1) * batch_size] = output.argmax(dim=-1)
                val_labels[i * batch_size : (i + 1) * batch_size] = labels

            val_loss = (total_val_loss / len(val_loader)).item()
            print(f"Epoch [{epoch + 1}/{epochs}]: val_loss = {val_loss}")

    val_pred, val_labels = val_pred.cpu(), val_labels.cpu()
    cm = confusion_matrix(val_labels, val_pred)
    print(cm)

    if dataset_name == "emotion":
        target_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        cr = classification_report(val_labels, val_pred, target_names=target_names)
        print(cr)
    else:
        cr = classification_report(val_labels, val_pred)
        print(cr)


if __name__ == "__main__":
    # emotion
    # main("bert-base-cased", "emotion", freeze=False)
    # main("bert-base-cased", "emotion", freeze=True)
    # main("distilbert/distilbert-base-cased", "emotion", freeze=False)
    # main("FacebookAI/roberta-base", "emotion", freeze=False)
    # poj104
    main("bert-base-cased", "poj104", freeze=False)
    main("bert-base-cased", "poj104", freeze=True)
    main("distilbert/distilbert-base-cased", "poj104", freeze=False)
    main("FacebookAI/roberta-base", "poj104", freeze=False)
