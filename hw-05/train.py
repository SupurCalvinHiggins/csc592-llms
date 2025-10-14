import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer

from data import cycle, get_loaders
from model import Transformer

device = torch.device("cuda")


def generate(
    model: nn.Module,
    tokens: Tensor,
    count: int,
    temperature: float = 1.0,
) -> Tensor:
    seq_len = tokens.size(0) - 1
    out = torch.zeros((1, seq_len + count), dtype=torch.int, device=device)
    out[:, :seq_len] = tokens[:seq_len]
    for i in range(count):
        logits = model(out[:, i : i + seq_len])[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, 1)
        out[:, seq_len + i] = token
    return out[0]


TESTING = False


def main() -> None:
    epochs = 2**10
    steps_per_epoch = 2**10 if not TESTING else 1
    val_steps_per_epoch = steps_per_epoch // 8 if not TESTING else 1
    seq_len = 256
    n_vocab = 28996
    batch_size = 16 if not TESTING else 1

    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=True, truncation=False
    )
    train_loader, val_loader, test_loader = get_loaders(
        Path("../datasets/wikitext-103/"), tokenizer, seq_len, batch_size
    )
    train_loader, val_loader, test_loader = (
        cycle(train_loader),
        cycle(val_loader),
        cycle(test_loader),
    )

    model = Transformer(
        d_model=512, n_heads=8, n_layers=8, seq_len=seq_len, n_vocab=n_vocab
    ).to(device)
    model = torch.compile(model, mode="max-autotune")

    opt = Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    start_epoch = 0

    checkpoint_path = Path("checkpoint.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        for _ in range(start_epoch - 1):
            for _ in range(steps_per_epoch):
                next(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = torch.zeros((1,), device=device)
        for _ in tqdm(range(steps_per_epoch)):
            data = next(train_loader)
            x = data[:, :-1]
            y = data[:, 1:].reshape(-1)

            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda"):
                pred_y = model(x).view(-1, n_vocab)
                loss = criterion(pred_y, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_train_loss += loss.detach()

        train_loss = total_train_loss.item() / steps_per_epoch
        print(f"Epoch [{epoch + 1}/{epochs}]: train_loss = {train_loss}")

        model.eval()
        total_val_loss = torch.zeros((1,), device=device)
        with torch.no_grad():
            for _ in tqdm(range(val_steps_per_epoch)):
                data = next(val_loader)
                x = data[:, :-1]
                y = data[:, 1:].reshape(-1)
                with torch.autocast("cuda"):
                    pred_y = model(x).view(-1, n_vocab)
                    loss = criterion(pred_y, y)

                total_val_loss += loss

            val_loss = total_val_loss.item() / val_steps_per_epoch
            perplexity = math.exp(val_loss)
            bpc = val_loss * math.log2(math.e)
            print(f"Epoch [{epoch + 1}/{epochs}]: val_loss = {val_loss}")
            print(f"Epoch [{epoch + 1}/{epochs}]: perplexity = {perplexity}")
            print(f"Epoch [{epoch + 1}/{epochs}]: bpc = {bpc}")

            x = next(val_loader)[-1]
            out = generate(model, x, 256).cpu().tolist()

            print()
            print("INPUT".center(80, "*"))
            print(tokenizer.decode(out[: seq_len + 1]))
            print("OUTPUT".center(80, "*"))
            print(tokenizer.decode(out[seq_len + 1 :]))
            print("*" * 80)
            print()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
