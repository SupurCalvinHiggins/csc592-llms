import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.amp import GradScaler
from tqdm import tqdm

from data import cycle, get_loaders
from model import Transformer

device_type = "cuda"
device = torch.device(device_type)


def generate(
    model: nn.Module,
    tokens: Tensor,
    count: int,
    temperature: float = 1.0,
) -> Tensor:
    seq_len = tokens.size(0) - 1
    out = torch.zeros((1, seq_len + count + 1), dtype=torch.int, device=device)
    out[:, : seq_len + 1] = tokens
    for i in range(count):
        logits = model(out[:, -seq_len:])[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, 1)
        out[:, seq_len + i + 1] = token
    return out[0]


def decode_token(token: int) -> str:
    c = chr(token)
    return c if c.isprintable() else ""


def decode_tokens(tokens: Iterable[int]) -> str:
    return "".join(map(decode_token, tokens))


def main() -> None:
    torch.set_float32_matmul_precision("high")

    model = Transformer(d_model=512).to(device)
    model.compile(mode="max-autotune")

    batch_size = 12
    seq_len = 1024
    train_loader, val_loader = get_loaders(
        Path("../datasets/enwik8.gz"), seq_len, batch_size
    )
    train_loader, val_loader = cycle(train_loader), cycle(val_loader)

    opt = optim.Adam(model.parameters(), lr=2e-4, fused=True)
    criteron = nn.CrossEntropyLoss()

    scaler = GradScaler()

    epochs = 1024
    steps_per_train_epoch = 1024
    steps_per_val_epoch = 128
    for epoch in range(epochs):
        model.train()
        total_train_loss = torch.zeros((1,), device=device)
        for _ in tqdm(range(steps_per_train_epoch)):
            opt.zero_grad(set_to_none=True)

            x = next(train_loader)
            # y is [b * t]
            y = x[:, 1:].reshape(-1)

            with torch.autocast(device_type=device_type, dtype=torch.float16):
                # pred_y is [b * t, v]
                pred_y = model(x[:, :-1]).view(-1, 256)
                loss = criteron(pred_y, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_train_loss += loss.detach()

        total_train_loss = total_train_loss.item()
        train_loss = total_train_loss / steps_per_train_epoch
        print(f"Epoch [{epoch + 1}/{epochs}]: train_loss = {train_loss}")

        model.eval()
        total_val_loss = torch.zeros((1,), device=device)
        with torch.no_grad():
            for _ in tqdm(range(steps_per_val_epoch)):
                x = next(val_loader)
                # y is [b * t]
                y = x[:, 1:].reshape(-1)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    # pred_y is [b * t]
                    pred_y = model(x[:, :-1]).view(-1, 256)
                    loss = criteron(pred_y, y)
                total_val_loss += loss

            total_val_loss = total_val_loss.item()
            val_loss = total_val_loss / steps_per_val_epoch
            perplexity = math.exp(val_loss)
            bpc = val_loss * math.log2(math.e)
            print(f"Epoch [{epoch + 1}/{epochs}]: val_loss = {val_loss}")
            print(f"Epoch [{epoch + 1}/{epochs}]: perplexity = {perplexity}")
            print(f"Epoch [{epoch + 1}/{epochs}]: bpc = {bpc}")

            x = next(val_loader)[-1]
            out = generate(model, x, 256).cpu().tolist()

            print()
            print("INPUT".center(80, "*"))
            print(decode_tokens(out[: seq_len + 1]))
            print("OUTPUT".center(80, "*"))
            print(decode_tokens(out[seq_len + 1 :]))
            print("*" * 80)
            print()


if __name__ == "__main__":
    main()
