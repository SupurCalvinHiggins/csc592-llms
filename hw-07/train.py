import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.amp import GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer

from data import cycle, get_loaders
from model import PerceiverAR

device_type = "cuda"
# device_type = "cpu"
device = torch.device(device_type)


@dataclass
class Config:
    epochs: int = 128
    steps_per_epoch: int = 1024
    val_steps_per_epoch: int = 32
    generate_step_per_epoch: int = 128
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.1
    seq_len: int = 128  # 1024
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    lat_len: int = 64  # 256


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


def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", truncation=True, max_length=cfg.seq_len
    )

    model = PerceiverAR(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        n_vocab=tokenizer.vocab_size,
        seq_len=cfg.seq_len,
        lat_len=cfg.lat_len,
    ).to(device)
    model.compile(mode="max-autotune")

    train_loader, val_loader, _ = get_loaders(
        Path("../datasets/wikitext-103/"), tokenizer, cfg.seq_len, cfg.batch_size
    )
    train_loader, val_loader = cycle(train_loader), cycle(val_loader)

    decay = []
    no_decay = []
    for param_name, param in model.named_parameters():
        if param_name.endswith("bias") or "embed" in param_name or "norm" in param_name:
            no_decay.append(param)
        else:
            decay.append(param)
    opt = optim.Adam(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=cfg.lr,
        fused=True,
    )
    criteron = nn.CrossEntropyLoss()

    scaler = GradScaler()

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = torch.zeros((1,), device=device)
        for _ in tqdm(range(cfg.steps_per_epoch)):
            opt.zero_grad(set_to_none=True)

            # xy is [b, t + 1]
            xy = next(train_loader)
            # x is [b, t]
            x = xy[:, :-1]
            # y is [b * l]
            y = xy[:, -cfg.lat_len :].reshape(-1)

            with torch.autocast(device_type=device_type, dtype=torch.float16):
                # pred_y is [b * l, v]
                pred_y = model(x).view(-1, tokenizer.vocab_size)
                loss = criteron(pred_y, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_train_loss += loss.detach()

        total_train_loss = total_train_loss.item()
        train_loss = total_train_loss / cfg.steps_per_epoch
        print(f"Epoch [{epoch + 1}/{cfg.epochs}]: train_loss = {train_loss}")

        model.eval()
        total_val_loss = torch.zeros((1,), device=device)
        with torch.no_grad():
            for _ in tqdm(range(cfg.val_steps_per_epoch)):
                # xy is [b, t + 1]
                xy = next(val_loader)
                # x is [b, t]
                x = xy[:, :-1]
                # y is [b * l]
                y = xy[:, -cfg.lat_len :].reshape(-1)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    # pred_y is [b * l, v]
                    pred_y = model(x).view(-1, tokenizer.vocab_size)
                    loss = criteron(pred_y, y)
                total_val_loss += loss

            total_val_loss = total_val_loss.item()
            val_loss = total_val_loss / cfg.val_steps_per_epoch
            perplexity = math.exp(val_loss)
            print(f"Epoch [{epoch + 1}/{cfg.epochs}]: val_loss = {val_loss}")
            print(f"Epoch [{epoch + 1}/{cfg.epochs}]: perplexity = {perplexity}")

            x = next(val_loader)[-1]
            out = generate(model, x, cfg.generate_step_per_epoch).cpu().tolist()

            print()
            print("INPUT".center(80, "*"))
            print(tokenizer.decode(out[: cfg.seq_len + 1]))
            print("OUTPUT".center(80, "*"))
            print(tokenizer.decode(out[cfg.seq_len + 1 :]))
            print("*" * 80)
            print()


if __name__ == "__main__":
    main(Config())
