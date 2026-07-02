from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import encode, stoi
from model import GPTLanguageModel

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=100):
    model.eval()

    out = {}

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split_name] = losses.mean()

    model.train()

    return out


def main():

    # =========================
    # Load Dataset
    # =========================

    text = Path("data/shakespeare.txt").read_text()

    data = torch.tensor(encode(text), dtype=torch.long)

    # =========================
    # Train / Validation Split
    # =========================

    n = int(0.9 * len(data))

    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    batch_size = 32
    lr = 1e-2

    vocab_size = len(stoi)

    model = GPTLanguageModel(vocab_size)

    # ==================
    # PARAMETER COUNT
    # ==================
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # ==================
    # GET BATCH
    # ==================
    x, y = get_batch(train_data, block_size, batch_size)

    logits, loss = model(x, y)

    # ===================
    # SELF ASSERTIONS
    # ===================
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert logits.shape == (batch_size * block_size, vocab_size)
    assert loss.ndim == 0

    print("✅ Shape assertions passed.")

    # ==================
    # BASIC TRAINING LOOP
    # ==================
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(2000):

        x, y = get_batch(train_data, block_size, batch_size)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            # BETTER TRANING -> IMPROVE
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                block_size,
                batch_size,
            )

            print(
                f"Step {step:4d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f}"
            )


if __name__ == "__main__":
    main()
