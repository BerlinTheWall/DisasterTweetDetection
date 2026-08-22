"""Fine-tune a pretrained transformer on the Disaster Tweets dataset.

    python train.py                                  # 5-fold DeBERTa-v3-base
    python train.py --folds 1 --epochs 1 --limit 500 # smoke test

Writes submission.csv and per-run probabilities under artifacts/.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

import data

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def resolve_model(model_id: str) -> str:
    """Fetch weights up front with a progress bar; from_pretrained does it silently."""
    if Path(model_id).exists():
        return model_id
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_id

    print(f"fetching {model_id} from the Hugging Face Hub (first run only)")
    if not os.environ.get("HF_TOKEN"):
        print("  no HF_TOKEN set - downloads are rate-limited; see "
              "https://huggingface.co/settings/tokens")
    try:
        path = snapshot_download(
            model_id, ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*tf_model*", "*.onnx"])
    except Exception as error:
        print(f"  could not prefetch ({type(error).__name__}: {error});"
              " falling back to lazy download")
        return model_id
    print(f"  cached at {path}")
    return path


def report_progress(epoch, epochs, step, steps, loss, started) -> None:
    elapsed = time.time() - started
    rate = step / elapsed if elapsed else 0.0
    eta = (steps - step) / rate if rate else 0.0
    print(f"\r    epoch {epoch + 1}/{epochs}  step {step}/{steps}  "
          f"loss {loss:.4f}  {rate:4.1f} it/s  eta {eta / 60:4.1f}m",
          end="", flush=True)


def tokenize(tokenizer, texts: list[str], max_length: int):
    encoded = tokenizer(texts, truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]


def predict(model, loader, device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids.to(device),
                               attention_mask=attention_mask.to(device)).logits
            probabilities.append(F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def train_one_fold(args, model_path, train_tensors, valid_tensors, test_loader, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    # Hub checkpoints are often fp16 and transformers 5.x keeps that dtype. AMP
    # needs fp32 master weights: half params make unscale_ raise on CUDA and
    # produce NaNs on CPU.
    model = model.float().to(device)

    loader = DataLoader(TensorDataset(*train_tensors), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    steps_per_epoch = len(loader)
    for epoch in range(args.epochs):
        model.train()
        running, started = 0.0, time.time()
        for step, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = model(input_ids=input_ids.to(device),
                             attention_mask=attention_mask.to(device),
                             labels=labels.to(device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 10 == 0 or step == steps_per_epoch:
                report_progress(epoch, args.epochs, step, steps_per_epoch, running / step, started)
        print(f"\r    epoch {epoch + 1}/{args.epochs}  loss {running / steps_per_epoch:.4f}"
              f"  ({time.time() - started:.0f}s)" + " " * 30)

    valid_loader = DataLoader(TensorDataset(*valid_tensors[:2]), batch_size=args.batch_size * 2)
    valid_probs = predict(model, valid_loader, device)
    test_probs = predict(model, test_loader, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return valid_probs, test_probs


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_true, (probabilities >= t).astype(int)) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        print("  WARNING: no CUDA GPU visible; a full 5-fold run takes hours on CPU.\n"
              "  Check with: python -c \"import torch; print(torch.cuda.is_available())\"")

    train_df, test_df = data.load("train"), data.load("test")
    if args.limit:
        train_df = train_df.head(args.limit)
    y = train_df["target"].to_numpy()

    model_path = resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_ids, train_mask = tokenize(tokenizer, train_df["input"].tolist(), args.max_length)
    test_ids, test_mask = tokenize(tokenizer, test_df["input"].tolist(), args.max_length)
    labels = torch.tensor(y, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(test_ids, test_mask), batch_size=args.batch_size * 2)

    oof = np.full(len(train_df), np.nan)
    test_probs = np.zeros(len(test_df))
    splitter = StratifiedKFold(n_splits=max(args.folds, 2), shuffle=True, random_state=args.seed)

    trained = 0
    for fold, (tr, va) in enumerate(splitter.split(train_ids, y), start=1):
        if trained >= args.folds:
            break
        print(f"\n  fold {fold}/{args.folds}  ({len(tr)} train / {len(va)} valid)")
        valid_probs, fold_test_probs = train_one_fold(
            args, model_path,
            (train_ids[tr], train_mask[tr], labels[tr]),
            (train_ids[va], train_mask[va], labels[va]),
            test_loader, device,
        )
        oof[va] = valid_probs
        test_probs += fold_test_probs
        trained += 1
        print(f"    fold F1 @ 0.50 = {f1_score(y[va], (valid_probs >= 0.5).astype(int)):.4f}")

    test_probs /= trained
    scored = ~np.isnan(oof)
    if not scored.any():
        raise RuntimeError("every out-of-fold prediction is NaN - training diverged")
    threshold, best = tune_threshold(y[scored], oof[scored])
    print(f"\nOOF F1 @ 0.50          = "
          f"{f1_score(y[scored], (oof[scored] >= 0.5).astype(int)):.4f}")
    print(f"OOF F1 @ {threshold:.2f} (tuned)   = {best:.4f}   over {scored.sum()} rows")

    ARTIFACTS.mkdir(exist_ok=True)
    tag = args.tag or args.model.rstrip("/").split("/")[-1]
    np.save(ARTIFACTS / f"{tag}_oof.npy", oof)
    np.save(ARTIFACTS / f"{tag}_test.npy", test_probs)
    print(f"saved artifacts/{tag}_oof.npy and artifacts/{tag}_test.npy")
    data.write_submission(test_df["id"], test_probs >= threshold)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=84)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
