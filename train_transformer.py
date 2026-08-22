"""Fine-tune a pretrained transformer on the Disaster Tweets dataset.

This is the part the repository was always missing: the original
``Model_RoBERTa_1.ipynb`` was a copy of a notebook for a *different* Kaggle
competition (LLM - Detect AI Generated Text) and never touched this dataset.

    pip install -r requirements-transformer.txt
    python prepare_data.py --preset transformer
    python train_transformer.py --model microsoft/deberta-v3-base --folds 5

Needs a GPU to be practical (~4 min/fold on a Kaggle T4).  Use
``--folds 1 --epochs 1`` for a quick smoke test on CPU.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parent
SEED = 42


class TweetDataset(Dataset):
    """Thin wrapper so we do not need the extra ``datasets`` dependency."""

    def __init__(self, encodings: dict, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def load(split: str) -> pd.DataFrame:
    prepared = ROOT / "data" / f"transformer_{split}.csv"
    path = prepared if prepared.exists() else ROOT / "dataset" / f"{split}.csv"
    return pd.read_csv(path).fillna("")


def build_text(df: pd.DataFrame) -> list[str]:
    """Keyword first, then the tweet -- a cheap way to give the model context."""
    keyword = df["keyword"].astype(str).str.replace("%20", " ", regex=False)
    return (keyword + " [SEP] " + df["text"].astype(str)).str.strip().tolist()


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    return {"f1": f1_score(labels, logits.argmax(axis=-1))}


def make_training_args(output_dir: str, args) -> TrainingArguments:
    """TrainingArguments renamed several fields across versions -- adapt."""
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs = dict(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    eval_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in supported})


def make_trainer(model, training_args, tokenizer, **kwargs) -> Trainer:
    """``tokenizer=`` became ``processing_class=`` in transformers 4.46."""
    key = ("processing_class"
           if "processing_class" in inspect.signature(Trainer.__init__).parameters
           else "tokenizer")
    return Trainer(model=model, args=training_args, **{key: tokenizer}, **kwargs)


def main(args) -> None:
    set_seed(SEED)
    train_df, test_df = load("train"), load("test")
    y = train_df["target"].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encode = lambda texts: dict(tokenizer(texts, truncation=True, padding="max_length",
                                          max_length=args.max_length))
    train_encodings = encode(build_text(train_df))
    test_dataset = TweetDataset(encode(build_text(test_df)))

    oof = np.zeros(len(train_df))
    test_probabilities = np.zeros(len(test_df))
    splitter = StratifiedKFold(n_splits=max(args.folds, 2), shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_encodings["input_ids"], y)):
        if fold >= args.folds:
            break
        print(f"\n===== fold {fold + 1}/{args.folds} =====")
        subset = lambda idx: TweetDataset({k: [v[i] for i in idx]
                                           for k, v in train_encodings.items()}, y[idx])
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        trainer = make_trainer(
            model,
            make_training_args(f"./checkpoints/fold{fold}", args),
            tokenizer,
            train_dataset=subset(train_idx),
            eval_dataset=subset(valid_idx),
            compute_metrics=compute_metrics,
        )
        trainer.train()

        oof[valid_idx] = softmax(trainer.predict(subset(valid_idx)).predictions)[:, 1]
        test_probabilities += softmax(trainer.predict(test_dataset).predictions)[:, 1] / args.folds
        print(f"fold {fold + 1} F1 = {f1_score(y[valid_idx], (oof[valid_idx] >= 0.5).astype(int)):.4f}")
        del model, trainer
        torch.cuda.empty_cache()

    scored = oof[oof > 0]
    if len(scored):
        mask = oof > 0
        print(f"\nOOF F1 = {f1_score(y[mask], (oof[mask] >= 0.5).astype(int)):.4f} "
              f"over {mask.sum()} rows")

    path = ROOT / "submission.csv"
    pd.DataFrame({"id": test_df["id"],
                  "target": (test_probabilities >= 0.5).astype(int)}).to_csv(path, index=False)
    print(f"Wrote {path}")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    main(parser.parse_args())
