"""Clean the raw Kaggle CSVs once and cache the result.

    python prepare_data.py --preset bow          # aggressive, for TF-IDF models
    python prepare_data.py --preset transformer  # light, for BERT-family models

Writes ``data/{preset}_train.csv`` and ``data/{preset}_test.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from preprocessor import Preprocessor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "dataset"
OUT = ROOT / "data"

# Aggressive cleaning helps sparse bag-of-words models, which cannot see past
# surface forms.  It *hurts* pretrained transformers, whose tokenizers were
# trained on cased, punctuated, un-lemmatized text -- hence two presets.
PRESETS = {
    "bow": dict(
        lowercase=True, contractions=True, urls=True, punctuation=True,
        html_tags=True, emoji=True, abbreviations=True, spelling=True, lemma=True,
    ),
    "transformer": dict(
        lowercase=False, contractions=True, urls=True, punctuation=False,
        html_tags=True, emoji=True, abbreviations=False, spelling=False, lemma=False,
    ),
}


def build(preset: str) -> None:
    options = PRESETS[preset]
    pre = Preprocessor()
    OUT.mkdir(exist_ok=True)

    for split in ("train", "test"):
        df = pd.read_csv(RAW / f"{split}.csv")
        started = time.time()
        for column in ("text", "keyword", "location"):
            df[column] = pre.process_series(df[column], **options)
        path = OUT / f"{preset}_{split}.csv"
        df.to_csv(path, index=False)
        print(f"{split}: {len(df):>5} rows in {time.time() - started:5.1f}s -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="bow")
    build(parser.parse_args().preset)
