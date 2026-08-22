"""Loading and light cleaning for the Disaster Tweets dataset.

Pretrained transformers were trained on cased, punctuated text, so cleaning
here stays deliberately minimal: undo HTML escaping, drop URLs (they carry no
signal once shortened), normalise whitespace.  No lowercasing, no lemmatizing,
no spell correction -- all of those measurably cost accuracy on this task.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_NORM_RE = re.compile(r"[^a-z0-9 ]")


def clean(text: str) -> str:
    """Undo HTML escapes, strip URLs, collapse whitespace."""
    text = html.unescape(str(text))
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Aggressive form used only for duplicate detection, never for training."""
    return _WS_RE.sub(" ", _NORM_RE.sub("", clean(text).lower())).strip()


def fix_conflicting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve duplicate tweets that carry contradictory labels.

    267 of the 7,613 training rows are near-duplicates annotated both ways.
    Left alone they teach the model that identical text has both labels, which
    caps how confident it can get.  Groups with a clear majority are snapped to
    it; exact ties are left untouched.
    """
    df = df.copy()
    groups = df.assign(_n=df["text"].map(normalize)).groupby("_n")["target"]
    means = groups.transform("mean")
    sizes = groups.transform("size")
    decided = (sizes > 1) & (means != 0.5)
    df.loc[decided, "target"] = (means[decided] > 0.5).astype(int)
    return df


def build_input(df: pd.DataFrame) -> list[str]:
    """Prepend the keyword -- it is present for 99% of rows and highly predictive."""
    keyword = (df["keyword"].fillna("").astype(str)
               .str.replace("%20", " ", regex=False).str.strip())
    text = df["text"].map(clean)
    return (keyword + ": " + text).str.strip(": ").tolist()


def load(split: str, fix_labels: bool = True) -> pd.DataFrame:
    """Read ``dataset/{split}.csv`` and attach a ready-to-tokenize ``input`` column."""
    df = pd.read_csv(DATASET / f"{split}.csv")
    if fix_labels and "target" in df:
        df = fix_conflicting_labels(df)
    df["input"] = build_input(df)
    return df


def write_submission(ids, predictions, path: Path | str = ROOT / "submission.csv") -> Path:
    """Write a Kaggle-shaped submission and report the positive rate."""
    path = Path(path)
    submission = pd.DataFrame({"id": ids, "target": [int(p) for p in predictions]})
    submission.to_csv(path, index=False)
    print(f"wrote {path}  ({len(submission)} rows, "
          f"{submission['target'].mean():.1%} positive)")
    return path
