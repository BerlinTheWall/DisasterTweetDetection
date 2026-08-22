"""Loading and cleaning for the Disaster Tweets dataset."""

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
    text = html.unescape(str(text))
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Aggressive form used for duplicate detection only, never for training."""
    return _WS_RE.sub(" ", _NORM_RE.sub("", clean(text).lower())).strip()


def fix_conflicting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Snap duplicate tweets with contradictory labels to their majority label."""
    df = df.copy()
    groups = df.assign(_n=df["text"].map(normalize)).groupby("_n")["target"]
    means = groups.transform("mean")
    sizes = groups.transform("size")
    decided = (sizes > 1) & (means != 0.5)
    df.loc[decided, "target"] = (means[decided] > 0.5).astype(int)
    return df


def build_input(df: pd.DataFrame) -> list[str]:
    keyword = (df["keyword"].fillna("").astype(str)
               .str.replace("%20", " ", regex=False).str.strip())
    return (keyword + ": " + df["text"].map(clean)).str.strip(": ").tolist()


def load(split: str, fix_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATASET / f"{split}.csv")
    if fix_labels and "target" in df:
        df = fix_conflicting_labels(df)
    df["input"] = build_input(df)
    return df


def write_submission(ids, predictions, path: Path | str = ROOT / "submission.csv") -> Path:
    path = Path(path)
    submission = pd.DataFrame({"id": ids, "target": [int(p) for p in predictions]})
    submission.to_csv(path, index=False)
    print(f"wrote {path}  ({len(submission)} rows, "
          f"{submission['target'].mean():.1%} positive)")
    return path
