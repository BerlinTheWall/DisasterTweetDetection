"""TF-IDF + linear classifier baseline for the Disaster Tweets competition.

Scored the way Kaggle scores it (F1 on the positive class), with stratified
5-fold cross-validation so the number printed here means something before you
spend a submission on it.

    python prepare_data.py --preset bow
    python train_baseline.py                 # CV score + submission.csv

Runs in well under a minute on a laptop CPU -- no GPU, no downloads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SEED = 42

MODELS = {
    "logreg": lambda: LogisticRegression(C=4.0, max_iter=2000, solver="liblinear"),
    "svm": lambda: CalibratedClassifierCV(LinearSVC(C=0.3), cv=3),
}


def load(preset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = ROOT / "data" / f"{preset}_train.csv"
    if prepared.exists():
        train = pd.read_csv(prepared).fillna("")
        test = pd.read_csv(ROOT / "data" / f"{preset}_test.csv").fillna("")
    else:  # fall back to raw text so the script works out of the box
        print(f"[warn] {prepared} not found -- using raw text. "
              f"Run `python prepare_data.py --preset {preset}` for cleaned input.")
        train = pd.read_csv(ROOT / "dataset" / "train.csv").fillna("")
        test = pd.read_csv(ROOT / "dataset" / "test.csv").fillna("")
    return train, test


def combine(df: pd.DataFrame) -> pd.Series:
    """The keyword column is a strong, almost-free signal -- fold it into the text."""
    return (df["keyword"].astype(str).str.replace("%20", " ", regex=False)
            + " " + df["text"].astype(str)).str.strip()


def vectorize(train_text: pd.Series, test_text: pd.Series):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True)
    x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """F1 is threshold-sensitive; 0.5 is rarely the best cut."""
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_true, (probabilities >= t).astype(int)) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def main(preset: str, model_name: str, folds: int) -> None:
    train, test = load(preset)
    y = train["target"].to_numpy()
    x_train, x_test = vectorize(combine(train), combine(test))

    oof = np.zeros(len(train))
    test_probabilities = np.zeros(len(test))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_train, y), start=1):
        model = MODELS[model_name]()
        model.fit(x_train[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x_train[valid_idx])[:, 1]
        test_probabilities += model.predict_proba(x_test)[:, 1] / folds
        print(f"  fold {fold}  F1@0.5 = "
              f"{f1_score(y[valid_idx], (oof[valid_idx] >= 0.5).astype(int)):.4f}")

    threshold, best_f1 = tune_threshold(y, oof)
    print(f"\nOOF F1 @ 0.50            = {f1_score(y, (oof >= 0.5).astype(int)):.4f}")
    print(f"OOF F1 @ {threshold:.2f} (tuned)     = {best_f1:.4f}")

    submission = pd.DataFrame({
        "id": test["id"],
        "target": (test_probabilities >= threshold).astype(int),
    })
    path = ROOT / "submission.csv"
    submission.to_csv(path, index=False)
    print(f"\nWrote {path}  ({submission['target'].mean():.1%} predicted positive)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="bow", choices=("bow", "transformer"))
    parser.add_argument("--model", default="logreg", choices=sorted(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.preset, args.model, args.folds)
