"""TF-IDF + linear baseline: a fast CPU check, not a competitive entry."""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

import data

SEED = 42
FOLDS = 5


def vectorize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True)
    x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def main() -> None:
    train, test = data.load("train"), data.load("test")
    y = train["target"].to_numpy()
    x_train, x_test = vectorize(train["input"], test["input"])

    oof = np.zeros(len(train))
    test_probabilities = np.zeros(len(test))
    for fold, (tr, va) in enumerate(
            StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x_train, y), 1):
        model = CalibratedClassifierCV(LinearSVC(C=0.3), cv=3)
        model.fit(x_train[tr], y[tr])
        oof[va] = model.predict_proba(x_train[va])[:, 1]
        test_probabilities += model.predict_proba(x_test)[:, 1] / FOLDS
        print(f"  fold {fold}  F1 = {f1_score(y[va], (oof[va] >= 0.5).astype(int)):.4f}")

    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y, (oof >= t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(scores))])
    print(f"\nOOF F1 @ 0.50        = {f1_score(y, (oof >= 0.5).astype(int)):.4f}")
    print(f"OOF F1 @ {threshold:.2f} (tuned) = {max(scores):.4f}")

    data.write_submission(test["id"], test_probabilities >= threshold)


if __name__ == "__main__":
    main()
