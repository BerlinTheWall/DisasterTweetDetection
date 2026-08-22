import numpy as np
import pandas as pd
import pytest

import data


def test_clean_strips_urls_and_html_escapes():
    assert data.clean("Fire &amp; smoke http://t.co/abc now") == "Fire & smoke now"
    assert data.clean("a\n\n  b\t c") == "a b c"


def test_clean_preserves_case_and_punctuation():
    assert data.clean("BREAKING: Forest fire near La Ronge, Sask.") == \
        "BREAKING: Forest fire near La Ronge, Sask."


def test_normalize():
    assert data.normalize("Fire!! http://t.co/x  FIRE") == "fire fire"


def test_fix_conflicting_labels_snaps_to_majority():
    df = pd.DataFrame({
        "text": ["same tweet", "same tweet!", "same tweet", "other"],
        "target": [1, 1, 0, 0],
    })
    assert data.fix_conflicting_labels(df)["target"].tolist() == [1, 1, 1, 0]


def test_fix_conflicting_labels_leaves_ties_alone():
    df = pd.DataFrame({"text": ["tie", "tie"], "target": [0, 1]})
    assert data.fix_conflicting_labels(df)["target"].tolist() == [0, 1]


def test_fix_conflicting_labels_does_not_touch_unique_rows():
    df = pd.DataFrame({"text": ["a", "b", "c"], "target": [1, 0, 1]})
    assert data.fix_conflicting_labels(df)["target"].tolist() == [1, 0, 1]


def test_build_input_prepends_keyword():
    df = pd.DataFrame({"keyword": ["forest%20fire", None], "text": ["Big blaze", "No key"]})
    assert data.build_input(df) == ["forest fire: Big blaze", "No key"]


@pytest.mark.parametrize("split,rows", [("train", 7613), ("test", 3263)])
def test_load_shapes(split, rows):
    df = data.load(split)
    assert len(df) == rows
    assert df["input"].str.len().gt(0).all()


def test_load_train_only_flips_conflicting_labels():
    raw = pd.read_csv(data.DATASET / "train.csv")
    fixed = data.load("train")
    changed = (raw["target"].to_numpy() != fixed["target"].to_numpy()).sum()
    assert 0 < changed < 200


def test_write_submission_matches_sample(tmp_path):
    sample = pd.read_csv(data.DATASET / "sample_submission.csv")
    test = data.load("test")
    written = pd.read_csv(data.write_submission(test["id"], np.zeros(len(test)), tmp_path / "s.csv"))
    assert list(written.columns) == list(sample.columns)
    assert written["id"].tolist() == sample["id"].tolist()
    assert written["target"].isin([0, 1]).all()
