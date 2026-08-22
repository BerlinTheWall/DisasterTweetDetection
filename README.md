# Disaster Tweet Detection

[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
— classify 3,263 test tweets as describing a real disaster or not. Scored by
**F1 on the positive class**.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                                          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
pytest                                                          # 11 tests, <1s
```

## Train

```bash
python train.py --folds 1 --epochs 1 --limit 500   # ~2 min smoke test first
python train.py                                    # 5-fold DeBERTa-v3-base
```

Then upload `submission.csv` on the competition page, or:

```bash
kaggle competitions submit -c nlp-getting-started -f submission.csv -m "deberta-v3-base 5-fold"
```

`train_baseline.py` is a 10-second CPU TF-IDF run kept as a sanity check — it
confirms the data loads and the submission is shaped right. It is not a
competitive entry.

## Layout

| Path | What it is |
| --- | --- |
| `dataset/` | The raw Kaggle CSVs. |
| `data.py` | Loading, light cleaning, label repair, submission writing. |
| `train.py` | Transformer fine-tuning, K-fold, tuned threshold → `submission.csv`. |
| `train_baseline.py` | TF-IDF sanity check, CPU-only. |
| `tests/` | pytest suite for `data.py`. |

## What the model does

**Light cleaning only.** `data.clean` undoes HTML escapes, drops URLs, and
collapses whitespace. It does not lowercase, lemmatize, or spell-correct —
pretrained tokenizers expect natural text, and spell correction mangles exactly
the proper nouns a disaster classifier needs ("La Ronge" → "la range"). An
earlier version of this repo did all of that; removing it raised the TF-IDF
baseline from 0.7690 to 0.7783 OOF.

**Label repair.** 267 training rows are near-duplicate tweets annotated
inconsistently. `data.fix_conflicting_labels` snaps each group to its majority
label (58 rows change) and leaves exact ties alone.

**Keyword as a prefix.** `keyword` is present for 99% of rows and its disaster
rate spans nearly 0–100%, so it is prepended to the tweet: `"wildfire: 13,000
residents evacuated"`.

**Threshold tuning.** F1 does not peak at 0.5. The cut is chosen on
out-of-fold predictions and applied to the averaged test probabilities.

## Reaching a competitive score

`microsoft/deberta-v3-base`, 5 folds, 3 epochs typically lands **0.83–0.84**.
Batch size 16 at sequence length 84 fits in about 4 GB of VRAM; raise
`--batch-size` if you have headroom.

If the DeBERTa-v3 tokenizer fails to build (it needs `sentencepiece` and
`protobuf`), fall back to `--model roberta-base` (~0.82) or, with ≥12 GB VRAM,
`--model roberta-large` (~0.84, use `--batch-size 8 --lr 1e-5`).

Every run saves its probabilities under `artifacts/`, so two different models
can be averaged — usually worth another ~0.005:

```python
import numpy as np, pandas as pd, data
probs = np.mean([np.load(f"artifacts/{t}_test.npy") for t in ("deberta-v3-base", "roberta-large")], axis=0)
data.write_submission(data.load("test")["id"], probs >= 0.45)
```

## History

This repo was a 2023 project that no longer ran: `autocorrect` stopped building
against modern setuptools, NLTK renamed its corpora, `main.ipynb` imported the
retired `keras_core`/`keras_nlp`, and the only model notebook present targeted
a different competition entirely. The preprocessing package and EDA notebook
were removed once measurement showed they cost accuracy rather than adding it;
they remain in git history.
