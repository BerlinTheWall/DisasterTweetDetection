# Disaster Tweet Detection

Classifying whether a tweet describes a real disaster or uses disaster language
figuratively: "this traffic is a nightmare" versus an actual emergency. The
distinction matters for emergency-response systems that monitor social media,
and it is hard because the surface language is often identical.

Built for the Kaggle competition
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

## Result

**0.84125 F1, rank 54 on the leaderboard** *(22 August 2026)*, up from a
previous best of 0.82776.

This competition's public leaderboard contains perfect scores obtained from
leaked test labels. Around 0.84 is close to the practical ceiling for an
honestly trained model on this task.

## Approach

### 1. Data cleaning (`data.py`)

- **Light cleaning only.** HTML entities are unescaped, URLs are removed and
  whitespace is collapsed. Case and punctuation are kept, because a pretrained
  transformer uses them ("BREAKING:" carries signal).
- **Keyword as context.** Each input is built as `keyword: tweet text`
  (for example `forest fire: Big blaze near the highway`), so the model sees the
  disaster keyword Kaggle provides alongside the tweet.
- **Fixing contradictory labels.** The training set contains near-duplicate
  tweets labelled both 0 and 1. Tweets are grouped by an aggressive normalized
  form (lowercase, letters and digits only), and each group with a clear
  majority is snapped to that label. This corrects 58 of 7,613 training rows.
  Exact ties are left alone. The normalized form is used only for this step,
  never as model input.

### 2. Baseline (`train_baseline.py`)

A fast CPU check before spending GPU time: word 1 to 2 gram and character
3 to 5 gram TF-IDF features into a calibrated linear SVM, with 5-fold
stratified cross-validation.

### 3. Transformer fine-tuning (`train.py`)

Fine-tunes `microsoft/deberta-v3-base` (any Hugging Face model can be swapped
in with `--model`) in PyTorch:

- 5-fold stratified cross-validation; test predictions are averaged across folds
- AdamW with cosine learning-rate schedule and 10% warmup, gradient clipping,
  mixed precision on GPU
- **Threshold tuning:** instead of assuming 0.5, the decision threshold is chosen
  by maximizing F1 on the out-of-fold predictions, which are never seen in
  training
- Out-of-fold and test probabilities are saved to `artifacts/` so runs from
  different models can be compared or ensembled later

## Model comparison

5-fold out-of-fold F1 on the training set (7,613 rows):

| Model | OOF F1 @ 0.50 | OOF F1, tuned threshold |
|---|---|---|
| TF-IDF + linear SVM (baseline) | 0.7703 | 0.7783 (at 0.45) |
| DeBERTa-v3-base | **0.8164** | **0.8165** (at 0.49) |

Fine-tuning DeBERTa-v3 adds about 4 F1 points over a strong classical
baseline on the same folds. For DeBERTa the tuned threshold barely differs
from 0.5, so its probabilities are already well calibrated for F1.

Best Kaggle leaderboard submission: **0.84125 F1**.

## Repository layout

```
data.py              loading, cleaning, label fixing, submission writer
train_baseline.py    TF-IDF + linear SVM baseline (CPU, under a minute)
train.py             transformer fine-tuning with k-fold CV
tests/test_data.py   unit tests for the data pipeline
dataset/             competition data (train, test, sample submission)
```

## Running it

Install PyTorch first, matched to your CUDA version
([instructions](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -r requirements.txt

python -m pytest                                    # 11 tests
python train_baseline.py                            # baseline, writes submission.csv
python train.py                                     # full 5-fold DeBERTa-v3-base run
python train.py --folds 1 --epochs 1 --limit 500    # quick smoke test
```

A full 5-fold run needs a CUDA GPU; on CPU it takes hours.

## Stack

Python · PyTorch · Hugging Face Transformers · DeBERTa-v3 · scikit-learn ·
pandas · NumPy · pytest
