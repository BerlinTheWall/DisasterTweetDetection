# Disaster Tweet Detection

[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
— classify tweets as describing a real disaster or not. Scored by F1 on the
positive class.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
pytest
```

Set `HF_TOKEN` for faster model downloads (`setx HF_TOKEN hf_xxxx`, then open a
new shell). Tokens: https://huggingface.co/settings/tokens

## Train

```bash
python train.py --folds 1 --epochs 1 --limit 500   # smoke test
python train.py                                    # 5-fold DeBERTa-v3-base
```

Check the first line of output — if it says `device: cpu`, fix the torch
install before starting a real run.

Submit with `kaggle competitions submit -c nlp-getting-started -f submission.csv -m "..."`
or upload `submission.csv` on the competition page.

`train_baseline.py` is a 10-second CPU TF-IDF run for checking the data loads
and the submission is shaped right. It scores around 0.80.

## Layout

| Path | What it is |
| --- | --- |
| `dataset/` | Raw Kaggle CSVs. |
| `data.py` | Loading, cleaning, label repair, submission writing. |
| `train.py` | Transformer fine-tuning, K-fold, tuned threshold. |
| `train_baseline.py` | TF-IDF baseline, CPU-only. |
| `tests/` | pytest suite for `data.py`. |

## Approach

Cleaning is deliberately light — HTML unescape, URL strip, whitespace. No
lowercasing, lemmatizing or spell correction: pretrained tokenizers expect
natural text, and spell correction mangles the proper nouns that matter
("La Ronge" → "la range"). Removing an earlier heavy pipeline raised the
TF-IDF baseline from 0.7690 to 0.7783 OOF.

267 training rows are near-duplicate tweets annotated inconsistently;
`fix_conflicting_labels` snaps each group to its majority (58 rows change).

`keyword` is present for 99% of rows with a disaster rate spanning nearly
0–100%, so it is prepended: `"wildfire: 13,000 residents evacuated"`.

F1 does not peak at 0.5, so the threshold is picked on out-of-fold predictions
and applied to the averaged test probabilities.

DeBERTa-v3-base at 5 folds scores 0.83–0.84. Batch 16 at length 84 needs about
4 GB of VRAM. If the DeBERTa tokenizer fails to build, `--model roberta-base`
(~0.82) or `--model roberta-large --batch-size 8 --lr 1e-5` (~0.84) work.

Each run saves probabilities under `artifacts/`, so two models can be averaged:

```python
import numpy as np, data
probs = np.mean([np.load(f"artifacts/{t}_test.npy")
                 for t in ("deberta-v3-base", "roberta-large")], axis=0)
data.write_submission(data.load("test")["id"], probs >= 0.45)
```
