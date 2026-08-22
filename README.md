# Disaster Tweet Detection

Solution code for the Kaggle playground competition
[**Natural Language Processing with Disaster Tweets**](https://www.kaggle.com/competitions/nlp-getting-started)
— given ~7,600 labelled tweets, predict whether each of 3,263 test tweets
describes a real disaster. Scored by **F1 on the positive class**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                 # 15 tests, ~5s
python prepare_data.py --preset bow    # cache cleaned CSVs (~16s)
python train_baseline.py               # 5-fold CV + submission.csv (~30s, CPU)
```

Then upload `submission.csv` to the competition page.

## Layout

| Path | What it is |
| --- | --- |
| `dataset/` | The raw Kaggle CSVs, as downloaded. |
| `preprocessor/` | `Preprocessor` — tweet cleaning behind boolean flags. |
| `prepare_data.py` | Applies a cleaning preset to train/test, caches to `data/`. |
| `train_baseline.py` | TF-IDF + linear model, stratified CV, writes `submission.csv`. |
| `train_transformer.py` | DeBERTa/RoBERTa fine-tuning, same interface, needs a GPU. |
| `notebooks/eda.ipynb` | Exploratory analysis of the labelled data. |
| `tests/` | pytest suite for the preprocessor. |

## The preprocessor

```python
from preprocessor import Preprocessor

pre = Preprocessor()
pre.process_text(
    "13,000 ppl receive #wildfires evacuation orders http://t.co/abc",
    lowercase=True, contractions=True, urls=True, punctuation=True,
    html_tags=True, emoji=True, abbreviations=True, spelling=True, lemma=True,
)
# -> '13,000 people receive wildfire evacuation order'
```

Every step is a flag, so one instance serves both models. The order is fixed:
strip (URLs, HTML, emoji) → normalize (case, abbreviations, contractions,
punctuation) → linguistic (lemmatize, spell-correct).

**Use different presets for different models.** `prepare_data.py` defines two.
Aggressive cleaning helps sparse bag-of-words models, which only see surface
forms. It *hurts* pretrained transformers, whose tokenizers expect cased,
punctuated, un-lemmatized text — and spell correction mangles proper nouns
("La Ronge" → "la range"), which are exactly the tokens a disaster classifier
wants.

## Results

Stratified 5-fold cross-validation on the training set:

| Model | Preset | OOF F1 @ 0.50 | OOF F1 @ tuned threshold |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | `bow` | 0.7629 | 0.7633 (t=0.42) |
| TF-IDF + calibrated LinearSVC | `bow` | 0.7612 | 0.7691 (t=0.43) |
| TF-IDF + logistic regression | `transformer` (light) | 0.7645 | 0.7670 (t=0.39) |

Two things worth noticing. First, the heavy cleaning pipeline does **not** beat
light cleaning even for the bag-of-words model — the lemmatization and spell
correction are costing about as much signal as they recover. Second, F1 is
threshold-sensitive, so `train_baseline.py` picks the cut on out-of-fold
predictions rather than assuming 0.5.

A fine-tuned DeBERTa-v3-base typically lands around 0.83–0.84 on this dataset;
that is what `train_transformer.py` is for.

## Notes on this codebase

The 2023 version of this repo no longer ran. What changed:

- **`autocorrect` was replaced with `pyspellchecker`.** The former is
  unmaintained and fails to build against current setuptools
  (`AttributeError: install_layout`), so the package could not even be
  imported.
- **NLTK resource names changed** in 3.8.2+ (`punkt` → `punkt_tab`,
  `averaged_perceptron_tagger` → `averaged_perceptron_tagger_eng`). Downloads
  now happen lazily, try both names, and fail soft when offline.
- **Spell correction went from ~70 minutes to ~35 seconds** over the full
  dataset, via per-word memoization, an edit distance of 1 by default, and
  skipping tokens that should never be corrected (short tokens, digits,
  @mentions, #hashtags).
- **Contraction expansion is one compiled pass** with word boundaries, instead
  of 113 unanchored `re.sub` calls per string that could match inside words.
- **`main.ipynb` depended on `keras_core` + `keras_nlp`**, both since folded
  into Keras 3 / KerasHub, and contained no model anyway. It is now
  `notebooks/eda.ipynb`, which is analysis only.
- **`Model_RoBERTa_1.ipynb` was for a different competition** — it loaded
  `llm-detect-ai-generated-text/test_essays.csv` and wrote a `generated`
  column, and referenced a `Dataset` import that was commented out. Replaced by
  `train_transformer.py`, which actually trains on this dataset.
- Missing values return `''` instead of the literal string `'None'`, which was
  being fed to the model as a real token.
- Added `requirements.txt`, `.gitignore`, and a runnable pytest suite; removed
  committed `__pycache__/`, `.idea/`, and scratch files.
