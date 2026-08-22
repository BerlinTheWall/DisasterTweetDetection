# Disaster Tweet Detection

Classifying whether a tweet describes a real disaster or uses disaster language
figuratively — "this traffic is a nightmare" versus an actual emergency. The
distinction matters for emergency-response systems that monitor social media,
and it is hard precisely because the surface language is often identical.

Built for the Kaggle competition
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

## Result

**0.84125 F1 — rank 54 on the leaderboard** *(22 August 2026)*.

Improved from a previous best of 0.82776.

Worth noting for anyone reading the number: this competition's public leaderboard
contains perfect scores obtained from leaked test labels. Around 0.84 is close to
the practical ceiling for an honestly-trained model on this task.

## Approach

1. **Preprocessing** (`preprocessor/`) — **[FILL: what cleaning you do. URL and
   mention stripping? Hashtag handling? Emoji? Lowercasing? Say which, and why —
   the choices are where the thinking shows.]**
2. **Baseline and iteration** (`main.ipynb`) — **[FILL: what you started with
   and what you tried]**
3. **Transformer fine-tuning** — DistilBERT via KerasNLP, with a RoBERTa variant
   in `Model_RoBERTa_1.ipynb`

## Model comparison

**[FILL: a small table. Which models you tried and what each scored. You have at
least DistilBERT and RoBERTa — putting their numbers side by side turns this
from "I trained a model" into "I ran an experiment", which is a different
signal entirely.]**

| Model | Validation score | Leaderboard score |
|---|---|---|
| DistilBERT (KerasNLP) | **[FILL]** | **[FILL]** |
| RoBERTa | **[FILL]** | **[FILL]** |
| *Best submission* | | **0.84125** |

**[FILL: fill in the per-model numbers. Two models with their scores side by side
turns this from "I trained a model" into "I ran an experiment" — a different
signal entirely. Also say which one produced the 0.84125.]**

## Repository layout

```
main.ipynb                  primary training and evaluation notebook
Model_RoBERTa_1.ipynb       RoBERTa variant
preprocessor/               text cleaning
dataset/                    competition data
preprocessed_df_train.csv   cleaned training set
```

## Stack

Python · TensorFlow · KerasNLP · DistilBERT · RoBERTa · NLTK · pandas · NumPy

## Running it

```bash
pip install -r requirements.txt   # [FILL: add a requirements.txt]
jupyter lab
```

Competition data is available from the
[Kaggle competition page](https://www.kaggle.com/competitions/nlp-getting-started/data).

## Notes

**[FILL: what was hardest, what didn't work, what you'd try next. Failed
experiments belong here — they're evidence you iterate rather than getting
lucky once.]**

---

### Housekeeping

Before publishing this README, clean the repo:

```bash
git rm -r --cached .idea __pycache__
printf '.idea/\n__pycache__/\n*.pyc\n' >> .gitignore
git commit -m "Stop tracking IDE and cache files"
git push
```
