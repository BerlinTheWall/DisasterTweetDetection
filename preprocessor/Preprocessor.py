"""
Text preprocessing for English tweets.

The public API is intentionally unchanged from the original version of this
file: ``Preprocessor().process_text(text, **flags)``.  What changed is that it
actually installs and runs on current Python / library versions:

* ``autocorrect`` (unmaintained, no longer builds on modern setuptools) was
  replaced by ``pyspellchecker``.
* NLTK resources are downloaded lazily and use the post-NLTK-3.8.2 names
  (``punkt_tab``, ``averaged_perceptron_tagger_eng``) with a fallback to the
  legacy names.
* Contraction expansion is a single compiled pass with word boundaries instead
  of 113 unanchored ``re.sub`` calls per string.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

_RESOURCES = os.path.join(os.path.dirname(__file__), "resources")

# (nltk.find path, modern package name, legacy package name)
_NLTK_REQUIREMENTS = (
    ("tokenizers/punkt_tab", "punkt_tab", "punkt"),
    ("taggers/averaged_perceptron_tagger_eng",
     "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"),
    ("corpora/wordnet", "wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4", "omw-1.4"),
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+|ftp://\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"   # symbols, pictographs, emoticons, transport
    "\U00002600-\U000027bf"   # misc symbols & dingbats
    "\U0001f1e0-\U0001f1ff"   # regional indicators (flags)
    "\U00002b00-\U00002bff"
    "\U0000fe00-\U0000fe0f"   # variation selectors
    "\U0001f000-\U0001f02f"
    "‍⃣⭐❤"
    "]+",
    flags=re.UNICODE,
)
_KEEP_RE = re.compile(r"[^a-zA-Z0-9,!?.\-'\" ]")
_DOTS_RE = re.compile(r"\.{2,}")
_SPELL_TOKEN_RE = re.compile(r"^(\W*)([A-Za-z][A-Za-z'-]*)(\W*)$")


@lru_cache(maxsize=1)
def ensure_nltk_data(quiet: bool = True) -> None:
    """Download the NLTK corpora this module needs, once per process.

    Safe to call repeatedly and offline: if a resource is already present or
    cannot be fetched, this returns instead of raising.
    """
    for path, modern, legacy in _NLTK_REQUIREMENTS:
        try:
            nltk.data.find(path)
            continue
        except LookupError:
            pass
        for name in (modern, legacy):
            try:
                if nltk.download(name, quiet=quiet):
                    break
            except Exception:  # offline, proxy, permissions...
                continue


class Preprocessor:
    """Cleans English tweet text for downstream NLP models.

    Every step is opt-in through :meth:`process_text` flags, so the same
    instance can produce light cleaning for a transformer and aggressive
    cleaning for a bag-of-words model.
    """

    def __init__(self, download_nltk: bool = True, spell_distance: int = 1) -> None:
        self.contractions = self._load_json("contractions.json")
        self.abbreviations = {k.lower(): v for k, v in
                              self._load_json("abbreviations.json").items()}
        self._contraction_re = self._compile_contractions(self.contractions)
        self._spell_distance = spell_distance
        self._spell = None  # built lazily; loading the frequency list is slow
        self._spell_cache: dict[str, str] = {}
        if download_nltk:
            ensure_nltk_data()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _load_json(name: str) -> dict:
        with open(os.path.join(_RESOURCES, name), encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _compile_contractions(contractions: dict) -> re.Pattern:
        """Build one alternation regex, longest key first so "i'd've" beats "i'd"."""
        keys = sorted({k.lower() for k in contractions}, key=len, reverse=True)
        alternation = "|".join(re.escape(k) for k in keys)
        # \b misbehaves around apostrophes, so bound on word chars manually.
        return re.compile(rf"(?<![\w'])({alternation})(?![\w'])", flags=re.IGNORECASE)

    @property
    def spell(self):
        """Lazily constructed :class:`spellchecker.SpellChecker`.

        ``spell_distance=1`` (the default) is roughly 40x faster than 2 and is
        the only setting that makes whole-dataset correction practical; pass
        ``spell_distance=2`` for a small accuracy gain at a large time cost.
        """
        if self._spell is None:
            from spellchecker import SpellChecker
            self._spell = SpellChecker(distance=self._spell_distance)
        return self._spell

    # ------------------------------------------------------------- operations

    def remove_urls(self, text: str) -> str:
        """Strip http(s)/www/ftp URLs."""
        return _URL_RE.sub("", text)

    def remove_html_tags(self, text: str) -> str:
        """Strip HTML/XML tags, keeping their inner text."""
        return _HTML_RE.sub("", text)

    def remove_emoji(self, text: str) -> str:
        """Strip emoji and pictographs."""
        return _EMOJI_RE.sub("", text)

    def to_lowercase(self, text: str) -> str:
        """Lowercase the whole string."""
        return text.lower()

    def replace_contractions(self, text: str) -> str:
        """Expand contractions ("he's" -> "he is") in a single pass.

        Straight and curly apostrophes are both accepted, and the replacement
        keeps the original casing of the first letter.
        """
        text = text.replace("’", "'")

        def _sub(match: re.Match) -> str:
            found = match.group(0)
            expansion = (self.contractions.get(found)
                         or self.contractions.get(found.lower())
                         or self.contractions.get(found.capitalize()))
            if expansion is None:
                return found
            if found[:1].isupper():
                return expansion[:1].upper() + expansion[1:]
            return expansion

        return self._contraction_re.sub(_sub, text)

    def convert_abbrev_in_text(self, text: str) -> str:
        """Expand chat abbreviations ("btw" -> "by the way"), token by token."""
        return " ".join(self.abbreviations.get(w.lower(), w) for w in text.split())

    def remove_punctuations_from_words(self, text: str) -> str:
        """Drop characters outside a small keep-list and collapse repeats.

        ``"book............!!!!!!  ????"`` becomes ``"book ... ! ?"``.
        """
        text = _KEEP_RE.sub("", text)
        text = _DOTS_RE.sub(" ... ", text)
        for char in (" ", "?", "!", ",", "-", "'", '"'):
            text = re.sub(rf"{re.escape(char)}{{2,}}", char, text)
        return text.strip()

    def _correct_word(self, word: str) -> str:
        """Correct one lowercase word, memoized -- tweets repeat vocabulary."""
        cached = self._spell_cache.get(word)
        if cached is None:
            cached = self.spell.correction(word) or word
            self._spell_cache[word] = cached
        return cached

    def autocorrect_text(self, text: str) -> str:
        """Spell-correct alphabetic tokens, preserving case and punctuation.

        Tokens shorter than four characters, tokens containing digits, and
        @mentions / #hashtags are left alone -- "correcting" them does more
        harm than good on tweets.  Results are memoized per word, which is what
        makes running this over the full dataset practical.
        """
        corrected = []
        for token in text.split():
            match = _SPELL_TOKEN_RE.match(token)
            if not match or token.startswith(("@", "#")):
                corrected.append(token)
                continue
            lead, word, trail = match.groups()
            lowered = word.lower()
            if len(word) < 4 or self.spell.known([lowered]):
                corrected.append(token)
                continue
            fixed = self._correct_word(lowered)
            if word.isupper():
                fixed = fixed.upper()
            elif word[:1].isupper():
                fixed = fixed.capitalize()
            corrected.append(f"{lead}{fixed}{trail}")
        return " ".join(corrected)

    def lemma(self, text: str) -> str:
        """POS-aware lemmatization; also separates punctuation into tokens."""
        ensure_nltk_data()
        lemmatizer = _get_lemmatizer()
        tokens = nltk.word_tokenize(text)
        return " ".join(
            lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(tag))
            for token, tag in pos_tag(tokens)
        )

    @staticmethod
    def get_wordnet_pos(treebank_tag: str) -> str:
        """Map a Penn Treebank tag onto a WordNet POS constant."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        if treebank_tag.startswith("V"):
            return wordnet.VERB
        if treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    # ---------------------------------------------------------------- pipeline

    def process_text(
        self,
        input_text,
        urls: bool = True,
        punctuation: bool = True,
        abbreviations: bool = True,
        html_tags: bool = True,
        emoji: bool = True,
        lowercase: bool = True,
        contractions: bool = True,
        spelling: bool = False,
        lemma: bool = False,
        na_value: str = "",
    ) -> str:
        """Run the enabled cleaning steps over ``input_text``.

        ``None`` and NaN inputs return ``na_value`` (empty string by default)
        rather than raising, because ``location`` is missing for roughly a
        third of the rows in this dataset.
        """
        if input_text is None or (isinstance(input_text, float) and input_text != input_text):
            return na_value
        text = str(input_text)

        if urls:
            text = self.remove_urls(text)
        if html_tags:
            text = self.remove_html_tags(text)
        if emoji:
            text = self.remove_emoji(text)
        if lowercase:
            text = self.to_lowercase(text)
        if abbreviations:
            text = self.convert_abbrev_in_text(text)
        if contractions:
            text = self.replace_contractions(text)
        if punctuation:
            text = self.remove_punctuations_from_words(text)
        if lemma:
            text = self.lemma(text)
        if spelling:
            text = self.autocorrect_text(text)
        return text

    def process_series(self, values: Iterable, **options) -> list:
        """Convenience wrapper: apply :meth:`process_text` over an iterable."""
        return [self.process_text(v, **options) for v in values]


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()
