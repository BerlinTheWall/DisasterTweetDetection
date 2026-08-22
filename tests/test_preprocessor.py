"""Unit tests for :class:`preprocessor.Preprocessor`.

Run with ``pytest`` from the repository root.
"""

import math

import pytest

from preprocessor import Preprocessor


@pytest.fixture(scope="module")
def pre() -> Preprocessor:
    return Preprocessor()


def test_remove_urls(pre):
    assert pre.remove_urls("Visit our website at https://www.example.com") == \
        "Visit our website at "
    assert pre.remove_urls("see www.foo.io/bar now") == "see  now"


def test_remove_html_tags(pre):
    assert pre.remove_html_tags("<p>This is a <b>bold</b> statement.</p>") == \
        "This is a bold statement."


def test_remove_emoji(pre):
    assert pre.remove_emoji("I love Python! \U0001f60d\U0001f40d") == "I love Python! "


def test_remove_punctuations_from_words(pre):
    assert pre.remove_punctuations_from_words(
        "What a            book............!!!!!!  ????") == "What a book ... ! ?"
    assert pre.remove_punctuations_from_words("#wildfire @user") == "wildfire user"


def test_autocorrect_text(pre):
    assert pre.autocorrect_text("Speling can be embarassing.") == \
        "Spelling can be embarrassing."


def test_autocorrect_preserves_case_and_handles(pre):
    assert pre.autocorrect_text("@embarassing #Speling") == "@embarassing #Speling"
    assert pre.autocorrect_text("EMBARASSING") == "EMBARRASSING"


def test_convert_abbrev_in_text(pre):
    assert pre.convert_abbrev_in_text("btw") == "by the way"
    assert pre.convert_abbrev_in_text("BTW it rained") == "by the way it rained"


def test_replace_contractions(pre):
    assert pre.replace_contractions("I can't do it") == "I cannot do it"
    assert pre.replace_contractions("I can’t do it") == "I cannot do it"


def test_replace_contractions_respects_word_boundaries(pre):
    # "he's" must not be expanded inside "she's" incorrectly, and a bare word
    # that merely contains a contraction key is left alone.
    assert pre.replace_contractions("she's late") == "she is late"
    assert pre.replace_contractions("shell") == "shell"


def test_lemma(pre):
    assert pre.lemma("running") == "run"


def test_to_lowercase(pre):
    assert pre.to_lowercase("Convert This Text To Lowercase") == \
        "convert this text to lowercase"


def test_process_text(pre):
    assert pre.process_text(
        "Hello, World! Visit https://www.example.com for more info.",
        spelling=True, lemma=True,
    ) == "hello , world ! visit for more info ."


def test_process_text_handles_missing_values(pre):
    assert pre.process_text(None) == ""
    assert pre.process_text(math.nan) == ""
    assert pre.process_text(math.nan, na_value="unknown") == "unknown"


def test_process_text_flags_are_independent(pre):
    raw = "OMG the FIRE is HUGE http://t.co/abc"
    assert pre.process_text(raw, lowercase=False, abbreviations=False) == \
        "OMG the FIRE is HUGE"
    assert "oh my god" in pre.process_text(raw)


def test_process_series(pre):
    assert pre.process_series(["A URL http://x.co", None]) == ["a url", ""]
