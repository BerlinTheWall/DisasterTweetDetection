import re
import string
import json
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

class Preprocessor:
    def __init__(self):
        self.load_contractions()
        self.load_abbreviations()
        self.spell = SpellChecker()

    def load_contractions(self):
        with open("contractions.json", "r") as json_file:
            self.contractions = json.load(json_file)

    def load_abbreviations(self):
        with open("abbreviations.json", "r") as json_file:
            self.abbreviations = json.load(json_file)

    def replace_contractions(self, text):
        for key, value in self.contractions.items():
            text = re.sub(re.escape(key), value, text)
        return text

    def remove_urls(self, text):
        return re.sub(r'https?://\S+|www\.\S+|ftp://\S+', '', text)

    def separate_punctuations_from_words(self, text):
        for p in string.punctuation:
            text = text.replace(p, f' {p} ')
        return text

    def replace_continues_dots(self, text):
        continues_dots = ['..', '...', '....', '.....', '......', '.......', '........', '.........', '..........']
        for c in continues_dots:
            text = text.replace(c, ' ... ')
        return text

    def remove_html_tags(self, text):
        return re.compile(r'<.*?>').sub('', text)

    def remove_emoji(self, text):
        emoji_pattern = re.compile('['
                                   u'\U0001F600-\U0001F64F'
                                   u'\U0001F300-\U0001F5FF'
                                   u'\U0001F680-\U0001F6FF'
                                   u'\U0001F1E0-\U0001F1FF'
                                   u'\U00002702-\U000027B0'
                                   u'\U000024C2-\U0001F251'
                                   ']+', flags=re.UNICODE)
        return emoji_pattern.sub('', text)

    def correct_spelling(self, input_text):
        words = input_text.split()
        corrected_words = [self.spell.correction(word) if word in self.spell.unknown(words) else word for word in words]
        corrected_text = " ".join(corrected_words)
        return corrected_text

    def convert_abbrev_in_text(self, text):
        words = text.split()
        converted_words = [self.abbreviations.get(word.lower(), word) for word in words]
        converted_text = ' '.join(converted_words)
        return converted_text

    def lemma(self, text):
        word_net_lemma = WordNetLemmatizer()
        return word_net_lemma.lemmatize(text)

    def to_lowercase(self, text):
        return text.to_lowercase()

    def process_text(self, input_text,
                     lowercase=False,
                     contractions=False,
                     urls=False,
                     punctuation=False,
                     dots=False,
                     html_tags=False,
                     emoji=False,
                     spelling=False,
                     abbreviations=False,
                     lemma=False
                     ):
        processed_text = input_text

        # remove
        if urls:
            processed_text = self.remove_urls(processed_text)
        if html_tags:
            processed_text = self.remove_html_tags(processed_text)
        if emoji:
            processed_text = self.remove_emoji(processed_text)

        # replace
        if abbreviations: #lmao
            processed_text = self.convert_abbrev_in_text(processed_text)
        if contractions: #It's
            processed_text = self.replace_contractions(processed_text)
        if dots: # ...
            processed_text = self.replace_continues_dots(processed_text)
        if punctuation: # good.so -> good . so
            processed_text = self.separate_punctuations_from_words(processed_text)
        if lemma:
            processed_text = self.lemma(processed_text)

        if lowercase:
            processed_text = self.to_lowercase(processed_text)

        if spelling:
            processed_text = self.correct_spelling(processed_text)

        return processed_text