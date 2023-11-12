import os
import re
import string
import json

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller


class Preprocessor:
    """
    Preprocessor class for English texts designed for use in NLP models,
    incorporating common tactics to clean the output texts.
    """
    def __init__(self):
        """
        Initializes an instance of the Preprocessor class.
        load contractions and abbreviations from JSON file.
        """
        self.load_contractions()
        self.load_abbreviations()
        self.spell = Speller()

    def load_contractions(self):
        """
        Loads a JSON file containing common English contractions and stores them in the contractions attribute.
        """
        filepath = os.path.join(os.path.dirname(__file__), "resources/contractions.json")
        with open(filepath, "r") as json_file:
            self.contractions = json.load(json_file)

    def load_abbreviations(self):
        """
        Loads a JSON file containing common abbreviations and stores them in the abbreviations attribute.
        """
        filepath = os.path.join(os.path.dirname(__file__), "resources/abbreviations.json")
        with open(filepath, "r") as json_file:
            self.abbreviations = json.load(json_file)

    # def load_contractions(self):
    #     with open("resources/contractions.json", "r") as json_file:
    #         self.contractions = json.load(json_file)
    #
    # def load_abbreviations(self):
    #     with open("resources/abbreviations.json", "r") as json_file:
    #         self.abbreviations = json.load(json_file)

    def replace_contractions(self, text):
        """
        Replaces contractions in the input text with their expanded forms using a predefined list of contractions.

        Arguments:
        text -- The input text with contractions.

        Returns:
        str -- The input text with contractions replaced.
        """
        for key, value in self.contractions.items():
            text = re.sub(re.escape(key), value, text)
        return text

    def remove_urls(self, text):
        """
        Removes URLs from the input text.

        Arguments:
        text -- The input text containing URLs.

        Returns:
        str -- The input text with URLs removed.
        """
        return re.sub(r'https?://\S+|www\.\S+|ftp://\S+', '', text)

    def remove_html_tags(self, text):
        """
        Removes HTML tags from the input text.

        Arguments:
        text -- The input text containing HTML tags.

        Returns:
        str -- The input text with HTML tags removed.
        """
        return re.compile(r'<.*?>').sub('', text)

    def remove_emoji(self, text):
        """
        Removes emojis from the input text.

        Arguments:
        text -- The input text containing emojis.

        Returns:
        str -- The input text with emojis removed.
        """
        emoji_pattern = re.compile('['
                                   u'\U0001F600-\U0001F64F'
                                   u'\U0001F300-\U0001F5FF'
                                   u'\U0001F680-\U0001F6FF'
                                   u'\U0001F1E0-\U0001F1FF'
                                   u'\U00002702-\U000027B0'
                                   u'\U000024C2-\U0001F251'
                                   ']+', flags=re.UNICODE)
        return emoji_pattern.sub('', text)

    def remove_punctuations_from_words(self, text):
        """
        Remove punctuation and non-alphabetic characters from words in the input text.
        Replace continuous characters to only one, and replace continuous dots with '...'

        Arguments:
        text -- The input text with punctuation characters and words.

        Returns:
        str -- The input text with punctuation characters removed from words.
        """
        exception_punctuation_list = [' ', '.', '?', '!', ',', '-', '\'', '\"', ]

        text = re.sub(r'[^a-zA-Z0-9,!?.\-\'\" ]', '', text)

        # replace continuous characters like . ! ?
        text = re.sub(r'\.{2,}', ' ... ', text)
        for c in exception_punctuation_list:
            if c != '.':
                text = re.sub(rf'{re.escape(c)}{{2,}}', c, text)
        return text

    def autocorrect_text(self, text):
        """
        Applies spelling correction to the input text using the autocorrect library.
        Command to install the library using pip:
        !pip install autocorrect

        Arguments:
        text -- The input text with potential spelling errors.

        Returns:
        str -- The input text with spelling errors corrected.
        """

        words = text.split()
        corrected_words = []

        for word in words:
            corrected_word = self.spell(word)
            if corrected_word != word:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

        corrected_text = " ".join(corrected_words)
        return corrected_text

    def convert_abbrev_in_text(self, text):
        """
        Converts common abbreviations in the input text to their full forms using a predefined list of abbreviations.

        Arguments:
        text -- The input text with abbreviations.

        Returns:
        str -- The input text with abbreviations converted to full forms.
        """
        words = text.split()
        converted_words = [self.abbreviations.get(word.lower(), word) for word in words]
        converted_text = ' '.join(converted_words)
        return converted_text

    def lemma(self, text):
        """
        Lemmatizes the words in the input text using NLTK's WordNet lemmatizer.
        You might need to download from NLTK using these commands:

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        Arguments:
        text -- The input text with words to be lemmatized.

        Returns:
        str -- The lemmatized version of the input text.
        """

        word_net_lemma = WordNetLemmatizer()

        tokens = nltk.word_tokenize(text)
        pos_tags = pos_tag(tokens)

        pos_tags = [(token, self.get_wordnet_pos(tag)) for token, tag in pos_tags]

        lemmatized_tokens = [word_net_lemma.lemmatize(token, pos=tag) for token, tag in pos_tags]

        return ' '.join(lemmatized_tokens)

    def get_wordnet_pos(self, treebank_tag):
        """
        Maps a Penn Treebank POS tag to a corresponding WordNet POS tag.

        Arguments:
        treebank_tag -- The POS tag from the Penn Treebank.

        Returns:
        str -- The corresponding WordNet POS tag.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def to_lowercase(self, text):
        """
        Converts all characters in the input text to lowercase.

        Arguments:
        text -- The input text.

        Returns:
        str -- The input text with all characters converted to lowercase.
        """
        return text.lower()

    def process_text(self, input_text,
                     urls=True,
                     punctuation=True,
                     abbreviations=True,
                     html_tags=True,
                     emoji=True,
                     lowercase=True,
                     contractions=True,

                     spelling=False,
                     lemma=False
                     ):

        """
        Processes the input text based on specified preprocessing options.

        Arguments:
        input_text -- The input text to be processed.

        Boolean flags to use the corresponding functions:
        lowercase - contractions - urls - punctuation - html_tags - emoji - spelling - abbreviations - lemma
        """

        processed_text = input_text

        # remove
        if urls:
            processed_text = self.remove_urls(processed_text)
        if html_tags:
            processed_text = self.remove_html_tags(processed_text)
        if emoji:
            processed_text = self.remove_emoji(processed_text)

        # replace
        if lowercase:
            processed_text = self.to_lowercase(processed_text)
        if abbreviations:  # lmao
            processed_text = self.convert_abbrev_in_text(processed_text)
        if contractions:  # It's
            processed_text = self.replace_contractions(processed_text)
        if punctuation:  # good.so -> good . so
            processed_text = self.remove_punctuations_from_words(processed_text)
        if lemma:
            processed_text = self.lemma(processed_text)
        if spelling:
            processed_text = self.autocorrect_text(processed_text)

        return processed_text

# %%
