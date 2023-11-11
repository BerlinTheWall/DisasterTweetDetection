import unittest
from Preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor()

    def test_remove_urls(self):
        input_text = "Visit our website at https://www.example.com"
        expected_output = "Visit our website at "
        print(input_text)
        self.assertEqual(expected_output, self.preprocessor.remove_urls(input_text))

    def test_remove_html_tags(self):
        input_text = "<p>This is a <b>bold</b> statement.</p>"
        expected_output = "This is a bold statement."
        self.assertEqual(expected_output, self.preprocessor.remove_html_tags(input_text))

    def test_remove_emoji(self):
        input_text = "I love Python! 😍🐍"
        expected_output = "I love Python! "
        self.assertEqual(expected_output, self.preprocessor.remove_emoji(input_text))

    def test_separate_punctuations_from_words(self):
        input_text = "Hello, how are you?"
        expected_output = "Hello ,  how are you ? "
        self.assertEqual(expected_output,
                         self.preprocessor.separate_punctuations_from_words(input_text)
                         )

    def test_autocorrect_text(self):
        input_text = "Speling misteaks can be embarassing."
        expected_output = "Spelling mistake can be embarrassing."
        self.assertEqual(expected_output, self.preprocessor.autocorrect_text(input_text)
                         )

    def test_convert_abbrev_in_text(self):
        input_text = "btw"
        expected_output = "by the way"
        self.assertEqual(expected_output,
                         self.preprocessor.convert_abbrev_in_text(input_text)
                         )

    def test_lemma(self):
        input_text = "running"
        expected_output = "run"
        self.assertEqual(expected_output, self.preprocessor.lemma(input_text))

    def test_to_lowercase(self):
        input_text = "Convert This Text To Lowercase"
        expected_output = "convert this text to lowercase"
        self.assertEqual(expected_output, self.preprocessor.to_lowercase(input_text))

    def test_process_text(self):
        input_text = "Hello, World! Visit https://www.example.com for more info."
        expected_output = "hello ,  world !  visit  for more info . "
        self.assertEqual(
            expected_output,
            self.preprocessor.process_text(
                input_text, urls=True, punctuation=True, lowercase=True
            )
        )


if __name__ == "__main__":
    unittest.main()
