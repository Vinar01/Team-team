import unittest
from text import TextNormalizer


class TestTextNormalizer(unittest.TestCase):

    def test_english(self):
        normalizer = TextNormalizer(language="en")
        result = normalizer.normalize("I can't buy 15 apples!!!")
        self.assertEqual(result, "i cannot buy fifteen apples")

    def test_french(self):
        normalizer = TextNormalizer(language="fr")
        result = normalizer.normalize("J'ai 12 pommes.                ")
        self.assertEqual(result, "je ai douze pommes")

    def test_hindi(self):
        normalizer = TextNormalizer(language="hi")
        result = normalizer.normalize("मेरे पास 5 किताबें हैं।")
        self.assertEqual(result, "मेरे पास पांच किताबें हैं")

    def test_spacing(self):
        normalizer = TextNormalizer(language="en")
        result = normalizer.normalize("Hello     world   ")
        self.assertEqual(result, "hello world")

    def test_no_numbers(self):
        normalizer = TextNormalizer(language="en")
        result = normalizer.normalize("Just text.")
        self.assertEqual(result, "just text")


if __name__ == "__main__":
    unittest.main()