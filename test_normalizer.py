import unittest
from text import normalize


class TestTextNormalizer(unittest.TestCase):

    # --------------------------
    # English
    # --------------------------
    def test_english_basic(self):
        text = "I can't buy 15 apples!!!"
        result = normalize(text, "en")
        self.assertEqual(result, "i cannot buy fifteen apples")

    def test_english_large_number(self):
        text = "The year is 2024."
        result = normalize(text, "en")
        self.assertEqual(result, "the year is two thousand twenty four")

    def test_english_negative(self):
        text = "Temperature is -5 degrees."
        result = normalize(text, "en")
        self.assertEqual(result, "temperature is negative five degrees")

    def test_english_model_number(self):
        text = "model3 is better than model2."
        result = normalize(text, "en")
        self.assertEqual(result, "modelthree is better than modeltwo")

    # --------------------------
    # Hindi
    # --------------------------
    def test_hindi_basic(self):
        text = "मेरे पास 25 किताबें हैं।"
        result = normalize(text, "hi")
        self.assertEqual(result, "मेरे पास पच्चीस किताबें हैं")

    def test_hindi_large(self):
        text = "1000 लोग आए।"
        result = normalize(text, "hi")
        self.assertEqual(result, "एक हज़ार लोग आए")

    # --------------------------
    # Chinese
    # --------------------------
    def test_chinese_basic(self):
        text = "我有25个苹果。"
        result = normalize(text, "zh")
        self.assertEqual(result, "我有二十五个苹果")

    def test_chinese_large(self):
        text = "2024年到了。"
        result = normalize(text, "zh")
        self.assertEqual(result, "二千零二十四年到了")

    # --------------------------
    # Arabic
    # --------------------------
    def test_arabic_plural(self):
        text = "لدي 3 كتب."
        result = normalize(text, "ar")
        self.assertIn("ثلاثة", result)

    # --------------------------
    # Russian
    # --------------------------
    def test_russian_plural(self):
        text = "У меня 21 книга."
        result = normalize(text, "ru")
        self.assertIn("двадцать один", result)

    # --------------------------
    # Hebrew
    # --------------------------
    def test_hebrew_dual(self):
        text = "יש לי 2 ספרים."
        result = normalize(text, "he")
        self.assertIn("שתיים", result)

    # --------------------------
    # Regional locale aliases
    # --------------------------
    def test_regional_alias(self):
        text = "I have 10 dogs."
        result = normalize(text, "en-us")
        self.assertEqual(result, "i have ten dogs")

    # --------------------------
    # Punctuation removal
    # --------------------------
    def test_punctuation(self):
        text = "Hello!!!   World???"
        result = normalize(text, "en")
        self.assertEqual(result, "hello world")

    # --------------------------
    # Idempotency
    # --------------------------
    def test_idempotent(self):
        text = "I can't buy 15 apples!!!"
        once = normalize(text, "en")
        twice = normalize(once, "en")
        self.assertEqual(once, twice)

    # --------------------------
    # Empty string
    # --------------------------
    def test_empty(self):
        self.assertEqual(normalize("", "en"), "")


if __name__ == "__main__":
    unittest.main()