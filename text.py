import re
import regex
import unicodedata
from num2words import num2words

# Optional (for Chinese segmentation)
try:
    import jieba
except ImportError:
    jieba = None


# ------------------------------
# 1️⃣ Unicode Standardization
# ------------------------------
def unicode_standardize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ------------------------------
# 2️⃣ Language-Aware Lowercasing
# ------------------------------
def lowercase(text: str, lang: str) -> str:
    # Turkish special casing
    if lang.startswith("tr"):
        text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


# ------------------------------
# 3️⃣ Remove Arabic Diacritics
# ------------------------------
def remove_arabic_diacritics(text: str) -> str:
    arabic_diacritics = regex.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return arabic_diacritics.sub('', text)


# ------------------------------
# 4️⃣ Number Normalization
# ------------------------------
def normalize_numbers(text: str, lang: str) -> str:
    def replace_number(match):
        number = match.group()
        try:
            return num2words(int(number), lang=lang)
        except:
            return number

    return re.sub(r'\b\d+\b', replace_number, text)


# ------------------------------
# 5️⃣ Remove Non-Meaningful Punctuation
# ------------------------------
def remove_punctuation(text: str) -> str:
    # Keep letters, numbers, whitespace, apostrophes
    return regex.sub(r"[^\p{L}\p{N}\s']", "", text)


# ------------------------------
# 6️⃣ Chinese Word Segmentation (Optional)
# ------------------------------
def segment_chinese(text: str) -> str:
    if jieba is None:
        return text
    return " ".join(jieba.cut(text))


# ------------------------------
# 7️⃣ Normalize Whitespace
# ------------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------
# 🔥 Main Normalization Function
# ------------------------------
def normalize(text: str, lang: str = "en") -> str:
    text = unicode_standardize(text)
    text = lowercase(text, lang)

    if lang.startswith("ar"):
        text = remove_arabic_diacritics(text)

    text = normalize_numbers(text, lang)
    text = remove_punctuation(text)

    if lang.startswith("zh"):
        text = segment_chinese(text)

    text = normalize_whitespace(text)

    return text