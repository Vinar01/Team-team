import regex
import unicodedata
from num2words import num2words


# ------------------------------
# 1️⃣ Unicode Standardization
# ------------------------------
def unicode_standardize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ------------------------------
# 2️⃣ Lowercasing
# ------------------------------
def lowercase(text: str) -> str:
    return text.lower()


# ------------------------------
# 3️⃣ Remove Diacritics (All Languages)
# ------------------------------
def remove_diacritics(text: str) -> str:
    # Remove all combining marks (category M)
    return regex.sub(r"\p{M}+", "", text)


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

    return regex.sub(r"\b\d+\b", replace_number, text)


# ------------------------------
# 5️⃣ Remove Punctuation (Unicode Safe)
# ------------------------------
def remove_punctuation(text: str) -> str:
    # Remove all Unicode punctuation characters
    return regex.sub(r"\p{P}+", "", text)


# ------------------------------
# 6️⃣ Normalize Whitespace
# ------------------------------
def normalize_whitespace(text: str) -> str:
    return regex.sub(r"\s+", " ", text).strip()


# ------------------------------
# 🔥 Main Function
# ------------------------------
def normalize(text: str, lang: str = "en") -> str:
    text = unicode_standardize(text)
    text = lowercase(text)
    if lang.startswith("ar"):
        text = regex.sub(r"\p{M}+", "", text)
    text = normalize_numbers(text, lang)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ------------------------------
# 🧪 Test
# ------------------------------
import json

# ------------------------------
# 🧪 Test + Save to JSON
# ------------------------------
if __name__ == "__main__":
    samples = {
        "en": "The meeting starts at 5.",
        "es": "La reunión empieza a las 5.",
        "ar": "الاجتماع يبدأ الساعة 5",
        "zh": "会议在5点开始",
        "hi": "नमस्कार, चलिए 5 बजे मिलते हैं।"
    }

    results = {}

    for lang, text in samples.items():
        results[lang] = {
            "original": text,
            "normalized": normalize(text, lang)
        }

    # Save to JSON file
    with open("normalized_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Results saved to normalized_output.json")