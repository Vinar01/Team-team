#!/usr/bin/env python3
"""
Text Normalization Script
Usage: python3 text.py <file_path> <language_code>
Example: python3 text.py input.txt en

Normalization pipeline:
  1. Unicode NFKC normalization
  2. Lowercase (skipped for caseless scripts, detected via babel)
  3. Contraction expansion  (loaded from contractions.json)
  4. Standalone integers to word form  (via num2words)
  5. Punctuation removal (via unicodedata categories, all scripts)
  6. Whitespace normalization
"""

import sys
import re
import json
import unicodedata
from pathlib import Path
from functools import lru_cache

from num2words import num2words
from babel import Locale, UnknownLocaleError



# ---------------------------------------------------------------------------
# CONTRACTIONS  (loaded from contractions.json — no hardcoding here)
# ---------------------------------------------------------------------------

def _load_contractions() -> dict:
    path = Path(__file__).with_name("contractions.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


CONTRACTIONS: dict = _load_contractions()


# ---------------------------------------------------------------------------
# CASELESS SCRIPT DETECTION  (fully babel-driven — no hardcoded language sets)
# ---------------------------------------------------------------------------

# ISO 15924 script codes (from Unicode CLDR) that have no case distinction.
_CASELESS_SCRIPTS: frozenset = frozenset({
    "Arab", "Hebr", "Deva", "Beng", "Guru", "Gujr", "Orya",
    "Taml", "Telu", "Knda", "Mlym", "Sinh", "Thai", "Laoo",
    "Mymr", "Khmr", "Tibt", "Ethi", "Thaa", "Geor",
    "Hani", "Hans", "Hant", "Hang",
})


@lru_cache(maxsize=256)
def _is_caseless_lang(lang: str) -> bool:
    """Return True when the locale's script has no case distinction."""
    try:
        loc = Locale.parse(lang.replace("-", "_"))
        script = loc.script
        if script is None:
            script = getattr(loc.likely_subtags, "script", None)
        return str(script or "") in _CASELESS_SCRIPTS
    except (UnknownLocaleError, ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# NUMBER TO WORDS  (num2words for supported langs; built-in for hi / zh)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _hi_data() -> dict:
    """Load Hindi number word data from num_words_hi.json."""
    return _load_json_data("num_words_hi.json")


@lru_cache(maxsize=1)
def _zh_data() -> dict:
    """Load Chinese number word data from num_words_zh.json."""
    return _load_json_data("num_words_zh.json")


def _load_json_data(filename: str) -> dict:
    path = Path(__file__).with_name(filename)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _hi_int_part(n: int) -> str:
    """Recursive helper — returns '' for 0 so scale suffixes stay clean."""
    if n == 0:
        return ""
    d = _hi_data()
    words = d["words"]
    if n <= 99:
        return words[n]
    if n < 1_000:
        rest = _hi_int_part(n % 100)
        return words[n // 100] + " " + d["sou"] + (" " + rest if rest else "")
    if n < 1_00_000:
        rest = _hi_int_part(n % 1_000)
        return _hi_int_part(n // 1_000) + " " + d["hazar"] + (" " + rest if rest else "")
    if n < 1_00_00_000:
        rest = _hi_int_part(n % 1_00_000)
        return _hi_int_part(n // 1_00_000) + " " + d["lakh"] + (" " + rest if rest else "")
    if n < 1_00_00_00_000:  # < 1,00,00,00,000 = 1 arab (1 billion)
        rest = _hi_int_part(n % 1_00_00_000)
        return _hi_int_part(n // 1_00_00_000) + " " + d["crore"] + (" " + rest if rest else "")
    rest = _hi_int_part(n % 1_00_00_00_000)
    return _hi_int_part(n // 1_00_00_00_000) + " " + d["arab"] + (" " + rest if rest else "")


def _hi_num_to_words(n: int) -> str:
    """Convert integer to Hindi word form (e.g. 25 → पच्चीस, 1000 → एक हज़ार)."""
    d = _hi_data()
    if n == 0:
        return d["words"][0]
    if n < 0:
        return d["neg"] + " " + _hi_int_part(-n)
    return _hi_int_part(n)


def _zh_num_to_words(n: int) -> str:
    """Convert integer to Chinese word form (e.g. 25 → 二十五, 2024 → 两千零二十四)."""
    d = _zh_data()
    digits, units = d["digits"], d["units"]
    if n == 0:
        return digits[0]
    if n < 0:
        return d["neg"] + _zh_num_to_words(-n)
    def _liang_if_two(s: str) -> str:
        """Use 两 instead of 二 when 2 is a standalone multiplier before 万/亿."""
        return d["two"] if s == digits[2] else s

    # Numbers ≥ 亿 (100 million)
    if n >= 1_0000_0000:
        mul = _liang_if_two(_zh_num_to_words(n // 1_0000_0000))
        yi_part = mul + d["yi"]
        rest = n % 1_0000_0000
        if rest == 0:
            return yi_part
        prefix = digits[0] if rest < 1_000_0000 else ""
        return yi_part + prefix + _zh_num_to_words(rest)
    # Numbers ≥ 万 (10 000)
    if n >= 1_0000:
        mul = _liang_if_two(_zh_num_to_words(n // 1_0000))
        wan_part = mul + d["wan"]
        rest = n % 1_0000
        if rest == 0:
            return wan_part
        prefix = digits[0] if rest < 1_000 else ""
        return wan_part + prefix + _zh_num_to_words(rest)
    # 1 – 9999: positional with zero-insertion
    s = str(n)
    length = len(s)
    result = ""
    need_zero = False
    for i, ch in enumerate(s):
        digit = int(ch)
        pos = length - 1 - i          # 0 = ones, 1 = tens, 2 = hundreds, 3 = thousands
        if digit == 0:
            if pos > 0:
                need_zero = True
        else:
            if need_zero:
                result += digits[0]
                need_zero = False
            # Use 两 for 2 in hundreds/thousands position (not tens)
            digit_char = d["two"] if (digit == 2 and pos >= 2) else digits[digit]
            # Omit leading 一 before 十  (e.g. 10 → 十, not 一十)
            if digit == 1 and pos == 1 and i == 0:
                digit_char = ""
            result += digit_char + (units[pos] if pos < len(units) else "")
    return result


def _num_to_words(n: int, lang: str) -> str:
    """Convert integer n to its word form in lang."""
    base = lang.lower().split("-")[0].split("_")[0]
    if base == "hi":
        return _hi_num_to_words(n)
    if base == "zh":
        return _zh_num_to_words(n)
    lang_key = lang.replace("-", "_")
    for key in (lang_key, lang_key.split("_")[0]):
        try:
            return num2words(n, lang=key)
        except Exception:
            continue
    return str(n)

# ---------------------------------------------------------------------------
# NORMALIZATION PIPELINE
# ---------------------------------------------------------------------------

def normalize(text: str, lang: str) -> str:
    """
    Full normalization pipeline for `text` in the given `lang`.

    Libraries in play:
      unicodedata.normalize  -> Step 1 NFKC
      babel.Locale           -> Step 2 caseless-script detection
      unicodedata.category   -> Step 5 unicode-aware punctuation removal
      babel.Locale.plural_form -> inside NumberToWords for Slavic/Arabic/Hebrew
    """
    base_lang = lang.lower().split("-")[0].split("_")[0]

    # Step 1 — Unicode NFKC
    # Decomposes ligatures (ﬁ -> fi), fullwidth chars, etc., then recomposes.
    text = unicodedata.normalize("NFKC", text)

    # Step 2 — Lowercase
    # babel resolves the locale; _is_caseless_lang checks its script code.
    # Scripts like Arabic, Hebrew, Devanagari, CJK have no case — skip lowercasing.
    if not _is_caseless_lang(lang):
        text = text.lower()

    # Step 3 — Contraction expansion
    contractions = CONTRACTIONS.get(base_lang, {})
    if contractions:
        for contraction in sorted(contractions, key=len, reverse=True):
            pattern = r'(?<!\S)' + re.escape(contraction) + r'(?!\S)'
            text = re.sub(pattern, contractions[contraction], text, flags=re.IGNORECASE)

    # Step 4 — Integer -> word form
    # Regex design:
    #   (?<![a-zA-Z\d_])  — not preceded by ASCII letter, digit, or underscore
    #                        so "model3" is skipped but "25歳" (CJK adjacency) is converted
    #   -?\d+             — optional minus sign then one or more digits
    #   (?![a-zA-Z\d_])   — not followed by ASCII letter, digit, or underscore
    def _replace_num(m: re.Match) -> str:
        return _num_to_words(int(m.group(0)), lang)

    text = re.sub(r'(?<![a-zA-Z\d_])-?\d+(?![a-zA-Z\d_])', _replace_num, text)

    # Step 5 — Punctuation removal via unicodedata categories
    # unicodedata.category(ch)[0] in ('L','M','N','Z') covers all scripts:
    #   L = Letters (Latin, CJK, Arabic, Devanagari ...)
    #   M = Combining marks (Devanagari vowel signs, Arabic diacritics ...)
    #   N = Numbers (now word-form, but handles any residual digit chars)
    #   Z = Separators (space variants)
    # Everything else (punctuation, symbols, control chars) becomes a space.
    def _remove_punct(s: str) -> str:
        out = []
        for ch in s:
            cat = unicodedata.category(ch)
            if cat[0] in ("L", "M", "N", "Z") or ch in " \t\n\r":
                out.append(ch)
            else:
                out.append(" ")
        return "".join(out)

    text = _remove_punct(text)

    # Step 6 — Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 text.py <file_path> <language_code>")
        print("Example: python3 text.py input.txt en")
        sys.exit(1)

    file_path, lang_code = sys.argv[1], sys.argv[2]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()

    print(normalize(text, lang_code))


if __name__ == "__main__":
    main()
