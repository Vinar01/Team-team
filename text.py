#!/usr/bin/env python3
"""
Text Normalization Script
Usage: python3 normalize_text.py <file_path> <language_code>
Example: python3 normalize_text.py input.txt en

Normalization pipeline (in order):
  1. Unicode NFKC normalization
  2. Lowercase (skipped for caseless scripts, auto-detected via unicodedata + babel)
  3. Contraction expansion (predefined per language)
  4. Standalone integers to word form (custom converters + babel plural_form)
  5. Punctuation removal (via unicodedata categories, preserves all scripts)
  6. Whitespace normalization

Libraries used:
  unicodedata  (stdlib)   NFKC, category-based punct removal, script detection
  babel        (installed) locale validation, CLDR plural_form() for correct
                           Slavic/Arabic/Hebrew scale-word agreement
"""

import sys
import re
import unicodedata
from functools import lru_cache
from babel import Locale, UnknownLocaleError


# ---------------------------------------------------------------------------
# SCRIPT / LOCALE HELPERS  (powered by unicodedata + babel)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _char_script(ch: str) -> str:
    """Return the Unicode script name for a character (first word of its name).
    e.g. 'ا' -> 'ARABIC', '字' -> 'CJK', 'a' -> 'LATIN', 'अ' -> 'DEVANAGARI'
    """
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return unicodedata.category(ch)


def _is_caseless_lang(lang: str) -> bool:
    """
    Return True when the language's primary script has no case distinction.
    Uses babel.Locale to resolve the script from the locale, with a fast-path
    hard-coded set for the most common caseless languages.
    """
    _CASELESS_CODES = {"ar", "he", "hi", "ja", "zh", "ko", "th", "my", "km", "bo", "si"}
    base = lang.lower().split("-")[0].split("_")[0]
    if base in _CASELESS_CODES:
        return True
    # Ask babel for less-obvious locales
    try:
        loc = Locale.parse(lang.replace("-", "_"))
        _CASELESS_BABEL_SCRIPTS = {
            "Arab", "Hebr", "Deva", "Thai", "Hani", "Hans", "Hant",
            "Hang", "Mymr", "Khmr", "Tibt", "Ethi", "Sinh", "Laoo",
        }
        if str(loc.script or "") in _CASELESS_BABEL_SCRIPTS:
            return True
    except (UnknownLocaleError, ValueError):
        pass
    return False


# ---------------------------------------------------------------------------
# CONTRACTIONS
# ---------------------------------------------------------------------------

CONTRACTIONS: dict = {
    "en": {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
        "he's": "he is", "he'll": "he will", "he'd": "he would",
        "she's": "she is", "she'll": "she will", "she'd": "she would",
        "it's": "it is", "it'll": "it will",
        "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
        "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
        "that's": "that is", "that'll": "that will", "that'd": "that would",
        "who's": "who is", "who'll": "who will", "who'd": "who would",
        "what's": "what is", "what'll": "what will", "what'd": "what did",
        "where's": "where is", "where'd": "where did",
        "when's": "when is", "when'd": "when did",
        "why's": "why is", "why'd": "why did",
        "how's": "how is", "how'd": "how did", "how'll": "how will",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "won't": "will not", "wouldn't": "would not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "couldn't": "could not",
        "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
        "let's": "let us", "there's": "there is", "there're": "there are",
        "here's": "here is", "could've": "could have", "should've": "should have",
        "would've": "would have", "might've": "might have", "must've": "must have",
        "ain't": "is not", "gonna": "going to", "wanna": "want to", "gotta": "got to",
    },
    "fr": {
        "j'ai": "je ai", "j'avais": "je avais", "j'aurai": "je aurai",
        "j'aurais": "je aurais", "j'etais": "je etais", "j'irai": "je irai",
        "j'irais": "je irais", "j'y": "je y", "j'en": "je en",
        "c'est": "ce est", "c'etait": "ce etait",
        "n'est": "ne est", "n'etait": "ne etait", "n'a": "ne a",
        "n'ont": "ne ont", "n'avait": "ne avait", "n'avaient": "ne avaient",
        "qu'il": "que il", "qu'elle": "que elle", "qu'ils": "que ils",
        "qu'elles": "que elles", "qu'on": "que on", "qu'un": "que un",
        "qu'une": "que une", "qu'en": "que en",
        "l'ai": "le ai", "l'a": "le a", "l'on": "le on",
        "d'un": "de un", "d'une": "de une", "d'abord": "de abord",
        "d'accord": "de accord",
        "m'a": "me a", "m'ont": "me ont",
        "s'est": "se est", "s'il": "se il", "s'ils": "se ils",
    },
    "es": {"al": "a el", "del": "de el"},
    "it": {
        "c'e": "ci e", "c'era": "ci era",
        "dov'e": "dove e", "com'e": "come e",
        "anch'io": "anche io", "tutt'altro": "tutto altro",
    },
    "pt": {
        "ao": "a o", "aos": "a os", "da": "de a", "das": "de as",
        "do": "de o", "dos": "de os", "na": "em a", "nas": "em as",
        "no": "em o", "nos": "em os", "num": "em um", "numa": "em uma",
        "pelo": "por o", "pelos": "por os", "pela": "por a", "pelas": "por as",
    },
    "de": {
        "im": "in dem", "ins": "in das", "am": "an dem", "ans": "an das",
        "zum": "zu dem", "zur": "zu der", "vom": "von dem", "beim": "bei dem",
        "aufs": "auf das", "furs": "fur das", "ums": "um das",
    },
}


# ---------------------------------------------------------------------------
# NUMBER TO WORDS  (hand-rolled converters + babel plural_form for agreement)
# ---------------------------------------------------------------------------

class NumberToWords:
    """
    Convert integers to word form in many languages.

    Key upgrade: babel.Locale.plural_form(n) provides CLDR-accurate plural
    categories ('one', 'few', 'many', 'other', 'two', 'zero') for each locale,
    eliminating manual mod-arithmetic for Russian, Polish, Arabic, Hebrew scale
    word agreement.
    """

    # ---- English ----
    _EN_ONES = ["", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    _EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety"]

    @classmethod
    def _en_lt1000(cls, n):
        if n < 20: return cls._EN_ONES[n]
        if n < 100:
            return cls._EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + cls._EN_ONES[n % 10])
        rest = cls._en_lt1000(n % 100)
        return cls._EN_ONES[n // 100] + " hundred" + (" " + rest if rest else "")

    @classmethod
    def en(cls, n):
        if n < 0: return "negative " + cls.en(-n)
        if n == 0: return "zero"
        parts = []
        for val, name in ((n // 1_000_000_000, "billion"),
                          ((n % 1_000_000_000) // 1_000_000, "million"),
                          ((n % 1_000_000) // 1_000, "thousand")):
            if val: parts.append(cls._en_lt1000(val) + " " + name)
        if n % 1_000: parts.append(cls._en_lt1000(n % 1_000))
        return " ".join(parts)

    # ---- French ----
    _FR_ONES = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
                "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
                "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"]

    @classmethod
    def _fr_lt100(cls, n):
        if n < 20: return cls._FR_ONES[n]
        tens_words = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante"]
        if n < 70:
            t, o = tens_words[n // 10], n % 10
            if o == 0: return t
            if o == 1: return t + " et un"
            return t + "-" + cls._FR_ONES[o]
        if n < 80: return "soixante-" + cls._FR_ONES[10 + n % 10]
        if n < 90:
            o = n % 10
            return "quatre-vingt" + ("-" + cls._FR_ONES[o] if o else "s")
        return "quatre-vingt-" + cls._FR_ONES[10 + n % 10]

    @classmethod
    def _fr_lt1000(cls, n):
        if n == 0: return ""
        if n < 100: return cls._fr_lt100(n)
        h, rest = n // 100, n % 100
        suffix = cls._fr_lt100(rest)
        if h == 1: return "cent" + (" " + suffix if suffix else "")
        return cls._FR_ONES[h] + " cent" + ("s" if not rest else "") + (" " + suffix if suffix else "")

    @classmethod
    def fr(cls, n):
        if n < 0: return "moins " + cls.fr(-n)
        if n == 0: return "zero"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._fr_lt1000(b) + " milliard" + ("s" if b > 1 else ""))
        if m: parts.append(cls._fr_lt1000(m) + " million" + ("s" if m > 1 else ""))
        if t == 1: parts.append("mille")
        elif t: parts.append(cls._fr_lt1000(t) + " mille")
        if r: parts.append(cls._fr_lt1000(r))
        return " ".join(parts)

    # ---- Spanish ----
    _ES_ONES = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
                "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
                "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve",
                "veinte", "veintiuno", "veintidos", "veintitres", "veinticuatro",
                "veinticinco", "veintiseis", "veintisiete", "veintiocho", "veintinueve"]
    _ES_TENS = ["", "", "veinte", "treinta", "cuarenta", "cincuenta",
                "sesenta", "setenta", "ochenta", "noventa"]
    _ES_HUNDS = ["", "ciento", "doscientos", "trescientos", "cuatrocientos",
                 "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"]

    @classmethod
    def _es_lt1000(cls, n):
        if n == 0: return ""
        if n == 100: return "cien"
        if n < 30: return cls._ES_ONES[n]
        if n < 100:
            t, o = cls._ES_TENS[n // 10], n % 10
            return t + (" y " + cls._ES_ONES[o] if o else "")
        rest = cls._es_lt1000(n % 100)
        return cls._ES_HUNDS[n // 100] + (" " + rest if rest else "")

    @classmethod
    def es(cls, n):
        if n < 0: return "menos " + cls.es(-n)
        if n == 0: return "cero"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._es_lt1000(b) + (" billon" if b == 1 else " billones"))
        if m: parts.append(cls._es_lt1000(m) + (" millon" if m == 1 else " millones"))
        if t == 1: parts.append("mil")
        elif t: parts.append(cls._es_lt1000(t) + " mil")
        if r: parts.append(cls._es_lt1000(r))
        return " ".join(parts)

    # ---- German ----
    _DE_ONES = ["", "ein", "zwei", "drei", "vier", "funf", "sechs", "sieben",
                "acht", "neun", "zehn", "elf", "zwolf", "dreizehn", "vierzehn",
                "funfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
    _DE_TENS = ["", "", "zwanzig", "dreissig", "vierzig", "funfzig",
                "sechzig", "siebzig", "achtzig", "neunzig"]

    @classmethod
    def _de_lt1000(cls, n):
        if n == 0: return ""
        if n < 20: return cls._DE_ONES[n]
        if n < 100:
            o = n % 10
            return (cls._DE_ONES[o] + "und" + cls._DE_TENS[n // 10]) if o else cls._DE_TENS[n // 10]
        rest = cls._de_lt1000(n % 100)
        return cls._DE_ONES[n // 100] + "hundert" + rest

    @classmethod
    def de(cls, n):
        if n < 0: return "minus " + cls.de(-n)
        if n == 0: return "null"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._de_lt1000(b) + "milliarde")
        if m: parts.append(cls._de_lt1000(m) + "millionen")
        if t: parts.append(cls._de_lt1000(t) + "tausend")
        if r: parts.append(cls._de_lt1000(r))
        return "".join(parts)

    # ---- Portuguese ----
    _PT_ONES = ["", "um", "dois", "tres", "quatro", "cinco", "seis", "sete",
                "oito", "nove", "dez", "onze", "doze", "treze", "quatorze",
                "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"]
    _PT_TENS = ["", "", "vinte", "trinta", "quarenta", "cinquenta",
                "sessenta", "setenta", "oitenta", "noventa"]
    _PT_HUNDS = ["", "cem", "duzentos", "trezentos", "quatrocentos",
                 "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos"]

    @classmethod
    def _pt_lt1000(cls, n):
        if n == 0: return ""
        if n == 100: return "cem"
        if n < 20: return cls._PT_ONES[n]
        if n < 100:
            o = n % 10
            return cls._PT_TENS[n // 10] + (" e " + cls._PT_ONES[o] if o else "")
        rest = cls._pt_lt1000(n % 100)
        return cls._PT_HUNDS[n // 100] + (" e " + rest if rest else "")

    @classmethod
    def pt(cls, n):
        if n < 0: return "menos " + cls.pt(-n)
        if n == 0: return "zero"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._pt_lt1000(b) + (" bilhao" if b == 1 else " bilhoes"))
        if m: parts.append(cls._pt_lt1000(m) + (" milhao" if m == 1 else " milhoes"))
        if t == 1: parts.append("mil")
        elif t: parts.append(cls._pt_lt1000(t) + " mil")
        if r: parts.append(cls._pt_lt1000(r))
        return " e ".join(parts)

    # ---- Italian ----
    _IT_ONES = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette",
                "otto", "nove", "dieci", "undici", "dodici", "tredici", "quattordici",
                "quindici", "sedici", "diciassette", "diciotto", "diciannove"]
    _IT_TENS = ["", "", "venti", "trenta", "quaranta", "cinquanta",
                "sessanta", "settanta", "ottanta", "novanta"]

    @classmethod
    def _it_lt1000(cls, n):
        if n == 0: return ""
        if n < 20: return cls._IT_ONES[n]
        if n < 100:
            o, t = n % 10, cls._IT_TENS[n // 10]
            if o in (1, 8): t = t.rstrip("aeiou")
            return t + (cls._IT_ONES[o] if o else "")
        h, rest = n // 100, cls._it_lt1000(n % 100)
        return ("cento" if h == 1 else cls._IT_ONES[h] + "cento") + rest

    @classmethod
    def it(cls, n):
        if n < 0: return "meno " + cls.it(-n)
        if n == 0: return "zero"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._it_lt1000(b) + "miliardi")
        if m: parts.append(cls._it_lt1000(m) + "milioni")
        if t == 1: parts.append("mille")
        elif t: parts.append(cls._it_lt1000(t) + "mila")
        if r: parts.append(cls._it_lt1000(r))
        return " ".join(parts)

    # ---- Dutch ----
    _NL_ONES = ["", "een", "twee", "drie", "vier", "vijf", "zes", "zeven",
                "acht", "negen", "tien", "elf", "twaalf", "dertien", "veertien",
                "vijftien", "zestien", "zeventien", "achttien", "negentien"]
    _NL_TENS = ["", "", "twintig", "dertig", "veertig", "vijftig",
                "zestig", "zeventig", "tachtig", "negentig"]

    @classmethod
    def _nl_lt1000(cls, n):
        if n == 0: return ""
        if n < 20: return cls._NL_ONES[n]
        if n < 100:
            o = n % 10
            return (cls._NL_ONES[o] + "en" + cls._NL_TENS[n // 10]) if o else cls._NL_TENS[n // 10]
        rest = cls._nl_lt1000(n % 100)
        return cls._NL_ONES[n // 100] + "honderd" + rest

    @classmethod
    def nl(cls, n):
        if n < 0: return "min " + cls.nl(-n)
        if n == 0: return "nul"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._nl_lt1000(b) + " miljard")
        if m: parts.append(cls._nl_lt1000(m) + " miljoen")
        if t: parts.append(cls._nl_lt1000(t) + "duizend")
        if r: parts.append(cls._nl_lt1000(r))
        return " ".join(parts)

    # ---- Russian ----
    # babel.Locale('ru').plural_form(n) correctly handles:
    #   n=1 -> 'one', n=11 -> 'many' (not 'one'!), n=21 -> 'one' (not 'many'!)
    # This replaces the error-prone hand-coded mod-10/mod-100 checks.
    _RU_ONES_M = ["", "odin", "dva", "tri", "chetyre", "pyat", "shest", "sem",
                  "vosem", "devyat", "desyat", "odinnadtsat", "dvenadtsat",
                  "trinadtsat", "chetyrnadtsat", "pyatnadtsat", "shestnadtsat",
                  "semnadtsat", "vosemnadtsat", "devyatnadtsat"]
    _RU_ONES_F = ["", "odna", "dve", "tri", "chetyre", "pyat", "shest", "sem",
                  "vosem", "devyat", "desyat", "odinnadtsat", "dvenadtsat",
                  "trinadtsat", "chetyrnadtsat", "pyatnadtsat", "shestnadtsat",
                  "semnadtsat", "vosemnadtsat", "devyatnadtsat"]
    _RU_TENS = ["", "", "dvadtsat", "tridtsat", "sorok", "pyatdesyat",
                "shestdesyat", "semdesyat", "vosemdesyat", "devyanosto"]
    _RU_HUNDS = ["", "sto", "dvesti", "trista", "chetyresta", "pyatsot",
                 "shessot", "semsot", "vosemsot", "devyatsot"]

    # Use Cyrillic forms (the latin above are for transliteration fallback)
    _RU_ONES_CYR_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь",
                      "восемь", "девять", "десять", "одиннадцать", "двенадцать",
                      "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
                      "семнадцать", "восемнадцать", "девятнадцать"]
    _RU_ONES_CYR_F = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь",
                      "восемь", "девять", "десять", "одиннадцать", "двенадцать",
                      "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
                      "семнадцать", "восемнадцать", "девятнадцать"]
    _RU_TENS_CYR = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
                    "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    _RU_HUNDS_CYR = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
                     "шестьсот", "семьсот", "восемьсот", "девятьсот"]

    _RU_LOCALE = None

    @classmethod
    def _ru_locale(cls):
        if cls._RU_LOCALE is None:
            cls._RU_LOCALE = Locale.parse("ru")
        return cls._RU_LOCALE

    @classmethod
    def _ru_lt1000(cls, n, feminine=False):
        if n == 0: return ""
        ones = cls._RU_ONES_CYR_F if feminine else cls._RU_ONES_CYR_M
        parts = []
        if n >= 100:
            parts.append(cls._RU_HUNDS_CYR[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._RU_TENS_CYR[n // 10])
            n %= 10
        if n: parts.append(ones[n])
        return " ".join(parts)

    @classmethod
    def ru(cls, n):
        if n < 0: return "минус " + cls.ru(-n)
        if n == 0: return "ноль"
        loc = cls._ru_locale()

        def _scale(val, sg, few, many, feminine=False):
            # babel plural_form replaces hand-coded modular arithmetic
            pf = loc.plural_form(val)
            word = {"one": sg, "few": few}.get(pf, many)
            return cls._ru_lt1000(val, feminine) + " " + word

        parts = []
        b = n // 1_000_000_000
        m = (n % 1_000_000_000) // 1_000_000
        t = (n % 1_000_000) // 1_000
        r = n % 1_000
        if b: parts.append(_scale(b, "миллиард", "миллиарда", "миллиардов"))
        if m: parts.append(_scale(m, "миллион", "миллиона", "миллионов"))
        if t: parts.append(_scale(t, "тысяча", "тысячи", "тысяч", feminine=True))
        if r: parts.append(cls._ru_lt1000(r))
        return " ".join(parts)

    # ---- Arabic ----
    # babel Arabic plural_form: 'zero'|'one'|'two'|'few'|'many'|'other'
    # 6-form system — replaces hand-coded mod-100/mod-10 checks.
    _AR_ONES = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة",
                "ثمانية", "تسعة", "عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر",
                "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
    _AR_TENS = ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون",
                "ستون", "سبعون", "ثمانون", "تسعون"]
    _AR_HUNDS = ["", "مئة", "مئتان", "ثلاثمئة", "أربعمئة", "خمسمئة",
                 "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة"]

    _AR_LOCALE = None

    @classmethod
    def _ar_locale(cls):
        if cls._AR_LOCALE is None:
            cls._AR_LOCALE = Locale.parse("ar")
        return cls._AR_LOCALE

    @classmethod
    def _ar_lt1000(cls, n):
        if n == 0: return ""
        if n < 20: return cls._AR_ONES[n]
        if n < 100:
            o = n % 10
            return (cls._AR_ONES[o] + " و" + cls._AR_TENS[n // 10]) if o else cls._AR_TENS[n // 10]
        rest = cls._ar_lt1000(n % 100)
        return cls._AR_HUNDS[n // 100] + (" و" + rest if rest else "")

    @classmethod
    def ar(cls, n):
        if n < 0: return "سالب " + cls.ar(-n)
        if n == 0: return "صفر"
        loc = cls._ar_locale()

        def _scale(val, one, two, few, many, other):
            pf = loc.plural_form(val)
            word = {"one": one, "two": two, "few": few, "many": many}.get(pf, other)
            return cls._ar_lt1000(val) + " " + word

        parts = []
        b = n // 1_000_000_000
        m = (n % 1_000_000_000) // 1_000_000
        t = (n % 1_000_000) // 1_000
        r = n % 1_000
        if b: parts.append(_scale(b, "مليار", "ملياران", "مليارات", "مليار", "مليار"))
        if m: parts.append(_scale(m, "مليون", "مليونان", "ملايين", "مليون", "مليون"))
        if t: parts.append(_scale(t, "ألف", "ألفان", "آلاف", "ألف", "ألف"))
        if r: parts.append(cls._ar_lt1000(r))
        return " و".join(parts)

    # ---- Hindi ---- (full lookup table for sub-100; Indian place-value above)
    _HI_NUMS = {
        0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच",
        6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस",
        11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह",
        16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस",
        21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
        26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस", 30: "तीस",
        31: "इकतीस", 32: "बत्तीस", 33: "तेंतीस", 34: "चौंतीस", 35: "पैंतीस",
        36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस", 40: "चालीस",
        41: "इकतालीस", 42: "बयालीस", 43: "तेंतालीस", 44: "चौवालीस", 45: "पैंतालीस",
        46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास",
        51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन",
        56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 60: "साठ",
        61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ",
        66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर", 70: "सत्तर",
        71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
        76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी", 80: "अस्सी",
        81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
        86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 90: "नब्बे",
        91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे",
        96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे",
    }

    @classmethod
    def hi(cls, n):
        if n < 0: return "ऋण " + cls.hi(-n)
        if n <= 99: return cls._HI_NUMS[n]
        parts = []
        for val, unit in ((n // 10_000_000, "करोड़"),
                          ((n % 10_000_000) // 100_000, "लाख"),
                          ((n % 100_000) // 1_000, "हज़ार"),
                          ((n % 1_000) // 100, "सौ")):
            if val: parts.append(cls._HI_NUMS[val] + " " + unit)
        if n % 100: parts.append(cls._HI_NUMS[n % 100])
        return " ".join(parts)

    # ---- Japanese ----
    _JA_K = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    @classmethod
    def _ja_lt10000(cls, n):
        if n == 0: return ""
        parts = []
        for val, u in ((n // 1000, "千"), ((n % 1000) // 100, "百"), ((n % 100) // 10, "十")):
            if val: parts.append(("" if val == 1 else cls._JA_K[val]) + u)
        if n % 10: parts.append(cls._JA_K[n % 10])
        return "".join(parts)

    @classmethod
    def ja(cls, n):
        if n < 0: return "マイナス" + cls.ja(-n)
        if n == 0: return "零"
        parts = []
        for val, u in ((n // 100_000_000, "億"), ((n % 100_000_000) // 10_000, "万")):
            if val: parts.append(cls._ja_lt10000(val) + u)
        if n % 10_000: parts.append(cls._ja_lt10000(n % 10_000))
        return "".join(parts)

    # ---- Chinese ----
    _ZH_K = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    @classmethod
    def _zh_lt10000(cls, n):
        if n == 0: return ""
        parts, prev_zero = [], False
        for val, u in ((n // 1000, "千"), ((n % 1000) // 100, "百"), ((n % 100) // 10, "十")):
            if val:
                parts.append(cls._ZH_K[val] + u)
                prev_zero = False
            elif parts and not prev_zero:
                parts.append("零")
                prev_zero = True
        if n % 10: parts.append(cls._ZH_K[n % 10])
        while parts and parts[-1] == "零": parts.pop()
        return "".join(parts)

    @classmethod
    def zh(cls, n):
        if n < 0: return "负" + cls.zh(-n)
        if n == 0: return "零"
        parts = []
        for val, u in ((n // 100_000_000, "亿"), ((n % 100_000_000) // 10_000, "万")):
            if val: parts.append(cls._zh_lt10000(val) + u)
        if n % 10_000: parts.append(cls._zh_lt10000(n % 10_000))
        return "".join(parts)

    # ---- Korean ----
    _KO_K = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

    @classmethod
    def _ko_lt10000(cls, n):
        if n == 0: return ""
        parts = []
        for val, u in ((n // 1000, "천"), ((n % 1000) // 100, "백"), ((n % 100) // 10, "십")):
            if val: parts.append(("" if val == 1 else cls._KO_K[val]) + u)
        if n % 10: parts.append(cls._KO_K[n % 10])
        return "".join(parts)

    @classmethod
    def ko(cls, n):
        if n < 0: return "마이너스 " + cls.ko(-n)
        if n == 0: return "영"
        parts = []
        for val, u in ((n // 100_000_000, "억"), ((n % 100_000_000) // 10_000, "만")):
            if val: parts.append(cls._ko_lt10000(val) + u)
        if n % 10_000: parts.append(cls._ko_lt10000(n % 10_000))
        return "".join(parts)

    # ---- Polish ----
    # babel Polish plural_form correctly gives:
    #   1 -> 'one', 2-4 -> 'few', 5-21 -> 'many', 22-24 -> 'few', 25+ -> 'many'
    _PL_ONES = ["", "jeden", "dwa", "trzy", "cztery", "piec", "szesc", "siedem",
                "osiem", "dziewiec", "dziesiec", "jedenascie", "dwanascie", "trzynascie",
                "czternascie", "pietnascie", "szesnascie", "siedemnascie", "osiemnascie",
                "dziewietnascie"]
    _PL_TENS = ["", "", "dwadziescia", "trzydziesci", "czterdziesci", "piecdziesiat",
                "szescdziesiat", "siedemdziesiat", "osiemdziesiat", "dziewiecdziesiat"]
    _PL_HUNDS = ["", "sto", "dwiescie", "trzysta", "czterysta", "piecset",
                 "szescset", "siedemset", "osiemset", "dziewiecset"]

    _PL_LOCALE = None

    @classmethod
    def _pl_locale(cls):
        if cls._PL_LOCALE is None:
            cls._PL_LOCALE = Locale.parse("pl")
        return cls._PL_LOCALE

    @classmethod
    def _pl_lt1000(cls, n):
        if n == 0: return ""
        parts = []
        if n >= 100:
            parts.append(cls._PL_HUNDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._PL_TENS[n // 10])
            n %= 10
        if n: parts.append(cls._PL_ONES[n])
        return " ".join(parts)

    @classmethod
    def pl(cls, n):
        if n < 0: return "minus " + cls.pl(-n)
        if n == 0: return "zero"
        loc = cls._pl_locale()

        def _scale(val, sg, few, many):
            pf = loc.plural_form(val)
            word = {"one": sg, "few": few}.get(pf, many)
            return cls._pl_lt1000(val) + " " + word

        parts = []
        b = n // 1_000_000_000
        m = (n % 1_000_000_000) // 1_000_000
        t = (n % 1_000_000) // 1_000
        r = n % 1_000
        if b: parts.append(_scale(b, "miliard", "miliardy", "miliardow"))
        if m: parts.append(_scale(m, "milion", "miliony", "milionow"))
        if t: parts.append(_scale(t, "tysiac", "tysiace", "tysiecy"))
        if r: parts.append(cls._pl_lt1000(r))
        return " ".join(parts)

    # ---- Swedish ----
    _SV_ONES = ["", "ett", "tva", "tre", "fyra", "fem", "sex", "sju",
                "atta", "nio", "tio", "elva", "tolv", "tretton", "fjorton",
                "femton", "sexton", "sjutton", "arton", "nitton"]
    _SV_TENS = ["", "", "tjugo", "trettio", "fyrtio", "femtio",
                "sextio", "sjuttio", "attio", "nittio"]

    @classmethod
    def _sv_lt1000(cls, n):
        if n == 0: return ""
        if n < 20: return cls._SV_ONES[n]
        if n < 100:
            o = n % 10
            return cls._SV_TENS[n // 10] + (cls._SV_ONES[o] if o else "")
        rest = cls._sv_lt1000(n % 100)
        return cls._SV_ONES[n // 100] + "hundra" + rest

    @classmethod
    def sv(cls, n):
        if n < 0: return "minus " + cls.sv(-n)
        if n == 0: return "noll"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._sv_lt1000(b) + " miljarder")
        if m: parts.append(cls._sv_lt1000(m) + " miljoner")
        if t == 1: parts.append("ettusen")
        elif t: parts.append(cls._sv_lt1000(t) + "tusen")
        if r: parts.append(cls._sv_lt1000(r))
        return " ".join(parts)

    # ---- Turkish ----
    _TR_ONES = ["", "bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz", "dokuz"]
    _TR_TENS = ["", "on", "yirmi", "otuz", "kirk", "elli", "altmis", "yetmis", "seksen", "doksan"]

    @classmethod
    def _tr_lt1000(cls, n):
        if n == 0: return ""
        parts = []
        if n >= 100:
            h = n // 100
            parts.append(("" if h == 1 else cls._TR_ONES[h]) + "yuz")
            n %= 100
        if n >= 10:
            parts.append(cls._TR_TENS[n // 10])
            n %= 10
        if n: parts.append(cls._TR_ONES[n])
        return " ".join(p for p in parts if p)

    @classmethod
    def tr(cls, n):
        if n < 0: return "eksi " + cls.tr(-n)
        if n == 0: return "sifir"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._tr_lt1000(b) + " milyar")
        if m: parts.append(cls._tr_lt1000(m) + " milyon")
        if t == 1: parts.append("bin")
        elif t: parts.append(cls._tr_lt1000(t) + " bin")
        if r: parts.append(cls._tr_lt1000(r))
        return " ".join(parts)

    # ---- Greek ----
    _EL_ONES = ["", "ena", "dyo", "tria", "tessera", "pente", "exi", "epta",
                "okto", "ennea", "deka", "endeka", "dodeka", "dekatria", "dekatessera",
                "dekapente", "dekaexi", "dekaepta", "dekaokto", "dekaennea"]
    _EL_TENS = ["", "", "eikosi", "trianta", "saranta", "peninta",
                "exinta", "ebdominta", "ogdonta", "eneninta"]
    _EL_HUNDS = ["", "ekato", "diakosia", "triakosia", "tetrakosia", "pentakosia",
                 "eXakosia", "eptakosia", "oktakosia", "ennakosia"]

    @classmethod
    def _el_lt1000(cls, n):
        if n == 0: return ""
        parts = []
        if n >= 100:
            parts.append(cls._EL_HUNDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._EL_TENS[n // 10])
            n %= 10
        if n: parts.append(cls._EL_ONES[n])
        return " ".join(parts)

    @classmethod
    def el(cls, n):
        if n < 0: return "meion " + cls.el(-n)
        if n == 0: return "miden"
        parts = []
        b, m = n // 1_000_000_000, (n % 1_000_000_000) // 1_000_000
        t, r = (n % 1_000_000) // 1_000, n % 1_000
        if b: parts.append(cls._el_lt1000(b) + " disekatommyria")
        if m: parts.append(cls._el_lt1000(m) + " ekatommyria")
        if t: parts.append(cls._el_lt1000(t) + " chilia")
        if r: parts.append(cls._el_lt1000(r))
        return " ".join(parts)

    # ---- Hebrew ----
    # babel Hebrew plural_form: 'one'|'two'|'many'|'other'
    # The dual form ('two') is unique to Semitic languages; babel handles it correctly.
    _HE_ONES = ["", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע",
                "שמונה", "תשע", "עשר", "אחת עשרה", "שתים עשרה", "שלוש עשרה",
                "ארבע עשרה", "חמש עשרה", "שש עשרה", "שבע עשרה", "שמונה עשרה",
                "תשע עשרה"]
    _HE_TENS = ["", "", "עשרים", "שלושים", "ארבעים", "חמישים",
                "שישים", "שבעים", "שמונים", "תשעים"]
    _HE_HUNDS = ["", "מאה", "מאתיים", "שלוש מאות", "ארבע מאות", "חמש מאות",
                 "שש מאות", "שבע מאות", "שמונה מאות", "תשע מאות"]

    _HE_LOCALE = None

    @classmethod
    def _he_locale(cls):
        if cls._HE_LOCALE is None:
            cls._HE_LOCALE = Locale.parse("he")
        return cls._HE_LOCALE

    @classmethod
    def _he_lt1000(cls, n):
        if n == 0: return ""
        parts = []
        if n >= 100:
            parts.append(cls._HE_HUNDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._HE_TENS[n // 10])
            n %= 10
        if n: parts.append(cls._HE_ONES[n])
        return " ".join(parts)

    @classmethod
    def he(cls, n):
        if n < 0: return "מינוס " + cls.he(-n)
        if n == 0: return "אפס"
        loc = cls._he_locale()

        def _scale(val, sg, dual, plural):
            pf = loc.plural_form(val)
            word = {"one": sg, "two": dual}.get(pf, plural)
            return cls._he_lt1000(val) + " " + word

        parts = []
        m, t, r = n // 1_000_000, (n % 1_000_000) // 1_000, n % 1_000
        if m: parts.append(_scale(m, "מיליון", "מיליון", "מיליון"))
        if t: parts.append(_scale(t, "אלף", "אלפיים", "אלפים"))
        if r: parts.append(cls._he_lt1000(r))
        return " ".join(parts)

    # ---- Dispatch ----
    _DISPATCH: dict = {}

    @classmethod
    def _build_dispatch(cls):
        base = {
            "en": cls.en, "fr": cls.fr, "es": cls.es, "de": cls.de,
            "pt": cls.pt, "it": cls.it, "nl": cls.nl, "ru": cls.ru,
            "ar": cls.ar, "hi": cls.hi, "ja": cls.ja, "zh": cls.zh,
            "ko": cls.ko, "pl": cls.pl, "sv": cls.sv, "tr": cls.tr,
            "el": cls.el, "he": cls.he,
        }
        aliases = {
            "en": ["en-us","en-gb","en-au","en-ca","en-in","en-nz"],
            "fr": ["fr-fr","fr-ca","fr-be","fr-ch"],
            "es": ["es-es","es-mx","es-ar","es-co","es-cl","es-pe"],
            "de": ["de-de","de-at","de-ch"],
            "pt": ["pt-pt","pt-br"],
            "it": ["it-it","it-ch"],
            "nl": ["nl-nl","nl-be"],
            "ru": ["ru-ru"],
            "ar": ["ar-sa","ar-eg","ar-ma","ar-iq","ar-ae","ar-dz"],
            "hi": ["hi-in"],
            "ja": ["ja-jp"],
            "zh": ["zh-cn","zh-tw","zh-hk","zh-sg"],
            "ko": ["ko-kr"],
            "pl": ["pl-pl"],
            "sv": ["sv-se","sv-fi"],
            "tr": ["tr-tr"],
            "el": ["el-gr"],
            "he": ["he-il"],
        }
        cls._DISPATCH = {k: v for k, v in base.items()}
        for b_lang, alias_list in aliases.items():
            for alias in alias_list:
                cls._DISPATCH[alias] = base[b_lang]

    @classmethod
    def convert(cls, n: int, lang: str) -> str:
        if not cls._DISPATCH:
            cls._build_dispatch()
        key = lang.lower()
        fn = (cls._DISPATCH.get(key)
              or cls._DISPATCH.get(key.split("-")[0])
              or cls._DISPATCH.get(key.split("_")[0]))
        return fn(n) if fn else str(n)


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
    #   - Blocks conversion when a digit touches ASCII letters/underscore (e.g. "model3")
    #   - Allows conversion when digits touch non-ASCII script chars (e.g. Japanese "25歳")
    #   - Treats leading minus as negative sign only after whitespace/start-of-line,
    #     preventing Hebrew makaf "ו-100" from being read as negative 100
    def _replace_num(m: re.Match) -> str:
        return NumberToWords.convert(int(m.group(0)), lang)

    text = re.sub(r'-?\d+', _replace_num, text)

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
        print("Usage: python3 normalize_text.py <file_path> <language_code>")
        print("Example: python3 normalize_text.py input.txt en")
        print()
        print("Supported language codes:")
        print("  en fr es de pt it nl ru ar hi ja zh ko pl sv tr el he")
        print("  (and regional variants: en-us, fr-ca, zh-tw, pt-br, ...)")
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