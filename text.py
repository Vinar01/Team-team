#!/usr/bin/env python3
"""
Text Normalization Script
Usage: python3 normalize_text.py <file_path> <language_code>
Example: python3 normalize_text.py input.txt en
"""

import sys
import re
import unicodedata


# ---------------------------------------------------------------------------
# 1. CONTRACTIONS
# ---------------------------------------------------------------------------

CONTRACTIONS: dict[str, dict[str, str]] = {
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
        "j'aurais": "je aurais", "j'étais": "je étais", "j'irai": "je irai",
        "j'irais": "je irais", "j'y": "je y", "j'en": "je en",
        "je n'ai": "je ne ai", "je n'avais": "je ne avais",
        "c'est": "ce est", "c'était": "ce était", "c'est-à-dire": "ce est à dire",
        "n'est": "ne est", "n'était": "ne était", "n'a": "ne a",
        "n'ont": "ne ont", "n'avait": "ne avait", "n'avaient": "ne avaient",
        "qu'il": "que il", "qu'elle": "que elle", "qu'ils": "que ils",
        "qu'elles": "que elles", "qu'on": "que on", "qu'un": "que un",
        "qu'une": "que une", "qu'à": "que à", "qu'en": "que en",
        "l'ai": "le ai", "l'avais": "le avais", "l'a": "le a",
        "l'on": "le on", "l'an": "le an", "l'autre": "le autre",
        "l'homme": "le homme", "l'heure": "le heure",
        "d'un": "de un", "d'une": "de une", "d'abord": "de abord",
        "d'accord": "de accord", "d'ailleurs": "de ailleurs",
        "m'a": "me a", "m'ont": "me ont", "m'avait": "me avait",
        "t'a": "te a", "t'ont": "te ont", "s'est": "se est",
        "s'il": "se il", "s'ils": "se ils",
    },
    "es": {
        "al": "a el", "del": "de el",
    },
    "it": {
        "dell'": "della ", "dell'": "dello ", "nell'": "nella ", "nell'": "nello ",
        "sull'": "sulla ", "all'": "alla ", "l'": "la ", "c'è": "ci è",
        "c'era": "ci era", "dov'è": "dove è", "com'è": "come è",
        "anch'io": "anche io", "tutt'altro": "tutto altro",
    },
    "pt": {
        "à": "a a", "às": "a as", "ao": "a o", "aos": "a os",
        "da": "de a", "das": "de as", "do": "de o", "dos": "de os",
        "na": "em a", "nas": "em as", "no": "em o", "nos": "em os",
        "numa": "em uma", "num": "em um",
        "pela": "por a", "pelas": "por as", "pelo": "por o", "pelos": "por os",
    },
    "de": {
        "im": "in dem", "ins": "in das", "am": "an dem", "ans": "an das",
        "zum": "zu dem", "zur": "zu der", "vom": "von dem", "beim": "bei dem",
        "aufs": "auf das", "fürs": "für das", "ums": "um das",
    },
}


# ---------------------------------------------------------------------------
# 2. NUMBER TO WORDS
# ---------------------------------------------------------------------------

class NumberToWords:
    """Convert integers to their word representation in various languages."""

    # ---- English ----
    _EN_ONES = ["", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    _EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety"]

    @classmethod
    def _en_below_1000(cls, n: int) -> str:
        if n < 20:
            return cls._EN_ONES[n]
        elif n < 100:
            return cls._EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + cls._EN_ONES[n % 10])
        else:
            rest = cls._en_below_1000(n % 100)
            return cls._EN_ONES[n // 100] + " hundred" + (" " + rest if rest else "")

    @classmethod
    def en(cls, n: int) -> str:
        if n < 0:
            return "negative " + cls.en(-n)
        if n == 0:
            return "zero"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._en_below_1000(billions) + " billion")
        if millions:
            parts.append(cls._en_below_1000(millions) + " million")
        if thousands:
            parts.append(cls._en_below_1000(thousands) + " thousand")
        if remainder:
            parts.append(cls._en_below_1000(remainder))
        return " ".join(parts)

    # ---- French ----
    _FR_ONES = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
                "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
                "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"]
    _FR_TENS = ["", "", "vingt", "trente", "quarante", "cinquante",
                "soixante", "soixante", "quatre-vingt", "quatre-vingt"]

    @classmethod
    def _fr_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._FR_ONES[n]
        if n < 70:
            tens = cls._FR_TENS[n // 10]
            ones = n % 10
            if ones == 0:
                return tens
            if ones == 1:
                return tens + " et un"
            return tens + "-" + cls._FR_ONES[ones]
        if n < 80:
            # 70-79: soixante + 10-19
            return "soixante-" + cls._FR_ONES[10 + n % 10]
        if n < 90:
            # 80-89: quatre-vingt + ones
            ones = n % 10
            return "quatre-vingt" + ("-" + cls._FR_ONES[ones] if ones else "s")
        # 90-99: quatre-vingt + 10-19
        return "quatre-vingt-" + cls._FR_ONES[10 + n % 10]

    @classmethod
    def _fr_hundreds(cls, n: int) -> str:
        if n < 100:
            return cls._fr_below_1000(n)
        h = n // 100
        rest = n % 100
        if h == 1:
            prefix = "cent"
        else:
            prefix = cls._FR_ONES[h] + " cent"
        if rest == 0:
            return prefix + ("s" if h > 1 else "")
        return prefix + " " + cls._fr_below_1000(rest)

    @classmethod
    def fr(cls, n: int) -> str:
        if n < 0:
            return "moins " + cls.fr(-n)
        if n == 0:
            return "zéro"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._fr_hundreds(billions) + " milliard" + ("s" if billions > 1 else ""))
        if millions:
            parts.append(cls._fr_hundreds(millions) + " million" + ("s" if millions > 1 else ""))
        if thousands == 1:
            parts.append("mille")
        elif thousands > 1:
            parts.append(cls._fr_hundreds(thousands) + " mille")
        if remainder:
            parts.append(cls._fr_hundreds(remainder))
        return " ".join(parts)

    # ---- Spanish ----
    _ES_ONES = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
                "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
                "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
                "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
                "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
    _ES_TENS = ["", "", "veinte", "treinta", "cuarenta", "cincuenta",
                "sesenta", "setenta", "ochenta", "noventa"]
    _ES_HUNDREDS = ["", "ciento", "doscientos", "trescientos", "cuatrocientos",
                    "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"]

    @classmethod
    def _es_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n == 100:
            return "cien"
        if n < 30:
            return cls._ES_ONES[n]
        if n < 100:
            tens = cls._ES_TENS[n // 10]
            ones = n % 10
            return tens + (" y " + cls._ES_ONES[ones] if ones else "")
        hundreds = cls._ES_HUNDREDS[n // 100]
        rest = cls._es_below_1000(n % 100)
        return hundreds + (" " + rest if rest else "")

    @classmethod
    def es(cls, n: int) -> str:
        if n < 0:
            return "menos " + cls.es(-n)
        if n == 0:
            return "cero"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            b = cls._es_below_1000(billions)
            parts.append(b + (" billón" if billions == 1 else " billones"))
        if millions:
            m = cls._es_below_1000(millions)
            parts.append(m + (" millón" if millions == 1 else " millones"))
        if thousands == 1:
            parts.append("mil")
        elif thousands > 1:
            parts.append(cls._es_below_1000(thousands) + " mil")
        if remainder:
            parts.append(cls._es_below_1000(remainder))
        return " ".join(parts)

    # ---- German ----
    _DE_ONES = ["", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
                "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
                "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
    _DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig",
                "sechzig", "siebzig", "achtzig", "neunzig"]

    @classmethod
    def _de_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._DE_ONES[n]
        if n < 100:
            ones = n % 10
            tens = cls._DE_TENS[n // 10]
            return (cls._DE_ONES[ones] + "und" + tens) if ones else tens
        h = n // 100
        rest = cls._de_below_1000(n % 100)
        return cls._DE_ONES[h] + "hundert" + rest

    @classmethod
    def de(cls, n: int) -> str:
        if n < 0:
            return "minus " + cls.de(-n)
        if n == 0:
            return "null"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._de_below_1000(billions) + "milliarde")
        if millions:
            parts.append(cls._de_below_1000(millions) + "millionen")
        if thousands:
            parts.append(cls._de_below_1000(thousands) + "tausend")
        if remainder:
            parts.append(cls._de_below_1000(remainder))
        return "".join(parts)

    # ---- Portuguese ----
    _PT_ONES = ["", "um", "dois", "três", "quatro", "cinco", "seis", "sete",
                "oito", "nove", "dez", "onze", "doze", "treze", "quatorze",
                "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"]
    _PT_TENS = ["", "", "vinte", "trinta", "quarenta", "cinquenta",
                "sessenta", "setenta", "oitenta", "noventa"]
    _PT_HUNDREDS = ["", "cem", "duzentos", "trezentos", "quatrocentos",
                    "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos"]

    @classmethod
    def _pt_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n == 100:
            return "cem"
        if n < 20:
            return cls._PT_ONES[n]
        if n < 100:
            ones = n % 10
            return cls._PT_TENS[n // 10] + (" e " + cls._PT_ONES[ones] if ones else "")
        rest = cls._pt_below_1000(n % 100)
        return cls._PT_HUNDREDS[n // 100] + (" e " + rest if rest else "")

    @classmethod
    def pt(cls, n: int) -> str:
        if n < 0:
            return "menos " + cls.pt(-n)
        if n == 0:
            return "zero"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            b = cls._pt_below_1000(billions)
            parts.append(b + (" bilhão" if billions == 1 else " bilhões"))
        if millions:
            m = cls._pt_below_1000(millions)
            parts.append(m + (" milhão" if millions == 1 else " milhões"))
        if thousands == 1:
            parts.append("mil")
        elif thousands > 1:
            parts.append(cls._pt_below_1000(thousands) + " mil")
        if remainder:
            parts.append(cls._pt_below_1000(remainder))
        return " e ".join(parts)

    # ---- Italian ----
    _IT_ONES = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette",
                "otto", "nove", "dieci", "undici", "dodici", "tredici", "quattordici",
                "quindici", "sedici", "diciassette", "diciotto", "diciannove"]
    _IT_TENS = ["", "", "venti", "trenta", "quaranta", "cinquanta",
                "sessanta", "settanta", "ottanta", "novanta"]

    @classmethod
    def _it_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._IT_ONES[n]
        if n < 100:
            ones = n % 10
            tens = cls._IT_TENS[n // 10]
            # Drop final vowel of tens before 1 and 8
            if ones in (1, 8):
                tens = tens.rstrip("aeiou")
            return tens + (cls._IT_ONES[ones] if ones else "")
        h = n // 100
        rest = cls._it_below_1000(n % 100)
        if h == 1:
            return "cento" + rest
        return cls._IT_ONES[h] + "cento" + rest

    @classmethod
    def it(cls, n: int) -> str:
        if n < 0:
            return "meno " + cls.it(-n)
        if n == 0:
            return "zero"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._it_below_1000(billions) + "miliardi")
        if millions:
            parts.append(cls._it_below_1000(millions) + "milioni")
        if thousands == 1:
            parts.append("mille")
        elif thousands > 1:
            parts.append(cls._it_below_1000(thousands) + "mila")
        if remainder:
            parts.append(cls._it_below_1000(remainder))
        return " ".join(parts)

    # ---- Dutch ----
    _NL_ONES = ["", "een", "twee", "drie", "vier", "vijf", "zes", "zeven",
                "acht", "negen", "tien", "elf", "twaalf", "dertien", "veertien",
                "vijftien", "zestien", "zeventien", "achttien", "negentien"]
    _NL_TENS = ["", "", "twintig", "dertig", "veertig", "vijftig",
                "zestig", "zeventig", "tachtig", "negentig"]

    @classmethod
    def _nl_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._NL_ONES[n]
        if n < 100:
            ones = n % 10
            tens = cls._NL_TENS[n // 10]
            return (cls._NL_ONES[ones] + "en" + tens) if ones else tens
        h = n // 100
        rest = cls._nl_below_1000(n % 100)
        return cls._NL_ONES[h] + "honderd" + rest

    @classmethod
    def nl(cls, n: int) -> str:
        if n < 0:
            return "min " + cls.nl(-n)
        if n == 0:
            return "nul"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._nl_below_1000(billions) + " miljard")
        if millions:
            parts.append(cls._nl_below_1000(millions) + " miljoen")
        if thousands:
            parts.append(cls._nl_below_1000(thousands) + "duizend")
        if remainder:
            parts.append(cls._nl_below_1000(remainder))
        return " ".join(parts)

    # ---- Russian ----
    _RU_ONES_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь",
                  "восемь", "девять", "десять", "одиннадцать", "двенадцать",
                  "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
                  "семнадцать", "восемнадцать", "девятнадцать"]
    _RU_ONES_F = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь",
                  "восемь", "девять", "десять", "одиннадцать", "двенадцать",
                  "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
                  "семнадцать", "восемнадцать", "девятнадцать"]
    _RU_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
                "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    _RU_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
                    "шестьсот", "семьсот", "восемьсот", "девятьсот"]

    @classmethod
    def _ru_below_1000(cls, n: int, feminine: bool = False) -> str:
        if n == 0:
            return ""
        ones_list = cls._RU_ONES_F if feminine else cls._RU_ONES_M
        parts = []
        if n >= 100:
            parts.append(cls._RU_HUNDREDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._RU_TENS[n // 10])
            n %= 10
        if n > 0:
            parts.append(ones_list[n])
        return " ".join(parts)

    @classmethod
    def ru(cls, n: int) -> str:
        if n < 0:
            return "минус " + cls.ru(-n)
        if n == 0:
            return "ноль"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000

        def _ru_scale(val, singular, few, many, feminine=False):
            w = cls._ru_below_1000(val, feminine)
            last2 = val % 100
            last1 = val % 10
            if 11 <= last2 <= 19:
                return w + " " + many
            if last1 == 1:
                return w + " " + singular
            if 2 <= last1 <= 4:
                return w + " " + few
            return w + " " + many

        if billions:
            parts.append(_ru_scale(billions, "миллиард", "миллиарда", "миллиардов"))
        if millions:
            parts.append(_ru_scale(millions, "миллион", "миллиона", "миллионов"))
        if thousands:
            parts.append(_ru_scale(thousands, "тысяча", "тысячи", "тысяч", feminine=True))
        if remainder:
            parts.append(cls._ru_below_1000(remainder))
        return " ".join(parts)

    # ---- Arabic ----
    _AR_ONES = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة",
                "ثمانية", "تسعة", "عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر",
                "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
    _AR_TENS = ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون",
                "ستون", "سبعون", "ثمانون", "تسعون"]

    @classmethod
    def _ar_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._AR_ONES[n]
        if n < 100:
            ones = n % 10
            tens = cls._AR_TENS[n // 10]
            return (cls._AR_ONES[ones] + " و" + tens) if ones else tens
        h = n // 100
        rest = cls._ar_below_1000(n % 100)
        hundreds = ["", "مئة", "مئتان", "ثلاثمئة", "أربعمئة", "خمسمئة",
                    "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة"]
        return hundreds[h] + (" و" + rest if rest else "")

    @classmethod
    def ar(cls, n: int) -> str:
        if n < 0:
            return "سالب " + cls.ar(-n)
        if n == 0:
            return "صفر"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._ar_below_1000(billions) + " مليار")
        if millions:
            parts.append(cls._ar_below_1000(millions) + " مليون")
        if thousands:
            parts.append(cls._ar_below_1000(thousands) + " ألف")
        if remainder:
            parts.append(cls._ar_below_1000(remainder))
        return " و".join(parts)

    # ---- Hindi ----
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
    def hi(cls, n: int) -> str:
        if n < 0:
            return "ऋण " + cls.hi(-n)
        if n == 0:
            return "शून्य"
        if n <= 99:
            return cls._HI_NUMS.get(n, str(n))
        parts = []
        crore = n // 10_000_000
        lakh = (n % 10_000_000) // 100_000
        thousand = (n % 100_000) // 1_000
        hundred = (n % 1_000) // 100
        remainder = n % 100
        if crore:
            parts.append(cls.hi(crore) + " करोड़")
        if lakh:
            parts.append(cls.hi(lakh) + " लाख")
        if thousand:
            parts.append(cls.hi(thousand) + " हज़ार")
        if hundred:
            parts.append(cls._HI_NUMS[hundred] + " सौ")
        if remainder:
            parts.append(cls.hi(remainder))
        return " ".join(parts)

    # ---- Japanese ----
    _JA_ONES = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    _JA_TENS = ["", "十", "二十", "三十", "四十", "五十", "六十", "七十", "八十", "九十"]

    @classmethod
    def _ja_below_10000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        thousands = n // 1000
        hundreds = (n % 1000) // 100
        tens = (n % 100) // 10
        ones = n % 10
        if thousands:
            parts.append(("" if thousands == 1 else cls._JA_ONES[thousands]) + "千")
        if hundreds:
            parts.append(("" if hundreds == 1 else cls._JA_ONES[hundreds]) + "百")
        if tens:
            parts.append(("" if tens == 1 else cls._JA_ONES[tens]) + "十")
        if ones:
            parts.append(cls._JA_ONES[ones])
        return "".join(parts)

    @classmethod
    def ja(cls, n: int) -> str:
        if n < 0:
            return "マイナス" + cls.ja(-n)
        if n == 0:
            return "零"
        parts = []
        oku = n // 100_000_000
        man = (n % 100_000_000) // 10_000
        remainder = n % 10_000
        if oku:
            parts.append(cls._ja_below_10000(oku) + "億")
        if man:
            parts.append(cls._ja_below_10000(man) + "万")
        if remainder:
            parts.append(cls._ja_below_10000(remainder))
        return "".join(parts)

    # ---- Chinese (Simplified) ----
    _ZH_ONES = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    @classmethod
    def _zh_below_10000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        thousands = n // 1000
        hundreds = (n % 1000) // 100
        tens = (n % 100) // 10
        ones = n % 10
        if thousands:
            parts.append(cls._ZH_ONES[thousands] + "千")
        if hundreds:
            if thousands and not hundreds:
                parts.append("零")
            else:
                parts.append(cls._ZH_ONES[hundreds] + "百")
        if tens:
            if (thousands or hundreds) and not tens:
                parts.append("零")
            else:
                parts.append(cls._ZH_ONES[tens] + "十")
        if ones:
            if (thousands or hundreds or tens) and not ones:
                parts.append("零")
            else:
                parts.append(cls._ZH_ONES[ones])
        return "".join(parts)

    @classmethod
    def zh(cls, n: int) -> str:
        if n < 0:
            return "负" + cls.zh(-n)
        if n == 0:
            return "零"
        parts = []
        yi = n // 100_000_000
        wan = (n % 100_000_000) // 10_000
        remainder = n % 10_000
        if yi:
            parts.append(cls._zh_below_10000(yi) + "亿")
        if wan:
            parts.append(cls._zh_below_10000(wan) + "万")
        if remainder:
            parts.append(cls._zh_below_10000(remainder))
        return "".join(parts)

    # ---- Korean ----
    _KO_ONES = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

    @classmethod
    def _ko_below_10000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        thousands = n // 1000
        hundreds = (n % 1000) // 100
        tens = (n % 100) // 10
        ones = n % 10
        if thousands:
            parts.append(("" if thousands == 1 else cls._KO_ONES[thousands]) + "천")
        if hundreds:
            parts.append(("" if hundreds == 1 else cls._KO_ONES[hundreds]) + "백")
        if tens:
            parts.append(("" if tens == 1 else cls._KO_ONES[tens]) + "십")
        if ones:
            parts.append(cls._KO_ONES[ones])
        return "".join(parts)

    @classmethod
    def ko(cls, n: int) -> str:
        if n < 0:
            return "마이너스 " + cls.ko(-n)
        if n == 0:
            return "영"
        parts = []
        uk = n // 100_000_000
        man = (n % 100_000_000) // 10_000
        remainder = n % 10_000
        if uk:
            parts.append(cls._ko_below_10000(uk) + "억")
        if man:
            parts.append(cls._ko_below_10000(man) + "만")
        if remainder:
            parts.append(cls._ko_below_10000(remainder))
        return "".join(parts)

    # ---- Polish ----
    _PL_ONES = ["", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem",
                "osiem", "dziewięć", "dziesięć", "jedenaście", "dwanaście", "trzynaście",
                "czternaście", "piętnaście", "szesnaście", "siedemnaście", "osiemnaście",
                "dziewiętnaście"]
    _PL_TENS = ["", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt",
                "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt"]
    _PL_HUNDREDS = ["", "sto", "dwieście", "trzysta", "czterysta", "pięćset",
                    "sześćset", "siedemset", "osiemset", "dziewięćset"]

    @classmethod
    def _pl_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            parts.append(cls._PL_HUNDREDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._PL_TENS[n // 10])
            n %= 10
        if n > 0:
            parts.append(cls._PL_ONES[n])
        return " ".join(parts)

    @classmethod
    def pl(cls, n: int) -> str:
        if n < 0:
            return "minus " + cls.pl(-n)
        if n == 0:
            return "zero"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._pl_below_1000(billions) + " miliardów")
        if millions:
            parts.append(cls._pl_below_1000(millions) + " milionów")
        if thousands:
            parts.append(cls._pl_below_1000(thousands) + " tysięcy")
        if remainder:
            parts.append(cls._pl_below_1000(remainder))
        return " ".join(parts)

    # ---- Swedish ----
    _SV_ONES = ["", "ett", "två", "tre", "fyra", "fem", "sex", "sju",
                "åtta", "nio", "tio", "elva", "tolv", "tretton", "fjorton",
                "femton", "sexton", "sjutton", "arton", "nitton"]
    _SV_TENS = ["", "", "tjugo", "trettio", "fyrtio", "femtio",
                "sextio", "sjuttio", "åttio", "nittio"]

    @classmethod
    def _sv_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._SV_ONES[n]
        if n < 100:
            ones = n % 10
            return cls._SV_TENS[n // 10] + (cls._SV_ONES[ones] if ones else "")
        h = n // 100
        rest = cls._sv_below_1000(n % 100)
        return cls._SV_ONES[h] + "hundra" + rest

    @classmethod
    def sv(cls, n: int) -> str:
        if n < 0:
            return "minus " + cls.sv(-n)
        if n == 0:
            return "noll"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._sv_below_1000(billions) + " miljarder")
        if millions:
            parts.append(cls._sv_below_1000(millions) + " miljoner")
        if thousands == 1:
            parts.append("ettusen")
        elif thousands > 1:
            parts.append(cls._sv_below_1000(thousands) + "tusen")
        if remainder:
            parts.append(cls._sv_below_1000(remainder))
        return " ".join(parts)

    # ---- Turkish ----
    _TR_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi",
                "sekiz", "dokuz"]
    _TR_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli",
                "altmış", "yetmiş", "seksen", "doksan"]

    @classmethod
    def _tr_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            h = n // 100
            parts.append(("" if h == 1 else cls._TR_ONES[h]) + "yüz")
            n %= 100
        if n >= 10:
            parts.append(cls._TR_TENS[n // 10])
            n %= 10
        if n:
            parts.append(cls._TR_ONES[n])
        return " ".join(p for p in parts if p)

    @classmethod
    def tr(cls, n: int) -> str:
        if n < 0:
            return "eksi " + cls.tr(-n)
        if n == 0:
            return "sıfır"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._tr_below_1000(billions) + " milyar")
        if millions:
            parts.append(cls._tr_below_1000(millions) + " milyon")
        if thousands == 1:
            parts.append("bin")
        elif thousands > 1:
            parts.append(cls._tr_below_1000(thousands) + " bin")
        if remainder:
            parts.append(cls._tr_below_1000(remainder))
        return " ".join(parts)

    # ---- Greek ----
    _EL_ONES = ["", "ένα", "δύο", "τρία", "τέσσερα", "πέντε", "έξι", "επτά",
                "οκτώ", "εννέα", "δέκα", "έντεκα", "δώδεκα", "δεκατρία", "δεκατέσσερα",
                "δεκαπέντε", "δεκαέξι", "δεκαεπτά", "δεκαοκτώ", "δεκαεννέα"]
    _EL_TENS = ["", "", "είκοσι", "τριάντα", "σαράντα", "πενήντα",
                "εξήντα", "εβδομήντα", "ογδόντα", "ενενήντα"]
    _EL_HUNDREDS = ["", "εκατό", "διακόσια", "τριακόσια", "τετρακόσια", "πεντακόσια",
                    "εξακόσια", "επτακόσια", "οκτακόσια", "εννιακόσια"]

    @classmethod
    def _el_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            parts.append(cls._EL_HUNDREDS[n // 100])
            n %= 100
        if n >= 20:
            parts.append(cls._EL_TENS[n // 10])
            n %= 10
        if n > 0:
            parts.append(cls._EL_ONES[n])
        return " ".join(parts)

    @classmethod
    def el(cls, n: int) -> str:
        if n < 0:
            return "μείον " + cls.el(-n)
        if n == 0:
            return "μηδέν"
        parts = []
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if billions:
            parts.append(cls._el_below_1000(billions) + " δισεκατομμύρια")
        if millions:
            parts.append(cls._el_below_1000(millions) + " εκατομμύρια")
        if thousands:
            parts.append(cls._el_below_1000(thousands) + " χίλια")
        if remainder:
            parts.append(cls._el_below_1000(remainder))
        return " ".join(parts)

    # ---- Hebrew ----
    _HE_ONES = ["", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע",
                "שמונה", "תשע", "עשר", "אחת עשרה", "שתים עשרה", "שלוש עשרה",
                "ארבע עשרה", "חמש עשרה", "שש עשרה", "שבע עשרה", "שמונה עשרה",
                "תשע עשרה"]
    _HE_TENS = ["", "", "עשרים", "שלושים", "ארבעים", "חמישים",
                "שישים", "שבעים", "שמונים", "תשעים"]

    @classmethod
    def _he_below_1000(cls, n: int) -> str:
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            h = n // 100
            h_words = ["", "מאה", "מאתיים", "שלוש מאות", "ארבע מאות", "חמש מאות",
                       "שש מאות", "שבע מאות", "שמונה מאות", "תשע מאות"]
            parts.append(h_words[h])
            n %= 100
        if n >= 20:
            parts.append(cls._HE_TENS[n // 10])
            n %= 10
        if n > 0:
            parts.append(cls._HE_ONES[n])
        return " ".join(parts)

    @classmethod
    def he(cls, n: int) -> str:
        if n < 0:
            return "מינוס " + cls.he(-n)
        if n == 0:
            return "אפס"
        parts = []
        millions = n // 1_000_000
        thousands = (n % 1_000_000) // 1_000
        remainder = n % 1_000
        if millions:
            parts.append(cls._he_below_1000(millions) + " מיליון")
        if thousands:
            parts.append(cls._he_below_1000(thousands) + " אלף")
        if remainder:
            parts.append(cls._he_below_1000(remainder))
        return " ".join(parts)

    # ---- Dispatch ----
    # Map language codes (and common aliases) to converter methods
    _DISPATCH = {}

    @classmethod
    def _build_dispatch(cls):
        cls._DISPATCH = {
            # English
            "en": cls.en, "en-us": cls.en, "en-gb": cls.en, "en-au": cls.en,
            # French
            "fr": cls.fr, "fr-fr": cls.fr, "fr-ca": cls.fr, "fr-be": cls.fr,
            # Spanish
            "es": cls.es, "es-es": cls.es, "es-mx": cls.es, "es-ar": cls.es,
            # German
            "de": cls.de, "de-de": cls.de, "de-at": cls.de, "de-ch": cls.de,
            # Portuguese
            "pt": cls.pt, "pt-pt": cls.pt, "pt-br": cls.pt,
            # Italian
            "it": cls.it, "it-it": cls.it,
            # Dutch
            "nl": cls.nl, "nl-nl": cls.nl, "nl-be": cls.nl,
            # Russian
            "ru": cls.ru, "ru-ru": cls.ru,
            # Arabic
            "ar": cls.ar, "ar-sa": cls.ar, "ar-eg": cls.ar,
            # Hindi
            "hi": cls.hi, "hi-in": cls.hi,
            # Japanese
            "ja": cls.ja, "ja-jp": cls.ja,
            # Chinese
            "zh": cls.zh, "zh-cn": cls.zh, "zh-tw": cls.zh, "zh-hk": cls.zh,
            # Korean
            "ko": cls.ko, "ko-kr": cls.ko,
            # Polish
            "pl": cls.pl, "pl-pl": cls.pl,
            # Swedish
            "sv": cls.sv, "sv-se": cls.sv,
            # Turkish
            "tr": cls.tr, "tr-tr": cls.tr,
            # Greek
            "el": cls.el, "el-gr": cls.el,
            # Hebrew
            "he": cls.he, "he-il": cls.he,
        }

    @classmethod
    def convert(cls, n: int, lang: str) -> str:
        if not cls._DISPATCH:
            cls._build_dispatch()
        key = lang.lower()
        fn = cls._DISPATCH.get(key)
        if fn is None:
            # Fallback: try base language code (strip region)
            base = key.split("-")[0]
            fn = cls._DISPATCH.get(base)
        if fn is None:
            return str(n)  # Unknown language: leave as digit string
        return fn(n)


# ---------------------------------------------------------------------------
# 3. MAIN NORMALIZATION PIPELINE
# ---------------------------------------------------------------------------

def normalize(text: str, lang: str) -> str:
    lang_lower = lang.lower()
    base_lang = lang_lower.split("-")[0]

    # Step 1: Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Lowercase (for all languages — skip for languages that don't use case)
    # Languages with no case distinction: Arabic, Hebrew, Hindi, Japanese, Chinese, Korean
    no_case_langs = {"ar", "he", "hi", "ja", "zh", "ko"}
    if base_lang not in no_case_langs:
        text = text.lower()

    # Step 3: Expand contractions
    contractions = CONTRACTIONS.get(base_lang, {})
    if contractions:
        # Sort by length descending to match longer contractions first
        sorted_contractions = sorted(contractions.keys(), key=len, reverse=True)
        for contraction in sorted_contractions:
            expansion = contractions[contraction]
            # Match whole word / token (word boundary aware, Unicode-safe)
            pattern = r'(?<![^\s])' + re.escape(contraction) + r'(?![^\s\'])'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

    # Step 4: Convert standalone integers to word form
    def replace_number(match: re.Match) -> str:
        token = match.group(0)
        try:
            n = int(token)
            return NumberToWords.convert(n, lang)
        except ValueError:
            return token

    # Standalone integer: not preceded/followed by letters, digits, or decimal point.
    # Using \w in lookbehind/ahead blocks embedded tokens like "model3" while still
    # matching numbers adjacent to CJK/non-Latin chars (\w is Unicode-aware).
    text = re.sub(r'(?:(?<=\s)|(?:^))\-\d+(?![.\d_a-zA-Z])|(?<![.\d_a-zA-Z])\d+(?![.\d_a-zA-Z])', replace_number, text, flags=re.MULTILINE)

    # Step 5: Remove punctuation while preserving Unicode letters, marks, digits, whitespace.
    # Use unicodedata categories for full Unicode script support (Devanagari combining marks,
    # CJK, Arabic, Hebrew, etc.) instead of regex \w which can miss combining characters.
    def remove_punct(s: str) -> str:
        out = []
        for ch in s:
            cat = unicodedata.category(ch)
            # L=Letter, M=Mark (combining), N=Number, Z=Separator, plus common whitespace
            if cat[0] in ('L', 'M', 'N', 'Z') or ch in ' \t\n\r':
                out.append(ch)
            else:
                out.append(' ')
        return ''.join(out)

    text = remove_punct(text)

    # Step 6: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ---------------------------------------------------------------------------
# 4. ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 normalize_text.py <file_path> <language_code>")
        print("Example: python3 normalize_text.py input.txt en")
        sys.exit(1)

    file_path = sys.argv[1]
    lang_code = sys.argv[2]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()

    result = normalize(text, lang_code)
    print(result)


if __name__ == "__main__":
    main()