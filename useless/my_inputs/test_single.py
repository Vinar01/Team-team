#!/usr/bin/env python3
"""
test_single.py — Test the pipeline on a single local audio file.

Same logic as runner.py: load audio → Whisper transcribe → normalize →
compute WER against 5 candidates → pick best → show result + transcript.

Usage:
    python test_single.py --audio fl.wav --lang en
    python test_single.py --audio fl.wav --lang en --model small
    python test_single.py --audio fl.wav --lang ar --model base

    # Pass candidates directly (no prompt):
    python test_single.py --audio fl.wav --lang en --correct 2 \
        --options "she sells seashells" "she sell sea shells" "she sell seashells" "she sells sea shells" "she sell seashell"

You will be prompted to enter 5 candidate transcriptions and the correct option
if --options is not provided.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisper

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

from jiwer import wer as jiwer_wer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers  (identical logic to runner.py)
# ---------------------------------------------------------------------------

def _clean_arabic(text: str) -> str:
    """Keep only Arabic script + shared chars; strip cross-script hallucinations."""
    cleaned = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
        r"\s\d.,،؛؟!؟:\-\"'()[\]{}]",
        "", text,
    )
    return re.sub(r"  +", " ", cleaned).strip()


def _clean_latin(text: str) -> str:
    """Keep Basic-Latin + Latin-Extended scripts; strip CJK/Arabic/Hangul hallucinations."""
    cleaned = re.sub(r"[^\u0000-\u024F\u1E00-\u1EFF\s\d.,!?;:\-\"'()[\]{}]", "", text)
    return re.sub(r"  +", " ", cleaned).strip()


def _clean_hindi(text: str) -> str:
    """Keep Devanagari script + shared chars; strip Latin/CJK/Arabic hallucinations."""
    cleaned = re.sub(r"[^\u0900-\u097F\s\d.,!?;:\-\"'()[\]{}]", "", text)
    return re.sub(r"  +", " ", cleaned).strip()


def _clean_chinese(text: str) -> str:
    """Keep CJK Unified Ideographs + CJK punctuation; strip Arabic/Hangul hallucinations."""
    cleaned = re.sub(
        r"[^\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF"
        r"\u3000-\u303F\uFF00-\uFFEF\s\d.,!?\uff1b\uff1a\u3001\u300c\u300d\u3010\u3011()[\]{}]",
        "", text,
    )
    return re.sub(r"  +", " ", cleaned).strip()


def _clean_generic(text: str) -> str:
    """Fallback: remove Whisper looping repetitions and collapse spaces.

    Does NOT strip any script characters — safe for any language not in the registry.
    """
    text = re.sub(r"(.{8,50}?)\1{3,}", r"\1", text)
    return re.sub(r"  +", " ", text).strip()


# Registry: ISO 639-1 code → script-specific cleaner.
# Languages absent fall back to _clean_generic (loop/space fix only).
_LANG_CLEANERS: dict = {
    "ar": _clean_arabic,
    "en": _clean_latin, "fr": _clean_latin, "de": _clean_latin,
    "es": _clean_latin, "it": _clean_latin, "pt": _clean_latin,
    "nl": _clean_latin, "pl": _clean_latin, "sv": _clean_latin,
    "tr": _clean_latin, "vi": _clean_latin, "id": _clean_latin,
    "ro": _clean_latin, "cs": _clean_latin, "fi": _clean_latin,
    "hu": _clean_latin, "da": _clean_latin, "no": _clean_latin,
    "hi": _clean_hindi, "mr": _clean_hindi, "ne": _clean_hindi,
    "zh": _clean_chinese,
}


def _clean_transcript(text: str, lang: str) -> str:
    """Apply the registered script cleaner for *lang*, then normalize.

    Two-step preparation for the Whisper hypothesis:
      1. Script cleaner  — strips cross-language hallucinations (or generic
         loop/space fix for unregistered languages).
      2. _normalize      — diacritics, casing, number-word expansion, etc.

    Returns fully cleaned *and* normalized text ready for WER comparison.
    """
    cleaner = _LANG_CLEANERS.get(lang, _clean_generic)
    cleaned = cleaner(text)
    return _normalize(cleaned, lang)


def _lang_base(lang: str) -> str:
    part = lang.lower().replace("-", "_").split("_")[0]
    _MAP = {
        "arabic": "ar", "english": "en", "french": "fr", "german": "de",
        "spanish": "es", "hindi": "hi", "chinese": "zh", "japanese": "ja",
        "korean": "ko", "russian": "ru", "portuguese": "pt", "italian": "it",
    }
    return _MAP.get(part, part)


def _lang_for_whisper(lang: str) -> str:
    base = lang.lower().replace("-", "_").split("_")[0]
    _MAP = {
        "en": "english",    "fr": "french",     "de": "german",    "es": "spanish",
        "it": "italian",    "pt": "portuguese", "ru": "russian",   "ja": "japanese",
        "ko": "korean",     "zh": "chinese",    "hi": "hindi",     "ar": "arabic",
        "nl": "dutch",      "pl": "polish",     "sv": "swedish",   "tr": "turkish",
        "vi": "vietnamese", "id": "indonesian", "th": "thai",      "uk": "ukrainian",
        "arabic": "arabic", "english": "english", "hindi": "hindi",
        "chinese": "chinese", "french": "french",
    }
    return _MAP.get(base, base)


def _normalize(text: str, lang: str) -> str:
    try:
        from text import normalize
        return normalize(text, lang)
    except Exception as exc:
        logger.warning("Normalization fallback (%s)", exc)
        return text.lower().strip()


def _load_audio(audio_path: Path) -> np.ndarray | None:
    """Load audio → 16 kHz float32 mono. ffmpeg → soundfile/torchaudio fallback."""
    try:
        return whisper.load_audio(str(audio_path))
    except Exception as e:
        logger.warning("whisper.load_audio failed (%s) — trying soundfile/torchaudio", e)
    try:
        if _SF_AVAILABLE:
            data, sr = sf.read(str(audio_path), dtype="float32")
        else:
            waveform, sr = torchaudio.load(str(audio_path))
            data = waveform.mean(dim=0).numpy().astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            waveform_t = torch.tensor(data).unsqueeze(0)
            resampler  = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            data = resampler(waveform_t).squeeze(0).numpy().astype(np.float32)
        return data
    except Exception as e:
        logger.error("Could not load audio: %s", e)
        return None


def _transcribe(model, audio: np.ndarray, whisper_lang: str, device: str) -> str:
    """Whisper transcription — same parameters as runner.py."""
    result = model.transcribe(
        audio,
        language=whisper_lang,
        task="transcribe",
        fp16=(device == "cuda"),
        condition_on_previous_text=False,   # reduces hallucination loops
        no_speech_threshold=0.6,            # skip near-silent segments
        logprob_threshold=-1.0,             # discard very uncertain tokens
        compression_ratio_threshold=2.4,    # drop repetition-heavy outputs
    )
    return result["text"].strip()


def _softmax_wer(wers: list[float]) -> list[float]:
    """Convert raw WER values to softmax scores where higher = better (WER 0.0 → ~1.0)."""
    finite = [w for w in wers if math.isfinite(w)]
    cap = (max(finite) + 10.0) if finite else 10.0
    clamped = [w if math.isfinite(w) else cap for w in wers]
    neg = [-w for w in clamped]
    shift = max(neg)
    exps = [math.exp(v - shift) for v in neg]
    total = sum(exps)
    return [round(e / total, 6) for e in exps]


def _score_candidates(hypothesis: str, candidates: list[str], norm_lang: str) -> list[dict]:
    """Clean+normalize hypothesis, normalize each candidate, compute WER→softmax, sort descending."""
    norm_hyp = _clean_transcript(hypothesis, norm_lang)
    scored = []
    for i, cand in enumerate(candidates):
        if not cand.strip():
            raw_wer = float("inf")
        else:
            norm_cand = _normalize(cand, norm_lang)
            try:
                raw_wer = jiwer_wer(norm_hyp, norm_cand)
            except Exception:
                raw_wer = float("inf")
        scored.append({
            "option_num": i + 1,
            "text":       cand,
            "norm":       _normalize(cand, norm_lang),
            "score":      raw_wer,   # temporary; replaced below
        })
    # Convert raw WERs → softmax scores (higher = better)
    raw_wers = [s["score"] for s in scored]
    for entry, sm in zip(scored, _softmax_wer(raw_wers)):
        entry["score"] = sm
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the golden-transcription pipeline on a single audio file"
    )
    parser.add_argument("--audio",   required=True,
                        help="Path to the audio file (.mp3, .wav, .flac, …)")
    parser.add_argument("--lang",    default="en",
                        help="Language code, e.g. en, ar, Arabic_SA  (default: en)")
    parser.add_argument("--model",   default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size  (default: base)")
    parser.add_argument("--options", nargs=5,
                        metavar=("OPT1", "OPT2", "OPT3", "OPT4", "OPT5"),
                        help="5 candidate transcriptions. If omitted, prompted interactively.")
    parser.add_argument("--correct", type=int, default=0, choices=range(0, 6),
                        help="Correct option number 1-5 (0 = unknown)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)
    if audio_path.stat().st_size == 0:
        print(f"ERROR: Audio file is empty (0 bytes): {audio_path}")
        sys.exit(1)

    # ── Collect candidates ──────────────────────────────────────────────────
    if args.options:
        candidates = list(args.options)
        correct    = args.correct
    else:
        print("\nEnter the 5 candidate transcriptions (press Enter after each):")
        candidates = [input(f"  Option {i}: ").strip() for i in range(1, 6)]
        correct = args.correct
        if correct == 0:
            try:
                correct = int(input("\nWhich option is correct? (1-5, or 0 if unknown): ").strip())
            except ValueError:
                correct = 0

    # ── Setup ───────────────────────────────────────────────────────────────
    norm_lang    = _lang_base(args.lang)
    whisper_lang = _lang_for_whisper(args.lang)
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading Whisper '{args.model}' on {device} …")
    model = whisper.load_model(args.model, device=device)

    # ── Step 1: Load audio ──────────────────────────────────────────────────
    print(f"Loading audio : {audio_path.name}")
    audio = _load_audio(audio_path)
    if audio is None:
        print("ERROR: Could not load audio file.")
        sys.exit(1)

    # ── Step 2: Transcribe ──────────────────────────────────────────────────
    print("Transcribing …")
    raw_hypothesis = _transcribe(model, audio, whisper_lang, device)

    # ── Step 3: Clean + normalize in one step ───────────────────────────────
    # _clean_transcript: strips cross-script hallucinations, then normalizes
    norm_hypothesis = _clean_transcript(raw_hypothesis, norm_lang)

    # ── Step 4: Score candidates ──────────────────────────────────────────
    scored = _score_candidates(raw_hypothesis, candidates, norm_lang)
    best            = scored[0]

    # ── Print results ────────────────────────────────────────────────────────
    SEP = "─" * 64
    print(f"\n{SEP}")
    print("  Whisper transcript (raw)")
    print(f"{SEP}")
    print(f"  {raw_hypothesis}")

    print(f"\n{SEP}")
    print("  Whisper transcript (normalized)")
    print(f"{SEP}")
    print(f"  {norm_hypothesis}")

    print(f"\n{SEP}")
    print(f"  Softmax scores per option  (higher = closer match to transcript, 1.0 = perfect)")
    print(f"{SEP}")
    for s in sorted(scored, key=lambda x: x["option_num"]):
        tags = []
        if correct and s["option_num"] == correct:
            tags.append("CORRECT")
        if s["option_num"] == best["option_num"]:
            tags.append("PREDICTED")
        tag_str = "  ← " + " | ".join(tags) if tags else ""
        score_str = f"{s['score']:.6f}"
        print(f"  Option {s['option_num']}  score={score_str}{tag_str}")
        print(f"    text : {s['text'][:90]}")
        print(f"    norm : {s['norm'][:90]}")
        print()

    print(f"{SEP}")
    print("  Result")
    print(f"{SEP}")
    print(f"  golden_ref : option_{best['option_num']}  (score={best['score']:.6f})"
    if correct:
        print(f"  correct    : option_{correct}")
        print(f"  is_correct : {best['option_num'] == correct}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
