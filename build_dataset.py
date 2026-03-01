#!/usr/bin/env python3
"""
build_dataset.py — Build inputs/inputN/ folder structure + CSV from google/fleurs
==================================================================================
Loads FLEURS audio + transcriptions, generates 4 corrupted options per clip,
places the correct transcription at a random slot (1–5), saves audio as WAV,
writes the inputs/inputN/ folder structure that acoustic.py consumes, AND
saves a flat CSV ready for direct use with: python acoustic.py --file ...

Usage:
    # 10 samples of Arabic with moderate corruptions
    python build_dataset.py --language ar_sa --samples 10

    # 50 samples of English with chaotic corruptions
    python build_dataset.py --language en_us --samples 50 --corruption chaotic

    # Multiple languages (5 samples each)
    python build_dataset.py --language ar_sa en_us hi_in --samples 5

    # Custom output paths
    python build_dataset.py --language ar_sa --samples 20 \\
        --out-dir my_inputs --csv my_dataset.csv

Then run the pipeline:
    python acoustic.py --file fleurs_dataset.csv --output output/output_golden.csv
    # OR using folder structure:
    python acoustic.py --inputs-dir inputs --output output/output_golden.csv
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Language mapping: FLEURS code → acoustic.py language string
# (acoustic.py accepts the raw FLEURS code like "ar_sa" directly — its
#  _lang_base() splits on "_" and maps "ar" → "ar", "en" → "en", etc.)
# ---------------------------------------------------------------------------
ALL_LANGUAGES = [
    "en_us", "es_419", "fr_fr", "de_de", "it_it",
    "pt_br", "ru_ru", "zh_cn", "ja_jp", "ko_kr",
    "hi_in", "ar_sa", "nl_nl", "pl_pl", "sv_se",
    "tr_tr", "vi_vn", "id_id", "uk_ua", "th_th",
]

UNRELATED_POOL = [
    "The global economy is shifting rapidly.",
    "Artificial intelligence continues to evolve.",
    "This sentence is completely unrelated.",
    "السياسة العالمية تتغير بسرعة.",
    "Quantum computing may change everything.",
]


# ---------------------------------------------------------------------------
# Moderate corruption helpers
# ---------------------------------------------------------------------------

def _delete_word(words: list[str]) -> list[str]:
    if len(words) <= 1:
        return words
    idx = random.randint(0, len(words) - 1)
    return words[:idx] + words[idx + 1:]


def _insert_word(words: list[str]) -> list[str]:
    fillers = ["very", "really", "actually", "basically", "just", "also"]
    idx = random.randint(0, len(words))
    return words[:idx] + [random.choice(fillers)] + words[idx:]


def _swap_adjacent(words: list[str]) -> list[str]:
    words = words.copy()
    if len(words) < 2:
        return words
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return words


def _char_noise(text: str) -> str:
    if len(text) < 5:
        return text
    idx = random.randint(0, len(text) - 1)
    new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:idx] + new_char + text[idx + 1:]


def moderate_corrupt(text: str) -> str:
    words = text.split()
    strategy = random.choice(["delete", "insert", "swap", "char"])
    if strategy == "delete":
        return " ".join(_delete_word(words))
    elif strategy == "insert":
        return " ".join(_insert_word(words))
    elif strategy == "swap":
        return " ".join(_swap_adjacent(words))
    elif strategy == "char":
        return _char_noise(text)
    return text


# ---------------------------------------------------------------------------
# Chaotic corruption helpers
# ---------------------------------------------------------------------------

def _random_truncate(text: str) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    cut = random.randint(1, len(words) - 1)
    return " ".join(words[:cut])


def _random_expand(text: str) -> str:
    return text + " " + random.choice(UNRELATED_POOL)


def _random_shuffle(text: str) -> str:
    words = text.split()
    random.shuffle(words)
    return " ".join(words)


def _random_scramble(text: str) -> str:
    chars = list(text)
    random.shuffle(chars)
    return "".join(chars)


def _duplicate(text: str) -> str:
    return text + " " + text


def chaotic_corrupt(text: str) -> str:
    if random.random() < 0.4:
        return random.choice(UNRELATED_POOL)
    func = random.choice([
        _random_truncate, _random_expand, _random_shuffle,
        _random_scramble, _duplicate,
    ])
    return func(text)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_inputs(
    language: str,
    samples: int,
    corruption: str,
    out_dir: Path,
    audio_dir: Path,
    start_idx: int = 1,
    seed: int = 42,
) -> tuple[int, list[dict]]:
    """
    Load `samples` clips of `language` from FLEURS, generate 4 corruptions,
    write inputs/inputN/ folders.  Returns (next_index, list_of_csv_rows).
    """
    corrupt_fn = moderate_corrupt if corruption == "moderate" else chaotic_corrupt

    print(f"\nLoading FLEURS '{language}' (split=train, samples={samples}) ...")
    dataset = load_dataset("google/fleurs", language, split="train", trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed)
    actual = min(samples, len(dataset))
    dataset = dataset.select(range(actual))

    lang_audio_dir = audio_dir / language
    lang_audio_dir.mkdir(parents=True, exist_ok=True)

    idx = start_idx
    csv_rows: list[dict] = []
    for i, sample in enumerate(dataset):
        audio_id   = sample.get("id", i)
        audio_arr  = np.array(sample["audio"]["array"], dtype=np.float32)
        sample_rate = sample["audio"]["sampling_rate"]
        correct    = sample["transcription"].strip()

        # Save audio as WAV
        wav_path = lang_audio_dir / f"{audio_id}.wav"
        sf.write(str(wav_path), audio_arr, sample_rate)

        # Generate 4 unique corruptions
        corruptions: list[str] = []
        attempts = 0
        seen = {correct}
        while len(corruptions) < 4 and attempts < 200:
            c = corrupt_fn(correct)
            if c not in seen and c.strip():
                corruptions.append(c)
                seen.add(c)
            attempts += 1

        # Pad with truncations if we ran out of unique corruptions
        words = correct.split()
        while len(corruptions) < 4:
            fallback = " ".join(words[: max(1, len(words) // 2)]) + " ..."
            corruptions.append(fallback)

        # Place correct at a random slot (1–5)
        correct_slot = random.randint(0, 4)   # 0-indexed
        options = corruptions.copy()
        options.insert(correct_slot, correct)  # now 5 items
        correct_option_num = correct_slot + 1  # 1-indexed

        # Write inputs/inputN/ folder
        row_dir = out_dir / f"input{idx}"
        row_dir.mkdir(parents=True, exist_ok=True)

        (row_dir / "audio_id.txt").write_text(str(audio_id), encoding="utf-8")
        (row_dir / "language.txt").write_text(language, encoding="utf-8")
        (row_dir / "audio_url.txt").write_text(str(wav_path.resolve()), encoding="utf-8")
        (row_dir / "correct_option.txt").write_text(str(correct_option_num), encoding="utf-8")
        for n, opt in enumerate(options, 1):
            (row_dir / f"option_{n}.txt").write_text(opt, encoding="utf-8")

        # Collect CSV row
        csv_rows.append({
            "audio_id":       audio_id,
            "language":       language,
            "audio":          str(wav_path.resolve()),
            "option_1":       options[0],
            "option_2":       options[1],
            "option_3":       options[2],
            "option_4":       options[3],
            "option_5":       options[4],
            "correct_option": correct_option_num,
        })

        idx += 1

    print(f"  {language}: wrote {actual} samples → input{start_idx}..input{idx - 1}")
    return idx, csv_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build inputs/inputN/ dataset from google/fleurs for acoustic.py"
    )
    parser.add_argument(
        "--language", nargs="+", default=["ar_sa"],
        metavar="LANG",
        help=(
            "FLEURS language code(s), e.g. ar_sa en_us hi_in  "
            f"(use 'all' for all 20 languages: {ALL_LANGUAGES})"
        ),
    )
    parser.add_argument("--samples",    type=int, default=10,
                        help="Number of clips per language (default: 10)")
    parser.add_argument("--corruption", choices=["moderate", "chaotic"], default="moderate",
                        help="Corruption difficulty (default: moderate)")
    parser.add_argument("--out-dir",    default="inputs",
                        help="Root folder for inputs/inputN/ folders (default: inputs)")
    parser.add_argument("--audio-dir",  default="fleurs_audio",
                        help="Folder to save WAV files (default: fleurs_audio)")
    parser.add_argument("--csv",        default="fleurs_dataset.csv",
                        help="Output CSV path (default: fleurs_dataset.csv)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    languages = ALL_LANGUAGES if args.language == ["all"] else args.language
    out_dir   = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find next available inputN index (don't overwrite existing)
    existing = [
        int(re.search(r"\d+", d.name).group())
        for d in out_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"input\d+", d.name)
    ]
    next_idx = (max(existing) + 1) if existing else 1

    print(f"Output folder  : {out_dir.resolve()}")
    print(f"Audio folder   : {audio_dir.resolve()}")
    print(f"CSV output     : {Path(args.csv).resolve()}")
    print(f"Corruption mode: {args.corruption}")
    print(f"Languages      : {languages}")
    print(f"Samples/lang   : {args.samples}")
    print(f"Starting index : {next_idx}")

    all_rows: list[dict] = []
    for lang in languages:
        next_idx, rows = build_inputs(
            language=lang,
            samples=args.samples,
            corruption=args.corruption,
            out_dir=out_dir,
            audio_dir=audio_dir,
            start_idx=next_idx,
            seed=args.seed,
        )
        all_rows.extend(rows)

    # Save CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=[
        "audio_id", "language", "audio",
        "option_1", "option_2", "option_3", "option_4", "option_5",
        "correct_option",
    ])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV saved → {csv_path.resolve()}  ({len(df)} rows)")

    total = next_idx - (max(existing) + 1 if existing else 1)
    print(f"Folder structure → {total} input folders under {out_dir.resolve()}")
    print(f"\nRun pipeline with:")
    print(f"  python acoustic.py --file {csv_path} --output output/output_golden.csv")
    print(f"  # OR via folder structure:")
    print(f"  python acoustic.py --inputs-dir {out_dir} --output output/output_golden.csv")


if __name__ == "__main__":
    main()
