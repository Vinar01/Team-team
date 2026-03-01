#!/usr/bin/env python3
"""
build_validation_csv.py
=======================
Reads fleurs_trimmed.csv (audio already saved as WAVs by trimmer.py),
generates 4 corrupted transcription options per row, shuffles all 5
(1 correct + 4 corruptions), and writes a validation CSV ready for
train_xgboost_fusion_model().

Output columns:
    audio_id, language, audio,
    option_1, option_2, option_3, option_4, option_5,
    correct_option,          ← same as true_golden_option_index (kept for acoustic.py)
    true_golden_option_index ← 1-based index of the correct option (for XGBoost trainer)

Usage:
    python build_validation_csv.py
    python build_validation_csv.py --input fleurs_trimmed.csv --output validation_dataset.csv
    python build_validation_csv.py --corruption moderate --seed 123
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Corruption pool (for chaotic expansions)
# ---------------------------------------------------------------------------

UNRELATED_POOL = [
    "The global economy is shifting rapidly.",
    "Artificial intelligence continues to evolve.",
    "This sentence is completely unrelated.",
    "السياسة العالمية تتغير بسرعة.",
    "Quantum computing may change everything.",
]

# ---------------------------------------------------------------------------
# Moderate corruptions
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
# Chaotic corruptions
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
# Mixed corruption: 2 moderate + 2 chaotic (or all of one type)
# ---------------------------------------------------------------------------

def generate_corruptions(text: str, mode: str, n: int = 4) -> list[str]:
    """
    Generate `n` unique corrupted variants of `text`.
    mode: 'moderate' | 'chaotic' | 'mixed'
    """
    if mode == "moderate":
        fns = [moderate_corrupt] * n
    elif mode == "chaotic":
        fns = [chaotic_corrupt] * n
    else:  # mixed: 2 moderate + 2 chaotic
        fns = [moderate_corrupt, moderate_corrupt, chaotic_corrupt, chaotic_corrupt]
        random.shuffle(fns)

    corruptions: list[str] = []
    seen = {text}
    attempts = 0
    fn_idx = 0

    while len(corruptions) < n and attempts < 400:
        fn = fns[fn_idx % len(fns)]
        c = fn(text)
        if c not in seen and c.strip():
            corruptions.append(c)
            seen.add(c)
            fn_idx += 1
        attempts += 1

    # Fallback: truncate if we couldn't get enough unique variants
    words = text.split()
    while len(corruptions) < n:
        fallback = " ".join(words[: max(1, len(words) // 2)]) + " ..."
        corruptions.append(fallback)

    return corruptions

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(input_csv: str, output_csv: str, corruption: str, seed: int) -> None:
    random.seed(seed)

    df_in = pd.read_csv(input_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df_in)} rows from '{input_csv}'")
    print(f"Languages: {sorted(df_in['language'].unique())}")
    print(f"Corruption mode: {corruption}")

    rows = []
    for _, row in df_in.iterrows():
        correct    = str(row["transcription"]).strip()
        corruptions = generate_corruptions(correct, mode=corruption)

        # Shuffle: place correct at a random slot among 5
        correct_slot = random.randint(0, 4)          # 0-based
        options = corruptions.copy()                  # 4 items
        options.insert(correct_slot, correct)         # now 5 items
        true_idx = correct_slot + 1                   # 1-based

        rows.append({
            "audio_id":                row["audio_id"],
            "language":                row["language"],
            "audio":                   row["audio"],
            "option_1":                options[0],
            "option_2":                options[1],
            "option_3":                options[2],
            "option_4":                options[3],
            "option_5":                options[4],
            "correct_option":          true_idx,       # for acoustic.py compatibility
            "true_golden_option_index": true_idx,      # for XGBoost trainer
        })

    out_df = pd.DataFrame(rows, columns=[
        "audio_id", "language", "audio",
        "option_1", "option_2", "option_3", "option_4", "option_5",
        "correct_option", "true_golden_option_index",
    ])

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(out_df)} rows → '{output_csv}'")
    print(f"Correct option distribution (should be ~uniform 1–5):")
    print(out_df["true_golden_option_index"].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build validation CSV from fleurs_trimmed.csv for XGBoost fusion training"
    )
    parser.add_argument("--input",      default="fleurs_trimmed.csv",
                        help="Input CSV from trimmer.py (default: fleurs_trimmed.csv)")
    parser.add_argument("--output",     default="validation_dataset.csv",
                        help="Output CSV path (default: validation_dataset.csv)")
    parser.add_argument("--corruption", choices=["moderate", "chaotic", "mixed"],
                        default="mixed",
                        help="Corruption strategy (default: mixed = 2 moderate + 2 chaotic)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    build(args.input, args.output, args.corruption, args.seed)


if __name__ == "__main__":
    main()
