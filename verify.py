#!/usr/bin/env python3
"""
verify.py — Sanity-check inputs/ folders and transcripts/ against the source Excel.

Checks:
  1. Per-row folder files match the Excel values (inputs/inputN/ vs dataset)
  2. Transcript files exist and contain only Arabic characters (no garbage)

Usage:
    python verify.py
    python verify.py --file my_dataset.xlsx --inputs-dir inputs --transcripts-dir transcripts
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def has_non_arabic(text: str) -> list[str]:
    """Return list of suspicious non-Arabic tokens found in text."""
    # Allow Arabic script, Arabic punctuation, digits, basic ASCII punctuation/spaces
    bad = re.findall(
        r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
        r"\s\d.,،؛؟!؟:\-\"'()[\]{}]",
        text,
    )
    return bad


COLUMN_MAP = {
    "audio_id.txt":       "audio_id",
    "language.txt":       "language",
    "audio_url.txt":      "audio",
    "option_1.txt":       "option_1",
    "option_2.txt":       "option_2",
    "option_3.txt":       "option_3",
    "option_4.txt":       "option_4",
    "option_5.txt":       "option_5",
    "correct_option.txt": "correct_option",
}


# ── 1. Check inputs/ folder against Excel ────────────────────────────────────

def check_inputs(df: pd.DataFrame, inputs_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"CHECK 1: inputs/ folders vs Excel  ({inputs_dir})")
    print(f"{'='*60}")

    errors = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        row_dir = inputs_dir / f"input{i}"
        if not row_dir.exists():
            print(f"  [MISSING]  {row_dir.name}/ folder does not exist")
            errors += 1
            continue

        for filename, col in COLUMN_MAP.items():
            if col not in df.columns:
                continue
            fpath = row_dir / filename
            if not fpath.exists():
                print(f"  [MISSING]  input{i}/{filename}")
                errors += 1
                continue

            excel_val = str(row[col]).strip()
            file_val  = fpath.read_text(encoding="utf-8").strip()

            if excel_val != file_val:
                print(f"  [MISMATCH] input{i}/{filename}")
                print(f"             Excel: {repr(excel_val[:80])}")
                print(f"             File : {repr(file_val[:80])}")
                errors += 1

    if errors == 0:
        print(f"  ✓  All {len(df)} row folders match the Excel exactly.")
    else:
        print(f"\n  ✗  {errors} error(s) found.")


# ── 2. Check transcripts/ for Arabic correctness ─────────────────────────────

def check_transcripts(df: pd.DataFrame, transcripts_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"CHECK 2: transcripts/ quality  ({transcripts_dir})")
    print(f"{'='*60}")

    if not transcripts_dir.exists():
        print("  [SKIP] transcripts/ folder does not exist yet — run the pipeline first.")
        return

    transcript_files = sorted(transcripts_dir.glob("*.txt"),
                               key=lambda p: int(p.stem) if p.stem.isdigit() else 0)

    if not transcript_files:
        print("  [SKIP] No transcript files found.")
        return

    issues = 0
    for tf in transcript_files:
        text = tf.read_text(encoding="utf-8").strip()

        # Find the matching Excel row
        audio_id = tf.stem
        match = df[df["audio_id"].astype(str) == audio_id]
        correct_opt = None
        if not match.empty:
            try:
                correct_opt = int(float(str(match.iloc[0]["correct_option"])))
                correct_text = str(match.iloc[0][f"option_{correct_opt}"]).strip()
            except Exception:
                correct_text = None
        else:
            correct_text = None

        bad_chars = has_non_arabic(text)

        status = "✓"
        notes  = []
        if not text:
            status = "✗"
            notes.append("EMPTY transcript")
        if bad_chars:
            status = "✗"
            unique_bad = list(dict.fromkeys(bad_chars))[:10]
            notes.append(f"non-Arabic chars: {''.join(unique_bad)!r}")

        note_str = "  |  " + " | ".join(notes) if notes else ""
        print(f"  [{status}] {tf.name:12s}  {text[:70]!r}{note_str}")

        if bad_chars or not text:
            issues += 1

    print()
    if issues == 0:
        print(f"  ✓  All {len(transcript_files)} transcripts look clean (Arabic only).")
    else:
        print(f"  ✗  {issues} transcript(s) have issues.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify inputs/ and transcripts/")
    parser.add_argument("--file",            default="dataset_download.xlsx")
    parser.add_argument("--inputs-dir",      default="inputs")
    parser.add_argument("--transcripts-dir", default="transcripts")
    args = parser.parse_args()

    dataset_path = Path(args.file)
    suffix = dataset_path.suffix.lower()
    df = pd.read_excel(dataset_path) if suffix in (".xlsx", ".xls") else pd.read_csv(dataset_path)
    print(f"Loaded: {len(df)} rows from {dataset_path}")

    check_inputs(df, Path(args.inputs_dir))
    check_transcripts(df, Path(args.transcripts_dir))


if __name__ == "__main__":
    main()
