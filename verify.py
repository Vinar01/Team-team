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


def check_output_matches_inputs(output_path: Path, inputs_dir: Path) -> None:
    """Compare `output_golden.csv` to the per-row `inputs/inputN/option_*.txt` files.

    For each `inputN/` we read `option_1.txt..option_5.txt` and check that the
    corresponding row in the output CSV (matched by `audio_id`) contains the
    exact same strings. Prints mismatches and a summary count.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 3: output CSV vs inputs/  ({output_path})")
    print(f"{'='*60}")

    if not output_path.exists():
        print(f"  [SKIP] Output file not found: {output_path}")
        return

    out_df = pd.read_csv(output_path, dtype=str).fillna("")

    import re
    row_dirs = sorted(
        [d for d in Path(inputs_dir).iterdir() if d.is_dir() and re.fullmatch(r"input\d+", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if not row_dirs:
        print(f"  [SKIP] No per-row input folders found in: {inputs_dir}")
        return

    mismatches = 0
    missing_rows = 0
    for row_dir in row_dirs:
        audio_id_file = row_dir / "audio_id.txt"
        if not audio_id_file.exists():
            print(f"  [MISSING] {row_dir.name}/audio_id.txt")
            continue
        audio_id = audio_id_file.read_text(encoding="utf-8").strip()

        match = out_df[out_df["audio_id"].astype(str) == str(audio_id)]
        if match.empty:
            print(f"  [MISSING ROW] No row with audio_id={audio_id} in {output_path.name}")
            missing_rows += 1
            continue

        out_row = match.iloc[0]
        for k in range(1, 6):
            fname = row_dir / f"option_{k}.txt"
            file_val = fname.read_text(encoding="utf-8").strip() if fname.exists() else ""
            out_val = str(out_row.get(f"option_{k}", "")).strip()
            if file_val != out_val:
                print(f"  [MISMATCH] {row_dir.name}/option_{k}.txt  (audio_id={audio_id})")
                print(f"             file : {repr(file_val[:120])}")
                print(f"             output: {repr(out_val[:120])}")
                mismatches += 1

    if missing_rows == 0 and mismatches == 0:
        print(f"  ✓  All {len(row_dirs)} input folders match the output CSV options exactly.")
    else:
        print()
        print(f"  ✗  {missing_rows} missing output row(s), {mismatches} mismatched option(s) found.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify inputs/ and transcripts/")
    parser.add_argument("--file",            default="dataset_download.xlsx")
    parser.add_argument("--inputs-dir",      default="inputs")
    parser.add_argument("--transcripts-dir", default="transcripts")
    parser.add_argument("--output",         default=str(Path("output") / "output_golden.csv"),
                        help="Path to pipeline output CSV to verify against inputs")
    parser.add_argument("--only-output",    action="store_true",
                        help="Run only the output-vs-inputs check (skip transcripts)")
    args = parser.parse_args()

    dataset_path = Path(args.file)
    suffix = dataset_path.suffix.lower()
    df = pd.read_excel(dataset_path) if suffix in (".xlsx", ".xls") else pd.read_csv(dataset_path)
    print(f"Loaded: {len(df)} rows from {dataset_path}")

    check_inputs(df, Path(args.inputs_dir))
    if args.only_output:
        # Skip transcript checking and run only output comparison
        check_output_matches_inputs(Path(args.output), Path(args.inputs_dir))
        return

    check_transcripts(df, Path(args.transcripts_dir))
    # Check that output CSV options were copied from per-row input files
    check_output_matches_inputs(Path(args.output), Path(args.inputs_dir))


if __name__ == "__main__":
    main()
