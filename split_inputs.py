#!/usr/bin/env python3
"""
split_inputs.py — Split dataset into per-row input folders
===========================================================
Reads the dataset Excel/CSV and creates one folder per row.
Each folder contains individual single-line text files for that row.

Output structure:
    inputs/
        input1/
            audio_id.txt
            language.txt
            audio_url.txt
            option_1.txt
            option_2.txt
            option_3.txt
            option_4.txt
            option_5.txt
            correct_option.txt
        input2/
            ...

Also writes flat column files alongside (inputs/audio_ids.txt etc.)
for use with --inputs-dir in runner.py.

Usage:
    python split_inputs.py                          # uses dataset_download.xlsx
    python split_inputs.py --file my_dataset.xlsx
    python split_inputs.py --file my_dataset.csv --out-dir my_inputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Map: filename inside each row folder  →  column name in the dataset
COLUMN_MAP = {
    "audio_id.txt":      "audio_id",
    "language.txt":      "language",
    "audio_url.txt":     "audio",
    "option_1.txt":      "option_1",
    "option_2.txt":      "option_2",
    "option_3.txt":      "option_3",
    "option_4.txt":      "option_4",
    "option_5.txt":      "option_5",
    "correct_option.txt":"correct_option",
}

# Flat files written at the top-level inputs/ folder (used by runner.py --inputs-dir)
FLAT_MAP = {
    "audio_ids.txt":       "audio_id",
    "languages.txt":       "language",
    "audio_urls.txt":      "audio",
    "option_1.txt":        "option_1",
    "option_2.txt":        "option_2",
    "option_3.txt":        "option_3",
    "option_4.txt":        "option_4",
    "option_5.txt":        "option_5",
    "correct_options.txt": "correct_option",
}


def load_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def split_inputs(dataset_path: Path, out_dir: Path) -> None:
    df = load_dataset(dataset_path)
    print(f"Loaded: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns found: {list(df.columns)}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-row folders ──────────────────────────────────────────────────────
    print("Creating per-row folders...")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        row_dir = out_dir / f"input{i}"
        row_dir.mkdir(parents=True, exist_ok=True)

        for filename, col in COLUMN_MAP.items():
            value = str(row[col]) if col in df.columns else ""
            (row_dir / filename).write_text(value, encoding="utf-8")

        print(f"  {row_dir.name}/  ({len(COLUMN_MAP)} files)")

    # ── Flat column files (for --inputs-dir in runner.py) ───────────────────
    print("\nWriting flat column files...")
    for filename, col in FLAT_MAP.items():
        if col not in df.columns:
            print(f"  [SKIP] Column '{col}' not found — skipping {filename}")
            continue
        out_path = out_dir / filename
        # Replace internal newlines with a space so each value stays on one line
        lines = df[col].fillna("").astype(str).str.replace(r"\n", " ", regex=True).tolist()
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Written: {out_path.name}  ({len(lines)} lines)")

    print(f"\nDone.")
    print(f"  Per-row folders : {out_dir.resolve()}/input1 .. input{len(df)}")
    print(f"  Flat files      : {out_dir.resolve()}/*.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into per-row input folders + flat column files"
    )
    parser.add_argument("--file",    default="dataset_download.xlsx",
                        help="Path to the input dataset (Excel or CSV)")
    parser.add_argument("--out-dir", default="inputs",
                        help="Output root directory (default: inputs/)")
    args = parser.parse_args()

    split_inputs(Path(args.file), Path(args.out_dir))


if __name__ == "__main__":
    main()
