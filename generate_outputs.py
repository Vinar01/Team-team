#!/usr/bin/env python3
"""
generate_outputs.py — Generate one output folder per row.

Reads output/output_golden.csv and writes, for each audio_id:

    output/<audio_id>/
        result.json      — full row data as JSON
        summary.txt      — human-readable result card
        wer_scores.csv   — option-level softmax score table

Usage:
    python generate_outputs.py
    python generate_outputs.py --csv output/output_golden.csv
    python generate_outputs.py --csv output/my_results.csv --out-dir output/rows
"""

from __future__ import annotations

import argparse
import json
import csv
import math
from pathlib import Path

import pandas as pd


OPTION_COLS = ["option_1", "option_2", "option_3", "option_4", "option_5"]
WER_COLS    = ["score_option1", "score_option2", "score_option3", "score_option4", "score_option5"]

SUMMARY_PREVIEW = 300   # max chars shown per option in summary.txt
CSV_PREVIEW     = 200   # max chars stored in wer_scores.csv text column


def _oneline(text: str, max_chars: int = SUMMARY_PREVIEW) -> str:
    """Collapse newlines to spaces and truncate for display."""
    flat = " ".join(text.split())   # collapses all whitespace incl. \n
    if len(flat) > max_chars:
        flat = flat[:max_chars].rstrip() + " …"
    return flat


def _fmt_wer(val) -> str:
    try:
        f = float(val)
        return "inf" if math.isinf(f) or math.isnan(f) else f"{f:.4f}"
    except (TypeError, ValueError):
        return str(val)


def generate(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {csv_path}")

    for _, row in df.iterrows():
        audio_id = str(row.get("audio_id", row.name)).strip()
        row_dir  = out_dir / audio_id
        row_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. result.json  (full row) ────────────────────────────────────
        row_dict = row.to_dict()
        with open(row_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(row_dict, f, ensure_ascii=False, indent=2, default=str)

        # ── 2. wer_scores.csv  ────────────────────────────────────────────
        correct_opt = row.get("correct_option")
        try:
            correct_num = int(str(correct_opt).replace("option_", "").strip())
        except ValueError:
            correct_num = 0

        wer_rows = []
        for i, (opt_col, score_col) in enumerate(zip(OPTION_COLS, WER_COLS), 1):
            raw_text = str(row.get(opt_col, ""))
            wer_rows.append({
                "option_num":   i,
                "text":         _oneline(raw_text, CSV_PREVIEW),
                "text_chars":   len(raw_text),
                "score":        _fmt_wer(row.get(score_col, "")),
                "is_correct":   "YES" if i == correct_num else "",
                "is_predicted": "YES" if f"option_{i}" == str(row.get("golden_ref", "")) else "",
            })

        with open(row_dir / "wer_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["option_num", "text", "text_chars",
                                                    "score", "is_correct", "is_predicted"])
            writer.writeheader()
            writer.writerows(wer_rows)

        # ── 3. summary.txt  ──────────────────────────────────────────────
        SEP  = "─" * 60
        golden = str(row.get("golden_ref", "N/A"))
        try:
            correct_correct = bool(row.get("is_correct"))
        except Exception:
            correct_correct = None

        lines = [
            SEP,
            f"  Audio ID   : {audio_id}",
            f"  Language   : {row.get('language', 'N/A')}",
            f"  Audio URL  : {row.get('audio', 'N/A')}",
            SEP,
            "  Options",
            SEP,
        ]
        for wr in wer_rows:
            tags = []
            if wr["is_correct"]:
                tags.append("CORRECT")
            if wr["is_predicted"]:
                tags.append("PREDICTED")
            tag_str   = "  ← " + " | ".join(tags) if tags else ""
            size_note = f"  [{wr['text_chars']} chars]" if wr["text_chars"] > CSV_PREVIEW else ""
            lines.append(f"  Option {wr['option_num']}  score={wr['score']}{tag_str}{size_note}")
            # wr["text"] is already collapsed (no embedded newlines) + truncated
            lines.append(f"    {wr['text']}")
            lines.append("")

        lines += [
            SEP,
            f"  golden_ref  : {golden}",
            f"  correct_opt : option_{correct_num}" if correct_num else "  correct_opt : (unknown)",
            f"  is_correct  : {correct_correct}",
            SEP,
        ]

        with open(row_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Done — {len(df)} folders written under {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-row output folders")
    parser.add_argument("--csv",     default="output/output_golden.csv",
                        help="Input CSV file  (default: output/output_golden.csv)")
    parser.add_argument("--out-dir", default="output/rows",
                        help="Parent folder for per-row subfolders  (default: output/rows)")
    args = parser.parse_args()

    generate(Path(args.csv), Path(args.out_dir))


if __name__ == "__main__":
    main()
