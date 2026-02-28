#!/usr/bin/env python3
"""
runner.py — End-to-End Golden Transcription Selection Pipeline
==============================================================
Usage:
    python runner.py --url "<sharepoint_sharing_url>" [options]
    python runner.py --file dataset.xlsx [options]

Steps:
    1. Download dataset (Excel/CSV) from SharePoint or local file
    2. For each row, locate/download its audio file
    3. Whisper transcribes the audio → model_hypothesis
    4. Normalize model_hypothesis + all 5 candidates via text.py
    5. Pick candidate with lowest WER vs normalised hypothesis → golden_ref
    6. Record golden option NUMBER (1-5) and WER scores
    7. Write output CSV

Output columns:
    audio_id, language, audio, option_1..5,
    golden_ref (option number 1-5),
    wer_option1..5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torchaudio
import whisper
try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False
from jiwer import wer as jiwer_wer

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
# Constants
# ---------------------------------------------------------------------------
OPTION_COLS = ["option_1", "option_2", "option_3", "option_4", "option_5"]
OUTPUT_COLS = [
    "audio_id", "language", "audio",
    "option_1", "option_2", "option_3", "option_4", "option_5",
    "correct_option",
    "golden_ref",
    "is_correct",
    "wer_option1", "wer_option2", "wer_option3", "wer_option4", "wer_option5",
]
AUDIO_DIR   = Path("temp_audio")
OUTPUT_FILE = Path("output_golden.csv")
MAX_SAMPLES = 10  # process all rows


# ---------------------------------------------------------------------------
# Text normalizer (from text.py)
# ---------------------------------------------------------------------------

def _normalize(text: str, lang: str) -> str:
    """Normalize text using the project's text.py pipeline."""
    try:
        from text import normalize
        return normalize(text, lang)
    except Exception as exc:
        logger.warning("Normalization failed (%s) — using raw text: %s", exc, text[:60])
        return text.lower().strip()


def _lang_base(lang: str) -> str:
    """Convert 'Arabic_SA' / 'en-US' to base code like 'ar' / 'en'."""
    part = lang.lower().replace("-", "_").split("_")[0]
    _FULL_TO_ISO = {
        "arabic": "ar", "english": "en", "french": "fr", "german": "de",
        "spanish": "es", "hindi": "hi", "chinese": "zh", "japanese": "ja",
        "korean": "ko", "russian": "ru", "portuguese": "pt", "italian": "it",
    }
    return _FULL_TO_ISO.get(part, part)


def _lang_for_whisper(lang: str) -> str:
    """Convert project language codes to Whisper language strings."""
    base = lang.lower().replace("-", "_").split("_")[0]
    _MAP = {
        "en": "english",    "fr": "french",     "de": "german",    "es": "spanish",
        "it": "italian",    "pt": "portuguese", "ru": "russian",   "ja": "japanese",
        "ko": "korean",     "zh": "chinese",    "hi": "hindi",     "ar": "arabic",
        "nl": "dutch",      "pl": "polish",     "sv": "swedish",   "tr": "turkish",
        "vi": "vietnamese", "id": "indonesian", "th": "thai",      "uk": "ukrainian",
        # full-name variants (e.g. column value is already "Arabic")
        "arabic": "arabic", "english": "english", "hindi": "hindi",
        "chinese": "chinese", "french": "french",
    }
    return _MAP.get(base, base)


# ---------------------------------------------------------------------------
# Acoustic Aligner
# ---------------------------------------------------------------------------

class AcousticAligner:
    """
    Transcribe & Compare strategy:
      1. Whisper transcribes the full audio (handles long files via sliding window).
      2. Normalize BOTH the hypothesis AND each candidate using text.py.
      3. Pick the candidate with the lowest WER vs the normalised hypothesis.
    """

    def __init__(self, model_name: str = "base", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Whisper model '%s' on %s ...", model_name, self.device)
        self.model = whisper.load_model(model_name, device=self.device)

    def score_candidates(
        self,
        audio_path: str | Path,
        candidates: list[str],
        lang: str = "en",
    ) -> list[dict]:
        """
        Returns list of dicts sorted by score ascending (lower WER = better):
            {text, option_num (1-5), score (WER)}
        """
        whisper_lang = _lang_for_whisper(lang)
        norm_lang    = _lang_base(lang)

        # ── Step 1: Load audio (ffmpeg → soundfile/torchaudio fallback) ────
        audio_input: str | Path | np.ndarray = audio_path
        try:
            # whisper.load_audio uses ffmpeg; if ffmpeg is missing this throws
            audio_input = whisper.load_audio(str(audio_path))  # 16 kHz float32 array
        except Exception as ffmpeg_exc:
            logger.warning("whisper.load_audio failed (%s) — trying soundfile/torchaudio",
                           ffmpeg_exc)
            try:
                if _SF_AVAILABLE:
                    data, sr = sf.read(str(audio_path), dtype="float32")
                else:
                    waveform, sr = torchaudio.load(str(audio_path))
                    data = waveform.mean(dim=0).numpy().astype(np.float32)
                # Make mono
                if data.ndim > 1:
                    data = data.mean(axis=1)
                # Resample to 16 kHz if needed
                if sr != 16000:
                    waveform_t = torch.tensor(data).unsqueeze(0)
                    resampler  = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    data = resampler(waveform_t).squeeze(0).numpy().astype(np.float32)
                audio_input = data
            except Exception as load_exc:
                logger.error("Could not load audio %s: %s", audio_path, load_exc)
                return [{"text": c, "option_num": i + 1, "score": float("inf")}
                        for i, c in enumerate(candidates)]

        # ── Step 2: Transcribe ──────────────────────────────────────────────
        try:
            result = self.model.transcribe(
                audio_input,
                language=whisper_lang,
                task="transcribe",
                fp16=(self.device == "cuda"),
            )
            raw_hypothesis = result["text"].strip()
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return [{"text": c, "option_num": i + 1, "score": float("inf")}
                    for i, c in enumerate(candidates)]

        # ── Step 3: Normalize hypothesis ────────────────────────────────────
        norm_hypothesis = _normalize(raw_hypothesis, norm_lang)
        logger.debug("Hypothesis (raw):  %s", raw_hypothesis[:120])
        logger.debug("Hypothesis (norm): %s", norm_hypothesis[:120])

        # ── Step 4: Normalize each candidate and compute WER ────────────────
        scored: list[dict] = []
        for i, cand in enumerate(candidates):
            option_num = i + 1
            if not cand or not cand.strip():
                score = float("inf")
            else:
                norm_cand = _normalize(cand, norm_lang)
                try:
                    score = jiwer_wer(norm_hypothesis, norm_cand)
                except Exception:
                    score = float("inf")
            scored.append({"text": cand, "option_num": option_num, "score": score})

        # ── Step 5: Sort ascending (lowest WER = best) ──────────────────────
        scored.sort(key=lambda x: x["score"])
        return scored


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_sharepoint_file(sharing_url: str, dest_path: Path) -> Path:
    download_url = (
        sharing_url.rstrip("&")
        + ("&" if "?" in sharing_url else "?")
        + "download=1"
    )
    logger.info("Downloading dataset from SharePoint...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(download_url, headers=headers, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)
    logger.info("Saved dataset -> %s  (%d bytes)", dest_path, len(resp.content))
    return dest_path


def load_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        try:
            df = pd.read_excel(file_path)
        except Exception:
            df = pd.read_csv(file_path)
    logger.info("Loaded dataset: %d rows x %d columns", len(df), len(df.columns))
    logger.info("Columns: %s", list(df.columns))
    return df


def _read_one(path: Path) -> str:
    """Read a single-value .txt file, returning its stripped content."""
    return path.read_text(encoding="utf-8").strip()


def load_dataset_from_row_folders(inputs_dir: Path) -> pd.DataFrame:
    """Assemble a DataFrame from per-row subfolders (input1/, input2/, ...).

    Each subfolder must contain:
        audio_id.txt, language.txt, audio_url.txt,
        option_1.txt – option_5.txt, correct_option.txt
    """
    import re
    ROW_FILE_MAP = {
        "audio_id":       "audio_id.txt",
        "language":       "language.txt",
        "audio":          "audio_url.txt",
        "option_1":       "option_1.txt",
        "option_2":       "option_2.txt",
        "option_3":       "option_3.txt",
        "option_4":       "option_4.txt",
        "option_5":       "option_5.txt",
        "correct_option": "correct_option.txt",
    }

    # Collect all inputN/ subdirectories, sorted numerically
    row_dirs = sorted(
        [d for d in inputs_dir.iterdir() if d.is_dir() and re.fullmatch(r"input\d+", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if not row_dirs:
        raise FileNotFoundError(
            f"No per-row subfolders (inputN/) found in: {inputs_dir}"
        )

    logger.info("Found %d per-row subfolders in %s", len(row_dirs), inputs_dir)

    rows: list[dict] = []
    for row_dir in row_dirs:
        record: dict[str, str] = {}
        for col, filename in ROW_FILE_MAP.items():
            fpath = row_dir / filename
            if fpath.exists():
                record[col] = _read_one(fpath)
            else:
                logger.warning("Missing file %s in %s — using empty string", filename, row_dir.name)
                record[col] = ""
        rows.append(record)

    df = pd.DataFrame(rows)
    logger.info("Assembled dataset from row folders: %d rows x %d columns", len(df), len(df.columns))
    return df


def load_dataset_from_files(inputs_dir: Path) -> pd.DataFrame:
    """Assemble a DataFrame from inputs_dir.

    Auto-detects format:
      • If inputN/ subfolders exist  → per-row folder mode
      • Otherwise                    → flat per-column .txt files
    """
    import re
    has_row_folders = any(
        d.is_dir() and re.fullmatch(r"input\d+", d.name)
        for d in inputs_dir.iterdir()
    )

    if has_row_folders:
        logger.info("Detected per-row subfolder structure — using row-folder loader.")
        return load_dataset_from_row_folders(inputs_dir)

    # ── Flat per-column files ────────────────────────────────────────────────
    logger.info("Using flat per-column file loader.")
    FILE_MAP = {
        "audio_id":       "audio_ids.txt",
        "language":       "languages.txt",
        "audio":          "audio_urls.txt",
        "option_1":       "option_1.txt",
        "option_2":       "option_2.txt",
        "option_3":       "option_3.txt",
        "option_4":       "option_4.txt",
        "option_5":       "option_5.txt",
        "correct_option": "correct_options.txt",
    }
    data: dict[str, list[str]] = {}
    for col, filename in FILE_MAP.items():
        path = inputs_dir / filename
        if not path.exists():
            logger.warning("Input file not found, skipping: %s", path)
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        data[col] = lines
        logger.info("Read %d lines from %s", len(lines), path)

    if not data:
        raise FileNotFoundError(f"No input files found in: {inputs_dir}")

    df = pd.DataFrame(data)
    logger.info("Assembled dataset from files: %d rows x %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Audio resolution
# ---------------------------------------------------------------------------

def resolve_audio(row: pd.Series, audio_dir: Path) -> Path | None:
    audio_col = str(row.get("audio", "")).strip()
    audio_id  = str(row.get("audio_id", "")).strip()

    if audio_col and Path(audio_col).exists():
        return Path(audio_col)

    candidate = audio_dir / audio_col
    if candidate.exists():
        return candidate

    if audio_col.startswith("http"):
        ext  = Path(audio_col.split("?")[0]).suffix or ".wav"
        dest = audio_dir / f"{audio_id}{ext}"
        if dest.exists():
            return dest
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
            r = requests.get(audio_col, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest
        except Exception as exc:
            logger.warning("Could not download audio for %s: %s", audio_id, exc)
            return None

    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        p = audio_dir / f"{audio_id}{ext}"
        if p.exists():
            return p

    return None


# ---------------------------------------------------------------------------
# WER helper (normalised)
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str, lang: str) -> float:
    """Compute WER after normalising both strings."""
    try:
        norm_ref = _normalize(reference, lang)
        norm_hyp = _normalize(hypothesis, lang)
        if not norm_ref.strip():
            return 1.0
        return round(jiwer_wer(norm_ref, norm_hyp), 4)
    except Exception as exc:
        logger.warning("WER computation failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, model_name: str = "base") -> pd.DataFrame:
    # ── Limit to first MAX_SAMPLES rows ─────────────────────────────────────
    df = df.head(MAX_SAMPLES).copy()
    logger.info("Processing first %d samples.", len(df))

    aligner = AcousticAligner(model_name=model_name)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict] = []

    for idx, row in df.iterrows():
        audio_id  = str(row.get("audio_id", idx))
        lang      = str(row.get("language", "en")).strip()
        norm_lang = _lang_base(lang)

        candidates = [str(row.get(col, "")).strip() for col in OPTION_COLS]

        # Parse correct_option (may be int, float string like "1.0", or plain "1")
        try:
            correct_option = int(float(str(row.get("correct_option", "")).strip()))
        except (ValueError, TypeError):
            correct_option = None

        logger.info("[%d/%d] audio_id=%s  lang=%s", idx + 1, len(df), audio_id, lang)

        try:
            audio_path = resolve_audio(row, AUDIO_DIR)

            if audio_path is None:
                logger.warning("  No audio found for %s — fallback to option_1", audio_id)
                golden_option_num = 1
                wers = [float("nan")] * 5
            else:
                scored = aligner.score_candidates(audio_path, candidates, lang=lang)
                best   = scored[0]
                golden_option_num = best["option_num"]   # integer 1-5
                logger.info("  Golden: option_%d  (WER=%.4f)", golden_option_num, best["score"])

                # WER of each option vs the golden candidate text (both normalised)
                golden_text = candidates[golden_option_num - 1]
                wers = [compute_wer(golden_text, c, norm_lang) for c in candidates]

        except Exception:
            logger.error("  Error processing %s:\n%s", audio_id, traceback.format_exc())
            golden_option_num = 1
            wers = [float("nan")] * 5

        is_correct = (golden_option_num == correct_option) if correct_option is not None else None

        output_rows.append({
            "audio_id":       audio_id,
            "language":       lang,
            "audio":          str(row.get("audio", "")),
            "option_1":       candidates[0],
            "option_2":       candidates[1],
            "option_3":       candidates[2],
            "option_4":       candidates[3],
            "option_5":       candidates[4],
            "correct_option": correct_option,
            "golden_ref":     golden_option_num,   # NUMBER (1-5), not text
            "is_correct":     is_correct,
            "wer_option1":    wers[0],
            "wer_option2":    wers[1],
            "wer_option3":    wers[2],
            "wer_option4":    wers[3],
            "wer_option5":    wers[4],
        })

    out_df = pd.DataFrame(output_rows, columns=OUTPUT_COLS)
    logger.info("Pipeline complete. %d rows processed.", len(out_df))

    # Accuracy summary
    scored_mask = out_df["is_correct"].notna()
    if scored_mask.any():
        n_scored  = scored_mask.sum()
        n_correct = out_df.loc[scored_mask, "is_correct"].sum()
        accuracy  = n_correct / n_scored * 100
        logger.info("Accuracy: %d / %d correct (%.1f%%)", n_correct, n_scored, accuracy)

    return out_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    global AUDIO_DIR
    parser = argparse.ArgumentParser(description="Golden Transcription Selection Pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url",        help="SharePoint public sharing URL for the dataset")
    src.add_argument("--file",       help="Local path to the dataset (Excel or CSV)")
    src.add_argument("--inputs-dir", help="Folder containing per-column .txt input files",
                     metavar="DIR")
    parser.add_argument("--model",     default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--output",    default=str(OUTPUT_FILE))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    args = parser.parse_args()

    AUDIO_DIR = Path(args.audio_dir)

    if args.url:
        local_dataset = Path("dataset_download.xlsx")
        download_sharepoint_file(args.url, local_dataset)
        df = load_dataset(local_dataset)
    elif args.inputs_dir:
        df = load_dataset_from_files(Path(args.inputs_dir))
    else:
        df = load_dataset(Path(args.file))

    out_df = run_pipeline(df, model_name=args.model)

    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Output saved -> %s", out_path.resolve())

    print("\n── Output preview (first 5 rows) ──")
    print(out_df[["audio_id", "language", "correct_option", "golden_ref", "is_correct",
                  "wer_option1", "wer_option2"]].head(5).to_string(index=False))

    # Final accuracy
    scored_mask = out_df["is_correct"].notna()
    if scored_mask.any():
        n_scored  = scored_mask.sum()
        n_correct = int(out_df.loc[scored_mask, "is_correct"].sum())
        accuracy  = n_correct / n_scored * 100
        print(f"\n── Accuracy: {n_correct}/{n_scored} correct ({accuracy:.1f}%) ──")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()