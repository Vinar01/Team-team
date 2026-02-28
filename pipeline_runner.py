#!/usr/bin/env python3
"""
pipeline_runner.py — Master Orchestrator for Golden Transcription Selection
============================================================================
Integrates four independent scoring workflows, fuses scores using XGBoost,
selects the "Golden Transcription," and outputs a final WER-annotated CSV.

Execution prerequisites
-----------------------
The two GPU-heavy Kaggle workflows must be run first to produce their CSVs:
  1. embedding_workflow.py  Cell 2  →  asr_transcriptions.csv  (SeamlessM4T ASR)
  2. embedding_workflow.py  Cell 3  →  e5_scores.csv           (E5 semantic)
  3. linguistic_worklow.py          →  linguistic_scores.csv   (mT5 linguistic)

This script then:
  • Computes acoustic scores  (Whisper + WER, from acoustic.py)
  • Computes char scores      (character_scorer.py)
  • Reads e5 + linguistic scores from the pre-computed CSVs
  • Trains XGBoost to fuse all four scores  (if --validation-csv supplied)
  • Selects the golden transcription per audio clip
  • Calculates WER of every option vs the selected golden
  • Saves a final CSV with the exact output schema required

Output schema
-------------
audio_id, language, audio,
option_1, option_2, option_3, option_4, option_5,
golden_ref,
wer_option1, wer_option2, wer_option3, wer_option4, wer_option5

Usage
-----
# With XGBoost training (recommended):
python pipeline_runner.py \\
    --csv dataset.csv \\
    --e5-scores e5_scores.csv \\
    --linguistic-scores linguistic_scores.csv \\
    --validation-csv validation.csv \\
    --output output/pipeline_output.csv

# Load a pre-trained XGBoost model:
python pipeline_runner.py \\
    --csv dataset.csv \\
    --e5-scores e5_scores.csv \\
    --linguistic-scores linguistic_scores.csv \\
    --model-path fusion_model.json \\
    --output output/pipeline_output.csv

# No training data (falls back to equal-weight average):
python pipeline_runner.py \\
    --csv dataset.csv \\
    --e5-scores e5_scores.csv \\
    --linguistic-scores linguistic_scores.csv \\
    --output output/pipeline_output.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from jiwer import wer as jiwer_wer

# ── Optional XGBoost ─────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn(
        "xgboost not installed.  Install it with: pip install xgboost\n"
        "Falling back to equal-weight score averaging.",
        stacklevel=2,
    )

# ── Project modules ───────────────────────────────────────────────────────────
# acoustic.py — AcousticAligner, _softmax_wer, _normalize, _lang_base
from acoustic import AcousticAligner, _softmax_wer, _normalize, _lang_base, resolve_audio
# character_scorer.py — compute_char_scores
from character_scorer import compute_char_scores

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
OPTION_COLS   = ["option_1", "option_2", "option_3", "option_4", "option_5"]
FEATURE_NAMES = ["acoustic_score", "char_score", "semantic_score", "linguistic_score"]
AUDIO_DIR     = Path("temp_audio")
OUTPUT_COLS   = [
    "audio_id", "language", "audio",
    "option_1", "option_2", "option_3", "option_4", "option_5",
    "golden_ref",
    "wer_option1", "wer_option2", "wer_option3", "wer_option4", "wer_option5",
]


# ==============================================================================
# PHASE 1 — Score Aggregator
# ==============================================================================

def aggregate_scores(
    csv_path: str,
    e5_scores_csv: str,
    linguistic_scores_csv: str,
    whisper_model: str = "base",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Execute / ingest all four scoring workflows and compile a score matrix.

    Parameters
    ----------
    csv_path : str
        Base dataset CSV (audio_id, language, audio, option_1 … option_5).
    e5_scores_csv : str
        Path to e5_scores.csv produced by embedding_workflow.py Cell 3.
        Expected columns: e5_score_option_1 … e5_score_option_5.
    linguistic_scores_csv : str
        Path to linguistic_scores.csv produced by linguistic_worklow.py.
        Expected columns: score_option_1 … score_option_5.
    whisper_model : str
        Whisper model size ('tiny', 'base', 'small', 'medium', 'large').

    Returns
    -------
    df : pd.DataFrame
        The base dataset with all columns.
    score_matrix : np.ndarray
        Shape [N_rows, 5_options, 4_features].
        Feature order: [acoustic, char, semantic, linguistic].
    """
    logger.info("=" * 70)
    logger.info("PHASE 1 — Score Aggregation")
    logger.info("=" * 70)

    # ── Load base CSV ─────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    n_rows = len(df)
    logger.info("Base CSV: %d rows", n_rows)

    # ── Load pre-computed E5 scores ───────────────────────────────────────
    logger.info("Loading E5 semantic scores from: %s", e5_scores_csv)
    e5_df = pd.read_csv(e5_scores_csv)
    e5_cols = [f"e5_score_option_{i}" for i in range(1, 6)]
    _check_cols(e5_df, e5_cols, e5_scores_csv)

    # ── Load pre-computed linguistic scores ───────────────────────────────
    logger.info("Loading linguistic scores from: %s", linguistic_scores_csv)
    ling_df = pd.read_csv(linguistic_scores_csv)
    ling_cols = [f"score_option_{i}" for i in range(1, 6)]
    _check_cols(ling_df, ling_cols, linguistic_scores_csv)

    # ── Initialise Whisper aligner (loaded ONCE for all rows) ────────────
    logger.info("Initialising Whisper aligner (model='%s') …", whisper_model)
    aligner = AcousticAligner(model_name=whisper_model)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build score matrix ────────────────────────────────────────────────
    # Shape: [N, 5, 4]  — 5 options × 4 features per row
    score_matrix = np.zeros((n_rows, 5, 4), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        logger.info("Row %d/%d  (audio_id=%s)", i + 1, n_rows, row.get("audio_id", "?"))

        lang       = str(row.get("language", "en")).strip()
        candidates = [str(row.get(col, "")).strip() for col in OPTION_COLS]

        # ── Acoustic scores (Whisper + softmax-WER) ───────────────────────
        acoustic_5 = _compute_acoustic_scores(aligner, row, candidates, lang)

        # ── Character scores (character_scorer.py) ────────────────────────
        char_series = compute_char_scores(row)
        char_5 = [
            float(char_series.get("char_score_1", 0.0)),
            float(char_series.get("char_score_2", 0.0)),
            float(char_series.get("char_score_3", 0.0)),
            float(char_series.get("char_score_4", 0.0)),
            float(char_series.get("char_score_5", 0.0)),
        ]

        # ── E5 semantic scores (pre-computed) ─────────────────────────────
        e5_row  = e5_df.iloc[i] if i < len(e5_df) else None
        e5_5    = [float(e5_row[c]) if e5_row is not None else 0.0 for c in e5_cols]

        # ── Linguistic scores (pre-computed) ──────────────────────────────
        ling_row = ling_df.iloc[i] if i < len(ling_df) else None
        ling_5   = [float(ling_row[c]) if ling_row is not None else 0.0 for c in ling_cols]

        # ── Assemble feature vectors ──────────────────────────────────────
        for opt_idx in range(5):
            score_matrix[i, opt_idx, 0] = acoustic_5[opt_idx]   # acoustic
            score_matrix[i, opt_idx, 1] = char_5[opt_idx]       # char
            score_matrix[i, opt_idx, 2] = e5_5[opt_idx]         # semantic
            score_matrix[i, opt_idx, 3] = ling_5[opt_idx]       # linguistic

        logger.info(
            "  Scores  acoustic=%s  char=%s  e5=%s  ling=%s",
            [round(x, 3) for x in acoustic_5],
            [round(x, 3) for x in char_5],
            [round(x, 3) for x in e5_5],
            [round(x, 3) for x in ling_5],
        )

    logger.info("Score matrix assembled: shape=%s", score_matrix.shape)
    return df, score_matrix


def _compute_acoustic_scores(
    aligner: AcousticAligner,
    row: pd.Series,
    candidates: List[str],
    lang: str,
) -> List[float]:
    """Run Whisper on one audio clip and return softmax scores [0,1] per option."""
    audio_path = resolve_audio(row, AUDIO_DIR)
    if audio_path is None:
        logger.warning("  No audio found — returning uniform acoustic scores.")
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    try:
        scored, _ = aligner.score_candidates(audio_path, candidates, lang=lang)
        # scored is sorted descending; restore original option order
        option_map = {s["option_num"]: s["score"] for s in scored}
        return [option_map.get(i + 1, 0.0) for i in range(5)]
    except Exception as exc:
        logger.error("  Acoustic scoring failed: %s", exc)
        return [0.2, 0.2, 0.2, 0.2, 0.2]


def _check_cols(df: pd.DataFrame, cols: List[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {source}.\nAvailable: {list(df.columns)}")


# ==============================================================================
# PHASE 2 — XGBoost Fusion Engine
# ==============================================================================

def train_xgboost_fusion_model(
    validation_csv: str,
    e5_scores_csv: str,
    linguistic_scores_csv: str,
    whisper_model: str = "base",
    save_path: Optional[str] = None,
) -> "xgb.XGBRanker":
    """
    Train an XGBoost ranking model to fuse all four workflow scores.

    Each candidate option is represented as a feature vector:
        [acoustic_score, char_score, semantic_score, linguistic_score]

    Training objective: rank:pairwise — learns to rank the correct option
    above the four incorrect ones for each audio clip.

    Parameters
    ----------
    validation_csv : str
        CSV with the same columns as the base dataset PLUS
        ``true_golden_option_index`` (1-based int, human-verified best option).
    e5_scores_csv : str
        Path to e5_scores.csv.
    linguistic_scores_csv : str
        Path to linguistic_scores.csv.
    whisper_model : str
        Whisper model size for acoustic scoring.
    save_path : str, optional
        If provided, the trained model is saved to this path (JSON format).

    Returns
    -------
    xgb.XGBRanker
        Trained XGBoost ranker model.
    """
    if not _XGB_AVAILABLE:
        raise RuntimeError("xgboost is required for training.  pip install xgboost")

    logger.info("=" * 70)
    logger.info("PHASE 2 — XGBoost Fusion Model Training")
    logger.info("=" * 70)

    # ── Aggregate scores for the validation set ───────────────────────────
    df, score_matrix = aggregate_scores(
        validation_csv, e5_scores_csv, linguistic_scores_csv, whisper_model
    )

    if "true_golden_option_index" not in df.columns:
        raise ValueError(
            "validation_csv must contain a 'true_golden_option_index' column "
            "(1-based integer: which option is the human-verified best)."
        )

    n_rows = len(df)

    # ── Build flat training table ─────────────────────────────────────────
    # One row per (audio_clip × option) = n_rows × 5 rows
    X_rows, y_rows, groups = [], [], []

    for i in range(n_rows):
        true_idx = int(df.iloc[i]["true_golden_option_index"]) - 1  # 0-based
        for opt_idx in range(5):
            X_rows.append(score_matrix[i, opt_idx, :])          # 4 features
            y_rows.append(1 if opt_idx == true_idx else 0)      # 1 = correct
        groups.append(5)  # group size (5 options per audio clip)

    X_train = np.array(X_rows, dtype=np.float32)
    y_train = np.array(y_rows, dtype=np.int32)
    groups   = np.array(groups, dtype=np.int32)

    logger.info(
        "Training data: %d samples (%d audio clips × 5 options), %d features",
        len(X_train), n_rows, X_train.shape[1],
    )
    logger.info("Positive labels (correct options): %d / %d", y_train.sum(), len(y_train))

    # ── Train XGBRanker ───────────────────────────────────────────────────
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="ndcg",
        random_state=42,
        verbosity=1,
    )

    model.fit(
        X_train, y_train,
        group=groups,
        feature_names=FEATURE_NAMES,
        verbose=True,
    )

    logger.info("XGBoost training complete.")

    # ── Feature importance ────────────────────────────────────────────────
    scores_dict = model.get_booster().get_fscore()
    logger.info("Feature importances: %s", scores_dict)

    # ── Save model ────────────────────────────────────────────────────────
    if save_path:
        model.save_model(save_path)
        logger.info("Model saved → %s", save_path)

    return model


def calculate_final_scores(
    model: "xgb.XGBRanker",
    score_matrix_5x4: np.ndarray,
) -> List[float]:
    """
    Use a trained XGBoost ranker to produce a final bounded score per option.

    Parameters
    ----------
    model : xgb.XGBRanker
        Trained ranker from ``train_xgboost_fusion_model``.
    score_matrix_5x4 : np.ndarray
        Shape [5, 4] — feature vectors for the 5 options of one audio clip.

    Returns
    -------
    list[float]
        5 scores in [0, 1], higher = model predicts this is the best option.
    """
    raw = model.predict(score_matrix_5x4.astype(np.float32))  # shape [5]
    # Apply softmax to map raw ranking scores → [0, 1] probabilities
    shifted = raw - raw.max()
    exps    = np.exp(shifted)
    probs   = exps / exps.sum()
    return probs.tolist()


def _equal_weight_scores(score_matrix_5x4: np.ndarray) -> List[float]:
    """Fallback: simple average of four normalised scores → softmax-normalised."""
    avg = score_matrix_5x4.mean(axis=1)   # shape [5]
    shifted = avg - avg.max()
    exps = np.exp(shifted)
    probs = exps / exps.sum()
    return probs.tolist()


# ==============================================================================
# PHASE 3 — Golden Selection & WER
# ==============================================================================

def select_golden_and_compute_wer(
    options: List[str],
    final_scores: List[float],
    lang: str,
) -> tuple[str, List[float]]:
    """
    Select the golden transcription and compute WER of every option against it.

    Parameters
    ----------
    options : list[str]
        The 5 candidate transcription strings.
    final_scores : list[float]
        Final fused scores (one per option).
    lang : str
        Language code (e.g. 'Arabic_SA', 'en').

    Returns
    -------
    golden_ref : str
        The text of the winning option.
    wer_list : list[float]
        WER of option_1 … option_5 vs golden_ref.
    """
    golden_idx = int(np.argmax(final_scores))   # 0-based index of best option
    golden_ref = options[golden_idx]

    norm_lang = _lang_base(lang)

    wer_list: List[float] = []
    for opt in options:
        try:
            norm_ref = _normalize(golden_ref, norm_lang)
            norm_opt = _normalize(opt, norm_lang)
            if not norm_ref.strip():
                w = 1.0
            else:
                w = round(jiwer_wer(norm_ref, norm_opt), 4)
        except Exception:
            w = float("nan")
        wer_list.append(w)

    return golden_ref, wer_list


# ==============================================================================
# PHASE 4 — Output Generation
# ==============================================================================

def run_full_pipeline(
    csv_path: str,
    e5_scores_csv: str,
    linguistic_scores_csv: str,
    output_csv: str,
    whisper_model: str = "base",
    validation_csv: Optional[str] = None,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline: aggregate → fuse → select golden → write CSV.

    Parameters
    ----------
    csv_path : str
        Base dataset CSV.
    e5_scores_csv : str
        Path to e5_scores.csv.
    linguistic_scores_csv : str
        Path to linguistic_scores.csv.
    output_csv : str
        Destination path for the final output CSV.
    whisper_model : str
        Whisper model size.
    validation_csv : str, optional
        If provided, trains a fresh XGBoost model using this labelled data.
    model_path : str, optional
        Path to a pre-trained XGBoost model (JSON).  Loaded if no validation_csv.

    Returns
    -------
    pd.DataFrame
        Final output dataframe.
    """
    # ── Phase 1: Aggregate scores ─────────────────────────────────────────
    df, score_matrix = aggregate_scores(
        csv_path, e5_scores_csv, linguistic_scores_csv, whisper_model
    )

    # ── Phase 2: Decide fusion strategy ──────────────────────────────────
    model = None

    if validation_csv and _XGB_AVAILABLE:
        logger.info("=" * 70)
        logger.info("PHASE 2 — Training XGBoost fusion model …")
        logger.info("=" * 70)
        save = model_path or "fusion_model.json"
        model = train_xgboost_fusion_model(
            validation_csv, e5_scores_csv, linguistic_scores_csv, whisper_model,
            save_path=save,
        )

    elif model_path and os.path.exists(model_path) and _XGB_AVAILABLE:
        logger.info("Loading pre-trained XGBoost model from: %s", model_path)
        model = xgb.XGBRanker()
        model.load_model(model_path)

    else:
        if not _XGB_AVAILABLE:
            logger.warning("xgboost not available — using equal-weight average fusion.")
        else:
            logger.warning(
                "No validation CSV or saved model provided — "
                "using equal-weight average fusion."
            )

    # ── Phases 3 & 4: Select golden, compute WER, build output ───────────
    logger.info("=" * 70)
    logger.info("PHASE 3 & 4 — Golden Selection, WER, Output Generation")
    logger.info("=" * 70)

    output_rows: List[dict] = []

    for i, (_, row) in enumerate(df.iterrows()):
        lang       = str(row.get("language", "en")).strip()
        candidates = [str(row.get(col, "")).strip() for col in OPTION_COLS]

        # Final scores for this row's 5 options
        row_matrix = score_matrix[i]   # shape [5, 4]

        if model is not None:
            final_scores = calculate_final_scores(model, row_matrix)
        else:
            final_scores = _equal_weight_scores(row_matrix)

        golden_ref, wer_list = select_golden_and_compute_wer(candidates, final_scores, lang)

        golden_idx = int(np.argmax(final_scores))
        logger.info(
            "Row %d: golden=option_%d  score=%.4f  wer_vs_golden=%s",
            i + 1, golden_idx + 1, final_scores[golden_idx],
            [round(w, 4) for w in wer_list],
        )

        output_rows.append({
            "audio_id":   str(row.get("audio_id", i + 1)),
            "language":   lang,
            "audio":      str(row.get("audio", "")),
            "option_1":   candidates[0],
            "option_2":   candidates[1],
            "option_3":   candidates[2],
            "option_4":   candidates[3],
            "option_5":   candidates[4],
            "golden_ref": golden_ref,       # text of winning option
            "wer_option1": wer_list[0],
            "wer_option2": wer_list[1],
            "wer_option3": wer_list[2],
            "wer_option4": wer_list[3],
            "wer_option5": wer_list[4],
        })

    out_df = pd.DataFrame(output_rows, columns=OUTPUT_COLS)

    # ── Save output ───────────────────────────────────────────────────────
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Pipeline complete. Output saved → %s  (%d rows)", out_path, len(out_df))

    return out_df


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master orchestrator: aggregate 4 scoring workflows → XGBoost fusion → golden selection"
    )
    parser.add_argument("--csv",                required=True,
                        help="Base dataset CSV (audio_id, language, audio, option_1…option_5)")
    parser.add_argument("--e5-scores",          required=True,
                        help="Path to e5_scores.csv from embedding_workflow.py")
    parser.add_argument("--linguistic-scores",  required=True,
                        help="Path to linguistic_scores.csv from linguistic_worklow.py")
    parser.add_argument("--output",             default="output/pipeline_output.csv",
                        help="Destination for the final output CSV")
    parser.add_argument("--whisper-model",      default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for acoustic scoring (default: base)")
    parser.add_argument("--validation-csv",     default=None,
                        help="Labelled CSV with 'true_golden_option_index' column "
                             "(triggers XGBoost training)")
    parser.add_argument("--model-path",         default=None,
                        help="Load/save path for the XGBoost model (JSON)")
    args = parser.parse_args()

    out_df = run_full_pipeline(
        csv_path=args.csv,
        e5_scores_csv=args.e5_scores,
        linguistic_scores_csv=args.linguistic_scores,
        output_csv=args.output,
        whisper_model=args.whisper_model,
        validation_csv=args.validation_csv,
        model_path=args.model_path,
    )

    print("\n── Output preview (first 5 rows) ──")
    preview_cols = ["audio_id", "language", "golden_ref",
                    "wer_option1", "wer_option2", "wer_option3", "wer_option4", "wer_option5"]
    print(out_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
