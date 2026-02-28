"""
==============================================================================
Linguistic Plausibility Scoring Module  (SeamlessM4T-free)
==============================================================================
Kaggle Notebook-ready script.

IMPORTANT — Execution order:
  1. Run embedding_workflow.py  Cell 2  →  produces  asr_transcriptions.csv
  2. Run embedding_workflow.py  Cell 3  →  produces  e5_scores.csv
  3. Run THIS script            Cell 2  →  reads     asr_transcriptions.csv
                                           produces  linguistic_scores.csv

SeamlessM4T is NOT loaded here.  The ASR baseline transcriptions (asr_reference)
are read directly from asr_transcriptions.csv (produced by embedding_workflow.py).
This eliminates the redundant 3 GB model load and cuts GPU time significantly.

Pipeline overview:
  1. Read asr_reference from asr_transcriptions.csv (already computed).
  2. Score the baseline AND each candidate string with mT5 cross-entropy loss.
  3. Rank candidates using an exponential-decay deviation penalty.

Mathematical Intuition
----------------------
Cross-Entropy Loss measures how "surprised" a language model is by a given
token sequence.
  - Lower loss  ➜  text is linguistically natural / plausible.
  - Higher loss ➜  text is broken, unusual, or nonsensical.

By comparing each candidate's loss to the baseline's, we quantify linguistic
agreement.  The exponential decay function:
    score_i = exp(-k * |loss_i - loss_baseline|)
maps deviations into (0, 1]:
  - deviation = 0  ➜  score = 1.0  (perfect match with baseline)
  - deviation → ∞  ➜  score → 0.0  (maximally divergent)
The parameter `k` controls sensitivity.
==============================================================================
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Run this FIRST in your Kaggle Notebook                       ║
# ║  (uncomment the lines below and paste into a code cell)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# !pip install -q torch transformers sentencepiece accelerate protobuf pandas
#

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Main Code (paste everything below into ONE cell)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import math
import logging
from typing import List

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EMPTY_TEXT_PENALTY_LOSS: float = 999.0    # Sentinel for empty / whitespace strings

# Path to the ASR transcriptions CSV produced by embedding_workflow.py Cell 2.
# This file must exist before running this script.
ASR_CSV_PATH: str = "asr_transcriptions.csv"  # output of embedding_workflow.py Cell 2


# ==============================================================================
# PHASE 2 — Linguistic Validity Scoring via Cross-Entropy Loss
# ==============================================================================

def calculate_text_loss(
    text: str,
    model: MT5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """
    Compute the Cross-Entropy Loss of a text string using mT5-small.

    How it works
    ------------
    mT5 is an encoder-decoder Transformer.  To get a "self-perplexity" score
    for a standalone string we feed the tokenised text as BOTH the encoder
    ``input_ids`` AND the decoder ``labels``.  The model computes standard
    teacher-forced cross-entropy loss — effectively measuring how well its
    internal knowledge can predict / reconstruct the token sequence.

    Parameters
    ----------
    text : str
        The string to score.
    model : MT5ForConditionalGeneration
        Pre-loaded mT5 model (already on ``device``).
    tokenizer : AutoTokenizer
        Matching mT5 tokenizer.
    device : torch.device
        GPU or CPU.

    Returns
    -------
    float
        Scalar cross-entropy loss.  Lower = more linguistically plausible.
        Returns ``EMPTY_TEXT_PENALTY_LOSS`` (999.0) for empty / whitespace text.
    """
    # ── Edge-case: empty or whitespace-only strings ──────────────────────
    if not text or text.strip() == "":
        logger.warning("Empty text → penalty loss %.1f", EMPTY_TEXT_PENALTY_LOSS)
        return EMPTY_TEXT_PENALTY_LOSS

    # ── Tokenize ─────────────────────────────────────────────────────────
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────
    # input_ids used as BOTH encoder input AND decoder labels ➜ self-loss.
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            labels=inputs.input_ids,
        )

    return outputs.loss.item()


# ==============================================================================
# PHASE 3 — Orchestration & Scoring
# ==============================================================================

def evaluate_linguistic_plausibility(
    asr_reference: str,
    candidate_options: List[str],
) -> List[float]:
    """
    Score each candidate option against the ASR baseline using mT5 loss.

    Workflow
    --------
    1. Load mT5-small **once** for all scoring.
    2. Compute loss_meta = loss(asr_reference).
    3. Compute loss_opt_i for each candidate.
    4. For each candidate i:
         deviation_i = |loss_opt_i − loss_meta|
         score_i     = exp(−k · deviation_i)

    NOTE: Phase 1 (SeamlessM4T ASR) has been removed from this function.
    The ``asr_reference`` is now read directly from asr_transcriptions.csv,
    which is produced by embedding_workflow.py Cell 2.

    Parameters
    ----------
    asr_reference : str
        The ASR baseline transcription (T_baseline) for this audio clip,
        previously generated by SeamlessM4T in embedding_workflow.py.
    candidate_options : list[str]
        Exactly 5 candidate transcription strings.

    Returns
    -------
    list[float]
        5 scores in (0, 1], one per candidate.
    """
    # ──────────────────────────────────────────────────────────────────────
    # 🎛  TUNING PARAMETER
    # k controls the steepness of the exponential decay.
    #   k = 0.5  ➜  lenient  (only large deviations matter)
    #   k = 1.0  ➜  balanced
    #   k = 2.0  ➜  harsh    (even small deviations are penalised)
    # ──────────────────────────────────────────────────────────────────────
    k: float = 1.0  # ← CHANGE THIS TO TUNE SENSITIVITY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load mT5-small (loaded ONCE, reused for all strings) ─────────────
    logger.info("=" * 60)
    logger.info("PHASE 2/3 — Loading mT5-small for linguistic scoring …")
    logger.info("=" * 60)
    mt5_model_id = "google/mt5-small"
    mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_id)
    mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_model_id).to(device)
    mt5_model.eval()

    # ── Score the baseline string ─────────────────────────────────────────
    loss_meta: float = calculate_text_loss(asr_reference, mt5_model, mt5_tokenizer, device)
    logger.info("Baseline loss (loss_meta) = %.4f", loss_meta)

    # ── Score each candidate option ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3/3 — Scoring %d candidates …", len(candidate_options))
    logger.info("=" * 60)

    candidate_losses: List[float] = []
    for idx, option_text in enumerate(candidate_options, start=1):
        loss_i = calculate_text_loss(option_text, mt5_model, mt5_tokenizer, device)
        candidate_losses.append(loss_i)
        logger.info("  Option %d │ loss=%.4f │ '%s'", idx, loss_i, option_text[:60])

    # ── Exponential-decay deviation scoring ──────────────────────────────
    #
    #   For each option i:
    #     deviation_i = |loss_opt_i − loss_meta|
    #     score_i     = exp(−k · deviation_i)
    #
    #   • score ≈ 1.0 ➜ candidate matches baseline quality  (good)
    #   • score → 0.0 ➜ candidate diverges from baseline    (bad)
    #
    scores: List[float] = []
    for idx, loss_i in enumerate(candidate_losses, start=1):
        deviation_i: float = abs(loss_i - loss_meta)
        score_i: float = math.exp(-k * deviation_i)
        scores.append(score_i)
        logger.info("  Option %d │ deviation=%.4f │ score=%.4f", idx, deviation_i, score_i)

    # ── Cleanup ──────────────────────────────────────────────────────────
    del mt5_model, mt5_tokenizer
    torch.cuda.empty_cache()

    return scores


# ==============================================================================
# CSV PIPELINE RUNNER
# ==============================================================================

def run_pipeline_on_csv(
    asr_csv_path: str,
    option_columns: List[str],
    output_csv: str = "linguistic_scores.csv",
) -> pd.DataFrame:
    """
    Process an entire asr_transcriptions.csv through the linguistic plausibility
    pipeline.

    Reads the asr_reference column written by embedding_workflow.py Cell 2 —
    SeamlessM4T is NOT invoked here.

    Parameters
    ----------
    asr_csv_path : str
        Path to asr_transcriptions.csv (output of embedding_workflow.py Cell 2).
        Must contain an ``asr_reference`` column plus the five option columns.
    option_columns : list[str]
        Names of the 5 columns containing candidate text options.
    output_csv : str
        Path to save the results CSV.

    Returns
    -------
    pd.DataFrame
        The original data with 5 new score columns appended.
    """
    df = pd.read_csv(asr_csv_path)
    logger.info("Loaded CSV: %d rows, columns: %s", len(df), list(df.columns))

    # Validate that the required columns exist
    required = option_columns + ["asr_reference"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}. Available: {list(df.columns)}")

    # Prepare score column names
    score_columns = [f"score_{col}" for col in option_columns]
    for col in score_columns:
        df[col] = 0.0

    # Process each row
    for idx, row in df.iterrows():
        logger.info("\n" + "#" * 60)
        logger.info("PROCESSING ROW %d / %d", idx + 1, len(df))
        logger.info("#" * 60)

        # ── Read asr_reference (no audio download needed) ─────────────────
        asr_reference = str(row.get("asr_reference", "")).strip()

        if not asr_reference or asr_reference.lower() in ("nan", "none", ""):
            logger.warning(
                "Row %d: asr_reference is empty or NaN — skipping, assigning zero scores.",
                idx + 1,
            )
            scores = [0.0] * len(option_columns)
        else:
            # ── Extract the 5 candidate texts ─────────────────────────────
            candidates = [str(row[col]).strip() for col in option_columns]

            # ── Run the scoring pipeline ──────────────────────────────────
            try:
                scores = evaluate_linguistic_plausibility(asr_reference, candidates)
            except Exception as exc:
                logger.error("Row %d failed: %s", idx + 1, exc)
                scores = [0.0] * len(option_columns)

        # ── Store scores ─────────────────────────────────────────────────
        for col_name, score_val in zip(score_columns, scores):
            df.at[idx, col_name] = score_val

        # ── Print results for this row ────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  ROW {idx + 1} — LINGUISTIC PLAUSIBILITY SCORES")
        print(f"{'='*60}")
        for opt_col, sc_col, score_val in zip(option_columns, score_columns, scores):
            text_preview = str(row[opt_col])[:80]
            print(f"  {opt_col:<20} │ Score = {score_val:.4f} │ '{text_preview}'")
        print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    logger.info("Results saved to: %s", output_csv)

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Run this cell to execute the pipeline on your CSV             ║
# ║                                                                         ║
# ║  ⚠️  MUST run embedding_workflow.py Cell 2 FIRST to produce             ║
# ║      asr_transcriptions.csv before running this cell.                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────────────────────
# 🎛  CONFIGURE THESE TO MATCH YOUR SETUP
# ──────────────────────────────────────────────────────────────────────────────
ASR_CSV_PATH   = "asr_transcriptions.csv"  # MUST run embedding_workflow.py Cell 2 first
OPTION_COLUMNS = ["option_1", "option_2", "option_3", "option_4", "option_5"]

# ──────────────────────────────────────────────────────────────────────────────
# Run the pipeline
# ──────────────────────────────────────────────────────────────────────────────
results_df = run_pipeline_on_csv(
    asr_csv_path=ASR_CSV_PATH,
    option_columns=OPTION_COLUMNS,
    output_csv="linguistic_scores.csv",
)

# Print final summary
print("\n" + "=" * 60)
print("  ALL SCORES SUMMARY")
print("=" * 60)
score_cols = [f"score_{c}" for c in OPTION_COLUMNS]
print(results_df[OPTION_COLUMNS + score_cols].to_string(index=True))
print("=" * 60)
print(f"\nResults saved to: linguistic_scores.csv")
