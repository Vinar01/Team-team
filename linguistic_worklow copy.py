"""
==============================================================================
Linguistic Plausibility & Acoustic Baseline Module
==============================================================================
Kaggle Notebook-ready script.  Copy-paste the CELL 1 block into your first
notebook cell, then the rest into Cell 2.

Pipeline overview:
  1. Generate a pseudo-ground-truth baseline transcription using Meta's
     SeamlessM4Tv2 ASR model.
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
# !pip install -q torch torchaudio transformers sentencepiece accelerate protobuf pandas requests
#

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Main Code (paste everything below into ONE cell)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import math
import os
import logging
import tempfile
from typing import List, Tuple, Optional

import pandas as pd
import requests
import torch
import torchaudio
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    SeamlessM4Tv2Model,            # Unified model (official, best-tested path)
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
SEAMLESS_TARGET_SR: int = 16_000          # SeamlessM4T strictly requires 16 kHz
EMPTY_TEXT_PENALTY_LOSS: float = 999.0    # Sentinel for empty / whitespace strings


# ==============================================================================
# PHASE 1 — Meta ASR Baseline Generation
# ==============================================================================

def _load_audio_as_mono_16khz(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, convert to mono, and resample to 16 kHz.

    Returns
    -------
    (waveform_2d, 16000)
        waveform_2d shape is [1, num_samples]; sample rate is always 16 000.
        Kept as 2D tensor to match official HuggingFace examples.
    """
    waveform, native_sr = torchaudio.load(audio_path)
    logger.info("Loaded audio — shape=%s, native_sr=%d Hz", list(waveform.shape), native_sr)

    # ── Convert to mono by averaging channels ────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        logger.info("Down-mixed to mono.")

    # ── Resample to 16 kHz if needed ─────────────────────────────────────
    # Uses torchaudio.functional.resample (matches official model card example)
    if native_sr != SEAMLESS_TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=native_sr,
            new_freq=SEAMLESS_TARGET_SR,
        )
        logger.info("Resampled %d Hz → %d Hz.", native_sr, SEAMLESS_TARGET_SR)

    # Keep shape as [1, num_samples] — this is what the processor expects
    return waveform, SEAMLESS_TARGET_SR


def generate_meta_baseline(audio_path: str) -> str:
    """
    Generate a pseudo-ground-truth English transcription from audio using
    Meta's SeamlessM4Tv2 unified model.

    Uses the official HuggingFace model card pattern:
      - SeamlessM4Tv2Model (unified class)
      - generate_speech=False for text-only output
      - Audio passed as tensor to processor

    Strategy
    --------
    1. Try loading in float16 (fits 16 GB VRAM on T4/P100).
    2. If that fails (OOM), retry with device_map="auto" for CPU+GPU sharding.
    3. If that also fails, retry in float32 with device_map="auto".

    Parameters
    ----------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    str
        The transcribed English text (T_baseline).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load & pre-process audio ─────────────────────────────────────────
    waveform, sr = _load_audio_as_mono_16khz(audio_path)

    # ── Load processor ───────────────────────────────────────────────────
    model_id = "facebook/seamless-m4t-v2-large"
    logger.info("Loading processor from %s …", model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # ── Prepare audio inputs (pass tensor directly, as in official examples)
    audio_inputs = processor(
        audio=waveform,           # 2-D tensor [1, samples] — matches model card
        sampling_rate=sr,
        return_tensors="pt",
    )

    # ── Load model (3 fallback attempts) ─────────────────────────────────
    model = None
    load_strategies = [
        {"label": "float16 on device",   "kwargs": {"torch_dtype": torch.float16}},
        {"label": "float16 device_map",  "kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}},
        {"label": "float32 device_map",  "kwargs": {"device_map": "auto"}},
    ]

    for strategy in load_strategies:
        try:
            logger.info("Loading %s (%s) …", model_id, strategy["label"])
            model = SeamlessM4Tv2Model.from_pretrained(model_id, **strategy["kwargs"])
            # If no device_map, manually move to GPU
            if "device_map" not in strategy["kwargs"]:
                model = model.to(device)
            logger.info("✓ Model loaded: %s", strategy["label"])
            break
        except Exception as exc:
            logger.warning("✗ Failed (%s): %s", strategy["label"], exc)
            model = None
            torch.cuda.empty_cache()

    if model is None:
        raise RuntimeError(
            "Could not load SeamlessM4Tv2 model with any strategy. "
            "Check GPU memory and internet connectivity."
        )

    model.eval()

    # ── Move input tensors to model's device ─────────────────────────────
    model_device = next(model.parameters()).device
    audio_inputs = {
        k: v.to(model_device) if isinstance(v, torch.Tensor) else v
        for k, v in audio_inputs.items()
    }

    # ── Generate text transcription ──────────────────────────────────────
    # Using generate_speech=False → returns text token IDs only.
    # This matches the official HuggingFace model card example exactly.
    with torch.no_grad():
        output_tokens = model.generate(
            **audio_inputs,
            tgt_lang="eng",              # target language: English
            generate_speech=False,       # text-only output (no speech synthesis)
        )

    # output_tokens[0] is the token ID tensor for the first batch element.
    # .tolist() gives a nested list → [0] extracts the sequence.
    baseline_text: str = processor.decode(
        output_tokens[0].tolist()[0],
        skip_special_tokens=True,
    )
    logger.info("T_baseline: '%s'", baseline_text)

    # ── Free VRAM for Phase 2 ────────────────────────────────────────────
    del model, processor, audio_inputs, output_tokens
    torch.cuda.empty_cache()

    return baseline_text


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
    audio_path: str,
    candidate_options: List[str],
) -> List[float]:
    """
    Master evaluation function.

    Workflow
    -------
    1. Generate T_baseline from audio  (Phase 1).
    2. Load mT5-small **once** for all scoring.
    3. Compute loss_meta = loss(T_baseline).
    4. Compute loss_opt_i for each candidate.
    5. For each candidate i:
         deviation_i = |loss_opt_i − loss_meta|
         score_i     = exp(−k · deviation_i)

    Parameters
    ----------
    audio_path : str
        Path to audio file.
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

    # ── Step 1: Generate baseline transcription from audio ───────────────
    logger.info("=" * 60)
    logger.info("PHASE 1/3 — Generating Meta ASR baseline …")
    logger.info("=" * 60)
    t_baseline: str = generate_meta_baseline(audio_path)

    # ── Step 2: Load mT5-small (loaded ONCE, reused for all strings) ─────
    logger.info("=" * 60)
    logger.info("PHASE 2/3 — Loading mT5-small for linguistic scoring …")
    logger.info("=" * 60)
    mt5_model_id = "google/mt5-small"
    mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_id)
    mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_model_id).to(device)
    mt5_model.eval()

    # ── Step 3: Score the baseline string ────────────────────────────────
    loss_meta: float = calculate_text_loss(t_baseline, mt5_model, mt5_tokenizer, device)
    logger.info("Baseline loss (loss_meta) = %.4f", loss_meta)

    # ── Step 4: Score each candidate option ──────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3/3 — Scoring %d candidates …", len(candidate_options))
    logger.info("=" * 60)

    candidate_losses: List[float] = []
    for idx, option_text in enumerate(candidate_options, start=1):
        loss_i = calculate_text_loss(option_text, mt5_model, mt5_tokenizer, device)
        candidate_losses.append(loss_i)
        logger.info("  Option %d │ loss=%.4f │ '%s'", idx, loss_i, option_text[:60])

    # ── Step 5: Exponential-decay deviation scoring ──────────────────────
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

def _download_audio(url: str, download_dir: str = "temp_audio") -> str:
    """
    Download an audio file from a URL to a local directory.
    Skips download if the file already exists locally.

    Returns
    -------
    str
        Local file path to the downloaded audio.
    """
    os.makedirs(download_dir, exist_ok=True)
    filename = os.path.basename(url).split("?")[0]  # strip query params
    if not filename:
        filename = "audio_download.wav"
    local_path = os.path.join(download_dir, filename)

    if os.path.exists(local_path):
        logger.info("Audio already cached: %s", local_path)
        return local_path

    logger.info("Downloading audio: %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    logger.info("Saved to: %s", local_path)
    return local_path


def run_pipeline_on_csv(
    csv_path: str,
    audio_column: str,
    option_columns: List[str],
    output_csv: str = "linguistic_scores.csv",
) -> pd.DataFrame:
    """
    Process an entire CSV file through the linguistic plausibility pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    audio_column : str
        Name of the column containing audio file paths or URLs.
    option_columns : list[str]
        Names of the 5 columns containing candidate text options.
    output_csv : str
        Path to save the results CSV.

    Returns
    -------
    pd.DataFrame
        The original data with 5 new score columns appended.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV: %d rows, columns: %s", len(df), list(df.columns))

    # Validate that the required columns exist
    missing = [c for c in [audio_column] + option_columns if c not in df.columns]
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

        # ── Get audio path (download if URL) ─────────────────────────
        audio_source = str(row[audio_column]).strip()
        if audio_source.startswith("http://") or audio_source.startswith("https://"):
            audio_path = _download_audio(audio_source)
        else:
            audio_path = audio_source  # already a local path

        # ── Extract the 5 candidate texts ────────────────────────────
        candidates = [str(row[col]).strip() for col in option_columns]

        # ── Run the scoring pipeline ─────────────────────────────────
        try:
            scores = evaluate_linguistic_plausibility(audio_path, candidates)
        except Exception as exc:
            logger.error("Row %d failed: %s", idx + 1, exc)
            scores = [0.0] * len(option_columns)

        # ── Store scores ─────────────────────────────────────────────
        for col_name, score_val in zip(score_columns, scores):
            df.at[idx, col_name] = score_val

        # ── Print results for this row ───────────────────────────────
        print(f"\n{'='*60}")
        print(f"  ROW {idx + 1} — LINGUISTIC PLAUSIBILITY SCORES")
        print(f"{'='*60}")
        for opt_col, sc_col, score_val in zip(option_columns, score_columns, scores):
            text_preview = str(row[opt_col])[:80]
            print(f"  {opt_col:<20} │ Score = {score_val:.4f} │ '{text_preview}'")
        print(f"{'='*60}\n")

    # ── Save results ─────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    logger.info("Results saved to: %s", output_csv)

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Run this cell to execute the pipeline on your CSV             ║
# ║                                                                         ║
# ║  ⚠️  EDIT THE 3 VARIABLES BELOW TO MATCH YOUR CSV COLUMN NAMES          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────────────────────
# 🎛  CONFIGURE THESE TO MATCH YOUR CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "/kaggle/input/datasets/visheshe/habibi/Transcription Assessment Arabic_SA Dataset.csv"  # ← CHANGE to your actual Kaggle CSV path
AUDIO_COLUMN    = "audio"                                     # ← column with audio URLs / paths
OPTION_COLUMNS  = ["option_1", "option_2", "option_3", "option_4", "option_5"]

# ──────────────────────────────────────────────────────────────────────────────
# Run the pipeline
# ──────────────────────────────────────────────────────────────────────────────
results_df = run_pipeline_on_csv(
    csv_path=CSV_PATH,
    audio_column=AUDIO_COLUMN,
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
