import math
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Attempt to use python-Levenshtein for speed; fall back to difflib (stdlib)
# ---------------------------------------------------------------------------
try:
    from Levenshtein import distance as _levenshtein_distance
    _LEV_BACKEND = "python-Levenshtein"
except ImportError:
    import difflib

    def _levenshtein_distance(a: str, b: str) -> int:
        """Pure-Python edit distance via difflib (zero extra dependencies)."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        # Use SequenceMatcher opcode count as an edit-distance approximation,
        # then fall back to a proper DP matrix for correctness.
        n, m = len(a), len(b)
        # Standard DP table — O(n*m) but fine for transcript lengths
        prev = list(range(m + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * m
            for j, cb in enumerate(b, 1):
                curr[j] = min(
                    prev[j] + 1,           # deletion
                    curr[j - 1] + 1,       # insertion
                    prev[j - 1] + (ca != cb),  # substitution
                )
            prev = curr
        return prev[m]

    _LEV_BACKEND = "difflib-dp"


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def compute_char_scores(row, epsilon: float = 1e-5, k: float = 1.0):
    """
    Compute a STRICTLY BOUNDED [0.0, 1.0] character-level score for each of
    the 5 transcription options in a DataFrame row.

    Scoring mode is selected automatically:

    Mode A — Referential Exactness  (preferred)
        Used when the row contains a non-empty ``asr_reference`` column.
        Formula:
            Similarity_i = 1.0 - edit_distance(option_i, reference)
                               / max(len(option_i), len(reference))
        • 1.0 → option is character-identical to the reference.
        • 0.0 → option shares no characters with the reference.

    Mode B — Inverse Exponential MAD  (fallback)
        Used when no ``asr_reference`` is available.
        Formula:
            deviation_i = |len(option_i) - median_length| / (MAD + epsilon)
            Score_i     = exp(-k * deviation_i²)
        • 1.0 → option length exactly matches the group median.
        • → 0.0 → option length diverges increasingly from the median.
        ``k`` controls steepness; default k=1.0 (balanced).

    Parameters
    ----------
    row : pandas.Series
        Must contain ``option_1`` … ``option_5``.
        May optionally contain ``asr_reference``.
    epsilon : float
        Small constant added to MAD to prevent division-by-zero (Mode B).
    k : float
        Decay steepness for Mode B (default 1.0).

    Returns
    -------
    pandas.Series
        Five scores in [0.0, 1.0] with index
        ['char_score_1', 'char_score_2', 'char_score_3', 'char_score_4', 'char_score_5'].
    """
    _out_index = ['char_score_1', 'char_score_2', 'char_score_3', 'char_score_4', 'char_score_5']
    _zero = pd.Series([0.0] * 5, index=_out_index)

    try:
        options = [
            str(row['option_1']) if pd.notna(row.get('option_1')) else "",
            str(row['option_2']) if pd.notna(row.get('option_2')) else "",
            str(row['option_3']) if pd.notna(row.get('option_3')) else "",
            str(row['option_4']) if pd.notna(row.get('option_4')) else "",
            str(row['option_5']) if pd.notna(row.get('option_5')) else "",
        ]

        # ── Mode A: Normalized Levenshtein vs asr_reference ─────────────────
        ref = row.get('asr_reference', None)
        if ref is not None and pd.notna(ref) and str(ref).strip():
            reference = str(ref).strip()
            scores = []
            for opt in options:
                if not opt:
                    scores.append(0.0)
                    continue
                denom = max(len(opt), len(reference))
                if denom == 0:
                    scores.append(1.0)
                else:
                    edit_dist = _levenshtein_distance(opt, reference)
                    similarity = 1.0 - edit_dist / denom
                    scores.append(max(0.0, min(1.0, similarity)))
            return pd.Series(scores, index=_out_index)

        # ── Mode B: Inverse Exponential MAD (no reference available) ────────
        C = np.array([len(opt) for opt in options], dtype=float)
        m = np.median(C)
        abs_deviations = np.abs(C - m)
        mad = np.median(abs_deviations)

        scores = []
        for dev in abs_deviations:
            normalized_dev = dev / (mad + epsilon)
            score = math.exp(-k * normalized_dev ** 2)
            scores.append(round(score, 6))

        return pd.Series(scores, index=_out_index)

    except Exception as e:
        print(f"Error processing row {row.get('audio_id', 'Unknown')}: {e}")
        return _zero


# ---------------------------------------------------------------------------
# Backward-compatible alias  (keeps any code that calls the old name working)
# ---------------------------------------------------------------------------

def compute_squared_mad_penalties(row, epsilon: float = 1e-5):
    """Deprecated alias for compute_char_scores().

    Previously returned unbounded squared-MAD penalties (0 = best, ∞ = worst).
    Now delegates to compute_char_scores() which returns bounded [0, 1] scores
    (1.0 = best, 0.0 = worst).
    """
    return compute_char_scores(row, epsilon=epsilon)


# ==========================================
# Phase 3 Execution Pipeline
# ==========================================

def run_character_scoring_pipeline(input_csv_path, output_csv_path):
    """
    Process an entire CSV through the character scorer and write results.

    Detects automatically whether 'asr_reference' is present and picks the
    appropriate scoring mode (Levenshtein vs. Inverse Exponential MAD).
    """
    print(f"Loading data from {input_csv_path}...")
    print(f"[char_scorer] Levenshtein backend: {_LEV_BACKEND}")

    df = pd.read_csv(input_csv_path)

    required_cols = ['option_1', 'option_2', 'option_3', 'option_4', 'option_5']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV is missing one of the required columns: {required_cols}")

    has_ref = 'asr_reference' in df.columns
    mode = "Referential Exactness (Levenshtein)" if has_ref else "Inverse Exponential MAD"
    print(f"Scoring mode: {mode}")

    score_columns = ['char_score_1', 'char_score_2', 'char_score_3', 'char_score_4', 'char_score_5']
    df[score_columns] = df.apply(compute_char_scores, axis=1)

    df.to_csv(output_csv_path, index=False)
    print(f"Success! Scored data saved to {output_csv_path}")

    print("\nPreview of the calculated scores:")
    print(df[['option_1', 'char_score_1', 'option_2', 'char_score_2']].head())

    # Sanity-check: all scores must be in [0, 1]
    for col in score_columns:
        out_of_range = df[(df[col] < 0.0) | (df[col] > 1.0)]
        if not out_of_range.empty:
            print(f"  WARNING: {len(out_of_range)} rows have {col} outside [0, 1]")


if __name__ == "__main__":
    INPUT_FILE  = "transcription_assessment.csv"
    OUTPUT_FILE = "transcription_assessment_scored.csv"
    run_character_scoring_pipeline(INPUT_FILE, OUTPUT_FILE)
