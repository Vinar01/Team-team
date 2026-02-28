#!/usr/bin/env python3
"""
Acoustic Alignment Module
=========================
Scores transcription candidates against an audio file using Whisper's
encoder + decoder in teacher-forcing mode.

Scoring approach:
  1. Whisper encodes the audio into mel-spectrogram embeddings.
  2. For each candidate, the decoder is run in teacher-forcing mode:
     the candidate tokens are fed one-by-one and the model predicts
     each next token.  The average log-probability over the sequence
     is the *acoustic log-likelihood* of that candidate given the audio.
  3. A *word-count penalty* is added to down-score candidates whose
     length deviates heavily from Whisper's own greedy transcription,
     catching omissions and hallucinations.
  4. The final score is normalised to [0, 1] across the 5 candidates.

Dependencies: openai-whisper, torch, numpy, torchaudio (or soundfile + librosa)
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model loader (singleton, cached per (model_name, device))
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_whisper(model_name: str = "base", device: str | None = None):
    """Load and cache a Whisper model.  Downloads on first call (~150 MB for base)."""
    try:
        import whisper  # openai-whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is required.  Install with: pip install openai-whisper"
        ) from exc

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Whisper model '%s' on %s …", model_name, _device)
    model = whisper.load_model(model_name, device=_device)
    return model, _device


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_as_array(audio_path: str | Path, target_sr: int = 16_000) -> np.ndarray:
    """
    Load any audio file, resample to `target_sr` Hz, collapse to mono.
    Returns float32 array in [-1, 1].
    """
    audio_path = Path(audio_path)

    # ---- Strategy 1: torchaudio (fast, supports mp3/flac/wav/ogg) ----------
    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))  # [C, T], int or float
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        waveform = waveform.mean(dim=0)                   # mono
        audio = waveform.numpy().astype(np.float32)
        # Normalise if integer PCM was returned
        if audio.max() > 1.5:
            audio = audio / 32768.0
        return audio
    except Exception:
        pass

    # ---- Strategy 2: librosa (universal fallback) ---------------------------
    try:
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception:
        pass

    # ---- Strategy 3: soundfile (lossless formats only) ----------------------
    try:
        import soundfile as sf

        data, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != target_sr:
            # Simple integer-ratio resampling via numpy (low quality, last resort)
            ratio = target_sr / sr
            new_len = int(len(data) * ratio)
            data = np.interp(
                np.linspace(0, len(data) - 1, new_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        return data
    except Exception as exc:
        raise RuntimeError(
            f"Could not load audio file '{audio_path}'.  "
            "Install torchaudio, librosa, or soundfile."
        ) from exc


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

class AcousticAligner:
    """
    Scores one or more transcription candidates against an audio clip
    using Whisper's encoder+decoder in teacher-forcing mode.

    Parameters
    ----------
    model_name : str
        Whisper model size: tiny | base | small | medium | large.
        ``base`` gives a good accuracy/speed trade-off for multilingual use.
    device : str | None
        ``"cuda"`` or ``"cpu"``.  Auto-detected when None.
    word_penalty_weight : float
        Weight applied to the word-count penalty (0 = off, 1 = equal weight).
    """

    WHISPER_SAMPLE_RATE: int = 16_000
    WHISPER_N_FRAMES: int = 3_000          # max 30 s at 100 frames/s
    WHISPER_HOP_LENGTH: int = 160          # 10 ms per frame

    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        word_penalty_weight: float = 0.3,
    ) -> None:
        self.model, self.device = _load_whisper(model_name, device)
        self.word_penalty_weight = word_penalty_weight
        # Import once; we need the tokenizer utilities
        import whisper
        self._whisper = whisper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # 30 s in samples at 16 kHz — Whisper's native context window
    WINDOW_SAMPLES: int = 30 * 16_000

    def score_candidates(
        self,
        audio_path: str | Path,
        candidates: Sequence[str],
        lang: str = "en",
    ) -> list[dict]:
        """
        Score each transcription candidate against the audio.

        For audio longer than 30 s the audio is split into non-overlapping
        30-second windows.  Whisper's greedy decode tells us how many words
        each window covers; each candidate is split proportionally so that
        window i is always scored against the matching slice of the candidate
        text.  This guarantees temporal alignment across the full recording.

        Returns
        -------
        list[dict]  — one entry per candidate, sorted high→low score.
            Each dict has:
              ``text``            the original candidate string
              ``acoustic_logprob`` average token log-prob (teacher-forcing)
              ``word_penalty``    word-count deviation penalty
              ``combined_score``  final normalised score in [0, 1]
              ``rank``            1 = best
        """
        audio = _load_audio_as_array(audio_path, self.WHISPER_SAMPLE_RATE)

        # ── Split audio into 30-second windows ───────────────────────────────
        windows = self._split_audio(audio)           # list[np.ndarray]
        n_win   = len(windows)

        # ── Greedy-decode each window to get per-window word counts ──────────
        greedy_texts: list[str] = []
        for w in windows:
            greedy_texts.append(self._greedy_transcribe(w, lang))

        greedy_word_counts = [max(1, _word_count(t)) for t in greedy_texts]
        total_ref_wc = sum(greedy_word_counts)
        logger.debug(
            "Audio split into %d window(s); greedy word counts: %s",
            n_win, greedy_word_counts,
        )

        # ── Score each candidate ─────────────────────────────────────────────
        raw_scores: list[dict] = []
        for cand in candidates:
            cand_words = cand.split()
            total_cand_wc = max(1, len(cand_words))

            window_logprobs: list[float] = []
            cursor = 0   # word cursor into the candidate

            for win_idx, (window, ref_wc_win) in enumerate(
                zip(windows, greedy_word_counts)
            ):
                # Proportional word slice for this window
                proportion = ref_wc_win / total_ref_wc
                n_words    = max(1, round(total_cand_wc * proportion))

                # Last window gets all remaining words
                if win_idx == n_win - 1:
                    slice_words = cand_words[cursor:]
                else:
                    slice_words = cand_words[cursor : cursor + n_words]
                    cursor += n_words

                slice_text = " ".join(slice_words).strip()
                if not slice_text:
                    slice_text = cand_words[0] if cand_words else ""

                mel = self._encode_audio(window)
                lp  = self._teacher_forced_logprob(mel, slice_text, lang)
                window_logprobs.append(lp)

            avg_logprob = sum(window_logprobs) / len(window_logprobs)
            penalty     = self._word_count_penalty(cand, total_ref_wc)

            raw_scores.append(
                {
                    "text":             cand,
                    "acoustic_logprob": avg_logprob,
                    "word_penalty":     penalty,
                    "_raw": avg_logprob - self.word_penalty_weight * penalty,
                }
            )

        # ── Normalise combined score to [0, 1] ───────────────────────────────
        vals = [r["_raw"] for r in raw_scores]
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
        for r in raw_scores:
            r["combined_score"] = (r["_raw"] - lo) / span
            del r["_raw"]

        # ── Rank ─────────────────────────────────────────────────────────────
        raw_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, r in enumerate(raw_scores, 1):
            r["rank"] = i

        return raw_scores

    # ------------------------------------------------------------------
    # Audio windowing
    # ------------------------------------------------------------------

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split a raw audio array into non-overlapping 30-second windows."""
        windows = []
        total   = len(audio)
        start   = 0
        while start < total:
            end = min(start + self.WINDOW_SAMPLES, total)
            windows.append(audio[start:end])
            start = end
        return windows

    def best_candidate(
        self,
        audio_path: str | Path,
        candidates: Sequence[str],
        lang: str = "en",
    ) -> str:
        """Return the highest-scoring candidate text."""
        results = self.score_candidates(audio_path, candidates, lang)
        return results[0]["text"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Convert raw audio array to Whisper mel-spectrogram tensor."""
        audio_tensor = self._whisper.pad_or_trim(audio)
        mel = self._whisper.log_mel_spectrogram(audio_tensor).to(self.device)
        return mel.unsqueeze(0)   # [1, 80, 3000]

    def _greedy_transcribe(self, audio: np.ndarray, lang: str) -> str:
        """Run Whisper greedy decode to get its own transcription."""
        try:
            options = self._whisper.DecodingOptions(
                language=_lang_for_whisper(lang),
                without_timestamps=True,
                beam_size=None,
                fp16=(self.device == "cuda"),
            )
            mel = self._encode_audio(audio)
            result = self._whisper.decode(self.model, mel.squeeze(0), options)
            return result.text.strip()
        except Exception as exc:
            logger.warning("Greedy transcription failed: %s", exc)
            return ""

    @torch.no_grad()
    def _teacher_forced_logprob(
        self,
        mel: torch.Tensor,
        candidate: str,
        lang: str,
    ) -> float:
        """
        Run Whisper decoder in teacher-forcing mode to get average token
        log-probability for `candidate` given `mel`.

        Long candidates are split into chunks of at most MAX_CAND_TOKENS each.
        Each chunk is scored independently against the same audio features and
        the per-token log-probs are averaged across ALL chunks, so no text is
        dropped regardless of transcript length.

        A *higher* (less negative) value means the audio is more consistent
        with the candidate transcription.
        """
        if not candidate.strip():
            return -1e6            # empty string → worst possible score

        tokenizer = self._whisper.tokenizer.get_tokenizer(
            multilingual=self.model.is_multilingual,
            language=_lang_for_whisper(lang),
            task="transcribe",
        )

        # Encode the full candidate into Whisper tokens
        all_tokens = tokenizer.encode(candidate.strip())  # list[int]
        if not all_tokens:
            return -1e6

        sot_sequence = list(tokenizer.sot_sequence)
        notimestamps = tokenizer.no_timestamps
        eot          = tokenizer.eot

        # Whisper decoder hard limit = 448 positions.
        # Reserve: sot_sequence + notimestamps token + eot token
        reserved        = len(sot_sequence) + 1 + 1
        max_cand_tokens = 448 - reserved          # tokens available per chunk

        # Run encoder once — shared across all chunks
        audio_features = self.model.encoder(mel)  # [1, T, D]

        all_log_probs: list[float] = []

        # Process candidate in chunks so every token is scored
        for chunk_start in range(0, len(all_tokens), max_cand_tokens):
            chunk = all_tokens[chunk_start : chunk_start + max_cand_tokens]

            full_tokens = sot_sequence + [notimestamps] + chunk + [eot]

            input_ids = torch.tensor(
                [full_tokens[:-1]], dtype=torch.long, device=self.device
            )

            logits = self.model.decoder(input_ids, audio_features)  # [1, seq, vocab]

            # Score only the candidate tokens (skip the SOT prefix positions)
            prefix_len = len(sot_sequence) + 1   # sot_sequence + notimestamps
            target_ids = torch.tensor(
                full_tokens[prefix_len:],         # candidate chunk tokens + eot
                dtype=torch.long,
                device=self.device,
            )
            pred_logits = logits[0, prefix_len - 1 : prefix_len - 1 + len(target_ids)]

            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[
                torch.arange(len(target_ids), device=self.device), target_ids
            ]
            # Collect raw per-token values (we'll average everything at the end)
            all_log_probs.extend(token_log_probs.tolist())

        return float(sum(all_log_probs) / len(all_log_probs))

    @staticmethod
    def _word_count_penalty(candidate: str, ref_word_count: int) -> float:
        """
        Returns a non-negative penalty that increases with relative deviation
        from Whisper's reference word count.

            penalty = |cand_wc - ref_wc| / ref_wc  (capped at 1.0)
        """
        cand_wc = _word_count(candidate)
        deviation = abs(cand_wc - ref_word_count) / max(1, ref_word_count)
        return min(deviation, 1.0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Rough word count that works for space-tokenised scripts."""
    return len(text.strip().split()) if text.strip() else 0


# Mapping from BCP-47 / ISO 639 codes used in the project to
# the language names expected by openai-whisper.
_LANG_MAP: dict[str, str] = {
    "en": "english",  "fr": "french",   "de": "german",   "es": "spanish",
    "it": "italian",  "pt": "portuguese","ru": "russian",  "ja": "japanese",
    "ko": "korean",   "zh": "chinese",  "hi": "hindi",    "ar": "arabic",
    "nl": "dutch",    "pl": "polish",   "sv": "swedish",  "tr": "turkish",
    "vi": "vietnamese","id": "indonesian","th": "thai",    "uk": "ukrainian",
}


def _lang_for_whisper(lang: str) -> str:
    """Convert a project language code to the string Whisper expects.

    Handles formats like: 'en', 'en-US', 'en_US', 'Arabic_SA', 'arabic'
    """
    # Normalise: lowercase, split on both '-' and '_', take first part
    base = lang.lower().replace("-", "_").split("_")[0]
    return _LANG_MAP.get(base, base)


# ---------------------------------------------------------------------------
# Standalone usage / quick test
# ---------------------------------------------------------------------------

def score_audio(
    audio_path: str | Path,
    candidates: list[str],
    lang: str = "en",
    model_name: str = "base",
) -> list[dict]:
    """
    Convenience wrapper.  Returns scored + ranked candidate list.

    Example
    -------
    >>> results = score_audio("clip.wav", ["hello world", "helo world", "hello word"])
    >>> for r in results:
    ...     print(r["rank"], r["combined_score"]:.3f, r["text"])
    """
    aligner = AcousticAligner(model_name=model_name)
    return aligner.score_candidates(audio_path, candidates, lang)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python acoustic_alignment.py <audio_file> <lang> [candidate1] [candidate2] ...")
        print()
        print("Example:")
        print('  python acoustic_alignment.py clip.wav en "hello world" "helo world" "hello word"')
        sys.exit(1)

    audio_file = sys.argv[1]
    language   = sys.argv[2]
    cands      = sys.argv[3:] if len(sys.argv) > 3 else ["(no candidates provided)"]

    print(f"\nScoring {len(cands)} candidate(s) against '{audio_file}' (lang={language})…\n")
    results = score_audio(audio_file, cands, lang=language)

    for r in results:
        print(
            f"  Rank {r['rank']}  score={r['combined_score']:.4f}"
            f"  logprob={r['acoustic_logprob']:.3f}"
            f"  penalty={r['word_penalty']:.3f}"
            f"  → {r['text']!r}"
        )

    print("\nBest candidate:", results[0]["text"])
    print(json.dumps(results, indent=2, ensure_ascii=False))
