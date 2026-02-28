"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  E5 SEMANTIC SCORING   —  2-Stage Kaggle Pipeline                          ║
║                                                                            ║
║  STAGE 1  (Cell 2):  SeamlessM4T  →  ASR the audio once, save to CSV      ║
║  STAGE 2  (Cell 3):  multilingual-E5  →  embed reference + candidates,    ║
║                      rank by cosine similarity, save scores to CSV         ║
║                                                                            ║
║  Why 2 stages?                                                             ║
║    SeamlessM4T is GPU-heavy.  Run it once, save the output CSV.            ║
║    Then you can iterate on the E5 scoring without re-running ASR.          ║
║                                                                            ║
║  Kaggle:  GPU T4 x2  |  Internet ON                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — pip installs  (run once)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# !pip install -q torch torchaudio transformers sentencepiece accelerate pandas requests


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — STAGE 1: ASR  (SeamlessM4T speech → text)                      ║
# ║                                                                            ║
# ║  Reads your CSV, transcribes every audio file, adds an 'asr_reference'   ║
# ║  column, saves to  asr_transcriptions.csv                                 ║
# ║                                                                            ║
# ║  !! Run this cell once.  It produces the cached reference transcriptions. ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────── configure ───────────────────────────────────────────────────
ASR_CSV_INPUT  = "/kaggle/input/datasets/visheshe/habibi/Transcription Assessment Arabic_SA Dataset.csv"
AUDIO_COLUMN   = "audio"
ASR_OUTPUT_CSV = "asr_transcriptions.csv"   # ← Stage 2 will read THIS file
ASR_MODEL_ID   = "facebook/seamless-m4t-v2-large"
ASR_TARGET_SR  = 16000   # SeamlessM4T expects 16 kHz
# ─────────────────────────────────────────────────────────────────────────────

import traceback, os, requests
import torch, torchaudio, pandas as pd
from transformers import AutoProcessor, SeamlessM4Tv2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ASR] Device: {device}")

# ── load model ────────────────────────────────────────────────────────────────
print(f"[ASR] Loading SeamlessM4T: {ASR_MODEL_ID} ...")
_asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
_asr_model = SeamlessM4Tv2Model.from_pretrained(
    ASR_MODEL_ID, torch_dtype=torch.float16
).to(device)
_asr_model.eval()
print("[ASR] ✓ Model ready\n")

# ── load CSV ──────────────────────────────────────────────────────────────────
_df = pd.read_csv(ASR_CSV_INPUT)
print(f"[ASR] CSV: {len(_df)} rows")

# pre-fill with empty strings in case some rows fail
_df["asr_reference"] = ""

def _download(url, cache="temp_audio"):
    os.makedirs(cache, exist_ok=True)
    fn = os.path.basename(url.split("?")[0]) or "audio.wav"
    lp = os.path.join(cache, fn)
    if os.path.exists(lp):
        return lp
    r = requests.get(url, timeout=120); r.raise_for_status()
    open(lp, "wb").write(r.content)
    return lp

def _transcribe(audio_path):
    """
    Transcribe the full audio file using SeamlessM4T.
    Audios are ~30s which is safely within SeamlessM4T's single-pass limit.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != ASR_TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, ASR_TARGET_SR)
    waveform = waveform.squeeze(0)   # 1-D

    total_sec = waveform.shape[0] / ASR_TARGET_SR
    print(f"     audio duration: {total_sec:.1f}s")

    inputs = _asr_processor(
        audio=waveform.numpy(),
        sampling_rate=ASR_TARGET_SR,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        output_tokens = _asr_model.generate(
            **inputs,
            tgt_lang="arb",
            generate_speech=False,
        )
    token_ids = output_tokens[0].tolist()[0]
    return _asr_processor.decode(token_ids, skip_special_tokens=True).strip()

# ── process each row ──────────────────────────────────────────────────────────
for _idx, _row in _df.iterrows():
    print(f"\n[ASR] Row {_idx+1}/{len(_df)}")
    try:
        _src = str(_row[AUDIO_COLUMN]).strip()
        _apath = _download(_src) if _src.startswith("http") else _src
        _ref = _transcribe(_apath)
        _df.at[_idx, "asr_reference"] = _ref
        print(f"     ✓ reference ({len(_ref)} chars): '{_ref[:100]}'")
    except Exception as _e:
        print(f"     ✗ FAILED: {_e}"); traceback.print_exc()

# ── save ──────────────────────────────────────────────────────────────────────
del _asr_model, _asr_processor; torch.cuda.empty_cache()

_df.to_csv(ASR_OUTPUT_CSV, index=False)
print(f"\n[ASR] ✓  Saved {len(_df)} rows → '{ASR_OUTPUT_CSV}'")
print("      Run Cell 3 to compute E5 scores.")



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — STAGE 2: E5 SEMANTIC SCORING                                    ║
# ║                                                                            ║
# ║  Reads  asr_transcriptions.csv  (output of Cell 2).                       ║
# ║  For each row, uses multilingual-E5-large to embed the asr_reference      ║
# ║  and each of the 5 candidate options, then ranks by cosine similarity.    ║
# ║  Saves results to  e5_scores.csv                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────── configure ───────────────────────────────────────────────────
E5_INPUT_CSV   = "asr_transcriptions.csv"  # ← output of Cell 2
OPTION_COLUMNS = ["option_1", "option_2", "option_3", "option_4", "option_5"]
E5_OUTPUT_CSV  = "e5_scores.csv"
E5_MODEL_ID    = "intfloat/multilingual-e5-large"

# Temperature for softmax sharpening.
# E5 cosine similarities cluster tightly (0.85-0.95) because all options are
# Arabic text from the same show.  Without scaling, softmax gives ≈0.2 for all.
# Lower temperature = sharper distribution = clearer winner.
# 0.05 works well; decrease to 0.02 if scores are still too flat.
E5_TEMPERATURE = 0.05
# ─────────────────────────────────────────────────────────────────────────────

import traceback
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[E5] Device: {device}")

# ── load E5 ───────────────────────────────────────────────────────────────────
print(f"[E5] Loading {E5_MODEL_ID} ...")
_e5_tok = AutoTokenizer.from_pretrained(E5_MODEL_ID)
_e5_model = AutoModel.from_pretrained(E5_MODEL_ID).to(device)
_e5_model.eval()
print("[E5] ✓ Model ready\n")

# ── load CSV with ASR references ──────────────────────────────────────────────
_df = pd.read_csv(E5_INPUT_CSV)
print(f"[E5] CSV: {len(_df)} rows")

# prepare score columns
_score_cols = [f"e5_score_{c}" for c in OPTION_COLUMNS]
for _sc in _score_cols:
    _df[_sc] = 0.0


def _average_pool(last_hidden_state, attention_mask):
    """Mean-pool the hidden states, ignoring padding tokens."""
    _mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * _mask).sum(dim=1) / _mask.sum(dim=1)


def _embed(texts, prefix="query"):
    """
    Embed a list of texts using E5.

    E5's intended usage is asymmetric:
      - Reference (what you're searching for) → prefix='query'
      - Candidates (the pool) → prefix='passage'
    This is how it was fine-tuned and gives the best separation.

    Returns: normalised embedding tensor of shape [len(texts), 1024]
    """
    prefixed = [f"{prefix}: " + t for t in texts]
    batch = _e5_tok(
        prefixed,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = _e5_model(**batch)
    emb = _average_pool(out.last_hidden_state, batch["attention_mask"])
    return F.normalize(emb, p=2, dim=1)          # unit vectors → dot = cosine


# ── process each row ──────────────────────────────────────────────────────────
for _idx, _row in _df.iterrows():
    print(f"\n{'='*55}")
    print(f"  ROW {_idx+1}/{len(_df)}")
    print(f"{'='*55}")

    _scores = [0.0] * len(OPTION_COLUMNS)

    # get reference transcription
    _ref = str(_row.get("asr_reference", "")).strip()
    if not _ref:
        print("  ✗ No asr_reference for this row — skipping")
        for _c, _s in zip(_score_cols, _scores):
            _df.at[_idx, _c] = _s
        continue

    print(f"  Reference: '{_ref[:80]}'")

    # truncate candidates to 800 chars (safe for 512-token limit)
    _cands = [str(_row[c]).strip()[:800] for c in OPTION_COLUMNS]

    try:
        # Embed reference as 'query' and candidates as 'passage'
        # This is E5's intended asymmetric retrieval mode → better separation
        _ref_emb  = _embed([_ref], prefix="query")    # [1, 1024]
        _cand_emb = _embed(_cands, prefix="passage")  # [5, 1024]

        # cosine similarity (dot product of unit vectors)
        _sims = (_ref_emb @ _cand_emb.T).squeeze(0)  # [5]

        # Temperature-scaled softmax — amplifies small differences
        # Raw sims are typically 0.85-0.95 for all options (very flat)
        # Dividing by 0.05 spreads them to 17-19 → huge softmax differences
        _scores = torch.softmax(_sims / E5_TEMPERATURE, dim=0).cpu().tolist()

        print(f"  Ref length:  {len(_ref)} chars")
        print(f"  Cosine sims: {[round(x,4) for x in _sims.cpu().tolist()]}")
        print(f"  ÷ temp={E5_TEMPERATURE}:  {[round(x/E5_TEMPERATURE,2) for x in _sims.cpu().tolist()]}")
        print(f"  E5 scores:   {[round(x,4) for x in _scores]}")

    except Exception as _e:
        print(f"  ✗ FAILED: {_e}"); traceback.print_exc()

    # store
    for _c, _s in zip(_score_cols, _scores):
        _df.at[_idx, _c] = _s

    # print table
    print(f"\n  {'Option':<12}  {'Score':>8}   Text preview")
    print(f"  {'-'*55}")
    for _oc, _sc, _sv in zip(OPTION_COLUMNS, _score_cols, _scores):
        print(f"  {_oc:<12}  {_sv:>8.4f}   {str(_row[_oc])[:45]}")

# ── save & cleanup ────────────────────────────────────────────────────────────
del _e5_model, _e5_tok; torch.cuda.empty_cache()

_df.to_csv(E5_OUTPUT_CSV, index=False)
print(f"\n[E5] ✓ Saved to '{E5_OUTPUT_CSV}'")
print(_df[["asr_reference"] + OPTION_COLUMNS + _score_cols].to_string())
