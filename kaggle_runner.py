#!/usr/bin/env python3
"""
kaggle_runner.py  —  Two-phase Golden Transcription Pipeline
=============================================================

PHASE A  (Training on FLEURS data):
    python kaggle_runner.py \\
        --mode train \\
        --train-csv  /kaggle/input/DATASET/validation_dataset.csv \\
        --train-audio /kaggle/input/DATASET/fleurs_audio \\
        --model-path  /kaggle/working/fusion_model.json

PHASE B  (Predicting on Arabic test set):
    python kaggle_runner.py \\
        --mode predict \\
        --test-csv   /kaggle/input/DATASET/transcription_assessment.csv \\
        --test-audio /kaggle/input/DATASET/temp_audio \\
        --model-path /kaggle/working/fusion_model.json \\
        --output     /kaggle/working/submission.csv

Do both in one shot:
    python kaggle_runner.py \\
        --mode both \\
        --train-csv   /kaggle/input/DATASET/validation_dataset.csv \\
        --train-audio /kaggle/input/DATASET/fleurs_audio \\
        --test-csv    /kaggle/input/DATASET/transcription_assessment.csv \\
        --test-audio  /kaggle/input/DATASET/temp_audio \\
        --model-path  /kaggle/working/fusion_model.json \\
        --output      /kaggle/working/submission.csv

Add --full to also run SeamlessM4T + E5 + mT5 (slower, more accurate).
Default fast mode: Whisper + character scoring only.

Notes
-----
- validation_dataset.csv  (FLEURS, 692 rows, multilingual) = TRAINING data
- transcription_assessment.csv / Arabic inputs/            = TEST data  (DO NOT train on this)
- fleurs_audio/en_us/, fleurs_audio/ar_eg/, ...            = FLEURS audio
- temp_audio/1.wav, 2.wav, ...                             = Arabic test audio
"""

from __future__ import annotations
import argparse, logging, math, os, re, sys, unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torchaudio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import whisper;          _WHISPER = True
except ImportError:
    _WHISPER = False;        log.warning("openai-whisper not installed")

try:
    from jiwer import wer as _jiwer_wer; _JIWER = True
except ImportError:
    _JIWER = False;          log.warning("jiwer not installed")

try:
    import xgboost as xgb;  _XGB = True
except ImportError:
    _XGB = False;            log.warning("xgboost not installed — equal-weight fallback")

try:
    from Levenshtein import distance as _lev
except ImportError:
    def _lev(a, b):
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b)+1))
        for ca in a:
            curr = [prev[0]+1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[-1]+1, prev[j]+(ca!=cb)))
            prev = curr
        return prev[-1]

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

# FLEURS BCP47 → Whisper language name
_FLEURS_TO_WHISPER = {
    "en_us": "english",    "ar_eg": "arabic",     "cmn_hans_cn": "chinese",
    "de_de": "german",     "es_419": "spanish",   "fr_fr": "french",
    "hi_in": "hindi",      "id_id": "indonesian", "it_it": "italian",
    "ja_jp": "japanese",   "ko_kr": "korean",     "nl_nl": "dutch",
    "pl_pl": "polish",     "pt_br": "portuguese", "ru_ru": "russian",
    "sv_se": "swedish",    "th_th": "thai",       "tr_tr": "turkish",
    "uk_ua": "ukrainian",  "vi_vn": "vietnamese",
    # also accept the Arabic_SA format from the test set
    "arabic_sa": "arabic", "arabic": "arabic",
    "english": "english",  "french": "french",    "german": "german",
    "spanish": "spanish",  "hindi": "hindi",      "chinese": "chinese",
}

# FLEURS BCP47 → SeamlessM4T tgt_lang
_FLEURS_TO_SEAMLESS = {
    "en_us": "eng",  "ar_eg": "arb",  "cmn_hans_cn": "cmn",
    "de_de": "deu",  "es_419": "spa", "fr_fr": "fra",
    "hi_in": "hin",  "id_id": "ind",  "it_it": "ita",
    "ja_jp": "jpn",  "ko_kr": "kor",  "nl_nl": "nld",
    "pl_pl": "pol",  "pt_br": "por",  "ru_ru": "rus",
    "sv_se": "swe",  "th_th": "tha",  "tr_tr": "tur",
    "uk_ua": "ukr",  "vi_vn": "vie",
    "arabic_sa": "arb", "arabic": "arb", "english": "eng",
}

# FLEURS BCP47 → ISO 639-1
_FLEURS_TO_ISO = {
    "en_us": "en", "ar_eg": "ar", "cmn_hans_cn": "zh",
    "de_de": "de", "es_419": "es", "fr_fr": "fr",
    "hi_in": "hi", "id_id": "id", "it_it": "it",
    "ja_jp": "ja", "ko_kr": "ko", "nl_nl": "nl",
    "pl_pl": "pl", "pt_br": "pt", "ru_ru": "ru",
    "sv_se": "sv", "th_th": "th", "tr_tr": "tr",
    "uk_ua": "uk", "vi_vn": "vi",
    "arabic_sa": "ar", "arabic": "ar", "english": "en",
}

_CASELESS_ISO = {"ar", "zh", "ja", "ko", "hi", "th", "he", "ka", "uk"}
OPT_COLS      = ["option_1", "option_2", "option_3", "option_4", "option_5"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _whisper_lang(lang: str) -> str:
    k = lang.strip().lower()
    if k in _FLEURS_TO_WHISPER: return _FLEURS_TO_WHISPER[k]
    base = k.split("_")[0]
    _fallback = {"en":"english","ar":"arabic","zh":"chinese","de":"german","es":"spanish",
                 "fr":"french","hi":"hindi","id":"indonesian","it":"italian","ja":"japanese",
                 "ko":"korean","nl":"dutch","pl":"polish","pt":"portuguese","ru":"russian",
                 "sv":"swedish","th":"thai","tr":"turkish","uk":"ukrainian","vi":"vietnamese"}
    return _fallback.get(base, base)

def _seamless_lang(lang: str) -> str:
    k = lang.strip().lower()
    if k in _FLEURS_TO_SEAMLESS: return _FLEURS_TO_SEAMLESS[k]
    base = k.split("_")[0]
    _fallback = {"en":"eng","ar":"arb","zh":"cmn","de":"deu","es":"spa","fr":"fra",
                 "hi":"hin","id":"ind","it":"ita","ja":"jpn","ko":"kor","nl":"nld",
                 "pl":"pol","pt":"por","ru":"rus","sv":"swe","th":"tha","tr":"tur",
                 "uk":"ukr","vi":"vie"}
    return _fallback.get(base, "arb")

def _iso_lang(lang: str) -> str:
    k = lang.strip().lower()
    if k in _FLEURS_TO_ISO: return _FLEURS_TO_ISO[k]
    return k.split("_")[0]

# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str, lang: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKC", text)
    iso  = _iso_lang(lang)
    if iso not in _CASELESS_ISO:
        text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def _softmax_wer(wers: List[float]) -> List[float]:
    finite  = [w for w in wers if math.isfinite(w)]
    # Cap WER at 1.0 — anything ≥ 1.0 is already "completely wrong";
    # avoids extreme values (198, 297) from screenplay options distorting softmax
    clamped = [min(w, 1.0) if math.isfinite(w) else 1.0 for w in wers]
    neg     = [-w for w in clamped]
    shift   = max(neg)
    exps    = [math.exp(v - shift) for v in neg]
    total   = sum(exps)
    return [round(e / total, 6) for e in exps]

def _wer(ref: str, hyp: str) -> float:
    if not _JIWER or not ref.strip(): return 0.0
    try:   return round(_jiwer_wer(ref, hyp), 4)
    except: return 1.0

def _softmax_arr(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.max()
    e   = np.exp(arr)
    return e / e.sum()

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_windows_path(audio_col: str):
    """
    Extract (lang_subfolder, filename) from a Windows-style audio path.
    'C:\\...\\fleurs_audio\\en_us\\141.wav' → ('en_us', '141.wav')
    """
    try:
        parts = audio_col.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if "fleur" in p.lower():
                if i + 2 < len(parts): return parts[i+1], parts[i+2]
                if i + 1 < len(parts): return "",          parts[i+1]
    except Exception:
        pass
    fname = audio_col.replace("\\", "/").split("/")[-1]
    return "", fname


def _find_audio(audio_id: str, audio_col: str,
                audio_dir: Path, language: str = "") -> Optional[str]:
    """
    Resolve audio path.  Tries (in order):
      1. {audio_dir}/{lang_from_windows_path}/{filename}
      2. {audio_dir}/{language_column}/{audio_id}.wav
      3. {audio_dir}/{audio_id}.wav   (flat, for temp_audio style)
      4. Download from URL
    """
    lang_sub, fname = _parse_windows_path(audio_col)
    candidates = []

    if audio_dir.exists():
        if lang_sub and fname:
            candidates.append(audio_dir / lang_sub / fname)
        if language:
            for ext in (".wav", ".mp3", ".flac"):
                candidates.append(audio_dir / language / f"{audio_id}{ext}")
        for ext in (".wav", ".mp3", ".flac"):
            candidates.append(audio_dir / f"{audio_id}{ext}")
        for p in candidates:
            if p.exists():
                return str(p)

    if audio_col.startswith("http"):
        # Downloads must go to a writable dir — audio_dir may be read-only in Kaggle
        # (/kaggle/input/ is always read-only; /kaggle/working/ is writable)
        _kaggle_cache = Path("/kaggle/working/audio_cache")
        dl_dir = _kaggle_cache if Path("/kaggle/working").exists() else audio_dir
        dl_dir.mkdir(parents=True, exist_ok=True)
        ext  = Path(audio_col.split("?")[0]).suffix or ".wav"
        dest = dl_dir / f"{audio_id}{ext}"
        if dest.exists(): return str(dest)
        try:
            log.info("  Downloading audio %s ...", audio_id)
            r = requests.get(audio_col, timeout=90)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return str(dest)
        except Exception as e:
            log.warning("  Download failed: %s", e)

    return None


def _load_16k(path: str) -> Optional[np.ndarray]:
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0).numpy().astype(np.float32)
    except Exception as e:
        log.warning("Audio load failed: %s", e)
        return None

# ══════════════════════════════════════════════════════════════════════════════
# SCORING STAGES
# ══════════════════════════════════════════════════════════════════════════════

def run_whisper_stage(df: pd.DataFrame, audio_dir: Path,
                      model_name: str = "base",
                      _model=None) -> tuple[np.ndarray, object]:
    """
    Returns (acoustic_scores [N,5], whisper_model).
    Pass _model to reuse an already-loaded model (avoids double download).
    """
    if not _WHISPER:
        return np.full((len(df), 5), 0.2, dtype=np.float32), None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _model is None:
        log.info("Loading Whisper '%s' on %s ...", model_name, device)
        _model = whisper.load_model(model_name, device=device)

    scores = np.full((len(df), 5), 0.2, dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        audio_id  = str(row.get("audio_id", i + 1))
        lang      = str(row.get("language", "en_us"))
        audio_col = str(row.get("audio", ""))
        candidates = [str(row.get(c, "")).strip() for c in OPT_COLS]

        log.info("[%d/%d] Whisper — audio_id=%s  lang=%s", i+1, len(df), audio_id, lang)
        apath = _find_audio(audio_id, audio_col, audio_dir, lang)

        if not apath:
            log.warning("  No audio found for %s — using neutral scores", audio_id)
            continue

        try:
            result = _model.transcribe(
                apath, language=_whisper_lang(lang), task="transcribe",
                fp16=(device == "cuda"),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            hypothesis = result["text"].strip()
            iso        = _iso_lang(lang)
            norm_hyp   = _normalize(hypothesis, iso)

            wers = []
            for cand in candidates:
                wers.append(float("inf") if not cand.strip()
                            else _wer(norm_hyp, _normalize(cand, iso)))

            sm = _softmax_wer(wers)
            scores[i] = sm
            log.info("  hyp: '%s'  scores: %s", hypothesis[:70], [round(x,3) for x in sm])

        except Exception as e:
            log.error("  Whisper error: %s", e)

    return scores, _model


def apply_structural_filter(df: pd.DataFrame, final_scores: np.ndarray,
                            k_len: float = 1.5, k_bracket: float = 0.05) -> np.ndarray:
    """
    Post-processing filter applied AFTER XGBoost prediction.
    Multiplies final scores by structural scores to zero-out screenplay/outlier options.
    Two signals:
      len_score     = exp(-k_len * log(len(opt) / median_len)^2)
                      → penalises options that are 10x+ longer than siblings
      bracket_score = exp(-k_bracket * count('(', opt))
                      → penalises screenplay stage directions like (يضحك)
    Combined: 0.4*len + 0.6*bracket  (bracket weighted higher — stronger signal)
    """
    out = final_scores.copy()
    for i, (_, row) in enumerate(df.iterrows()):
        opts = [str(row.get(c, "")).strip() for c in OPT_COLS]
        lens = np.array([max(len(o), 1) for o in opts], dtype=float)
        median_len = float(np.median(lens)) + 1.0
        for j, opt in enumerate(opts):
            log_ratio  = abs(math.log(max(len(opt), 1) / median_len))
            len_sc     = math.exp(-k_len * log_ratio ** 2)
            bracket_sc = math.exp(-k_bracket * opt.count('('))
            out[i, j] *= (0.4 * len_sc + 0.6 * bracket_sc)
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out


def run_char_stage(df: pd.DataFrame) -> np.ndarray:
    scores = np.zeros((len(df), 5), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        opts = [str(row.get(c, "")).strip() for c in OPT_COLS]
        ref  = str(row.get("asr_reference", "")).strip()
        if ref:
            row_s = []
            for opt in opts:
                if not opt: row_s.append(0.0)
                else:
                    d = max(len(opt), len(ref))
                    row_s.append(max(0.0, 1.0 - _lev(opt, ref) / d) if d else 1.0)
        else:
            lens  = np.array([len(o) for o in opts], dtype=float)
            med   = np.median(lens)
            mad   = np.median(np.abs(lens - med)) + 1e-5
            row_s = [float(np.exp(-((abs(l - med) / mad) ** 2))) for l in lens]
        scores[i] = row_s
    return scores


def run_seamless_stage(df: pd.DataFrame, audio_dir: Path) -> pd.Series:
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc   = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model  = SeamlessM4Tv2Model.from_pretrained(
        "facebook/seamless-m4t-v2-large", torch_dtype=torch.float16).to(device)
    model.eval()
    refs = []
    for i, (_, row) in enumerate(df.iterrows()):
        audio_id  = str(row.get("audio_id", i+1))
        lang      = str(row.get("language", "en_us"))
        audio_col = str(row.get("audio", ""))
        tgt       = _seamless_lang(lang)
        log.info("[%d/%d] Seamless — %s  tgt=%s", i+1, len(df), audio_id, tgt)
        apath = _find_audio(audio_id, audio_col, audio_dir, lang)
        ref = ""
        if apath:
            try:
                wav = _load_16k(apath)
                if wav is not None:
                    inp = proc(audio=wav, sampling_rate=16000, return_tensors="pt")
                    inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inp.items()}
                    with torch.no_grad():
                        toks = model.generate(**inp, tgt_lang=tgt, generate_speech=False)
                    ref = proc.decode(toks[0].tolist()[0], skip_special_tokens=True).strip()
            except Exception as e:
                log.error("  Seamless failed: %s", e)
        refs.append(ref)
    del model, proc; torch.cuda.empty_cache()
    s = pd.Series(refs, index=df.index, name="asr_reference")
    df.copy().assign(asr_reference=s).to_csv("asr_transcriptions.csv", index=False)
    return s


def run_e5_stage(df: pd.DataFrame, asr_refs: pd.Series,
                 temperature: float = 0.05) -> np.ndarray:
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok    = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    model  = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(device)
    model.eval()
    def _embed(texts, prefix):
        b = tok([f"{prefix}: " + t for t in texts], padding=True, truncation=True,
                max_length=512, return_tensors="pt")
        b = {k: v.to(device) for k,v in b.items()}
        with torch.no_grad(): out = model(**b)
        mask = b["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return F.normalize(emb, p=2, dim=1)
    scores = np.full((len(df), 5), 0.2, dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        ref = str(asr_refs.iloc[i]).strip()
        if not ref: continue
        try:
            cands    = [str(row.get(c,"")).strip()[:800] for c in OPT_COLS]
            sims     = (_embed([ref],"query") @ _embed(cands,"passage").T).squeeze(0)
            scores[i] = torch.softmax(sims / temperature, dim=0).cpu().tolist()
        except Exception as e:
            log.error("  E5 row %d: %s", i+1, e)
    del model, tok; torch.cuda.empty_cache()
    return scores


def run_mt5_stage(df: pd.DataFrame, asr_refs: pd.Series, k: float = 1.0) -> np.ndarray:
    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok    = AutoTokenizer.from_pretrained("google/mt5-small")
    model  = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)
    model.eval()
    def _loss(text):
        if not text or not text.strip(): return 999.0
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad(): out = model(input_ids=inp.input_ids, labels=inp.input_ids)
        return out.loss.item()
    scores = np.full((len(df), 5), 0.2, dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        ref = str(asr_refs.iloc[i]).strip()
        if not ref: continue
        try:
            lr = _loss(ref)
            scores[i] = [math.exp(-k * abs(_loss(str(row.get(c,"")).strip()) - lr))
                         for c in OPT_COLS]
        except Exception as e:
            log.error("  mT5 row %d: %s", i+1, e)
    del model, tok; torch.cuda.empty_cache()
    return scores

# ══════════════════════════════════════════════════════════════════════════════
# SCORE ALL ROWS  (shared helper used for both train and test)
# ══════════════════════════════════════════════════════════════════════════════

def score_dataset(df: pd.DataFrame, audio_dir: Path,
                  whisper_model: str, run_full: bool,
                  whisper_instance=None) -> tuple[np.ndarray, object]:
    """
    Returns (score_matrix [N, 5, 4], whisper_instance).
    score_matrix[:,i,:] = [acoustic, char, e5, linguistic] for option i.
    """
    N = len(df)

    log.info("  Whisper acoustic scoring ...")
    acoustic, whisper_instance = run_whisper_stage(df, audio_dir, whisper_model,
                                                   _model=whisper_instance)
    log.info("  Character scoring ...")
    char = run_char_stage(df)

    e5   = np.full((N, 5), 0.2, dtype=np.float32)
    ling = np.full((N, 5), 0.2, dtype=np.float32)
    asr_refs = pd.Series([""] * N)

    if run_full:
        log.info("  SeamlessM4T ASR ...")
        try:
            asr_refs = run_seamless_stage(df, audio_dir)
            log.info("  E5 semantic scoring ...")
            e5       = run_e5_stage(df, asr_refs)
        except Exception as e:
            log.error("  Seamless/E5 failed: %s", e)
        log.info("  mT5 linguistic scoring ...")
        try:
            ling = run_mt5_stage(df, asr_refs)
        except Exception as e:
            log.error("  mT5 failed: %s", e)

    matrix = np.stack([acoustic, char, e5, ling], axis=2)   # [N, 5, 4]
    return matrix, whisper_instance

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(matrix: np.ndarray, correct_options: List[int],
                model_path: str,
                sample_weights: Optional[List[float]] = None,
                warmstart_path: Optional[str] = None) -> "xgb.XGBRanker":
    """
    Train XGBRanker on the score matrix and save to model_path.

    Args:
        sample_weights: per-row weights (len == N). Higher = more influence.
                        e.g. synthetic=3, labeled_arabic=5, test_rows=10
        warmstart_path: path to existing XGBoost model to fine-tune from.
                        Preserves prior knowledge while adapting to new data.
    """
    if not _XGB:
        raise RuntimeError("xgboost not installed. pip install xgboost")

    N       = matrix.shape[0]
    X_flat  = matrix.reshape(N * 5, 4).astype(np.float32)
    y_flat  = np.zeros(N * 5, dtype=np.int32)
    groups  = []

    for i, co in enumerate(correct_options):
        if co is not None:
            try: y_flat[i * 5 + (int(co) - 1)] = 1
            except Exception: pass
        groups.append(5)

    # Expand row-level weights to per-option weights
    if sample_weights is not None:
        w_flat = np.repeat(np.array(sample_weights, dtype=np.float32), 5)
    else:
        w_flat = None

    split = max(1, int(N * 0.8))
    is_warmstart = warmstart_path and os.path.exists(warmstart_path)
    log.info("XGBoost %straining: %d rows (80%% train / 20%% val)%s...",
             "warm-start " if is_warmstart else "",
             N,
             f" — loading base model from {warmstart_path}" if is_warmstart else "")

    # Use lower LR and fewer trees when fine-tuning (warm-start)
    n_est = 200 if is_warmstart else 300
    lr    = 0.03 if is_warmstart else 0.05

    ranker = xgb.XGBRanker(
        objective="rank:pairwise", n_estimators=n_est,
        max_depth=4, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="ndcg", random_state=42, verbosity=0,
    )
    ranker.fit(
        X_flat[:split*5], y_flat[:split*5],
        group=groups[:split],
        sample_weight=w_flat[::5][:split] if w_flat is not None else None,
        xgb_model=warmstart_path if is_warmstart else None,
    )

    # Validation accuracy on held-out 20%
    raw_val   = ranker.predict(X_flat[split*5:]).reshape(N - split, 5)
    preds_val = np.argmax(raw_val, axis=1) + 1
    co_val    = [correct_options[i] for i in range(split, N)]
    acc       = sum(p == a for p, a in zip(preds_val, co_val) if a) / max(1, len(co_val))
    log.info("Validation accuracy: %.1f%%  (%d/%d correct)", acc*100,
             int(acc*len(co_val)), len(co_val))

    ranker.save_model(model_path)
    log.info("Model saved → %s", model_path)
    return ranker

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict(matrix: np.ndarray,
            ranker: Optional["xgb.XGBRanker"] = None,
            model_path: Optional[str] = None) -> np.ndarray:
    """Return final_scores [N, 5]."""
    N      = matrix.shape[0]
    X_flat = matrix.reshape(N * 5, 4).astype(np.float32)

    if ranker is None and model_path and os.path.exists(model_path) and _XGB:
        log.info("Loading XGBoost model: %s", model_path)
        ranker = xgb.XGBRanker()
        ranker.load_model(model_path)

    if ranker is not None:
        raw   = ranker.predict(X_flat).reshape(N, 5)
        return np.array([_softmax_arr(raw[i]) for i in range(N)], dtype=np.float32)

    log.info("No model — equal-weight average fusion.")
    avg = matrix.mean(axis=2)   # [N, 5]
    return np.array([_softmax_arr(avg[i]) for i in range(N)], dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_output(df: pd.DataFrame, final_scores: np.ndarray,
                 correct_options: Optional[List[int]]) -> pd.DataFrame:
    rows = []
    total, correct = 0, 0
    for i, (_, row) in enumerate(df.iterrows()):
        lang       = str(row.get("language", "en_us"))
        iso        = _iso_lang(lang)
        candidates = [str(row.get(c, "")).strip() for c in OPT_COLS]
        scores     = final_scores[i]
        gidx       = int(np.argmax(scores))
        golden_ref = candidates[gidx]
        norm_gold  = _normalize(golden_ref, iso)
        wers       = [_wer(norm_gold, _normalize(c, iso)) for c in candidates]
        co = (correct_options[i] if correct_options else None)
        if co is not None:
            total   += 1
            correct += int(gidx + 1 == int(co))
        rows.append({
            "audio_id": str(row.get("audio_id", i+1)), "language": lang,
            "audio":    str(row.get("audio", "")),
            "option_1": candidates[0], "option_2": candidates[1],
            "option_3": candidates[2], "option_4": candidates[3],
            "option_5": candidates[4],
            "golden_ref": golden_ref, "predicted_option_num": gidx + 1,
            "correct_option": co,
            "is_correct": (gidx+1 == int(co)) if co is not None else None,
            "score_1": round(float(scores[0]),4), "score_2": round(float(scores[1]),4),
            "score_3": round(float(scores[2]),4), "score_4": round(float(scores[3]),4),
            "score_5": round(float(scores[4]),4),
            "wer_option1": wers[0], "wer_option2": wers[1], "wer_option3": wers[2],
            "wer_option4": wers[3], "wer_option5": wers[4],
        })
    if total > 0:
        log.info("=" * 60)
        log.info("ACCURACY: %d / %d  (%.1f%%)", correct, total, correct/total*100)
        log.info("=" * 60)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Golden Transcription Pipeline")
    p.add_argument("--mode", choices=["train","predict","both"], default="both",
                   help="train=FLEURS only, predict=test only, both=train then predict")

    # Training data (FLEURS)
    p.add_argument("--train-csv",   help="FLEURS validation_dataset.csv (has correct_option)")
    p.add_argument("--train-audio", help="fleurs_audio/ root (subfolders per language)")

    # Test data (Arabic / target set)
    p.add_argument("--test-csv",    help="Test CSV to predict on (no correct_option needed)")
    p.add_argument("--test-audio",  help="Audio folder for test set (e.g., temp_audio/)")

    p.add_argument("--output",        default="/kaggle/working/submission.csv")
    p.add_argument("--whisper-model", default="base",
                   choices=["tiny","base","small","medium","large"])
    p.add_argument("--full",          action="store_true",
                   help="Also run SeamlessM4T + E5 + mT5 scoring")
    p.add_argument("--model-path",    default="/kaggle/working/fusion_model.json")
    p.add_argument("--warmstart-model", default=None,
                   help="Path to existing XGBoost model to fine-tune from (preserves prior knowledge)")
    p.add_argument("--structural-filter", action="store_true",
                   help="Post-processing: multiply scores by structural scores to kill screenplay/outlier options")
    p.add_argument("--limit",         type=int, default=None,
                   help="Limit rows per dataset (for quick testing)")
    args = p.parse_args()

    ranker = None
    whisper_instance = None   # reuse across train + predict to avoid double-loading

    # ── PHASE A: Train on FLEURS ──────────────────────────────────────────
    if args.mode in ("train", "both"):
        if not args.train_csv:
            p.error("--train-csv required for mode=train/both")
        if not args.train_audio:
            p.error("--train-audio required for mode=train/both")

        log.info("\n" + "=" * 60)
        log.info("PHASE A — TRAINING  (FLEURS dataset)")
        log.info("=" * 60)

        train_df = pd.read_csv(args.train_csv)
        if args.limit:
            train_df = train_df.head(args.limit).copy()
        log.info("Training rows: %d | Languages: %s",
                 len(train_df), sorted(train_df["language"].unique().tolist()))

        # Read labels (prefer true_golden_option_index, fallback correct_option)
        label_col = ("true_golden_option_index" if "true_golden_option_index" in train_df.columns
                     else "correct_option" if "correct_option" in train_df.columns else None)
        if label_col is None:
            p.error(f"Training CSV must have 'correct_option' or 'true_golden_option_index'. "
                    f"Columns found: {list(train_df.columns)}")

        correct_options = []
        for v in train_df[label_col]:
            try:   correct_options.append(int(float(str(v).strip())))
            except: correct_options.append(None)

        train_matrix, whisper_instance = score_dataset(
            train_df, Path(args.train_audio),
            args.whisper_model, args.full)

        if _XGB:
            # Read per-row sample weights if column present
            sample_weights = None
            if 'sample_weight' in train_df.columns:
                sample_weights = train_df['sample_weight'].fillna(1.0).tolist()
                log.info("Sample weights: min=%.1f  max=%.1f  mean=%.1f",
                         min(sample_weights), max(sample_weights),
                         sum(sample_weights)/len(sample_weights))
            ranker = train_model(train_matrix, correct_options, args.model_path,
                                 sample_weights=sample_weights,
                                 warmstart_path=args.warmstart_model)
        else:
            log.warning("xgboost not available — model not trained, will use equal weights.")

    # ── PHASE B: Predict on test set ──────────────────────────────────────
    if args.mode in ("predict", "both"):
        if not args.test_csv:
            p.error("--test-csv required for mode=predict/both")
        if not args.test_audio:
            p.error("--test-audio required for mode=predict/both")

        log.info("\n" + "=" * 60)
        log.info("PHASE B — PREDICTION  (test dataset)")
        log.info("=" * 60)

        test_df = pd.read_csv(args.test_csv)
        if args.limit:
            test_df = test_df.head(args.limit).copy()
        log.info("Test rows: %d", len(test_df))

        # Read labels if present (for accuracy measurement only — NOT for training)
        test_labels = None
        for col in ("correct_option", "true_golden_option_index"):
            if col in test_df.columns:
                test_labels = []
                for v in test_df[col]:
                    try:   test_labels.append(int(float(str(v).strip())))
                    except: test_labels.append(None)
                log.info("Test labels found ('%s') — will report accuracy.", col)
                break

        test_matrix, _ = score_dataset(
            test_df, Path(args.test_audio),
            args.whisper_model, args.full,
            whisper_instance=whisper_instance)

        final_scores = predict(test_matrix, ranker=ranker, model_path=args.model_path)

        if args.structural_filter:
            log.info("Applying structural filter (kills screenplay/length-outlier options)...")
            final_scores = apply_structural_filter(test_df, final_scores)

        out_df       = build_output(test_df, final_scores, test_labels)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info("Submission saved → %s  (%d rows)", out_path, len(out_df))

        print("\n" + "=" * 60)
        print("RESULTS PREVIEW")
        print("=" * 60)
        cols = ["audio_id","language","predicted_option_num","correct_option","is_correct",
                "score_1","score_2","score_3","score_4","score_5"]
        print(out_df[[c for c in cols if c in out_df.columns]].head(15).to_string(index=False))

        if test_labels:
            ok  = int(out_df["is_correct"].sum())
            tot = int(out_df["is_correct"].notna().sum())
            print(f"\n{'='*60}")
            print(f"  ACCURACY: {ok}/{tot}  ({100*ok/tot:.1f}%)")
            print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
