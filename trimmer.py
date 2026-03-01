import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset

HF_CACHE_BASE = Path.home() / ".cache" / "huggingface" / "datasets"

# ============================================
# CONFIGURATION
# ============================================

SECONDS_PER_LANGUAGE = 400
OUTPUT_CSV  = "fleurs_trimmed.csv"
AUDIO_DIR   = Path("fleurs_audio")

LANGUAGES = [
    "en_us", "es_419", "fr_fr", "de_de", "it_it",
    "pt_br", "ru_ru", "cmn_hans_cn", "ja_jp", "ko_kr",
    "hi_in", "ar_eg", "nl_nl", "pl_pl", "sv_se",
    "tr_tr", "vi_vn", "id_id", "uk_ua", "th_th"
]

# ============================================
# DATA COLLECTION
# ============================================

# Resume: load any already-completed rows
rows = []
done_langs = set()
if Path(OUTPUT_CSV).exists():
    existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    rows = existing.to_dict("records")
    done_langs = set(existing["language"].unique())
    print(f"Resuming — already done: {sorted(done_langs)}")

grand_total = sum(r["duration"] for r in rows)

for lang in LANGUAGES:
    if lang in done_langs:
        print(f"Skipping {lang} (already done)")
        continue

    print(f"\nLoading {lang}...")
    # streaming=True: data is fetched on-the-fly — zero disk cache usage
    dataset = load_dataset("google/fleurs", lang, split="train",
                           streaming=True, trust_remote_code=True)

    lang_audio_dir = AUDIO_DIR / lang
    lang_audio_dir.mkdir(parents=True, exist_ok=True)

    lang_total = 0
    sample_count = 0

    for sample in dataset:
        audio       = sample["audio"]
        audio_arr   = np.array(audio["array"], dtype=np.float32)
        sample_rate = audio["sampling_rate"]
        duration    = len(audio_arr) / sample_rate

        if lang_total + duration > SECONDS_PER_LANGUAGE:
            break

        # Save WAV
        audio_id  = sample.get("id", sample_count)
        wav_path  = lang_audio_dir / f"{audio_id}.wav"
        sf.write(str(wav_path), audio_arr, sample_rate)

        rows.append({
            "audio_id":      audio_id,
            "language":      lang,
            "audio":         str(wav_path.resolve()),
            "transcription": sample["transcription"],
            "duration":      round(duration, 3),
        })

        lang_total    += duration
        grand_total   += duration
        sample_count  += 1

    print(f"{lang}: {lang_total:.2f} sec (~{lang_total/60:.2f} min), {sample_count} clips")

    # Save incrementally after each language
    pd.DataFrame(rows, columns=["audio_id", "language", "audio", "transcription", "duration"]).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig"
    )

    # Clean up any residual HF cache for this language (streaming leaves very little)
    del dataset
    for cache_subdir in HF_CACHE_BASE.glob(f"*fleurs*{lang}*"):
        shutil.rmtree(cache_subdir, ignore_errors=True)
    for cache_subdir in HF_CACHE_BASE.glob(f"downloads/extracted/*{lang}*"):
        shutil.rmtree(cache_subdir, ignore_errors=True)

print(f"\nTotal duration: {grand_total:.2f} sec (~{grand_total/3600:.2f} hours)")

# ============================================
# SAVE CSV
# ============================================

df = pd.DataFrame(rows, columns=["audio_id", "language", "audio", "transcription", "duration"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")