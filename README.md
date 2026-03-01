# Golden Transcription — Hackenza 2026

A multi-signal machine learning pipeline that identifies the **best transcription** of an Arabic audio clip from 5 candidate options, using acoustic, character-level, semantic, and linguistic scoring fused via gradient-boosted ranking.

---

## Problem Statement

Given an  audio clip and 5 transcription candidates (option 1–5), the system must predict which option is the most accurate transcription. 

---

## Architecture

```
Audio + 5 Transcription Options
            │
     ┌──────┴──────────────────────────────────┐
     │                                          │
  ① Whisper ASR             ② Character Scorer │
  (acoustic WER scores)     (Levenshtein/MAD)   │
     │                                          │
  ③ E5 Embeddings           ④ mT5 Linguistic   │
  (semantic cosine sim)      (cross-entropy)    │
     │                                          │
     └──────────┬───────────────────────────────┘
                │
         XGBoost Ranker
         (rank:pairwise, trained on labeled data)
                │
         ┌──────┴──────┐
         │             │
    Structural      Final Scores
    Filter          (softmax over 5 options)
    (kills length        │
     outliers +      Predicted Option
     bracket spam)
```

### Scoring Workflows

| # | Name | Method | Output |
|---|------|--------|--------|
| 1 | **Whisper Acoustic** | Transcribe audio with Whisper-small; compute WER vs each candidate | 5 softmax scores |
| 2 | **Character** | Levenshtein edit distance normalised by MAD, with golden-ref alignment | 5 softmax scores |
| 3 | **E5 Semantic** | `intfloat/multilingual-e5-base` cosine similarity between audio transcript and each option | 5 softmax scores |
| 4 | **mT5 Linguistic** | `google/mt5-base` cross-entropy loss of generating each option | 5 inverted-loss scores |

These 4×5 = 20 features are fed to **XGBoost `rank:pairwise`** which learns the optimal fusion weights from labeled training data.

### Structural Post-Processing Filter
After XGBoost scoring, options that are extreme length outliers (>2σ from median) or contain many parenthetical expressions (screenplay-style text) receive a multiplicative score penalty. This eliminates spurious "screenplay" options that confuse all 4 scorers.

---

## Training Data

| Dataset | Rows | Dialect | Weight | Source |
|---------|------|---------|--------|--------|
| FLEURS `validation_dataset.csv` | 692 | 20 languages | Warm-start base | [Google FLEURS](https://huggingface.co/datasets/google/fleurs) |
| `filtered_dataset.csv` (synthetic) | 470 | ar_eg (Egyptian) | w = 3 | Synthetic corruptions of FLEURS Arabic |
| `inputs/input1–49` (labeled) | 49 | Arabic_SA | w = 5 | Competition-provided ground truth |

Training uses **XGBoost warm-start**: the FLEURS-trained model checkpoint is fine-tuned on in-domain Arabic data, preserving multilingual knowledge while specialising for Arabic SA.

---

## Repository Structure

```
├── kaggle_runner.py          # Main pipeline — train and predict modes
├── kaggle_notebook.ipynb     # Kaggle notebook — end-to-end run
├── filtered_dataset.csv      # Synthetic Arabic training data (470 rows)
├── inputs/                   # Competition input rows (input1–input99)
│   ├── input1/               # audio_id, language, option_1-5, correct_option
│   ├── ...
│   └── correct_options.txt   # Ground truth for inputs 1–49
├── submission.csv            # Final predictions (100 rows)
├── model_final.json          # Trained XGBoost model weights
├── acoustic.py               # Whisper + WER scoring
├── character_scorer.py       # Levenshtein / MAD character scoring
├── linguistic_worklow.py     # mT5 cross-entropy scoring
├── text.py                   # Arabic text normalisation utilities
└── requirements.txt
```

---

## Setup & Installation

### Requirements
- Python 3.9+
- CUDA GPU recommended (Whisper is slow on CPU)
- ~8 GB RAM, ~4 GB VRAM

### Install dependencies

```bash
git clone https://github.com/Vinar01/Team-team
cd Team-team
pip install -r requirements.txt
```

---

## Running on Kaggle (Recommended)

Kaggle provides free GPU T4 instances. This is the recommended way to run the full pipeline.

### Step 1 — Add datasets to your Kaggle notebook
In the right panel of your notebook, click **+ Add Input** and attach:
- Your dataset containing `inputs/`, `transcription_assessment.csv`, `temp_audio/`
- The synthetic audio dataset (`final_synthetic_audiio.zip` uploaded as a Kaggle dataset)

### Step 2 — Configure settings
- **Settings → Accelerator → GPU T4 x2** ✅
- **Settings → Internet → ON** ✅

### Step 3 — Edit Cell 3 paths
After running Cell 2 (which prints all available file paths), update Cell 3:

```python
SYNTH_AUDIO_ROOT = '/kaggle/input/YOUR-SYNTH-DATASET/final_synthetic_audiio'
TEST_CSV         = '/kaggle/input/YOUR-DATASET/transcription_assessment.csv'
TEST_AUDIO       = '/kaggle/input/YOUR-DATASET/temp_audio'
MODEL_R0         = None   # or path to existing fusion_model.json for warm-start
```

### Step 4 — Run All
Click **Run All**. Expected runtime: ~60 minutes (full mode with GPU).

| Cell | Action | Time |
|------|--------|------|
| 1 | Install packages | 2 min |
| 2 | Clone repo | 30 sec |
| 3 | Set paths | instant |
| 4 | Build 519-row Arabic training CSV | 2 min |
| 5 | Score + train XGBoost | ~45 min |
| 6 | Predict on 100 test rows | ~10 min |
| 7 | Show accuracy vs baseline | instant |

---

## Running Locally

For local inference using a pre-trained model:

```bash
# Predict on a CSV of test rows (requires model_final.json)
python kaggle_runner.py \
    --mode predict \
    --test-csv  transcription_assessment.csv \
    --test-audio temp_audio/ \
    --model-path model_final.json \
    --output submission.csv \
    --whisper-model small \
    --structural-filter
```

For training from scratch:

```bash
python kaggle_runner.py \
    --mode train \
    --train-csv  filtered_dataset.csv \
    --train-audio /path/to/synthetic/audio/ \
    --model-path model_final.json \
    --whisper-model small \
    --full
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--full` | Enable all 4 workflows (default: Whisper + char only) |
| `--whisper-model` | Whisper model size: `tiny`, `base`, `small`, `medium` |
| `--structural-filter` | Apply post-processing structural filter |
| `--warmstart-model` | Path to existing model to fine-tune from |
| `--limit N` | Process only first N rows (for testing) |

---

## Key Design Decisions

### Why XGBoost over simple averaging?
Simple averaging treats all 4 scoring signals equally. XGBoost learns that for Arabic audio, the Whisper acoustic signal is most reliable when WER is low, but the character-level signal dominates when Whisper mishears words. This learned fusion outperforms any fixed weighting scheme.

### Why structural filtering?
Several options in the dataset are >50KB screenplay transcripts (complete with stage directions and `(laugh)` notations). These confuse all 4 scorers by having artificially low WER on partial matches. The structural filter explicitly penalises extreme length outliers and bracket-heavy text, eliminating this class of noise.

### Why WER capping?
Whisper WER values for the screenplay options can reach 198–297 (198× longer than the audio). Passing these raw into softmax causes the denominator to explode, compressing the scores of all plausible options to near-zero. Capping at WER=1.0 before softmax restores discrimination between real candidates.

### Why sample weighting?
Training data comes from 3 sources of different quality. The 49 real Arabic_SA labeled rows are from the exact same distribution as the test set and receive weight 5. The 470 synthetic Egyptian Arabic rows are a different dialect and receive weight 3. XGBoost's `rank:pairwise` loss is scaled per-group by these weights.

---

## Results

| Model | Accuracy (49 labeled rows) |
|-------|--------------------------|
| Baseline (Whisper-base, FLEURS only) | 22/49 (44.9%) |
| + Arabic fine-tuning + structural filter | TBD after final run |

---

## Technical Report

A full technical report covering methodology, experiments, and results is included in the submission.

---

## License

This project was built for HAckenza 2026. All model weights and generated data are for competition use only.
