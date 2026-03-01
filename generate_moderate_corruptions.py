import random
import json
import argparse
from datasets import load_dataset

# ============================================
# Moderate corruption functions
# ============================================

def delete_word(words):
    if len(words) <= 1:
        return words
    idx = random.randint(0, len(words) - 1)
    return words[:idx] + words[idx+1:]

def insert_word(words):
    fillers = ["very", "really", "actually", "basically"]
    idx = random.randint(0, len(words))
    return words[:idx] + [random.choice(fillers)] + words[idx:]

def swap_adjacent(words):
    if len(words) < 2:
        return words
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx+1] = words[idx+1], words[idx]
    return words

def char_noise(text):
    if len(text) < 5:
        return text
    idx = random.randint(0, len(text) - 1)
    new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:idx] + new_char + text[idx+1:]

def moderate_corrupt(text):
    words = text.split()
    strategy = random.choice(["delete", "insert", "swap", "char"])

    if strategy == "delete":
        return " ".join(delete_word(words))
    elif strategy == "insert":
        return " ".join(insert_word(words))
    elif strategy == "swap":
        return " ".join(swap_adjacent(words.copy()))
    elif strategy == "char":
        return char_noise(text)

    return text

# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, help="Language code (e.g. en_us, ar_sa)")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--corruptions", type=int, default=4)
    parser.add_argument("--output", default="moderate_dataset.json")

    args = parser.parse_args()

    dataset = load_dataset("google/fleurs", args.language, split="train")
    dataset = dataset.shuffle(seed=42).select(range(args.samples))

    data = []

    for sample in dataset:
        correct = sample["transcription"].strip()

        corruptions = set()
        while len(corruptions) < args.corruptions:
            corrupted = moderate_corrupt(correct)
            if corrupted != correct:
                corruptions.add(corrupted)

        data.append({
            "language": args.language,
            "audio_path": sample.get("path", None),
            "correct": correct,
            "corrupted": list(corruptions)
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Moderate corruption dataset generated.")

if __name__ == "__main__":
    main()