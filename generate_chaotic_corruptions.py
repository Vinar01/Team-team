import random
import json
import argparse
from datasets import load_dataset

UNRELATED_POOL = [
    "The global economy is shifting rapidly.",
    "Artificial intelligence continues to evolve.",
    "This sentence is completely unrelated.",
    "السياسة العالمية تتغير بسرعة.",
    "Quantum computing may change everything."
]

def random_truncate(text):
    words = text.split()
    if len(words) < 3:
        return text
    cut = random.randint(1, len(words) - 1)
    return " ".join(words[:cut])

def random_expand(text):
    return text + " " + random.choice(UNRELATED_POOL)

def random_shuffle(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def random_scramble(text):
    chars = list(text)
    random.shuffle(chars)
    return "".join(chars)

def duplicate(text):
    return text + " " + text

def chaotic_corrupt(text):
    strategies = [
        random_truncate,
        random_expand,
        random_shuffle,
        random_scramble,
        duplicate,
    ]

    if random.random() < 0.4:
        return random.choice(UNRELATED_POOL)

    func = random.choice(strategies)
    return func(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--corruptions", type=int, default=4)
    parser.add_argument("--output", default="chaotic_dataset.json")

    args = parser.parse_args()

    dataset = load_dataset("google/fleurs", args.language, split="train")
    dataset = dataset.shuffle(seed=42).select(range(args.samples))

    data = []

    for sample in dataset:
        correct = sample["transcription"].strip()

        corruptions = set()
        while len(corruptions) < args.corruptions:
            corrupted = chaotic_corrupt(correct)
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

    print("Chaotic corruption dataset generated.")

if __name__ == "__main__":
    main()