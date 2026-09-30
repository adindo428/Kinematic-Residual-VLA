#!/usr/bin/env python3
"""Derive a bounded training subset from the validated full speed split."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("configs/dom_speed_ood_0p10_0p18_split.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=8)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    train = source["train"]
    if args.train_episodes < 1 or args.train_episodes > len(train):
        parser.error("--train-episodes must be within the full training split")
    selected = sorted(random.Random(1234).sample(train, args.train_episodes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": {
            "source": str(args.source),
            "train_max_mps": 0.10,
            "test_min_mps": 0.18,
            "purpose": "bounded GPU training smoke test",
        },
        "train": selected,
        "test": [],
    }, indent=2) + "\n")
    print(f"Smoke training episodes: {selected}")


if __name__ == "__main__":
    main()
