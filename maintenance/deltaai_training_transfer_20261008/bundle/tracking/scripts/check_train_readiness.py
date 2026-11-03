#!/usr/bin/env python3
"""Check local DOM training assets without downloading or allocating a GPU."""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dynamicvla_dom_full_finetune_speed_ood_0p10_0p18.yaml",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    split_path = Path(config["DATASET"]["EPISODE_SPLIT_FILE"])
    derived = json.loads(split_path.read_text())
    source = json.loads((split_path.parent / derived["source_file"]).read_text())
    train = source["train"]
    test = source["test"]
    if (
        len(train) != 3810
        or len(test) != 119778
        or len(set(train)) != len(train)
        or len(set(test)) != len(test)
        or not set(train).isdisjoint(test)
        or source["metadata"]["train_max_mps"] != 0.10
        or source["metadata"]["test_min_mps"] != 0.18
    ):
        raise SystemExit("Unexpected or invalid 0.10/0.18 split.")

    missing = []
    required_videos = [
        feature for feature in config["DATASET"]["REQUIRED_FEATURES"]
        if feature.startswith("observation.images.")
    ]
    for episode in train:
        stem = f"episode_{episode:06d}"
        chunk = f"chunk-{episode // 1000:03d}"
        paths = [args.cache / "data" / chunk / f"{stem}.parquet"]
        paths.extend(
            args.cache / "videos" / chunk / camera / f"{stem}.mp4"
            for camera in required_videos
        )
        missing.extend(str(path) for path in paths if not path.is_file())
    if missing:
        print(f"Missing {len(missing)} training assets; first examples:")
        print("\n".join(missing[:10]))
        raise SystemExit(2)
    print(f"Ready: {len(train)} train episodes, {len(test)} held-out test episodes; all training files cached.")


if __name__ == "__main__":
    main()
