#!/usr/bin/env python
"""Prefetch DynamicVLA evaluation splits without constructing a policy.

This is meant for CPU-only Slurm jobs that warm the LeRobot/DOM cache before a
GPU evaluation job. In the speed-OOD configs, `iid` maps to the configured train
split and `ood` maps to the configured test split.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_HOME = Path(__file__).resolve().parents[1]
if str(PROJECT_HOME) not in sys.path:
    sys.path.insert(0, str(PROJECT_HOME))


DEFAULT_CFG = "configs/residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13.yaml"
SPLIT_ALIASES = {
    "iid": ["train"],
    "in_distribution": ["train"],
    "id": ["train"],
    "train": ["train"],
    "ood": ["test"],
    "out_of_distribution": ["test"],
    "test": ["test"],
    "both": ["train", "test"],
    "all": ["train", "test"],
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not load to a mapping.")
    return data


def _resolve_path(path: str) -> Path:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    return cfg_path


def _expand_splits(raw_splits: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for raw in raw_splits:
        key = raw.strip().lower()
        if key not in SPLIT_ALIASES:
            choices = ", ".join(sorted(SPLIT_ALIASES))
            raise ValueError(f"Unknown split alias {raw!r}; expected one of: {choices}.")
        expanded.extend(SPLIT_ALIASES[key])

    deduped: list[str] = []
    for split in expanded:
        if split not in deduped:
            deduped.append(split)
    return deduped


def _episode_summary(episodes: list[int] | None) -> str:
    if episodes is None:
        return "metadata only"
    if not episodes:
        return "0 episodes"
    unique = sorted(set(int(ep_idx) for ep_idx in episodes))
    return f"{len(unique)} episodes, range {unique[0]}..{unique[-1]}"


def prefetch_config(cfg_path: Path, splits: list[str], dataset_override: str | None, dry_run: bool) -> None:
    import utils.datasets

    cfg = _load_yaml(cfg_path)
    dataset_cfg = cfg.get("DATASET")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Config {cfg_path} does not contain a DATASET mapping.")
    if dataset_override:
        dataset_cfg["NAME"] = dataset_override

    dataset_name = dataset_cfg.get("NAME")
    if not dataset_name:
        raise ValueError(f"Config {cfg_path} does not set DATASET.NAME.")
    required_features = dataset_cfg.get("REQUIRED_FEATURES")

    logging.info("Using config: %s", cfg_path)
    logging.info("Dataset: %s", dataset_name)
    logging.info("Splits: %s", ", ".join(splits))

    for split in splits:
        episodes = utils.datasets.resolve_episode_selection(dataset_cfg, split)
        logging.info("Resolved %s split: %s.", split, _episode_summary(episodes))
        if dry_run:
            continue
        utils.datasets.ensure_lerobot_episodes_cached(
            dataset_name,
            episodes,
            required_features,
        )
        logging.info("Prefetch complete for %s split from %s.", split, cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        action="append",
        default=None,
        help="Config to resolve. Repeat to warm multiple compatible configs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["ood"],
        help="Split aliases to prefetch: iid/train, ood/test, both/all. Default: ood.",
    )
    parser.add_argument("--dataset", default=None, help="Optional DATASET.NAME override.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve splits without downloading files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    cfg_paths = [_resolve_path(path) for path in (args.cfg or [DEFAULT_CFG])]
    splits = _expand_splits(args.splits)

    for cfg_path in cfg_paths:
        prefetch_config(cfg_path, splits, args.dataset, args.dry_run)


if __name__ == "__main__":
    main()
