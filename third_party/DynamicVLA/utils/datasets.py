# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2025-06-17 16:10:33
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-21 15:06:36
# @Email:  root@haozhexie.com

import collections
import contextlib
import json
import logging
import os
import pathlib
import random
import threading
import time
import typing

import lerobot.constants
import lerobot.datasets.compute_stats
import lerobot.datasets.lerobot_dataset
import lerobot.datasets.utils
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchcodec.decoders
import torchvision.io
import torchvision.transforms.v2
import torchvision.transforms.v2.functional as F
from huggingface_hub import hf_hub_download, snapshot_download

import utils.memcached
from utils.instruction_generator import InstructionGenerator
from proactive_vla.residual_flow import extract_dom_dynamic_fields


LEROBOT_MAIN_REVISION = "main"
LEROBOT_META_PATTERNS = [
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
]
PREFETCH_STATUS_INTERVAL_ENV = "DYNAMICVLA_PREFETCH_STATUS_INTERVAL"
PREFETCH_STATUS_INTERVAL_S = 30.0
PREFETCH_MODE_ENV = "DYNAMICVLA_PREFETCH_MODE"
PREFETCH_RETRIES_ENV = "DYNAMICVLA_PREFETCH_RETRIES"
PREFETCH_RETRIES = 8
PREFETCH_FILE_DELAY_ENV = "DYNAMICVLA_PREFETCH_FILE_DELAY"
PREFETCH_FILE_DELAY_S = 1.0
PREFETCH_RATE_LIMIT_DELAY_ENV = "DYNAMICVLA_PREFETCH_RATE_LIMIT_DELAY"
PREFETCH_RATE_LIMIT_DELAY_S = 60.0
PREFETCH_CHUNK_PATTERN_THRESHOLD_ENV = "DYNAMICVLA_PREFETCH_CHUNK_PATTERN_THRESHOLD"
PREFETCH_CHUNK_PATTERN_THRESHOLD = 1000
EPISODE_SPEED_CACHE_VERSION = 1


def _episode_chunk(ep_idx: int) -> str:
    return f"chunk-{ep_idx // 1000:03d}"


def _episode_file_stem(ep_idx: int) -> str:
    return f"episode_{ep_idx:06d}"


def _required_video_keys(required_features: list[str] | None) -> list[str]:
    return [
        key
        for key in (required_features or [])
        if key.startswith("observation.images.")
    ]


def _subset_download_patterns(
    episodes: list[int] | None, required_features: list[str] | None
) -> list[str]:
    patterns = list(LEROBOT_META_PATTERNS)
    if episodes is None:
        return patterns

    for ep_idx in sorted(set(episodes)):
        patterns.extend(_episode_download_patterns(ep_idx, required_features))

    return patterns


def _episode_download_patterns(
    ep_idx: int, required_features: list[str] | None
) -> list[str]:
    video_keys = _required_video_keys(required_features)
    chunk = _episode_chunk(ep_idx)
    stem = _episode_file_stem(ep_idx)
    patterns = [f"data/{chunk}/{stem}.parquet"]
    patterns.extend(
        f"videos/{chunk}/{video_key}/{stem}.mp4" for video_key in video_keys
    )

    return patterns


def _read_chunk_pattern_threshold() -> int:
    raw_threshold = os.environ.get(PREFETCH_CHUNK_PATTERN_THRESHOLD_ENV)
    if raw_threshold is None:
        return PREFETCH_CHUNK_PATTERN_THRESHOLD
    try:
        return max(0, int(raw_threshold))
    except ValueError:
        logging.warning(
            "Ignoring invalid %s=%r; using %d.",
            PREFETCH_CHUNK_PATTERN_THRESHOLD_ENV,
            raw_threshold,
            PREFETCH_CHUNK_PATTERN_THRESHOLD,
        )
        return PREFETCH_CHUNK_PATTERN_THRESHOLD


def _snapshot_allow_patterns(
    patterns: list[str],
    episodes: list[int] | None,
    required_features: list[str] | None,
) -> list[str]:
    if episodes is None or _required_video_keys(required_features):
        return patterns

    unique_episodes = sorted(set(episodes))
    threshold = _read_chunk_pattern_threshold()
    if threshold > 0 and len(unique_episodes) < threshold:
        return patterns

    chunks = sorted({_episode_chunk(ep_idx) for ep_idx in unique_episodes})
    allow_patterns = list(LEROBOT_META_PATTERNS)
    allow_patterns.extend(f"data/{chunk}/*.parquet" for chunk in chunks)
    logging.info(
        "Using %d chunk-level data allow patterns instead of %d per-episode "
        "data patterns for snapshot prefetch. Set %s above %d to disable.",
        len(chunks),
        len(unique_episodes),
        PREFETCH_CHUNK_PATTERN_THRESHOLD_ENV,
        len(unique_episodes),
    )
    return allow_patterns


def _download_pattern_category(pattern: str) -> str:
    if pattern.startswith("meta/"):
        return "meta"
    if pattern.startswith("data/"):
        return "data"
    if pattern.startswith("videos/"):
        return "video"
    return "other"


def _count_patterns_by_category(patterns: list[str]) -> collections.Counter[str]:
    return collections.Counter(
        _download_pattern_category(pattern) for pattern in patterns
    )


def _format_category_totals(counter: collections.Counter[str]) -> str:
    parts = []
    for category in ("meta", "data", "video", "other"):
        total = counter.get(category, 0)
        if total > 0:
            parts.append(f"{category} {total}")
    return ", ".join(parts)


def _format_category_counts(
    done_by_category: collections.Counter[str],
    total_by_category: collections.Counter[str],
) -> str:
    parts = []
    for category in ("meta", "data", "video", "other"):
        total = total_by_category.get(category, 0)
        if total > 0:
            parts.append(f"{category} {done_by_category.get(category, 0)}/{total}")
    return ", ".join(parts)


def _format_bytes(num_bytes: float | int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _format_signed_bytes(num_bytes: float | int) -> str:
    sign = "+" if num_bytes >= 0 else "-"
    return f"{sign}{_format_bytes(abs(num_bytes))}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _read_prefetch_status_interval() -> float:
    raw_interval = os.environ.get(PREFETCH_STATUS_INTERVAL_ENV)
    if raw_interval is None:
        return PREFETCH_STATUS_INTERVAL_S
    try:
        return float(raw_interval)
    except ValueError:
        logging.warning(
            "Ignoring invalid %s=%r; using %.0fs.",
            PREFETCH_STATUS_INTERVAL_ENV,
            raw_interval,
            PREFETCH_STATUS_INTERVAL_S,
        )
        return PREFETCH_STATUS_INTERVAL_S


def _read_prefetch_retries() -> int:
    raw_retries = os.environ.get(PREFETCH_RETRIES_ENV)
    if raw_retries is None:
        return PREFETCH_RETRIES
    try:
        return max(1, int(raw_retries))
    except ValueError:
        logging.warning(
            "Ignoring invalid %s=%r; using %d.",
            PREFETCH_RETRIES_ENV,
            raw_retries,
            PREFETCH_RETRIES,
        )
        return PREFETCH_RETRIES


def _read_prefetch_delay(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        logging.warning("Ignoring invalid %s=%r; using %.1fs.", name, raw_value, default)
        return default


def _read_prefetch_mode(missing: list[str]) -> str:
    mode = os.environ.get(PREFETCH_MODE_ENV, "auto").strip().lower()
    if mode not in {"auto", "file", "snapshot"}:
        logging.warning(
            "Ignoring invalid %s=%r; using auto.",
            PREFETCH_MODE_ENV,
            mode,
        )
        mode = "auto"

    if mode == "auto" and any(
        _download_pattern_category(pattern) == "video" for pattern in missing
    ):
        return "file"
    return mode


def _safe_stat(path: pathlib.Path) -> os.stat_result | None:
    with contextlib.suppress(OSError):
        return path.stat()
    return None


def _cfg_get(cfg: dict | None, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _PrefetchStatusMonitor:
    def __init__(
        self,
        root: pathlib.Path,
        patterns: list[str],
        initially_missing: list[str],
        interval_s: float,
    ):
        self.root = root
        self.patterns = patterns
        self.initially_missing = initially_missing
        self.interval_s = interval_s
        self._paths = [self.root / pattern for pattern in self.patterns]
        self._initial_missing_paths = [
            self.root / pattern for pattern in self.initially_missing
        ]
        self._total_by_category = _count_patterns_by_category(self.patterns)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._last_log_at: float | None = None
        self._last_done: int | None = None
        self._last_bytes: int | None = None

    def __enter__(self) -> "_PrefetchStatusMonitor":
        if self.interval_s <= 0:
            return self

        self._started_at = time.monotonic()
        logging.info(
            "DOM prefetch visualizer enabled: status every %.0fs "
            "(set %s=0 to disable).",
            self.interval_s,
            PREFETCH_STATUS_INTERVAL_ENV,
        )
        self._log_status("started")
        self._thread = threading.Thread(
            target=self._run,
            name="dom-prefetch-status",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.interval_s <= 0:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        label = "finished" if exc_type is None else "stopped"
        self._log_status(label)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._log_status("running")

    def _iter_download_cache_files(self) -> typing.Iterator[pathlib.Path]:
        download_root = self.root / ".cache" / "huggingface" / "download"
        if not download_root.exists():
            return
        with contextlib.suppress(OSError):
            for path in download_root.rglob("*"):
                with contextlib.suppress(OSError):
                    if path.is_file():
                        yield path

    def _snapshot(self) -> dict[str, typing.Any]:
        done_by_category: collections.Counter[str] = collections.Counter()
        done = 0
        target_bytes = 0
        latest_path = None
        latest_mtime = -1.0

        for pattern, path in zip(self.patterns, self._paths):
            stat = _safe_stat(path)
            if stat is None:
                continue

            done += 1
            done_by_category[_download_pattern_category(pattern)] += 1
            target_bytes += stat.st_size
            if stat.st_mtime > latest_mtime:
                latest_path = path
                latest_mtime = stat.st_mtime

        temp_bytes = 0
        for path in self._iter_download_cache_files():
            stat = _safe_stat(path)
            if stat is None:
                continue

            temp_bytes += stat.st_size
            if stat.st_mtime > latest_mtime:
                latest_path = path
                latest_mtime = stat.st_mtime

        initial_done = sum(
            1 for path in self._initial_missing_paths if _safe_stat(path) is not None
        )

        return {
            "done": done,
            "done_by_category": done_by_category,
            "initial_done": initial_done,
            "bytes": target_bytes + temp_bytes,
            "target_bytes": target_bytes,
            "temp_bytes": temp_bytes,
            "latest_path": latest_path,
        }

    def _relative_path(self, path: pathlib.Path | None) -> str:
        if path is None:
            return "none yet"
        with contextlib.suppress(ValueError):
            return str(path.relative_to(self.root))
        return str(path)

    def _log_status(self, label: str) -> None:
        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now

        snapshot = self._snapshot()
        total = len(self.patterns)
        missing_at_start = len(self.initially_missing)
        remaining = max(0, total - snapshot["done"])
        elapsed = _format_duration(now - self._started_at)

        if self._last_log_at is None:
            delta_seconds = 0.0
            done_delta = 0
            bytes_delta = 0
        else:
            delta_seconds = max(0.001, now - self._last_log_at)
            done_delta = snapshot["done"] - (self._last_done or 0)
            bytes_delta = snapshot["bytes"] - (self._last_bytes or 0)

        if delta_seconds > 0:
            byte_rate = _format_bytes(abs(bytes_delta) / delta_seconds) + "/s"
        else:
            byte_rate = "n/a"

        eta = "unknown"
        if done_delta > 0 and delta_seconds > 0 and remaining > 0:
            eta = _format_duration(remaining / (done_delta / delta_seconds))
        elif remaining == 0:
            eta = "0s"

        logging.info(
            "DOM prefetch %s %s requested %d/%d complete; "
            "this run %d/%d downloaded; remaining %d; %s; "
            "elapsed %s; eta %s; cache %s (%s, %s); latest %s",
            label,
            _progress_bar(snapshot["done"], total),
            snapshot["done"],
            total,
            snapshot["initial_done"],
            missing_at_start,
            remaining,
            _format_category_counts(
                snapshot["done_by_category"],
                self._total_by_category,
            ),
            elapsed,
            eta,
            _format_bytes(snapshot["bytes"]),
            _format_signed_bytes(bytes_delta),
            byte_rate,
            self._relative_path(snapshot["latest_path"]),
        )

        self._last_log_at = now
        self._last_done = snapshot["done"]
        self._last_bytes = snapshot["bytes"]


def _download_patterns_individually(
    repo_id: str,
    root: pathlib.Path,
    patterns: list[str],
    revision: str,
) -> None:
    total = len(patterns)
    retries = _read_prefetch_retries()
    file_delay_s = _read_prefetch_delay(PREFETCH_FILE_DELAY_ENV, PREFETCH_FILE_DELAY_S)
    rate_limit_delay_s = _read_prefetch_delay(
        PREFETCH_RATE_LIMIT_DELAY_ENV, PREFETCH_RATE_LIMIT_DELAY_S
    )
    started_at = time.monotonic()
    completed = 0

    logging.info(
        "Using per-file LeRobot/DOM prefetch for %d files "
        "(set %s=snapshot to use snapshot_download; retries=%d).",
        total,
        PREFETCH_MODE_ENV,
        retries,
    )

    for file_idx, pattern in enumerate(patterns, start=1):
        target = root / pattern
        stat = _safe_stat(target)
        if stat is not None:
            completed += 1
            logging.info(
                "DOM prefetch file %d/%d already cached %s (%s).",
                file_idx,
                total,
                pattern,
                _format_bytes(stat.st_size),
            )
            continue

        file_started_at = time.monotonic()
        logging.info(
            "DOM prefetch file %d/%d downloading %s ...",
            file_idx,
            total,
            pattern,
        )
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=pattern,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=root,
                    etag_timeout=30,
                )
                stat = _safe_stat(target)
                if stat is None:
                    raise RuntimeError(
                        f"Downloaded {pattern}, but it is not present at {target}."
                    )
                completed += 1
                elapsed = time.monotonic() - started_at
                file_elapsed = time.monotonic() - file_started_at
                remaining = max(0, total - completed)
                eta = "unknown"
                if completed > 0 and remaining > 0 and elapsed > 0:
                    eta = _format_duration(remaining / (completed / elapsed))
                elif remaining == 0:
                    eta = "0s"
                logging.info(
                    "DOM prefetch file %d/%d done %s (%s in %s); "
                    "overall %s %d/%d complete; eta %s.",
                    file_idx,
                    total,
                    pattern,
                    _format_bytes(stat.st_size),
                    _format_duration(file_elapsed),
                    _progress_bar(completed, total),
                    completed,
                    total,
                    eta,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    raise RuntimeError(
                        f"Could not download {pattern} after {retries} attempts."
                    ) from exc

                is_rate_limit = "429" in str(exc)
                sleep_s = rate_limit_delay_s if is_rate_limit else min(60.0, 2.0**attempt)
                logging.warning(
                    "DOM prefetch file %d/%d failed attempt %d/%d for %s: %s. "
                    "Retrying in %s.",
                    file_idx,
                    total,
                    attempt,
                    retries,
                    pattern,
                    exc,
                    _format_duration(sleep_s),
                )
                time.sleep(sleep_s)

        if file_idx < total:
            time.sleep(file_delay_s)

        if last_exc is not None and _safe_stat(target) is None:
            raise RuntimeError(f"Could not download {pattern}.") from last_exc


def _ensure_lerobot_subset_cached(
    repo_id: str,
    root: pathlib.Path,
    episodes: list[int] | None,
    required_features: list[str] | None,
    revision: str = LEROBOT_MAIN_REVISION,
) -> None:
    """Download metadata and the selected episode files used by the tiny MVP."""

    patterns = _subset_download_patterns(episodes, required_features)
    missing = [
        pattern for pattern in patterns if _safe_stat(root / pattern) is None
    ]
    if not missing:
        logging.info(
            "LeRobot/DOM subset already cached at %s (%d requested files).",
            root,
            len(patterns),
        )
        return

    episode_msg = (
        "metadata only" if episodes is None else f"{len(set(episodes))} episodes"
    )
    missing_by_category = _count_patterns_by_category(missing)
    logging.info(
        "Downloading %s from %s into %s (%d missing files: %s).",
        episode_msg,
        repo_id,
        root,
        len(missing),
        _format_category_totals(missing_by_category),
    )
    try:
        mode = _read_prefetch_mode(missing)
        with _PrefetchStatusMonitor(
            root,
            patterns,
            missing,
            interval_s=_read_prefetch_status_interval(),
        ):
            if mode == "file":
                _download_patterns_individually(repo_id, root, missing, revision)
            else:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=root,
                    allow_patterns=_snapshot_allow_patterns(
                        patterns, episodes, required_features
                    ),
                )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the requested LeRobot/DOM subset. Check that the "
            "node has Hugging Face network access and that you are logged in if "
            f"the dataset requires auth. Repository: {repo_id}, revision: {revision}."
        ) from exc


def ensure_lerobot_episodes_cached(
    repo_id: str,
    episodes: list[int] | None,
    required_features: list[str] | None,
    root: str | pathlib.Path | None = None,
    revision: str = LEROBOT_MAIN_REVISION,
) -> None:
    cache_root = (
        pathlib.Path(root) if root else lerobot.constants.HF_LEROBOT_HOME / repo_id
    )
    _ensure_lerobot_subset_cached(
        repo_id,
        cache_root,
        episodes,
        required_features,
        revision=revision,
    )


def filter_cached_lerobot_episodes(
    repo_id: str,
    episodes: list[int],
    required_features: list[str] | None,
    root: str | pathlib.Path | None = None,
) -> tuple[list[int], dict[str, typing.Any]]:
    """Return episodes whose required parquet and video files are already local."""

    cache_root = (
        pathlib.Path(root) if root else lerobot.constants.HF_LEROBOT_HOME / repo_id
    )
    missing_meta = [
        pattern for pattern in LEROBOT_META_PATTERNS if _safe_stat(cache_root / pattern) is None
    ]
    if missing_meta:
        raise FileNotFoundError(
            "Cannot run cached-only LeRobot/DOM evaluation because metadata is "
            f"missing under {cache_root}: {missing_meta}"
        )

    cached: list[int] = []
    missing_by_category: collections.Counter[str] = collections.Counter()
    missing_episode_count = 0
    unique_episodes = sorted(set(int(ep_idx) for ep_idx in episodes))

    for ep_idx in unique_episodes:
        missing = [
            pattern
            for pattern in _episode_download_patterns(ep_idx, required_features)
            if _safe_stat(cache_root / pattern) is None
        ]
        if missing:
            missing_episode_count += 1
            missing_by_category.update(_count_patterns_by_category(missing))
            continue
        cached.append(ep_idx)

    summary = {
        "root": str(cache_root),
        "requested_episodes": len(unique_episodes),
        "cached_episodes": len(cached),
        "missing_episodes": missing_episode_count,
        "missing_files": int(sum(missing_by_category.values())),
        "missing_by_category": dict(missing_by_category),
    }
    logging.info(
        "Cached-only LeRobot/DOM filter at %s kept %d/%d episodes; "
        "missing %d episodes (%s files).",
        cache_root,
        len(cached),
        len(unique_episodes),
        missing_episode_count,
        _format_category_totals(missing_by_category) or "0",
    )
    return cached, summary


def _split_file_path(split_file: str | pathlib.Path) -> pathlib.Path:
    split_path = pathlib.Path(split_file)
    if not split_path.is_absolute():
        split_path = pathlib.Path.cwd() / split_path
    return split_path


def _load_split_file(split_path: pathlib.Path) -> dict:
    try:
        with split_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"DATASET.EPISODE_SPLIT_FILE not found: {split_path}"
        ) from exc


def _split_map(split_data: dict) -> dict:
    return split_data.get("splits", split_data) if isinstance(split_data, dict) else {}


def _episode_list_from_split_spec(
    split_data: dict,
    split: str,
    split_path: pathlib.Path,
    seen: set[tuple[str, str]] | None = None,
) -> list[int]:
    seen = seen or set()
    key = (str(split_path), str(split))
    if key in seen:
        raise ValueError(f"Cyclic episode split reference for {split!r} in {split_path}.")
    seen.add(key)

    split_map = _split_map(split_data)
    if split not in split_map:
        raise ValueError(
            f"DATASET.EPISODE_SPLIT_FILE {split_path} does not contain split {split!r}."
        )

    spec = split_map[split]
    if isinstance(spec, (list, tuple)):
        return [int(ep_idx) for ep_idx in spec]

    if not isinstance(spec, dict):
        raise ValueError(
            f"Split {split!r} in {split_path} must be a list or object, got {type(spec).__name__}."
        )

    source_file = (
        spec["source_file"] if "source_file" in spec else split_data.get("source_file")
    )
    source_split = spec.get("source_split")
    if source_split is None:
        episodes = [int(ep_idx) for ep_idx in spec.get("episodes", [])]
    else:
        if source_file is None:
            source_data = split_data
            source_path = split_path
        else:
            source_path = pathlib.Path(source_file)
            if not source_path.is_absolute():
                source_path = split_path.parent / source_path
            source_data = _load_split_file(source_path)
        episodes = _episode_list_from_split_spec(
            source_data,
            str(source_split),
            source_path,
            seen=seen,
        )

    for excluded_split in spec.get("exclude_splits", []):
        excluded = set(
            _episode_list_from_split_spec(
                split_data,
                str(excluded_split),
                split_path,
                seen=set(),
            )
        )
        episodes = [ep_idx for ep_idx in episodes if ep_idx not in excluded]

    sample_size = spec.get("sample", spec.get("max_episodes"))
    if sample_size is not None:
        sample_size = int(sample_size)
        if sample_size < 0:
            raise ValueError(f"Split {split!r} sample size must be >= 0, got {sample_size}.")
        if sample_size < len(episodes):
            seed = int(spec.get("seed", split_data.get("seed", 0)))
            episodes = random.Random(seed).sample(list(episodes), sample_size)

    if bool(spec.get("sort", True)):
        episodes = sorted(int(ep_idx) for ep_idx in episodes)
    else:
        episodes = [int(ep_idx) for ep_idx in episodes]
    return episodes


def _resolve_episode_split_file(dataset_cfg: dict, split: str) -> list[int] | None:
    split_file = _cfg_get(dataset_cfg, "EPISODE_SPLIT_FILE")
    if split_file is None:
        return None

    split_path = _split_file_path(split_file)
    split_data = _load_split_file(split_path)
    return _episode_list_from_split_spec(split_data, split, split_path)


def _resolve_explicit_episode_selection(dataset_cfg: dict, split: str) -> list[int] | None:
    split_file_episodes = _resolve_episode_split_file(dataset_cfg, split)
    if split_file_episodes is not None:
        return split_file_episodes

    if split == "train":
        explicit_key = "TRAIN_EPISODES"
        range_key = "TRAIN_EPISODE_RANGE"
    elif split == "test":
        explicit_key = "TEST_EPISODES"
        range_key = "TEST_EPISODE_RANGE"
    else:
        return None

    episodes = _cfg_get(dataset_cfg, explicit_key)
    if episodes is not None:
        return [int(ep_idx) for ep_idx in episodes]

    episode_range = _cfg_get(dataset_cfg, range_key)
    if episode_range is None:
        return None
    if len(episode_range) != 2:
        raise ValueError(f"{range_key} must contain [start, end], got {episode_range!r}.")
    start, end = int(episode_range[0]), int(episode_range[1])
    if end < start:
        raise ValueError(f"{range_key} end must be >= start, got {episode_range!r}.")
    return list(range(start, end + 1))


def _speed_split_cfg(dataset_cfg: dict) -> dict | None:
    speed_cfg = _cfg_get(dataset_cfg, "SPEED_SPLIT")
    if speed_cfg is None:
        speed_cfg = _cfg_get(dataset_cfg, "EPISODE_SPEED_SPLIT")
    if speed_cfg is None:
        return None
    if _cfg_get(speed_cfg, "ENABLED", True) is False:
        return None
    return speed_cfg


def _speed_split_bound(speed_cfg: dict, split: str, bound: str):
    split = split.upper()
    bound = bound.upper()
    aliases = (
        f"{split}_{bound}_MPS",
        f"{split}_{bound}_SPEED_MPS",
        f"{split}_SPEED_{bound}_MPS",
    )
    for key in aliases:
        value = _cfg_get(speed_cfg, key)
        if value is not None:
            return float(value)
    return None


def _speed_split_cache_path(root: pathlib.Path) -> pathlib.Path:
    return root / ".cache" / "dynamicvla_episode_speed_stats.json"


def _load_speed_split_cache(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"version": EPISODE_SPEED_CACHE_VERSION, "entries": {}}
    try:
        with path.open("r", encoding="utf-8") as fp:
            cache = json.load(fp)
    except Exception as exc:
        logging.warning("Ignoring unreadable episode speed cache %s: %s", path, exc)
        return {"version": EPISODE_SPEED_CACHE_VERSION, "entries": {}}
    if cache.get("version") != EPISODE_SPEED_CACHE_VERSION:
        return {"version": EPISODE_SPEED_CACHE_VERSION, "entries": {}}
    cache.setdefault("entries", {})
    return cache


def _save_speed_split_cache(path: pathlib.Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(cache, fp, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _episode_speed_cache_key(
    velocity_key: str,
    velocity_slice: tuple[int, int],
    speed_stat: str,
) -> str:
    return f"{velocity_key}:{velocity_slice[0]}:{velocity_slice[1]}:{speed_stat}"


def _compute_episode_speed_mps(
    path: pathlib.Path,
    velocity_key: str,
    velocity_slice: tuple[int, int],
    speed_stat: str,
) -> float:
    table = pq.read_table(path, columns=[velocity_key])
    values = np.stack(table.column(velocity_key).to_pylist()).astype(np.float32)
    start, end = velocity_slice
    velocity = values[..., start:end]
    if velocity.ndim < 2 or velocity.shape[-1] == 0:
        raise ValueError(
            f"Could not slice velocity dims {velocity_slice!r} from {velocity_key} "
            f"in {path}; got shape {tuple(values.shape)}."
        )
    speed = np.linalg.norm(velocity, axis=-1)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        raise ValueError(f"No finite object speeds found in {path}.")
    if speed_stat == "mean":
        return float(np.mean(speed))
    if speed_stat == "median":
        return float(np.median(speed))
    if speed_stat == "max":
        return float(np.max(speed))
    if speed_stat == "min":
        return float(np.min(speed))
    raise ValueError(
        f"Unsupported SPEED_SPLIT SPEED_STAT={speed_stat!r}; "
        "expected one of mean, median, max, min."
    )


def _episode_data_path(root: pathlib.Path, ep_idx: int) -> pathlib.Path:
    return root / "data" / _episode_chunk(ep_idx) / f"{_episode_file_stem(ep_idx)}.parquet"


def _episode_speed_stats(
    repo_id: str,
    root: pathlib.Path,
    episodes: list[int],
    velocity_key: str,
    velocity_slice: tuple[int, int],
    speed_stat: str,
) -> dict[int, float]:
    ensure_lerobot_episodes_cached(
        repo_id,
        episodes,
        [velocity_key],
        root=root,
    )

    cache_path = _speed_split_cache_path(root)
    cache = _load_speed_split_cache(cache_path)
    cache_key = _episode_speed_cache_key(velocity_key, velocity_slice, speed_stat)
    cache_entries = cache.setdefault("entries", {}).setdefault(cache_key, {})
    changed = False
    speeds: dict[int, float] = {}

    for ep_idx in episodes:
        path = _episode_data_path(root, ep_idx)
        stat = _safe_stat(path)
        if stat is None:
            raise FileNotFoundError(f"Missing episode parquet after prefetch: {path}")

        entry = cache_entries.get(str(ep_idx))
        if (
            entry is not None
            and int(entry.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            and int(entry.get("size", -1)) == int(stat.st_size)
            and "speed_mps" in entry
        ):
            speeds[ep_idx] = float(entry["speed_mps"])
            continue

        speed_mps = _compute_episode_speed_mps(
            path,
            velocity_key,
            velocity_slice,
            speed_stat,
        )
        speeds[ep_idx] = speed_mps
        cache_entries[str(ep_idx)] = {
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "speed_mps": speed_mps,
        }
        changed = True

    if changed:
        _save_speed_split_cache(cache_path, cache)
        logging.info("Updated DOM episode speed cache at %s.", cache_path)

    return speeds


def _format_speed_summary(speeds: list[float]) -> str:
    if not speeds:
        return "empty"
    values = np.asarray(speeds, dtype=np.float32)
    return (
        f"min {float(np.min(values)):.4f}, "
        f"median {float(np.median(values)):.4f}, "
        f"max {float(np.max(values)):.4f} m/s"
    )


def _apply_episode_speed_split(
    dataset_cfg: dict,
    split: str,
    episodes: list[int] | None,
) -> list[int] | None:
    speed_cfg = _speed_split_cfg(dataset_cfg)
    if speed_cfg is None:
        return episodes

    if episodes is None:
        raise ValueError(
            "DATASET.SPEED_SPLIT requires an explicit candidate pool via "
            f"{split.upper()}_EPISODES or {split.upper()}_EPISODE_RANGE. "
            "Refusing to scan the full DOM dataset by speed."
        )

    min_speed = _speed_split_bound(speed_cfg, split, "MIN")
    max_speed = _speed_split_bound(speed_cfg, split, "MAX")
    if min_speed is None and max_speed is None:
        return episodes

    repo_id = _cfg_get(dataset_cfg, "NAME")
    if not repo_id:
        raise ValueError("DATASET.NAME is required when DATASET.SPEED_SPLIT is enabled.")
    root_cfg = _cfg_get(dataset_cfg, "ROOT")
    root = pathlib.Path(root_cfg) if root_cfg else lerobot.constants.HF_LEROBOT_HOME / repo_id
    velocity_key = _cfg_get(speed_cfg, "VELOCITY_KEY", "observation.environment_state")
    velocity_slice_raw = _cfg_get(speed_cfg, "VELOCITY_SLICE", [6, 9])
    if len(velocity_slice_raw) != 2:
        raise ValueError(
            f"DATASET.SPEED_SPLIT.VELOCITY_SLICE must be [start, end], got {velocity_slice_raw!r}."
        )
    velocity_slice = (int(velocity_slice_raw[0]), int(velocity_slice_raw[1]))
    speed_stat = str(_cfg_get(speed_cfg, "SPEED_STAT", "mean")).lower()

    logging.info(
        "Resolving %s speed split from %d candidate episodes with %s[%d:%d] "
        "%s speed; min=%s max=%s.",
        split,
        len(episodes),
        velocity_key,
        velocity_slice[0],
        velocity_slice[1],
        speed_stat,
        "none" if min_speed is None else f"{min_speed:.4f} m/s",
        "none" if max_speed is None else f"{max_speed:.4f} m/s",
    )
    speeds = _episode_speed_stats(
        repo_id,
        root,
        sorted(set(episodes)),
        velocity_key,
        velocity_slice,
        speed_stat,
    )

    selected = []
    for ep_idx in episodes:
        speed = speeds[int(ep_idx)]
        if min_speed is not None and speed < min_speed:
            continue
        if max_speed is not None and speed > max_speed:
            continue
        selected.append(int(ep_idx))

    if not selected:
        raise ValueError(
            f"DATASET.SPEED_SPLIT selected zero {split} episodes from {len(episodes)} "
            f"candidates. Candidate speed summary: {_format_speed_summary(list(speeds.values()))}."
        )

    selected_speeds = [speeds[ep_idx] for ep_idx in selected]
    logging.info(
        "Resolved %s speed split: selected %d/%d episodes; selected speeds %s; "
        "candidate speeds %s.",
        split,
        len(selected),
        len(episodes),
        _format_speed_summary(selected_speeds),
        _format_speed_summary(list(speeds.values())),
    )
    return selected


def resolve_episode_selection(dataset_cfg: dict, split: str) -> list[int] | None:
    """Resolve explicit episode lists or inclusive ranges from a config block."""
    episodes = _resolve_explicit_episode_selection(dataset_cfg, split)
    if _cfg_get(dataset_cfg, "EPISODE_SPLIT_FILE") is not None:
        return episodes
    return _apply_episode_speed_split(dataset_cfg, split, episodes)


def get_dataset(
    dataset_name: str,
    split: str,
    pin_memory: bool,
    delta_action: bool,
    required_features: list[str] | None = None,
    episodes: list[int] | None = None,
    image_transforms: typing.Callable | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
) -> torch.utils.data.Dataset:
    return LeRobotDataset(
        dataset_name,
        split=split,
        pin_memory=pin_memory,
        delta_action=delta_action,
        required_features=required_features,
        episodes=episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
    )


class ImageTransforms:
    def __init__(self, img_size=None, cfg={}):
        self.img_size = img_size
        self.brightness = cfg.get("BRIGHTNESS")
        self.contrast = cfg.get("CONTRAST")
        self.saturation = cfg.get("SATURATION")
        self.hue = cfg.get("HUE")
        self.sharpness = cfg.get("SHARPNESS")

    def __call__(self, item, camera_keys: list[str]) -> dict:
        transform_args = torchvision.transforms.v2.ColorJitter.get_params(
            brightness=self.brightness if self.brightness else None,
            contrast=self.contrast if self.contrast else None,
            saturation=self.saturation if self.saturation else None,
            hue=self.hue if self.hue else None,
        )
        for k, v in item.items():
            if k not in camera_keys:
                continue

            item[k] = self._apply_transforms(v, transform_args)

        return item

    def _apply_transforms(self, image, transform_args):
        if self.img_size is not None:
            image = F.resize(image, self.img_size)

        for tr_idx in transform_args[0]:
            if tr_idx == 0 and transform_args[1] is not None:
                image = F.adjust_brightness(image, transform_args[1])
            elif tr_idx == 1 and transform_args[2] is not None:
                image = F.adjust_contrast(image, transform_args[2])
            elif tr_idx == 2 and transform_args[3] is not None:
                image = F.adjust_saturation(image, transform_args[3])
            elif tr_idx == 3 and transform_args[4] is not None:
                image = F.adjust_hue(image, transform_args[4])

        return image


class LeRobotDataset(torch.utils.data.Dataset):
    LEROBOT_VERSION = "v2.1"
    """
    A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...
    """

    def __init__(
        self,
        repo_id: str,
        root: str | pathlib.Path | None = None,
        split: str = "train",
        pin_memory: bool = False,
        delta_action: bool = True,
        required_features: list[str] | None = None,
        episodes: list[int] | None = None,
        image_transforms: typing.Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str = LEROBOT_MAIN_REVISION,
    ):
        super().__init__()
        self._memcached = utils.memcached.MCClient(enabled=pin_memory)
        self.repo_id = repo_id
        self.root = (
            pathlib.Path(root) if root else lerobot.constants.HF_LEROBOT_HOME / repo_id
        )
        self.delta_action = delta_action
        self.required_features = required_features or []
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s
        self.delta_indices = None
        self.revision = revision

        # Load metadata
        _ensure_lerobot_subset_cached(
            self.repo_id,
            self.root,
            self.episodes,
            self.required_features,
            revision=self.revision,
        )
        logging.info("Loading LeRobot metadata for %s from %s ...", self.repo_id, self.root)
        self.meta = lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata(
            self.repo_id, self.root, revision=self.revision
        )
        logging.info("LeRobot metadata loaded for %s.", self.repo_id)
        self.image_keys = [
            key for key in self.meta.image_keys if key in self.required_features
        ]
        self.video_keys = [
            key for key in self.meta.video_keys if key in self.required_features
        ]
        self.camera_keys = [
            key for key in self.meta.camera_keys if key in self.required_features
        ]
        if self.episodes is None:
            self.episodes = [ep_idx for ep_idx in range(self.meta.total_episodes)]
            if split == "train":
                self.episodes = [e for e in self.episodes if e % 1000 != 0]
            elif split == "test":
                self.episodes = [e for e in self.episodes if e % 1000 == 0]
            else:
                raise ValueError(f"Unknown split {split}.")

        # Compute dataset stats
        self.stats = lerobot.datasets.compute_stats.aggregate_stats(
            [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
        )
        # Load data indices
        self.episode_data_index = lerobot.datasets.utils.get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        self.episode_data_index["length"] = (
            self.episode_data_index["to"] - self.episode_data_index["from"]
        )
        logging.info(
            "Built %s split with %d episodes and %d samples.",
            split,
            len(self.episodes),
            len(self),
        )
        # Setup delta indices
        if delta_timestamps is not None:
            # Resolve delta timestamps for specific features
            ds_delta_timestamps = {}
            for key in self.meta.features:
                if key not in self.required_features:
                    continue
                # Ref: lerobot.datasets.factory.resolve_delta_timestamps
                if key == "next.reward" and "reward" in delta_timestamps:
                    ds_delta_timestamps[key] = [
                        i / self.meta.fps for i in delta_timestamps["reward"]
                    ]
                if key == "action" and "action" in delta_timestamps:
                    ds_delta_timestamps[key] = [
                        i / self.meta.fps for i in delta_timestamps["action"]
                    ]
                if key.startswith("observation.") and "observation" in delta_timestamps:
                    ds_delta_timestamps[key] = [
                        i / self.meta.fps for i in delta_timestamps["observation"]
                    ]

            self.delta_indices = lerobot.datasets.utils.get_delta_indices(
                ds_delta_timestamps, self.meta.fps
            )

    def __repr__(self) -> str:
        feature_keys = list(self.meta.features.keys())
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.meta.total_episodes}',\n"
            f"    Number of selected samples: '{self.meta.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def __len__(self) -> int:
        return torch.sum(self.episode_data_index["length"]).item()

    def _get_episode(self, ep_idx: int) -> dict[str, torch.Tensor | bytes]:
        episode = {}
        # Load episode data
        data_frame = pq.read_table(
            pa.BufferReader(
                self._memcached.get(self.root / self.meta.get_data_file_path(ep_idx))
            )
        ).to_pandas()
        for col in data_frame.columns:
            if col in self.meta.image_keys:
                episode[col] = data_frame[col].values
            else:
                episode[col] = torch.tensor(np.stack(data_frame[col].values))

        # Load video data
        for video_key in self.video_keys:
            episode[video_key] = self._get_video_bytes(ep_idx, video_key)

        return episode

    def _get_video_bytes(self, ep_idx: int, video_key: str) -> torch.Tensor:
        video_path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
        return self._memcached.get(video_path)

    def __getitem__(self, idx) -> dict:
        ep_idx = self._get_episode_index(
            idx, self.episode_data_index["from"], self.episode_data_index["to"]
        )
        frame_idx = idx - self.episode_data_index["from"][ep_idx].item()

        g_ep_idx = self.episodes[ep_idx]
        episode = self._get_episode(g_ep_idx)
        item = {
            k: v[frame_idx] for k, v in episode.items() if k in self.required_features
        }
        item["sample_index"] = torch.tensor(idx, dtype=torch.long)
        item["episode_index"] = torch.tensor(g_ep_idx, dtype=torch.long)
        item["frame_index"] = torch.tensor(frame_idx, dtype=torch.long)

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(
                frame_idx,
                self.episode_data_index["length"][ep_idx].item(),
                self.delta_indices,
            )
            query_result = self._query_episode(episode, query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.image_keys) > 0:
            for key in self.image_keys:
                item[key] = (
                    torchvision.io.decode_image(
                        torch.frombuffer(
                            episode[key][frame_idx]["bytes"], dtype=torch.uint8
                        )
                    ).float()
                    / 255.0
                )

        if len(self.video_keys) > 0:
            current_ts = episode["timestamp"][frame_idx].item()
            query_timestamps = self._get_query_timestamps(
                episode["timestamp"], current_ts, query_indices
            )
            video_frames = self._query_videos(episode, query_timestamps)
            item = {**item, **video_frames}

        if self.image_transforms is not None:
            item = self.image_transforms(item, self.camera_keys)

        if self.delta_action:
            act_dim = item["action"].shape[-1] - 1
            if (
                self.delta_indices is not None
                and "observation.state" in self.delta_indices
            ):
                curr_indice = [i == 0 for i in self.delta_indices["observation.state"]]
                curr_state = item["observation.state"][curr_indice, :act_dim]
            else:
                curr_state = item["observation.state"][..., :act_dim]

            # Only delta the XYZ and rotation components of the action
            item["action"][..., :act_dim] -= curr_state

        if "observation.environment_state" in item:
            item.update(extract_dom_dynamic_fields(item["observation.environment_state"]))
            if query_indices is not None and "action" in query_indices:
                future_env = episode["observation.environment_state"][query_indices["action"]]
                item["dynamic_object.future_position_xyz"] = future_env[..., :3]
                item["dynamic_object.future_velocity_xyz"] = future_env[..., 6:9]

        # Add task as a string
        task_idx = episode["task_index"][frame_idx].item()
        item["task"] = InstructionGenerator.generate_instruction(
            self.meta.tasks[task_idx]
        )
        return item

    def _get_episode_index(
        self,
        frame_idx: int,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> int:
        """
        Get the episode index for a given sample index.
        """
        ep_idx = torch.searchsorted(start_indices, frame_idx, right=True) - 1
        if ep_idx < 0 or frame_idx >= end_indices[ep_idx]:
            raise IndexError(f"Frame Index {frame_idx} is out of bounds.")

        return ep_idx.item()

    def _get_query_indices(
        self,
        frame_idx: int,
        ep_length: int,
        delta_indices: dict[str, list[int]],
    ) -> tuple[dict[str, list[int | bool]]]:
        # Ref: lerobot.datasets.lerobot_dataset._get_query_indices
        query_indices = {
            key: [max(0, min(ep_length - 1, frame_idx + delta)) for delta in delta_idx]
            for key, delta_idx in delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (frame_idx + delta < 0) | (frame_idx + delta >= ep_length)
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in delta_indices.items()
        }
        return query_indices, padding

    def _query_episode(
        self,
        episode: dict[str, torch.Tensor | bytes],
        query_indices: dict[str, list[int]],
    ) -> dict:
        return {
            key: episode[key][q_idx]
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _get_query_timestamps(
        self,
        episode_ts: torch.Tensor,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        # Ref: lerobot.datasets.lerobot_dataset._get_query_timestamps
        query_timestamps = {}
        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = episode_ts[query_indices[key]].tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(
        self,
        episode: dict[str, torch.Tensor | bytes],
        query_timestamps: dict[str, list[float]],
    ) -> dict[str, torch.Tensor]:
        # Ref: lerobot.datasets.lerobot_dataset._query_videos
        videos = {}
        for video_key, query_ts in query_timestamps.items():
            video_decoder = torchcodec.decoders.VideoDecoder(
                episode[video_key],
                seek_mode="approximate",
            )
            videos[video_key] = self._get_video_frames(video_decoder, query_ts).squeeze(
                0
            )

        return videos

    def _get_video_frames(
        self, video_decoder: torchcodec.decoders.VideoDecoder, timestamps: list[float]
    ) -> torch.Tensor:
        # Ref: lerobot.datasets.video_utils.decode_video_frames_torchcodec
        loaded_frames = []
        loaded_ts = []
        # convert timestamps to frame indices
        frame_indices = [round(ts * self.meta.fps) for ts in timestamps]
        # retrieve frames based on indices
        frames_batch = video_decoder.get_frames_at(indices=frame_indices)
        for frame, pts in zip(
            frames_batch.data, frames_batch.pts_seconds, strict=False
        ):
            loaded_frames.append(frame)
            loaded_ts.append(pts.item())

        query_ts = torch.tensor(timestamps)
        loaded_ts = torch.tensor(loaded_ts)

        # compute distances between each query timestamp and loaded timestamps
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_, argmin_ = dist.min(1)
        assert (min_ < self.tolerance_s).all()

        # get closest frames to the query timestamps
        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        # closest_ts = loaded_ts[argmin_]

        # convert to float32 in [0,1] range (channel first)
        closest_frames = closest_frames.type(torch.float32) / 255.0
        assert len(timestamps) == len(closest_frames)
        return closest_frames
