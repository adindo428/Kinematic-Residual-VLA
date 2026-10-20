# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-03-14 15:09:46
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-19 15:35:20
# @Email:  root@haozhexie.com

import argparse
import datetime
import importlib
import logging
import os
import pprint
import sys

import cv2
import easydict
import torch
import yaml

import core
import utils.datasets
import utils.distributed

# Fix deadlock in DataLoader
cv2.setNumThreads(0)


def get_args_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="exp_name",
        help="The name of the experiment",
        default="%s" % datetime.datetime.now(),
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_file",
        help="Path to the config.yaml file",
        default="config.yaml",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        help="The dataset name to train or test.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        help="The GPU device to use (e.g., 0,1,2,3).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--ckpt",
        dest="ckpt",
        help="Resume training or evaluation from a checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--init-ckpt",
        dest="init_ckpt",
        help="Initialize weights from a checkpoint but start training from epoch 0.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run",
        dest="run_id",
        help="The unique run ID for WandB",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test", dest="test", help="Test the network.", action="store_true"
    )
    parser.add_argument(
        "--prefetch-only",
        dest="prefetch_only",
        help="Download the configured LeRobot dataset subset and exit.",
        action="store_true",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="The rank ID of the GPU. Automatically assigned by torch.distributed.",
        default=os.getenv("LOCAL_RANK", 0),
    )
    args = parser.parse_args()
    return args


def _eval_test_during_train(cfg) -> bool:
    train_cfg = getattr(cfg, "TRAIN", {})
    if isinstance(train_cfg, dict):
        return bool(train_cfg.get("EVAL_TEST_DURING_TRAIN", True))
    return bool(getattr(train_cfg, "EVAL_TEST_DURING_TRAIN", True))


def _train_eval_splits(cfg) -> list[str]:
    train_cfg = getattr(cfg, "TRAIN", {})
    configured = (
        train_cfg.get("EVAL_SPLITS")
        if isinstance(train_cfg, dict)
        else getattr(train_cfg, "EVAL_SPLITS", None)
    )
    if configured is not None:
        if isinstance(configured, str):
            return [item.strip() for item in configured.replace(",", " ").split() if item.strip()]
        return [str(item) for item in configured if str(item)]
    if _eval_test_during_train(cfg):
        return ["test"]
    return []


def _parse_bool_env(name: str):
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value")


def prefetch_configured_dataset(cfg):
    split_episode_map = {
        "train": utils.datasets.resolve_episode_selection(cfg.DATASET, "train")
    }
    for split in _train_eval_splits(cfg):
        split_episode_map[split] = utils.datasets.resolve_episode_selection(
            cfg.DATASET, split
        )

    episodes = None
    if any(value is not None for value in split_episode_map.values()):
        merged = set()
        for split_episodes in split_episode_map.values():
            merged.update(split_episodes or [])
        episodes = sorted(merged)

    split_msg = ", ".join(
        f"{split}={len(split_episodes) if split_episode_map[split] is not None else 'all'}"
        for split, split_episodes in split_episode_map.items()
    )
    episode_msg = "metadata only" if episodes is None else f"{len(episodes)} episodes"
    logging.info(
        "Prefetch-only mode: downloading %s for %s (%s), then exiting.",
        episode_msg,
        cfg.DATASET.NAME,
        split_msg,
    )
    utils.datasets.ensure_lerobot_episodes_cached(
        cfg.DATASET.NAME,
        episodes,
        cfg.DATASET.REQUIRED_FEATURES,
    )
    logging.info("Prefetch-only mode complete.")


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Read the experimental config
    with open(args.cfg_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
        cfg = easydict.EasyDict(cfg)
    cfg.DIR.OUTPUT = os.environ.get("DVLA_OUTPUT_ROOT", cfg.DIR.OUTPUT)

    # Parse runtime arguments
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.exp_name is not None:
        cfg.CONST.EXP_NAME = args.exp_name
    if args.dataset is not None:
        cfg.DATASET.NAME = args.dataset
    if args.init_ckpt is not None:
        cfg.CONST.CKPT = args.init_ckpt
        cfg.CONST.INIT_ONLY = True
    elif args.ckpt is not None:
        cfg.CONST.CKPT = args.ckpt
        cfg.CONST.INIT_ONLY = False
    if args.run_id is not None:
        cfg.WANDB.RUN_ID = args.run_id
    if args.run_id is not None and args.ckpt is None and args.init_ckpt is None:
        raise Exception("No checkpoints")

    if os.getenv("DVLA_N_WORKERS") is not None:
        cfg.CONST.N_WORKERS = int(os.environ["DVLA_N_WORKERS"])
        if cfg.CONST.N_WORKERS < 0:
            raise ValueError("DVLA_N_WORKERS must be >= 0")
    if os.getenv("DVLA_DATALOADER_PREFETCH_FACTOR") is not None:
        cfg.CONST.DATALOADER_PREFETCH_FACTOR = int(
            os.environ["DVLA_DATALOADER_PREFETCH_FACTOR"]
        )
        if cfg.CONST.DATALOADER_PREFETCH_FACTOR < 1:
            raise ValueError("DVLA_DATALOADER_PREFETCH_FACTOR must be >= 1")
    if os.getenv("DVLA_PERSISTENT_WORKERS") is not None:
        persistent_workers = os.environ["DVLA_PERSISTENT_WORKERS"].strip().lower()
        if persistent_workers not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError(
                "DVLA_PERSISTENT_WORKERS must be a boolean-like value"
            )
        cfg.CONST.PERSISTENT_WORKERS = persistent_workers in {
            "1",
            "true",
            "yes",
            "on",
        }
    batch_size = os.getenv("DVLA_BATCH_SIZE")
    if batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = int(batch_size)
        if cfg.TRAIN.BATCH_SIZE < 1:
            raise ValueError("DVLA_BATCH_SIZE must be >= 1")
    grad_accum_steps = os.getenv("DVLA_GRAD_ACCUM_STEPS")
    if grad_accum_steps is not None:
        cfg.TRAIN.GRAD_ACCUM_STEPS = int(grad_accum_steps)
        if cfg.TRAIN.GRAD_ACCUM_STEPS < 1:
            raise ValueError("DVLA_GRAD_ACCUM_STEPS must be >= 1")
    n_epochs = os.getenv("DVLA_N_EPOCHS")
    if n_epochs is not None:
        cfg.TRAIN.N_EPOCHS = int(n_epochs)
        if cfg.TRAIN.N_EPOCHS < 1:
            raise ValueError("DVLA_N_EPOCHS must be >= 1")
    optimizer_lr = os.getenv("DVLA_OPTIMIZER_LR")
    if optimizer_lr is not None:
        cfg.TRAIN.OPTIMIZER.LR = float(optimizer_lr)
        if cfg.TRAIN.OPTIMIZER.LR <= 0:
            raise ValueError("DVLA_OPTIMIZER_LR must be > 0")
    eval_splits = os.getenv("DVLA_EVAL_SPLITS")
    if eval_splits is not None:
        cfg.TRAIN.EVAL_SPLITS = [
            item.strip() for item in eval_splits.replace(",", " ").split() if item.strip()
        ]
    eval_freq_epoch = os.getenv("DVLA_EVAL_FREQ_EPOCH")
    if eval_freq_epoch is not None:
        cfg.TRAIN.EVAL_FREQ_EPOCH = int(eval_freq_epoch)
        if cfg.TRAIN.EVAL_FREQ_EPOCH < 0:
            raise ValueError("DVLA_EVAL_FREQ_EPOCH must be >= 0")
    eval_max_batches = os.getenv("DVLA_EVAL_MAX_BATCHES")
    if eval_max_batches is not None:
        cfg.TRAIN.EVAL_MAX_BATCHES = int(eval_max_batches)
        if cfg.TRAIN.EVAL_MAX_BATCHES < 0:
            raise ValueError("DVLA_EVAL_MAX_BATCHES must be >= 0")
    eval_primary_metric = os.getenv("DVLA_EVAL_PRIMARY_METRIC")
    if eval_primary_metric is not None:
        cfg.TRAIN.EVAL_PRIMARY_METRIC = eval_primary_metric
    eval_primary_split = os.getenv("DVLA_EVAL_PRIMARY_SPLIT")
    if eval_primary_split is not None:
        cfg.TRAIN.EVAL_PRIMARY_SPLIT = eval_primary_split

    residual_env_overrides = {
        "DVLA_RESIDUAL_LAMBDA_TYPE": ("RESIDUAL_LAMBDA_TYPE", str),
        "DVLA_RESIDUAL_DRIFT_MODE": ("RESIDUAL_DRIFT_MODE", str),
        "DVLA_RESIDUAL_DRIFT_SCALE": ("RESIDUAL_DRIFT_SCALE", float),
        "DVLA_RESIDUAL_LOOKAHEAD_HORIZON_S": ("RESIDUAL_LOOKAHEAD_HORIZON_S", float),
    }
    for env_name, (cfg_key, caster) in residual_env_overrides.items():
        raw_value = os.getenv(env_name)
        if raw_value is not None:
            cfg.POLICY[cfg_key] = caster(raw_value)
    residual_axis_scale = os.getenv("DVLA_RESIDUAL_AXIS_SCALE_XYZ")
    if residual_axis_scale is not None:
        values = [
            float(item.strip())
            for item in residual_axis_scale.replace(",", " ").split()
            if item.strip()
        ]
        if len(values) != 3:
            raise ValueError("DVLA_RESIDUAL_AXIS_SCALE_XYZ must contain exactly 3 values")
        cfg.POLICY.RESIDUAL_AXIS_SCALE_XYZ = values

    eval_test_during_train = _parse_bool_env("DVLA_EVAL_TEST_DURING_TRAIN")
    if eval_test_during_train is not None:
        cfg.TRAIN.EVAL_TEST_DURING_TRAIN = eval_test_during_train

    # Print the current config
    local_rank = args.local_rank
    if local_rank == 0:
        pprint.pprint(cfg)

    if args.prefetch_only:
        prefetch_configured_dataset(cfg)
        return

    # Initialize the DDP environment
    if torch.cuda.is_available() and not args.test:
        if utils.distributed.launched_with_torchrun():
            utils.distributed.set_affinity(local_rank)
            utils.distributed.init_dist(local_rank)
        else:
            logging.info(
                "Running CUDA training in single-process mode. Use torchrun to enable DDP."
            )
            torch.cuda.set_device(local_rank)

    # Start train/test processes
    if not args.test:
        try:
            core.train(cfg)
        finally:
            utils.distributed.cleanup_dist()
    else:
        if "CKPT" not in cfg.CONST:
            logging.error("Please specify the file path of checkpoint.")
            sys.exit(2)
        try:
            core.test(cfg)
        finally:
            utils.distributed.cleanup_dist()


if __name__ == "__main__":
    # References: https://stackoverflow.com/a/53553516/1841143
    importlib.reload(logging)
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s"
    )
    main()
