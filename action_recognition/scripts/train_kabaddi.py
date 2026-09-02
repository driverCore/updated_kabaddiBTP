"""
train_kabaddi.py
================
Launcher for ST-GCN Kabaddi training.
Patches mmengine's mmcv_full_available() to return False so the runner
skips trying to load the broken mmcv._ext DLL on Windows.
ST-GCN does NOT need any custom CUDA ops — only pure PyTorch.
"""

import sys

# ── Patch mmcv_full_available BEFORE mmengine.runner is imported ─────────────
# This must happen before any mmengine.runner import triggers it.
import mmengine.utils.dl_utils.misc as _misc
_misc.mmcv_full_available = lambda: False

# Now safe to import runner
import os
import argparse
import mmengine
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-GCN on Kabaddi")
    parser.add_argument("config",      help="Config file path")
    parser.add_argument("--work-dir",  help="Override work dir", default=None)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument(
        "--cfg-options", nargs="+", action=mmengine.DictAction,
        help="override config entries, e.g. --cfg-options model.cls_head.num_classes=12")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = mmengine.Config.fromfile(args.config)
    cfg.launcher = "none"

    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.resume:
        cfg.resume = True

    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
