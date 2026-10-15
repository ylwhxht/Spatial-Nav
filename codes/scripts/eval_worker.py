#!/usr/bin/env python3
"""Run one deterministic shard of the bundled ObjectNav navigation runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hm3d_v1", "hm3d_v2", "mp3d"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--index-list", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--image-res", type=int, default=512)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if not runtime.is_dir():
        raise FileNotFoundError(runtime)
    sys.path.insert(0, str(runtime))
    from env import ObjectNavEnv  # type: ignore

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    env_cfg = config["env_cfg"]
    env_cfg.update({
        "name": args.name,
        "dataset": args.dataset,
        "split": "val",
        "instance": args.worker,
        "instances": args.workers,
        "num_episodes": len(json.loads(args.index_list.read_text(encoding="utf-8"))),
        "max_steps": args.max_steps,
        "success_threshold": 1.0,
        "parallel": False,
    })
    config.setdefault("sim_cfg", {})["gpu_device_id"] = args.gpu_device_id
    config["agent_cfg"]["vlm_cfg"]["model_kwargs"]["max_image_res"] = args.image_res
    os.environ["VLMNAV_EPISODE_INDEX_LIST_PATH"] = str(args.index_list.resolve())
    ObjectNavEnv(cfg=config).run_experiment()


if __name__ == "__main__":
    main()
