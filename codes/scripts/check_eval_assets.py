#!/usr/bin/env python3
"""Check ObjectNav validation scenes before starting GPU model servers."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import validate_eval_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hm3d_v1", "hm3d_v2", "mp3d"), required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--objectnav-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_eval_assets(args.dataset, args.scene_root, args.objectnav_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Evaluation assets complete: {args.dataset}")


if __name__ == "__main__":
    main()
