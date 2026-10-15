#!/usr/bin/env python3
"""Install the paper's exact MP3D validation navmeshes beside scene files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from datetime import datetime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing the 11 official <scene>/<scene>.navmesh files")
    parser.add_argument("--scene-root", type=Path, required=True,
                        help="MP3D validation scene root used by evaluate.py")
    parser.add_argument("--replace", action="store_true",
                        help="Back up a conflicting destination navmesh before replacing it")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_root, scene_root = args.source_root.resolve(), args.scene_root.resolve()
    if not source_root.is_dir() or not scene_root.is_dir():
        parser.error("Both --source-root and --scene-root must be existing directories")
    sources = sorted(source_root.glob("*/**/*.navmesh"))
    if len(sources) != 11:
        parser.error(f"Expected the 11 paper MP3D navmeshes; found {len(sources)} under {source_root}")

    copied = skipped = 0
    for source in sources:
        destination = scene_root / source.parent.name / source.name
        if not (scene_root / source.parent.name).is_dir():
            parser.error(f"Scene directory is missing: {destination.parent}")
        if destination.is_file() and sha256(destination) == sha256(source):
            skipped += 1
            continue
        if destination.exists() and not args.replace:
            parser.error(f"Conflicting file exists (use --replace to back it up): {destination}")
        print(f"COPY {source} -> {destination}")
        if args.dry_run:
            copied += 1
            continue
        if destination.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = destination.with_name(destination.name + f".backup-{stamp}")
            suffix = 1
            while backup.exists():
                backup = destination.with_name(destination.name + f".backup-{stamp}-{suffix}")
                suffix += 1
            shutil.move(str(destination), str(backup))
            print(f"BACKUP {backup}")
        shutil.copy2(source, destination)
        if sha256(destination) != sha256(source):
            raise IOError(f"Copied file hash mismatch: {destination}")
        copied += 1
    print(f"Done: copied={copied}, already-correct={skipped}, required={len(sources)}")


if __name__ == "__main__":
    main()
