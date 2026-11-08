"""Export the audited native-resolution training data without re-encoding images.

This maintainer utility accepts source locations explicitly. The exported dataset
has no dependency on the source repository. Existing outputs are never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

STAGES = {
    "foundation": ("stage1_foundation", {"train": 40000, "val": 0}),
    "move_no_kd": ("stage2_move_nokd", {"train": 40000, "dev": 4000}),
    "move_opd": ("stage2_move_opd", {"train": 13317, "dev": 683}),
    "stop_v23": ("stage2_stop", {"train": 14125, "dev": 761}),
}
DIRECT_MOVE_ROWS = 10000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def portable_metadata(value):
    # Remove machine-local provenance only; prompts, responses, IDs, labels and
    # all loss/curriculum fields remain byte-for-byte equivalent as JSON values.
    if isinstance(value, dict):
        return {k: portable_metadata(v) for k, v in value.items()
                if not (isinstance(v, str) and v.startswith(("/home/", "/dev/shm/", "/tmp/")))}
    if isinstance(value, list):
        return [portable_metadata(v) for v in value]
    return value


def export(source: Path, cache: Path, output: Path, workers: int, resolvers=()):
    from PIL import Image

    contract_path = source / "manifest.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    inventory_path = source / "original_images.sha256.jsonl"
    if (contract.get("schema") != "raw_training_contract_restored_v1"
            or contract.get("passed") is not True
            or contract.get("uses_resized_512_dataset") is not False
            or contract.get("labels_and_row_order_unchanged") is not True
            or sha256(inventory_path) != contract.get("image_manifest_sha256")):
        raise ValueError("Source must certify original image bytes; resized exports are not equivalent")
    inventory = {}
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["path"] in inventory:
            raise ValueError(f"Duplicate original-image inventory entry: {item['path']}")
        inventory[item["path"]] = item
    if len(inventory) != contract.get("unique_images"):
        raise ValueError("Original-image inventory count mismatch")
    if sorted(sha256(p) for p in resolvers) != sorted(contract.get("resolver_files", {}).values()):
        raise ValueError("Supply the exact --resolver JSONL files hashed by the raw contract")
    resolved = {}
    for path in resolvers:
        with path.open(encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row.get("sample_id") or "")
                if not sid:
                    continue
                image = os.path.abspath(os.path.expanduser(row["image_path"]))
                if resolved.setdefault(sid, image) != image:
                    raise ValueError(f"Ambiguous original image for sample: {sid}")
    output.mkdir(parents=True, exist_ok=False)
    image_dir = output / "images"
    image_dir.mkdir()
    images = {}
    stages = {}
    move_ids = []
    for source_name, (target_name, expected_splits) in STAGES.items():
        target = output / target_name
        target.mkdir()
        splits = {}
        for split, expected in expected_splits.items():
            src = source / "stages" / source_name / f"{split}.jsonl"
            source_record = contract["stages"][source_name]["splits"][f"{split}.jsonl"]
            if sha256(src) != source_record["sha256"] or source_record["rows"] != expected:
                raise ValueError(f"Source split no longer matches the raw contract: {src}")
            dst = target / f"{split}.jsonl"
            count, ids, direct_count = 0, set(), 0
            resolutions = Counter()
            with src.open(encoding="utf-8") as reader, dst.open("w", encoding="utf-8") as writer:
                for line in reader:
                    if not line.strip():
                        continue
                    original = json.loads(line)
                    sid = original["sample_id"]
                    if sid in ids:
                        raise ValueError(f"duplicate sample: {src}: {sid}")
                    ids.add(sid)
                    image = Path(original["image_path"])
                    # Match the historical stager: tmpfs paths were ephemeral;
                    # resolve them by exact sample ID using the pinned JSONL,
                    # never by filename similarity or a resized replacement.
                    if not image.is_absolute() or str(image).startswith("/dev/shm/"):
                        if sid not in resolved:
                            raise ValueError(f"Original image requires an audited resolver: {sid}")
                        image = Path(resolved[sid])
                    else:
                        image = Path(os.path.abspath(image))
                    info = inventory[str(image)]
                    digest = info["sha256"]
                    if original.get("image_sha256", digest) != digest:
                        raise ValueError(f"Row/image inventory hash mismatch: {sid}")
                    suffix = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png"}.get(image.suffix.lower())
                    if suffix is None:
                        raise ValueError(f"Unsupported original image format: {image}")
                    filename = f"{digest}{suffix}"
                    images.setdefault(filename, info)
                    resolutions[f"{info['width']}x{info['height']}"] += 1
                    row = portable_metadata(original)
                    row["image_path"] = f"../images/{filename}"
                    row["image_sha256"] = digest
                    writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                    if source_name == "move_no_kd" and split == "train":
                        move_ids.append(sid)
                        direct_count += int(row.get("mix_role") == "direct_move_only_policy")
            if count != expected:
                raise ValueError(f"{src}: {count} rows, expected {expected}")
            if source_name == "move_no_kd" and split == "train" and direct_count != DIRECT_MOVE_ROWS:
                raise ValueError(f"Move must contain {DIRECT_MOVE_ROWS} direct decisions, got {direct_count}")
            if dict(resolutions) != source_record["resolution_counts"]:
                raise ValueError(f"Original resolution counts differ from the contract: {src}")
            splits[split] = {"rows": count, "sha256": sha256(dst), "source_sha256": sha256(src),
                             "resolution_counts": dict(sorted(resolutions.items()))}
        if "dev" in splits:
            shutil.copyfile(target / "dev.jsonl", target / "val.jsonl")
        write_json(target / "manifest.json", {"schema": "spatial-nav-data-v2", "splits": splits})
        stages[target_name] = splits
        print(f"exported {target_name}: {splits}", flush=True)

    def copy_image(item):
        filename, info = item
        src = Path(info["path"])
        digest = info["sha256"]
        dst = image_dir / filename
        if sha256(src) != digest:
            raise ValueError(f"source image checksum mismatch: {src}")
        with Image.open(src) as im:
            expected_format = "PNG" if dst.suffix == ".png" else "JPEG"
            if (im.size != (info["width"], info["height"]) or im.mode != info["mode"]
                    or im.format != expected_format or src.stat().st_size != info["bytes"]):
                raise ValueError(f"Original image metadata changed: {src}")
            im.verify()
        shutil.copyfile(src, dst)
        if sha256(dst) != digest:
            raise ValueError(f"exported image checksum mismatch: {dst}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(copy_image, images.items()), 1):
            if i % 2000 == 0:
                print(f"verified/copied images {i}/{len(images)}", flush=True)

    manifest = json.loads((cache / "manifest.json").read_text())
    cache_ids = json.loads((cache / "sample_ids.json").read_text())
    if not manifest.get("passed") or not manifest.get("training_only") or manifest.get("runtime_loads_cache") is not False:
        raise ValueError("evidence cache must be audited and training-only")
    if len(cache_ids) != len(set(cache_ids)) or not set(cache_ids).issubset(move_ids):
        raise ValueError("evidence sample IDs do not match Move training data")
    target_cache = output / "move_evidence"
    target_cache.mkdir()
    for name, digest in {"sample_ids.json": manifest["sample_ids_sha256"], **manifest["shard_sha256"]}.items():
        if Path(name).name != name or sha256(cache / name) != digest:
            raise ValueError(f"invalid evidence file: {name}")
        shutil.copyfile(cache / name, target_cache / name)
    manifest = portable_metadata(manifest)
    manifest["train_sha256"] = sha256(output / "stage2_move_nokd" / "train.jsonl")
    manifest["data_dir"] = "../stage2_move_nokd"
    manifest["source_cache_manifest_sha256"] = sha256(cache / "manifest.json")
    write_json(target_cache / "manifest.json", manifest)
    with (output / "images_manifest.jsonl").open("w", encoding="utf-8") as writer:
        for filename, info in sorted(images.items()):
            record = {k: v for k, v in info.items() if k != "path"}
            record["file"] = filename
            writer.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    write_json(output / "manifest.json", {
        "schema": "spatial-nav-data-v2", "passed": True,
        "image_contract": "original_bytes", "unique_images": len(images),
        "source_contract_sha256": sha256(contract_path),
        "source_image_inventory_sha256": sha256(inventory_path),
        "source_resolver_sha256": sorted(sha256(p) for p in resolvers),
        "images_manifest_sha256": sha256(output / "images_manifest.jsonl"),
        "image_resolution_counts": dict(sorted(Counter(
            f"{info['width']}x{info['height']}" for info in images.values()).items())),
        "image_format_counts": dict(sorted(Counter(Path(name).suffix for name in images).items())),
        "all_image_hashes_verified": True, "stages": stages,
        "training_rows": sum(s["train"]["rows"] for s in stages.values()),
        "evidence_manifest_sha256": sha256(target_cache / "manifest.json"),
    })
    print(f"Export complete: {output}; {len(images)} verified images", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="Audited raw contract: manifest.json, original_images.sha256.jsonl, stages/")
    p.add_argument("--evidence-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--resolver", type=Path, action="append", default=[],
                   help="Persistent sample-ID/image JSONL pinned by the raw contract; repeat as needed")
    a = p.parse_args()
    if a.workers < 1:
        p.error("--workers must be positive")
    export(a.source.resolve(), a.evidence_cache.resolve(), a.output.resolve(), a.workers,
           [p.resolve() for p in a.resolver])
