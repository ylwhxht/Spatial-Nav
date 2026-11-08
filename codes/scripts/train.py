#!/usr/bin/env python3
"""One reproducible entry point for the Spatial-Nav training chain."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
STAGES = ("foundation", "move_no_kd", "move_opd", "compose", "stop", "merge_stop")
DATA_DIR = {
    "foundation": "stage1_foundation",
    "move_no_kd": "stage2_move_nokd",
    "move_opd": "stage2_move_opd",
    "stop": "stage2_stop",
}
EXPECTED_SPLITS = {
    "stage1_foundation": {"train": 40000, "val": 0},
    "stage2_move_nokd": {"train": 40000, "dev": 4000},
    "stage2_move_opd": {"train": 13317, "dev": 683},
    "stage2_stop": {"train": 14125, "dev": 761},
}
LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_EXCLUDE = ".*(vision_tower|vision_model|visual).*"
DEFAULT_SEEDS = {
    "qwen3-vl-8b": (20260728, 20260808, 20260807, 20260809),
    "internvl3-8b": (20261020, 20261021, 20261022, 20261023),
}


def architecture_from_config(base_model: Path) -> str:
    config_path = base_model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing base-model config: {config_path}")
    model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
    architecture = {"qwen3_vl": "qwen3-vl-8b", "internvl": "internvl3-8b"}.get(model_type)
    if architecture is None:
        raise ValueError(f"Unsupported base-model type {model_type!r} in {config_path}")
    return architecture


def accumulation(global_batch: int, microbatch: int, gpus: int) -> int:
    if global_batch % (microbatch * gpus):
        raise ValueError(
            f"Global batch {global_batch} is not divisible by microbatch {microbatch} × GPUs {gpus}"
        )
    return global_batch // (microbatch * gpus)


def training_accumulation(
    args: argparse.Namespace,
    global_batch: int,
    microbatch: int,
    stage: str,
) -> int:
    """Return accumulation for the paper recipe or an explicit 7-GPU variant."""
    if args.gpus == 7 and not args.smoke:
        if args.architecture == "internvl3-8b" or args.allow_7gpu_batch_adjustment:
            # 98 examples for foundation and 28 for specialist stages are the
            # nearest divisible batches to the paper's 96/32 on seven ranks.
            adjusted_batch = 98 if stage == "foundation" else 28
            return accumulation(adjusted_batch, microbatch, args.gpus)
        raise ValueError(
            "Qwen3-VL on 7 GPUs requires --allow-7gpu-batch-adjustment "
            "(effective batches 98/28); use 8 GPUs for the paper recipe"
        )
    return accumulation(global_batch, microbatch, args.gpus)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing exported dataset: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "spatial-nav-data-v2" or not manifest.get("passed")
            or manifest.get("image_contract") != "original_bytes"
            or not manifest.get("all_image_hashes_verified")):
        raise RuntimeError("Dataset must be a v2 original-bytes export; the resized 512x288 bundle is not equivalent")
    if set(manifest.get("stages", {})) != set(EXPECTED_SPLITS):
        raise RuntimeError("Dataset manifest has missing or unexpected training stages")
    if manifest.get("training_rows") != sum(
        count for splits in EXPECTED_SPLITS.values() for name, count in splits.items() if name == "train"
    ):
        raise RuntimeError("Dataset manifest has an unexpected total training-row count")

    image_manifest_path = data_root / "images_manifest.jsonl"
    if (not image_manifest_path.is_file()
            or sha256(image_manifest_path) != manifest.get("images_manifest_sha256")):
        raise RuntimeError("Original-image manifest missing or hash mismatch")
    image_records = {}
    for line in image_manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        filename = record["file"]
        if (filename in image_records or Path(filename).name != filename
                or Path(filename).suffix not in {".jpg", ".png"}
                or Path(filename).stem != record["sha256"]):
            raise RuntimeError(f"Invalid original-image manifest entry: {filename}")
        image_records[filename] = record
    expected_images: set[str] = set()
    move_sample_ids: set[str] = set()
    for stage, expected_splits in EXPECTED_SPLITS.items():
        files = manifest["stages"][stage]
        if set(files) != set(expected_splits):
            raise RuntimeError(f"Unexpected split list for {stage}: {sorted(files)}")
        for split, expected_rows in expected_splits.items():
            record = files[split]
            if record.get("rows") != expected_rows:
                raise RuntimeError(f"Unexpected row count for {stage}/{split}: {record.get('rows')}")
            path = data_root / stage / f"{split}.jsonl"
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"Dataset hash mismatch: {path}")

            with path.open(encoding="utf-8") as handle:
                rows = 0
                resolutions = Counter()
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    image_hash = row.get("image_sha256")
                    if (not isinstance(image_hash, str) or len(image_hash) != 64
                            or any(char not in "0123456789abcdef" for char in image_hash)):
                        raise RuntimeError(f"Invalid image SHA-256 in {path} at row {rows + 1}")
                    filename = Path(row.get("image_path", "")).name
                    expected_path = f"../images/{filename}"
                    if row.get("image_path") != expected_path:
                        raise RuntimeError(f"Invalid relative image reference in {path} at row {rows + 1}")
                    info = image_records.get(filename)
                    if info is None or info["sha256"] != image_hash:
                        raise RuntimeError(f"Image missing from original-byte manifest: {filename}")
                    resolutions[f"{info['width']}x{info['height']}"] += 1
                    expected_images.add(filename)
                    if stage == "stage2_move_nokd" and split == "train":
                        move_sample_ids.add(str(row["sample_id"]))
                    rows += 1
                if rows != expected_rows:
                    raise RuntimeError(f"Row count mismatch for {path}: {rows} != {expected_rows}")
                if dict(resolutions) != record.get("resolution_counts"):
                    raise RuntimeError(f"Original-resolution counts differ for {path}")

    image_dir = data_root / "images"
    image_paths = list(image_dir.iterdir())
    actual_images = {path.name for path in image_paths}
    if (len(actual_images) != manifest.get("unique_images")
            or actual_images != expected_images or actual_images != set(image_records)):
        raise RuntimeError("Dataset image count mismatch")
    for image_path in image_paths:
        if (not image_path.is_file() or sha256(image_path) != image_path.stem
                or image_path.stat().st_size != image_records[image_path.name]["bytes"]):
            raise RuntimeError(f"Dataset image hash mismatch: {image_path}")
    if dict(Counter(f"{r['width']}x{r['height']}" for r in image_records.values())) != manifest.get("image_resolution_counts"):
        raise RuntimeError("Original-image resolution inventory mismatch")

    evidence_root = data_root / "move_evidence"
    evidence_manifest_path = evidence_root / "manifest.json"
    if (not evidence_manifest_path.is_file()
            or sha256(evidence_manifest_path) != manifest.get("evidence_manifest_sha256")):
        raise RuntimeError("Move evidence manifest missing or hash mismatch")
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    move_train_path = data_root / "stage2_move_nokd" / "train.jsonl"
    if (not evidence_manifest.get("passed") or not evidence_manifest.get("training_only")
            or evidence_manifest.get("runtime_loads_cache") is not False
            or evidence_manifest.get("train_sha256") != sha256(move_train_path)):
        raise RuntimeError("Move evidence cache does not match the audited training data")
    sample_ids_path = evidence_root / "sample_ids.json"
    if (not sample_ids_path.is_file()
            or sha256(sample_ids_path) != evidence_manifest.get("sample_ids_sha256")):
        raise RuntimeError("Move evidence sample ID file missing or hash mismatch")
    evidence_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if (len(evidence_ids) != len(set(evidence_ids))
            or not set(map(str, evidence_ids)).issubset(move_sample_ids)):
        raise RuntimeError("Move evidence sample IDs do not match Move no-KD training rows")
    shards = evidence_manifest.get("shards", {})
    shard_hashes = evidence_manifest.get("shard_sha256", {})
    if not shards or set(shards) != set(shard_hashes):
        raise RuntimeError("Move evidence shard manifest is incomplete")
    for name, expected_rows in shards.items():
        if Path(name).name != name or expected_rows < 0:
            raise RuntimeError(f"Invalid Move evidence shard entry: {name}")
        path = evidence_root / name
        if not path.is_file() or sha256(path) != shard_hashes[name]:
            raise RuntimeError(f"Move evidence shard missing or hash mismatch: {name}")
    if sum(shards.values()) != evidence_manifest.get("evidence_items"):
        raise RuntimeError("Move evidence shard counts do not match its manifest")

    # Both specialized trainers require local audit markers. Only issue them
    # after the entire export, images and evidence have been verified.
    for stage in ("stage2_move_nokd", "stage2_move_opd", "stage2_stop"):
        audit_path = data_root / stage / "dataset_audit.json"
        audit = {
            "passed": True,
            "source": "manifest.json",
            "manifest_sha256": sha256(manifest_path),
            "stage": stage,
            "train_sha256": manifest["stages"][stage]["train"]["sha256"],
        }
        if not audit_path.is_file() or json.loads(audit_path.read_text()) != audit:
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return manifest


def adapter_file(output_root: Path, stage: str) -> Path:
    return output_root / stage / "final_adapter" / "adapter_model.safetensors"


def adapter_dir(output_root: Path, stage: str) -> Path:
    return adapter_file(output_root, stage).parent


def stage_marker(output_root: Path, stage: str) -> Path:
    if stage == "merge_stop":
        return output_root / "stop_merged" / "model.safetensors.index.json"
    return adapter_file(output_root, stage)


def verify_run_identity(args: argparse.Namespace) -> None:
    """Do not silently reuse smoke or unrelated completed training stages."""
    output = args.output.resolve()
    record_path = output / "training_run.json"
    identity = {
        "schema": "spatial-nav-training-run-v1",
        "architecture": args.architecture,
        "base_model": str(args.base_model.resolve()),
        "base_config_sha256": sha256(args.base_model / "config.json"),
        "data_manifest_sha256": sha256(args.data / "manifest.json"),
        "smoke": args.smoke,
        "gpus": args.gpus,
        "seeds": {name: getattr(args, name) for name in
                  ("foundation_seed", "move_seed", "move_opd_seed", "stop_seed")},
        "recipe_sha256": {name: sha256(SCRIPTS / name) for name in
                          ("train.py", "train_cold_start_vision_sft.py",
                           "train_sc_esr_move.py", "train_sc_mrd_sft.py",
                           "compose_sc_mrd_lora.py", "merge_stop.py")},
    }
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        changed = sorted(key for key in set(previous) | set(identity)
                         if previous.get(key) != identity.get(key))
        if changed:
            raise RuntimeError(
                f"Training output identity differs ({', '.join(changed)}): {output}. "
                "Use a new output directory; completed smoke stages cannot become a full run."
            )
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Existing output lacks training_run.json: {output}. "
            "Use a new output directory; do not assume existing adapters match this run."
        )
    output.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(identity, handle, indent=2)
        handle.write("\n")


def require_adapter(output_root: Path, stage: str, dry_run: bool) -> str:
    path = adapter_file(output_root, stage)
    if path.is_file():
        return sha256(path)
    if dry_run:
        return "0" * 64
    raise FileNotFoundError(f"Required {stage} adapter missing: {path}")


def verify_smoke_update(output_root: Path, stage: str) -> dict:
    """Reject a saved but unmodified adapter (e.g. a zero-LR warmup-only run)."""
    import torch
    from safetensors.torch import load_file

    weights = load_file(str(adapter_file(output_root, stage)), device="cpu")
    if not weights or not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise RuntimeError(f"Smoke {stage}: empty or non-finite adapter")
    lora_b = {key: value for key, value in weights.items() if "lora_B" in key}
    if not lora_b or not any(torch.count_nonzero(value).item() for value in lora_b.values()):
        raise RuntimeError(f"Smoke {stage}: LoRA B remains zero; no effective weight update")
    parent = "foundation" if stage in ("move_no_kd", "stop") else None
    if parent:
        anchor = load_file(str(adapter_file(output_root, parent)), device="cpu")
        if set(weights) != set(anchor):
            raise RuntimeError(f"Smoke {stage}: adapter keys differ from parent {parent}")
        changed = sum(not torch.equal(value, anchor[key]) for key, value in weights.items())
        if not changed:
            raise RuntimeError(f"Smoke {stage}: all adapter tensors equal parent {parent}")
    else:
        changed = sum(bool(torch.count_nonzero(value).item()) for value in lora_b.values())
    return {"stage": stage, "parent": parent, "changed_tensors": changed,
            "nonzero_lora_b_tensors": sum(bool(torch.count_nonzero(v).item()) for v in lora_b.values())}


def training_command(args: argparse.Namespace, stage: str) -> list[str]:
    data = args.data.resolve()
    output = args.output.resolve()
    common = ["--model-path", str(args.base_model), "--output-dir", str(output / stage)]
    lora = [
        "--lora-r", "64", "--lora-alpha", "128", "--lora-dropout", "0.05",
        "--lora-target-modules", LORA_MODULES,
        "--lora-exclude-modules", LORA_EXCLUDE, "--forbid-vision-lora",
    ]
    config_path = args.base_model / "config.json"
    image_token_id = (
        str(json.loads(config_path.read_text(encoding="utf-8"))["image_token_id"])
        if config_path.is_file() else "151655"
    )
    is_internvl = args.architecture == "internvl3-8b"
    num_workers = "4" if is_internvl else ("2" if stage == "move_no_kd" else "0")
    foundation_micro = 2 if is_internvl and not args.smoke else 1
    move_micro = 1 if args.smoke else (1 if is_internvl else 4)
    opd_micro = 1
    accum = 1 if args.smoke else training_accumulation(args, 32, opd_micro, stage)
    if stage == "foundation":
        stage_args = common + [
            "--require-fresh-base-model-path", str(args.base_model),
            "--data-dir", str(data / DATA_DIR[stage]), "--finetune-mode", "lora",
            *lora, "--epochs", "2", "--per-device-batch-size", str(foundation_micro),
            "--grad-accum-steps", str(1 if args.smoke else training_accumulation(args, 96, foundation_micro, "foundation")),
            "--learning-rate", "1e-5", "--lr-scheduler", "cosine",
            "--warmup-ratio", "0.05", "--max-length", "4096",
            "--num-workers", num_workers,
            "--seed", str(args.foundation_seed), "--normalize-loss-per-row",
            "--no-system-prompt", "--text-before-image", "--disable-eval",
            "--save-every-steps", "10000", "--eval-every-steps", "10000",
            "--log-every-steps", "5", "--train-loss-every-steps", "5",
            "--wandb-mode", "disabled",
        ]
        if not is_internvl and not args.smoke:
            stage_args.append("--no-gradient-checkpointing")
        script = "train_cold_start_vision_sft.py"
    elif stage == "move_no_kd":
        stage_args = common + [
            "--b0-adapter", str(adapter_dir(output, "foundation")),
            "--expected-b0-sha256", require_adapter(output, "foundation", args.dry_run),
            "--data-dir", str(data / DATA_DIR[stage]),
            "--evidence-cache", str(data / "move_evidence"),
            "--expected-train-rows", "40000", "--expected-dev-rows", "4000",
            "--epochs", "1", "--per-device-batch-size", str(move_micro),
            "--grad-accum-steps", str(1 if args.smoke else training_accumulation(args, 32, move_micro, "move_no_kd")),
            "--num-workers", num_workers,
            "--max-length", "4096" if is_internvl else "1536",
            "--image-token-id", image_token_id, "--seed", str(args.move_seed),
            "--backbone-learning-rate", "1e-5", "--residual-learning-rate", "2e-4",
            "--gate-learning-rate", "5e-3", "--warmup-ratio", "0.03",
            "--route-weight", "0", "--route-ranking-weight", "1",
            "--route-ranking-margin", "0.2", "--ability-pos-weight-mode", "none",
            "--route-class-balanced", "--evidence-weight", "0",
            "--residual-action-weight", "0", "--utility-weight", "0",
            "--gain-vs-off-weight", "0", "--gain-vs-off-margin", "0.05",
            "--preservation-weight", "0", "--candidate-coupling-weight", "0",
            "--need-weight", "0", "--need-margin-threshold", "0.10",
            "--counterfactual-every", "2", "--preservation-every", "2",
            "--gate-warmup-ratio", "0.05", "--candidate-conditioned",
            "--independent-ability-gates", "--adaptive-need-gate",
            "--runtime-aligned-injection", "--native-lm-weight-semantics",
            "historical_sft", "--auxiliary-gradient-scope", "sc_esr",
            "--historical-distributed-course", "--diagnostic-split", "train",
            "--diagnostic-action-only", "--diagnostic-samples", "512",
            "--max-dev-samples", "0", "--gradient-separated-native-lm",
            "--log-every-steps", "5",
        ]
        script = "train_sc_esr_move.py"
    elif stage in ("move_opd", "stop"):
        is_stop = stage == "stop"
        parent = "foundation" if is_stop else "move_no_kd"
        stage_args = common + [
            "--b0-adapter", str(adapter_dir(output, parent)),
            "--expected-b0-sha256", require_adapter(output, parent, args.dry_run),
            "--expected-anchor-r", "64", "--expected-anchor-alpha", "128",
            "--data-dir", str(data / DATA_DIR[stage]),
            "--objective", "opd", "--decision-kind", "stop" if is_stop else "move",
            "--expected-train-rows", "14125" if is_stop else "13317",
            "--expected-dev-rows", "761" if is_stop else "683",
            "--epochs", "1", "--per-device-batch-size", str(opd_micro),
            "--grad-accum-steps", str(accum),
            "--learning-rate", "1e-6" if is_stop else "5e-7",
            "--warmup-ratio", "0.05", "--max-length", "4096" if is_internvl else "2048",
            "--num-workers", num_workers,
            "--seed", str(args.stop_seed if is_stop else args.move_opd_seed),
            *lora, "--direct-weight", "0.10" if is_stop else "1.0",
            "--direct-action-token-weight", "4", "--kd-weight", "0.25" if is_stop else "0.15",
            "--kd-positive-weight", "0.5316194474427967" if is_stop else "1",
            "--kd-negative-weight", "1.4670730698036616" if is_stop else "1",
            "--temperature", "2", "--kd-warmup-ratio", "0.05" if is_stop else "0.2",
            "--kd-require-teacher-correct", "--kd-min-teacher-gold-margin",
            "0.10" if is_stop else "2", "--reason-weight", "0",
            "--reasoned-action-weight", "0", "--counterfactual-weight", "0",
            "--max-dev-samples", "256", "--max-train-diagnostic-samples",
            "256" if is_stop else "128", "--log-every-steps", "5" if is_stop else "1",
        ]
        if is_stop:
            stage_args.extend([
                "--continue-b0-adapter-directly", "--direct-negative-weight",
                "1.9507101763824326", "--direct-positive-weight", "0.2541345533333914",
            ])
        if not is_internvl and not args.smoke:
            stage_args.append("--no-gradient-checkpointing")
        script = "train_sc_mrd_sft.py"
    else:
        raise ValueError(stage)
    if args.smoke:
        # The first scheduled optimizer step has zero LR during warmup.
        # Two steps exercise an actual parameter update as well as backward.
        stage_args += ["--max-steps", "2"]
        for option in ("--diagnostic-samples", "--max-train-diagnostic-samples", "--max-dev-samples"):
            if option in stage_args:
                index = stage_args.index(option) + 1
                stage_args[index] = str(min(16, int(stage_args[index])))
    return [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={args.gpus}", str(SCRIPTS / script), *stage_args,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=None, help="Local Hugging Face base model")
    parser.add_argument("--architecture", choices=("auto", *DEFAULT_SEEDS), default="auto",
                        help="Infer from the base-model config by default")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--gpus", type=int, default=4,
                        help="Number of training GPUs (default: 4; set explicitly for another host)")
    parser.add_argument("--gpu-ids", default="", help="Optional CUDA_VISIBLE_DEVICES list")
    parser.add_argument("--foundation-seed", type=int, default=None)
    parser.add_argument("--move-seed", type=int, default=None)
    parser.add_argument("--move-opd-seed", type=int, default=None)
    parser.add_argument("--stop-seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Two optimizer steps per stage (past zero-LR warmup)")
    parser.add_argument(
        "--allow-7gpu-batch-adjustment", action="store_true",
        help="Use effective batches 98/28 on seven GPUs (not the exact paper batch)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--verify-only", action="store_true", help="Validate exported data and exit")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    if args.gpu_ids and len(args.gpu_ids.split(",")) != args.gpus:
        parser.error("--gpu-ids length must equal --gpus")
    if args.verify_only:
        manifest = verify_data(args.data.resolve())
        print(json.dumps({"passed": True, "data": str(args.data.resolve()), "stages": list(manifest["stages"])}, indent=2))
        return
    if args.base_model is None:
        parser.error("--base-model is required unless --verify-only is set")
    try:
        inferred_architecture = architecture_from_config(args.base_model)
    except ValueError as exc:
        parser.error(str(exc))
    if args.architecture != "auto" and args.architecture != inferred_architecture:
        parser.error(f"--architecture {args.architecture} disagrees with base model {inferred_architecture}")
    args.architecture = inferred_architecture
    defaults = DEFAULT_SEEDS[args.architecture]
    for name, default in zip(("foundation_seed", "move_seed", "move_opd_seed", "stop_seed"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not args.dry_run:
        if not args.base_model.is_dir():
            parser.error(f"Base model not found: {args.base_model}")
        manifest = verify_data(args.data.resolve())
    selected = STAGES if args.stage == "all" else (args.stage,)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "codes" / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "4"
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if not args.dry_run:
        verify_run_identity(args)
    for stage in selected:
        marker = stage_marker(args.output.resolve(), stage)
        if marker.is_file() and not args.dry_run:
            if args.smoke and stage in DATA_DIR:
                print(json.dumps(verify_smoke_update(args.output.resolve(), stage)), flush=True)
            print(f"SKIP {stage}: {marker}", flush=True)
            continue
        if stage == "merge_stop":
            command = [
                sys.executable, str(SCRIPTS / "merge_stop.py"),
                "--base-model", str(args.base_model.resolve()),
                "--foundation-adapter", str(adapter_dir(args.output.resolve(), "foundation")),
                "--stop-adapter", str(adapter_dir(args.output.resolve(), "stop")),
                "--provenance", str(args.output.resolve() / "stop" / "pipeline_complete.json"),
                "--output-dir", str(args.output.resolve() / "stop_merged"),
            ]
        elif stage == "compose":
            command = [
                sys.executable, str(SCRIPTS / "compose_sc_mrd_lora.py"),
                "--b0-adapter", str(adapter_dir(args.output.resolve(), "move_no_kd")),
                "--delta-adapter", str(adapter_dir(args.output.resolve(), "move_opd")),
                "--delta-scale", "1", "--output-dir", str(adapter_dir(args.output.resolve(), stage)),
            ]
        else:
            command = training_command(args, stage)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            env["OMP_NUM_THREADS"] = "1" if stage == "move_no_kd" else "4"
            stage_seed = {
                "foundation": args.foundation_seed,
                "move_no_kd": args.move_seed,
                "move_opd": args.move_opd_seed,
                "stop": args.stop_seed,
            }.get(stage)
            if stage_seed is not None:
                env["PYTHONHASHSEED"] = str(stage_seed)
            stage_output = args.output / ("stop_merged" if stage == "merge_stop" else stage)
            if stage_output.exists() and not marker.is_file():
                raise RuntimeError(f"Incomplete output exists at {stage_output}; move it aside before retrying")
            subprocess.run(command, env=env, check=True)
            if not marker.is_file():
                raise RuntimeError(f"Stage {stage} finished without its output marker: {marker}")
            if args.smoke and stage in DATA_DIR:
                print(json.dumps(verify_smoke_update(args.output.resolve(), stage)), flush=True)


if __name__ == "__main__":
    main()
