#!/usr/bin/env python3
"""Merge a cumulative Qwen/InternVL Foundation+Stop adapter for evaluation.

Stop training continues the Foundation adapter in place.  Its output must be
applied to the untouched base exactly once, not on top of a merged Foundation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--foundation-adapter", type=Path, required=True)
    parser.add_argument("--stop-adapter", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check adapter lineage without loading model weights")
    args = parser.parse_args()

    base = args.base_model.resolve()
    foundation = args.foundation_adapter.resolve()
    stop = args.stop_adapter.resolve()
    provenance_path = args.provenance.resolve()
    output = args.output_dir.resolve()
    for path in (
        base / "config.json", base / "model.safetensors.index.json",
        foundation / "adapter_model.safetensors",
        stop / "adapter_config.json", stop / "adapter_model.safetensors",
        provenance_path,
    ):
        require_file(path)
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    adapter_config = json.loads((stop / "adapter_config.json").read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    foundation_hash = sha256(foundation / "adapter_model.safetensors")
    stop_hash = sha256(stop / "adapter_model.safetensors")
    checks = {
        "supported_base": base_config.get("model_type") in {"internvl", "qwen3_vl"},
        "complete_stop_training": (
            provenance.get("schema") == "sc_mrd_native_lm_complete_v1"
            and provenance.get("passed") is True
            and isinstance(provenance.get("global_step"), int)
            and provenance["global_step"] > 0
            and provenance.get("global_step") == provenance.get("expected_optimizer_steps")
        ),
        "stop_objective": provenance.get("objective") == "opd" and provenance.get("decision_kind") == "stop",
        "cumulative_adapter": (
            provenance.get("continued_b0_adapter_directly") is True
            and provenance.get("single_adapter_output") is True
        ),
        "foundation_hash": provenance.get("fixed_b0_sha256") == foundation_hash,
        "stop_hash": provenance.get("trained_adapter_sha256") == stop_hash,
        "adapter_shape": adapter_config.get("r") == 64 and adapter_config.get("lora_alpha") == 128,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stop merge preflight failed: " + ", ".join(failed))
    if args.verify_only:
        print(json.dumps({"passed": True, "checks": checks}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged model: {output}")

    # PEFT 0.19 can otherwise place adapter tensors on visible GPUs even when
    # the base was requested on CPU, causing a cross-device merge failure.
    # This process only prepares an evaluation artifact; keep it CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".stop-merge-", dir=output.parent))
    started = time.time()
    try:
        processor = AutoProcessor.from_pretrained(stop, trust_remote_code=True, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            base, dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map={"": "cpu"},
        )
        model = PeftModel.from_pretrained(
            model, stop, is_trainable=False, device_map={"": "cpu"}
        )
        model = model.merge_and_unload(safe_merge=True)
        if any("lora" in name.lower() for name, _ in model.named_parameters()):
            raise RuntimeError("LoRA parameters remain after merge")
        model.save_pretrained(partial, safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(partial)
        for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
            source = base / name
            destination = partial / name
            if source.is_file() and not destination.is_file():
                shutil.copy2(source, destination)
        index_path = partial / "model.safetensors.index.json"
        require_file(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise RuntimeError("Merged model index contains no weight shards")
        for name in shards:
            require_file(partial / name)
        manifest = {
            "schema": "cumulative_b0_stop_vllm_merged_v1",
            "passed": True,
            "base_model": str(base),
            "base_config_sha256": sha256(base / "config.json"),
            "source_adapter": str(stop),
            "source_adapter_sha256": stop_hash,
            "source_b0_adapter_sha256": foundation_hash,
            "training_provenance": str(provenance_path),
            "training_provenance_sha256": sha256(provenance_path),
            "merge_dtype": "bfloat16",
            "safe_merge": True,
            "b0_applied_exactly_once": True,
            "weights": {name: {"bytes": (partial / name).stat().st_size,
                               "sha256": sha256(partial / name)} for name in shards},
            "config_sha256": sha256(partial / "config.json"),
            "elapsed_seconds": time.time() - started,
        }
        (partial / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(partial, output)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        if partial.exists():
            shutil.rmtree(partial)


if __name__ == "__main__":
    main()
