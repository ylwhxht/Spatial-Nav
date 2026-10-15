#!/usr/bin/env python3
"""Compose frozen B0 and an SC-MRD delta into one exact additive LoRA.

Both source adapters use the same target modules and the same LoRA scaling
(``alpha / r == 2``).  Concatenating A on the rank dimension and B on its rank
dimension therefore represents exactly

    scale * B_b0 @ A_b0 + scale * B_delta @ A_delta

with one rank-128 adapter.  This is a deployment representation only: neither
the Qwen weights nor either source adapter is modified or merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-adapter", required=True)
    parser.add_argument("--delta-adapter", required=True)
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="Multiply the deployed delta contribution without changing the anchor.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as source:
        return {
            key: source.get_tensor(key).contiguous()
            for key in source.keys()
        }


def module_prefix(key: str) -> tuple[str, str]:
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")], "A"
    if key.endswith(".lora_B.weight"):
        return key[: -len(".lora_B.weight")], "B"
    raise ValueError(f"non-LoRA tensor in source adapter: {key}")


def validate_config(config: dict[str, Any], label: str) -> tuple[int, float]:
    if config.get("peft_type") != "LORA" or config.get("bias") != "none":
        raise ValueError(f"{label} is not a plain bias-free LoRA")
    if config.get("use_dora") or config.get("use_rslora"):
        raise ValueError(f"{label} uses an unsupported scaled LoRA variant")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{label} uses per-module rank/alpha patterns")
    rank = int(config.get("r", 0))
    alpha = float(config.get("lora_alpha", 0))
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"{label} has invalid rank/scaling")
    return rank, alpha / rank


def main() -> None:
    args = arguments()
    if not math.isfinite(args.delta_scale) or args.delta_scale < 0:
        raise ValueError("--delta-scale must be a finite non-negative value")
    b0 = Path(args.b0_adapter).resolve()
    delta = Path(args.delta_adapter).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite composed adapter: {output}")

    b0_config_path = b0 / "adapter_config.json"
    delta_config_path = delta / "adapter_config.json"
    b0_weights = b0 / "adapter_model.safetensors"
    delta_weights = delta / "adapter_model.safetensors"
    for path in (
        b0_config_path,
        delta_config_path,
        b0_weights,
        delta_weights,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    b0_config = load_json(b0_config_path)
    delta_config = load_json(delta_config_path)
    b0_rank, b0_scale = validate_config(b0_config, "B0")
    delta_rank, delta_scale = validate_config(delta_config, "delta")
    if abs(b0_scale - delta_scale) > 1e-12:
        raise ValueError(
            f"source LoRA scales differ: B0={b0_scale}, delta={delta_scale}"
        )
    if set(b0_config.get("target_modules") or []) != set(
        delta_config.get("target_modules") or []
    ):
        raise ValueError("source target-module sets differ")

    left = load_tensors(b0_weights)
    right = load_tensors(delta_weights)
    if set(left) != set(right):
        raise ValueError("source adapter tensor keys differ")
    composed: dict[str, torch.Tensor] = {}
    module_counts = {"A": 0, "B": 0}
    for key in sorted(left):
        _, kind = module_prefix(key)
        a = left[key]
        b = right[key]
        if kind == "A":
            if a.shape[0] != b0_rank or b.shape[0] != delta_rank:
                raise ValueError(f"unexpected A rank for {key}")
            value = torch.cat((a, b), dim=0)
        else:
            if a.shape[1] != b0_rank or b.shape[1] != delta_rank:
                raise ValueError(f"unexpected B rank for {key}")
            value = torch.cat((a, b.mul(args.delta_scale)), dim=1)
        composed[key] = value.contiguous()
        module_counts[kind] += 1
    if not composed or module_counts["A"] != module_counts["B"]:
        raise ValueError("incomplete source LoRA tensor pairs")

    output.mkdir(parents=True)
    output_weights = output / "adapter_model.safetensors"
    save_file(composed, output_weights)
    combined_rank = b0_rank + delta_rank
    config = dict(b0_config)
    config.update(
        {
            "r": combined_rank,
            "lora_alpha": b0_scale * combined_rank,
            "lora_dropout": 0.0,
            "inference_mode": True,
            "init_lora_weights": True,
            "rank_pattern": {},
            "alpha_pattern": {},
        }
    )
    if float(config["lora_alpha"]).is_integer():
        config["lora_alpha"] = int(config["lora_alpha"])
    (output / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The processor is unchanged. Copy only ordinary processor/tokenizer
    # artifacts; never copy a source adapter config or weight file.
    for source_root in (delta, b0):
        for source in source_root.iterdir():
            if source.name in {
                "adapter_config.json",
                "adapter_model.safetensors",
            }:
                continue
            target = output / source.name
            if target.exists():
                continue
            if source.is_file():
                shutil.copy2(source, target)

    manifest = {
        "schema": "sc_mrd_exact_additive_lora_composition_v1",
        "b0_adapter": str(b0),
        "b0_sha256": sha256(b0_weights),
        "delta_adapter": str(delta),
        "delta_sha256": sha256(delta_weights),
        "delta_scale": args.delta_scale,
        "output_sha256": sha256(output_weights),
        "b0_rank": b0_rank,
        "delta_rank": delta_rank,
        "composed_rank": combined_rank,
        "lora_scale": b0_scale,
        "tensor_count": len(composed),
        "module_pair_count": module_counts["A"],
        "mathematically_exact_additive_composition": True,
        "base_weights_merged": False,
        "native_visual_encoder_transformer_lm_head_preserved": True,
    }
    (output / "composition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
