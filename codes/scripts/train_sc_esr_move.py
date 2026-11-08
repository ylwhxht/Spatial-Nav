#!/usr/bin/env python3
"""Train native-LM Move with one state-conditioned evidence-slot residual.

The fixed B0 hash defines the reproducible starting point. The historical mode
continues B0's r64 LoRA. The frozen-B0 mode instead starts from an audited
B0-merged model and trains a fresh Move delta-LoRA, so Move closure cannot
overwrite the shared spatial adapter. SC-ESR never emits an action itself: it
adds one bounded residual at a single native decoder location and the original
Qwen Transformer/LM head remains the only decision path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from train_cold_start_vision_sft import (  # noqa: E402
    VisionPromptResponseDataset,
    VisionSFTCollator,
    find_jsonish_scalar_value_span,
    hidden_size_from_config,
    load_model,
    load_processor,
)
from embodied_nav_cot.sc_esr import (  # noqa: E402
    ABILITIES,
    SCESRConfig,
    SCESRController,
    StateConditionedEvidenceSlotResidual,
    candidate_relation_alignment_loss,
    all_candidate_relation_alignment_loss,
    evidence_slot_contrastive_loss,
    residual_preservation_loss,
    residual_utility_margin_loss,
)
from embodied_nav_cot.sc_esr_generation import (  # noqa: E402
    candidate_terminal_positions,
)
from embodied_nav_cot.sc_mrd import (  # noqa: E402
    action_token_spec,
    gold_margin,
    legal_action_logits,
)
from embodied_nav_cot.sc_lsr_losses import (  # noqa: E402
    fixed_orthonormal_projection,
    relation_delta_contrast_loss,
)
from embodied_nav_cot.candidate_overlay_regions import (  # noqa: E402
    candidate_overlay_token_masks,
    local_gold_candidate_mask,
)


B0_SHA256 = "e192db7ce068c011b7a1de6e317c65e35473c92005f7711d81f8399020afa74c"
MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
    "position_ids",
    "labels",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--b0-adapter", required=True)
    parser.add_argument(
        "--expected-b0-sha256",
        default=B0_SHA256,
        help=(
            "Audited initialization adapter digest. The historical default "
            "is unchanged; compatibility experiments must opt in explicitly."
        ),
    )
    parser.add_argument(
        "--fresh-move-delta-on-merged-b0",
        action="store_true",
        help=(
            "Treat --model-path as an audited Base+B0 merged model, keep it "
            "frozen, and initialize a fresh r64 Move delta-LoRA instead of "
            "continuing B0's adapter parameters."
        ),
    )
    parser.add_argument(
        "--merged-b0-manifest",
        default="",
        help=(
            "Required audit manifest for "
            "--fresh-move-delta-on-merged-b0."
        ),
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--evidence-cache", required=True)
    parser.add_argument(
        "--candidate-relation-cache",
        default="",
        help=(
            "Optional training-only action-blind visible-candidate relation "
            "targets. Runtime never loads this cache."
        ),
    )
    parser.add_argument(
        "--all-candidate-relation-cache",
        default="",
        help=(
            "Training-only SC-LSR cache with action-blind relation targets for "
            "every legal candidate and all five spatial operators."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-train-rows", type=int, default=9725)
    parser.add_argument("--expected-dev-rows", type=int, default=972)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--expected-closure-epochs",
        type=int,
        default=1,
        help=(
            "Declared deterministic epoch budget for "
            "--require-sc-lsr-closure-contract."
        ),
    )
    parser.add_argument(
        "--expected-closure-world-size",
        type=int,
        default=8,
        help=(
            "Declared DDP world size for "
            "--require-sc-lsr-closure-contract. Global batch remains 32."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--residual-learning-rate", type=float, default=5e-5)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--relation-decision-learning-rate",
        type=float,
        default=0.0,
        help=(
            "Optional LR for the zero-initialized semantic-to-decision "
            "residual mappers. Zero reuses --residual-learning-rate."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--route-weight", type=float, default=0.05)
    parser.add_argument(
        "--route-ranking-weight",
        type=float,
        default=0.0,
        help="Pairwise positive-over-negative ability ranking weight.",
    )
    parser.add_argument("--route-ranking-margin", type=float, default=0.20)
    parser.add_argument(
        "--ability-pos-weight-mode",
        choices=("balanced", "sqrt_balanced", "none"),
        default="balanced",
        help=(
            "Positive weighting for route BCE. Pairwise top-k routing usually "
            "uses none; balanced preserves SC-ESR v1/v2 behavior."
        ),
    )
    parser.add_argument("--evidence-weight", type=float, default=0.05)
    parser.add_argument(
        "--candidate-relation-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--all-candidate-relation-weight", type=float, default=0.0
    )
    parser.add_argument("--utility-weight", type=float, default=0.15)
    parser.add_argument(
        "--gain-vs-off-weight",
        type=float,
        default=0.0,
        help=(
            "On audited closure rows, require current-image SC-ESR to improve "
            "the gold-action margin over the identical residual-off path."
        ),
    )
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument(
        "--candidate-coupling-weight",
        type=float,
        default=0.0,
        help=(
            "Supervise native action-token attention over candidate-conditioned "
            "visual evidence slots; valid only with --candidate-conditioned."
        ),
    )
    parser.add_argument("--utility-margin", type=float, default=0.05)
    parser.add_argument(
        "--counterfactual-gradient-weight",
        type=float,
        default=0.0,
        help=(
            "Backpropagate through the same-image shuffled-candidate branch "
            "inside SC-ESR only, so mismatched candidate evidence cannot "
            "improve the gold-action margin over residual-off."
        ),
    )
    parser.add_argument(
        "--counterfactual-neutrality-margin",
        type=float,
        default=0.0,
        help=(
            "Required amount by which shuffled-candidate evidence remains "
            "below the residual-off gold-action margin."
        ),
    )
    parser.add_argument("--gain-vs-off-margin", type=float, default=0.05)
    parser.add_argument("--preservation-tolerance", type=float, default=0.02)
    parser.add_argument("--action-token-weight", type=float, default=4.0)
    parser.add_argument("--counterfactual-every", type=int, default=2)
    parser.add_argument("--preservation-every", type=int, default=4)
    parser.add_argument("--gate-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--slot-rank", type=int, default=64)
    parser.add_argument("--source-layer", type=int, default=24)
    parser.add_argument("--injection-layer", type=int, default=25)
    parser.add_argument("--max-residual-scale", type=float, default=0.02)
    parser.add_argument("--state-dropout", type=float, default=0.30)
    parser.add_argument("--candidate-conditioned", action="store_true")
    parser.add_argument(
        "--candidate-local-masks",
        action="store_true",
        help=(
            "Ground each positive Move candidate in its visible public "
            "overlay ray/endpoint region. Failed detections fall back to the "
            "global image and are excluded from local utility supervision."
        ),
    )
    parser.add_argument(
        "--global-conditioned-local-slots",
        action="store_true",
        help=(
            "Condition each candidate-local visual slot on an ability-specific "
            "global spatial slot from the same current RGB."
        ),
    )
    parser.add_argument(
        "--candidate-value-interaction",
        action="store_true",
        help=(
            "Add a zero-initialized, bounded multiplicative interaction "
            "between each candidate's public label and its visual evidence "
            "inside the residual value path."
        ),
    )
    parser.add_argument(
        "--candidate-memory-injection",
        action="store_true",
        help=(
            "Inject each candidate's visual spatial evidence into its existing "
            "prompt label token, leaving downstream native Transformer layers "
            "and the original LM head to generate the action."
        ),
    )
    parser.add_argument(
        "--latent-spatial-rollout",
        action="store_true",
        help=(
            "Replace candidate-local evidence values with five independent "
            "global-to-local spatial-relation operators before the bounded "
            "native decision residual or optional candidate-memory write."
        ),
    )
    parser.add_argument(
        "--relation-semantic-decision-separation",
        action="store_true",
        help=(
            "Supervise an action-blind relation semantic latent while a "
            "zero-initialized residual mapper compiles that same latent into "
            "the native-LM decision increment. Runtime remains one forward."
        ),
    )
    parser.add_argument(
        "--local-evidence-control",
        action="store_true",
        help=(
            "Equal-parameter control that keeps all five operators and the "
            "native residual path but replaces global-to-local relation "
            "inputs with goal-conditioned candidate-local evidence."
        ),
    )
    parser.add_argument(
        "--relation-semantic-target-space",
        choices=("lift_to_teacher", "fixed_lowrank"),
        default="lift_to_teacher",
        help=(
            "For action-blind relation contrast, either lift the rank-64 "
            "semantic through slot_to_hidden (legacy V8) or project the same "
            "teacher target into a fixed rank-64 coordinate and supervise the "
            "actual runtime semantic directly."
        ),
    )
    parser.add_argument(
        "--relation-semantic-projection-seed",
        type=int,
        default=20260727,
        help="Deterministic seed for the training-only fixed low-rank target space.",
    )
    parser.add_argument(
        "--straight-through-sparse-route",
        action="store_true",
        help=(
            "Use the same hard 1--3 ability route during training and runtime "
            "while retaining the legacy dense router gradient with a "
            "zero-forward straight-through term."
        ),
    )
    parser.add_argument(
        "--soft-ability-route",
        action="store_true",
        help=(
            "Use a dense learned mixture of all five abilities during both "
            "training and inference. This is an exploration setting used "
            "before tightening a useful route back to top-1--3."
        ),
    )
    parser.add_argument(
        "--open-capacity-exploration-contract",
        action="store_true",
        help=(
            "Audit the deliberate open-before-tighten SC-LSR course: fixed "
            "B0 and closure data remain unchanged, while routing is dense, "
            "the need gate is open, and the bounded residual budget is 10%%."
        ),
    )
    parser.add_argument(
        "--independent-ability-gates",
        action="store_true",
        help="Use one learned current-image evidence scorer per spatial ability.",
    )
    parser.add_argument(
        "--adaptive-need-gate",
        action="store_true",
        help=(
            "Modulate the bounded spatial residual per decision token instead "
            "of applying one global residual amplitude everywhere."
        ),
    )
    parser.add_argument(
        "--route-class-balanced",
        action="store_true",
        help="Average positive/negative route BCE within each ability.",
    )
    parser.add_argument(
        "--residual-action-weight",
        type=float,
        default=0.0,
        help=(
            "Legal-action CE applied only to SC-ESR parameters in the full "
            "path; it never trains an independent decision head."
        ),
    )
    parser.add_argument("--need-weight", type=float, default=0.0)
    parser.add_argument("--need-margin-threshold", type=float, default=0.10)
    parser.add_argument(
        "--exact-stable-course",
        action="store_true",
        help=(
            "Consume every 40k course row exactly once per epoch while keeping "
            "matched spatial action rows local for counterfactual supervision."
        ),
    )
    parser.add_argument(
        "--historical-distributed-course",
        action="store_true",
        help=(
            "Use the original DistributedSampler(seed=0) over the order-"
            "preserving 40k course for the native residual-off LM path."
        ),
    )
    parser.add_argument(
        "--require-historical-course-contract",
        action="store_true",
        help="Fail unless the audited 40k/8-GPU/1,250-step Move course is exact.",
    )
    parser.add_argument(
        "--require-v3-course-contract",
        action="store_true",
        help=(
            "Fail unless corrected historical loss weights, SC-ESR-only "
            "auxiliary gradients and runtime-aligned injection are enabled."
        ),
    )
    parser.add_argument(
        "--gradient-separated-native-lm",
        action="store_true",
        help=(
            "Train the B0 LoRA only through a native residual-off LM forward; "
            "train SC-ESR only through the residual-full navigation losses."
        ),
    )
    parser.add_argument(
        "--require-v4-course-contract",
        action="store_true",
        help="Fail unless the order-preserving gradient-separated V4 contract holds.",
    )
    parser.add_argument(
        "--require-sc-lsr-v1-contract",
        action="store_true",
        help=(
            "Fail unless V4's 40k native-LM course is preserved while only "
            "the spatial path is replaced by all-candidate latent rollout."
        ),
    )
    parser.add_argument(
        "--require-sc-lsr-closure-contract",
        action="store_true",
        help=(
            "Fail unless the course contains exactly the 10k direct Move "
            "anchors, no standalone spatial replay, and the audited "
            "same-row all-candidate relation closure."
        ),
    )
    parser.add_argument("--image-token-id", type=int, default=151655)
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=256)
    parser.add_argument("--diagnostic-samples", type=int, default=256)
    parser.add_argument(
        "--native-lm-weight-semantics",
        choices=("legacy_normalized", "historical_sft"),
        default="legacy_normalized",
        help=(
            "legacy_normalized reproduces SC-ESR v1/v2, where token weights "
            "also entered the denominator. historical_sft matches the stable "
            "Vision SFT trainer: weighted token losses are normalized by the "
            "unweighted number of assistant tokens."
        ),
    )
    parser.add_argument(
        "--auxiliary-gradient-scope",
        choices=("all", "sc_esr"),
        default="all",
        help=(
            "Limit route/evidence/counterfactual/preservation gradients to "
            "SC-ESR parameters. Native LM loss always trains the continued "
            "B0 LoRA and the residual path."
        ),
    )
    parser.add_argument(
        "--diagnostic-split",
        choices=("train", "dev"),
        default="train",
        help="Dataset used by the post-training mechanism diagnostic.",
    )
    parser.add_argument(
        "--diagnostic-action-only",
        action="store_true",
        help="Evaluate diagnostics only on rows with a legal Move target.",
    )
    parser.add_argument(
        "--runtime-aligned-injection",
        action="store_true",
        help=(
            "On Move rows, inject from the prompt terminal through every "
            "teacher-forced assistant token, matching cached generation."
        ),
    )
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if "image_path" in row and not Path(row["image_path"]).is_absolute():
                    row["image_path"] = str((path.parent / row["image_path"]).resolve())
                rows.append(row)
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_device() -> tuple[torch.device, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("SC-ESR training requires CUDA")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank, rank, world


class EvidenceCache:
    """CPU-resident sparse evidence vectors indexed by data row and ability."""

    def __init__(self, root: Path, *, data: Path) -> None:
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest.get("passed") or not manifest.get("training_only"):
            raise RuntimeError("invalid SC-ESR evidence cache")
        if manifest.get("runtime_loads_cache") is not False:
            raise RuntimeError("evidence cache must be training-only")
        if manifest["train_sha256"] != sha256(data / "train.jsonl"):
            raise RuntimeError("evidence cache/data hash mismatch")
        sample_ids = json.loads((root / "sample_ids.json").read_text())
        self.sample_to_row = {
            str(sample_id): index for index, sample_id in enumerate(sample_ids)
        }
        values: list[torch.Tensor] = []
        coordinates: list[torch.Tensor] = []
        for name, expected_rows in sorted(manifest["shards"].items()):
            path = root / name
            if sha256(path) != manifest["shard_sha256"][name]:
                raise RuntimeError(f"evidence shard hash drift: {name}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                shard_values = handle.get_tensor("values")
                shard_coords = handle.get_tensor("row_ability_index")
            if shard_values.shape[0] != expected_rows:
                raise RuntimeError(f"evidence shard row drift: {name}")
            values.append(shard_values)
            coordinates.append(shard_coords)
        # Keep the immutable cache in BF16.  The contrastive loss promotes
        # targets to FP32 only for the current mini-batch.
        self.values = torch.cat(values)
        coords = torch.cat(coordinates).long()
        self.lookup = torch.full(
            (len(sample_ids), len(ABILITIES)), -1, dtype=torch.long
        )
        for flat_index, (row_index, ability_index) in enumerate(coords.tolist()):
            if self.lookup[row_index, ability_index] >= 0:
                raise RuntimeError("duplicate evidence cache coordinate")
            self.lookup[row_index, ability_index] = flat_index
        self.hidden_size = int(manifest["hidden_size"])
        self.manifest = manifest

    def batch(
        self, sample_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = torch.tensor(
            [sample_id in self.sample_to_row for sample_id in sample_ids],
            dtype=torch.bool,
        )
        row_indices = torch.tensor(
            [self.sample_to_row.get(sample_id, 0) for sample_id in sample_ids],
            dtype=torch.long,
        )
        flat = self.lookup[row_indices].clone()
        flat[~present] = -1
        active = flat.ge(0)
        safe = flat.clamp_min(0)
        targets = self.values[safe]
        targets = targets * active[..., None]
        return targets, active


class CandidateRelationCache:
    """Centered action-blind relation vectors indexed by certificate row."""

    def __init__(self, root: Path, *, data: Path) -> None:
        manifest = json.loads((root / "manifest.json").read_text())
        required = {
            "passed": True,
            "training_only": True,
            "runtime_loads_cache": False,
            "action_blind_text": True,
            "candidate_id_removed": True,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise RuntimeError("invalid candidate-relation cache contract")
        if manifest["train_sha256"] != sha256(data / "train.jsonl"):
            raise RuntimeError("candidate-relation cache/data hash mismatch")
        sample_ids = json.loads((root / "sample_ids.json").read_text())
        self.sample_to_row = {
            str(sample_id): index for index, sample_id in enumerate(sample_ids)
        }
        values_path = root / "candidate_relation.safetensors"
        if sha256(values_path) != manifest["values_sha256"]:
            raise RuntimeError("candidate-relation target hash drift")
        with safe_open(values_path, framework="pt", device="cpu") as handle:
            self.values = handle.get_tensor("values")
        if self.values.shape != (
            len(sample_ids),
            int(manifest["hidden_size"]),
        ):
            raise RuntimeError("candidate-relation target shape drift")
        self.hidden_size = int(manifest["hidden_size"])
        self.manifest = manifest

    def batch(
        self, sample_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = torch.tensor(
            [sample_id in self.sample_to_row for sample_id in sample_ids],
            dtype=torch.bool,
        )
        rows = torch.tensor(
            [self.sample_to_row.get(sample_id, 0) for sample_id in sample_ids],
            dtype=torch.long,
        )
        targets = self.values[rows] * present[:, None]
        return targets, present


class AllCandidateRelationCache:
    """Sparse action-blind relation vectors for every legal candidate/ability."""

    def __init__(self, root: Path, *, data: Path) -> None:
        manifest = json.loads((root / "manifest.json").read_text())
        required = {
            "schema": "sc_lsr_all_candidate_relation_cache_v1",
            "passed": True,
            "training_only": True,
            "runtime_loads_cache": False,
            "action_blind_text": True,
            "candidate_id_removed": True,
            "all_legal_candidates_supervised": True,
            "policy_action_in_targets": False,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise RuntimeError("invalid all-candidate relation cache contract")
        if manifest["train_sha256"] != sha256(data / "train.jsonl"):
            raise RuntimeError("all-candidate relation cache/data hash mismatch")
        values_path = root / "all_candidate_relations.safetensors"
        index_path = root / "item_index.jsonl"
        if (
            sha256(values_path) != manifest["values_sha256"]
            or sha256(index_path) != manifest["index_sha256"]
        ):
            raise RuntimeError("all-candidate relation cache hash drift")
        with safe_open(values_path, framework="pt", device="cpu") as handle:
            self.values = handle.get_tensor("values")
            self.active = (
                handle.get_tensor("active").bool()
                if "active" in handle.keys()
                else torch.ones(len(self.values), dtype=torch.bool)
            )
        index = read_jsonl(index_path)
        if self.values.shape != (
            len(index),
            int(manifest["hidden_size"]),
        ):
            raise RuntimeError("all-candidate relation cache shape drift")
        if self.active.shape != (len(index),):
            raise RuntimeError("all-candidate relation activity shape drift")
        self.lookup: dict[str, list[tuple[int, int, int]]] = {}
        for flat, item in enumerate(index):
            sample_id = str(item["sample_id"])
            candidate = int(item["candidate_index"])
            ability = int(item["ability_index"])
            if candidate < 0 or ability not in range(len(ABILITIES)):
                raise RuntimeError("invalid all-candidate cache coordinate")
            self.lookup.setdefault(sample_id, []).append(
                (candidate, ability, flat)
            )
        self.hidden_size = int(manifest["hidden_size"])
        self.manifest = manifest

    def batch(
        self, sample_ids: list[str], candidate_width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros(
            len(sample_ids),
            candidate_width,
            len(ABILITIES),
            self.hidden_size,
            dtype=self.values.dtype,
        )
        active = torch.zeros(
            len(sample_ids),
            candidate_width,
            len(ABILITIES),
            dtype=torch.bool,
        )
        seen = torch.zeros_like(active)
        for row, sample_id in enumerate(sample_ids):
            for candidate, ability, flat in self.lookup.get(sample_id, []):
                if candidate >= candidate_width:
                    raise RuntimeError(
                        "cached candidate exceeds current public action width"
                    )
                if seen[row, candidate, ability]:
                    raise RuntimeError("duplicate all-candidate relation target")
                seen[row, candidate, ability] = True
                targets[row, candidate, ability] = self.values[flat]
                active[row, candidate, ability] = bool(self.active[flat])
        return targets, active


class SCESRDataset(VisionPromptResponseDataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        if len(self.samples) != len(rows):
            raise RuntimeError("SC-ESR dataset alignment failed")
        for sample, row in zip(self.samples, rows):
            sample.sc_esr_metadata = {
                "sample_id": str(row["sample_id"]),
                "goal": str(row["goal"]),
                "state_label": int(row.get("state_label", -1)),
                "ability_targets": [
                    float(value)
                    for value in row.get(
                        "ability_targets", [0.0] * len(ABILITIES)
                    )
                ],
                "route_valid": bool(
                    row.get("route_supervision_valid", False)
                ),
                "state_valid": bool(
                    row.get(
                        "state_supervision_valid",
                        row.get("route_supervision_valid", False),
                    )
                ),
                "ability_valid": bool(
                    row.get(
                        "ability_supervision_valid",
                        row.get("route_supervision_valid", False),
                    )
                ),
                "evidence_valid": bool(
                    row.get("evidence_supervision_valid", False)
                ),
                "gain_valid": bool(
                    row.get("spatial_gain_supervision_valid", False)
                ),
                "gold_action": str(row.get("gold_action") or ""),
                "legal_actions": list(row.get("legal_actions") or []),
                "image_path": str(row["image_path"]),
                "row_loss_weight": float(
                    row.get("action_loss_weight", 1.0)
                ),
                "decision_loss_weight": float(
                    row.get(
                        "decision_loss_weight",
                        1.5 if row.get("gold_action") else 1.0,
                    )
                ),
                "action_field_loss_weight": float(
                    row.get(
                        "action_field_loss_weight",
                        4.0 if row.get("gold_action") else 1.0,
                    )
                ),
            }


@dataclass
class ActionSurface:
    positions: torch.Tensor
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    gold_indices: torch.Tensor
    valid: torch.Tensor


class SCESRCollator:
    def __init__(
        self,
        processor: Any,
        evidence: EvidenceCache,
        candidate_relation: CandidateRelationCache | None,
        all_candidate_relation: AllCandidateRelationCache | None,
        *,
        max_length: int,
        image_token_id: int,
        runtime_aligned_injection: bool = False,
        candidate_local_masks: bool = False,
    ) -> None:
        self.evidence = evidence
        self.candidate_relation = candidate_relation
        self.all_candidate_relation = all_candidate_relation
        self.image_token_id = int(image_token_id)
        self.runtime_aligned_injection = bool(runtime_aligned_injection)
        self.candidate_local_masks = bool(candidate_local_masks)
        self.base = VisionSFTCollator(
            processor,
            max_length=max_length,
            system_prompt="You are a navigation spatial reasoning assistant.",
            text_before_image=False,
        )
        self.tokenizer = self.base.tokenizer

    def _action_token_mask(
        self, labels: torch.Tensor, samples: list[Any]
    ) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for row, sample in enumerate(samples):
            if not sample.sc_esr_metadata["gold_action"]:
                continue
            span = find_jsonish_scalar_value_span(
                sample.response,
                field_names=("action",),
                value_pattern=r"candidate_\d+",
            )
            if span is None:
                raise ValueError(f"missing Move action: {sample.sample_id}")
            active = labels[row].ne(-100).nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                raise ValueError(f"truncated response: {sample.sample_id}")
            prefix = self.tokenizer(
                sample.response[: span[0]], add_special_tokens=False
            )["input_ids"]
            value = self.tokenizer(
                sample.response[span[0] : span[1]], add_special_tokens=False
            )["input_ids"]
            start = int(active[0]) + len(prefix)
            end = start + len(value)
            if labels[row, start:end].tolist() != value:
                # A small legacy replay subset was serialized with a tokenizer
                # prefix whose token count differs from the current processor.
                # Recover the span from the active (non-masked) labels instead
                # of aborting a completed course during rank-zero diagnostics.
                active_values = labels[row, active].tolist()
                fallback = None
                for offset in range(len(active_values) - len(value) + 1):
                    if active_values[offset : offset + len(value)] == value:
                        fallback = offset
                        break
                if fallback is None:
                    # This row cannot contribute to the action-token mask, but
                    # the native LM/action-surface diagnostics remain valid.
                    continue
                start = int(active[fallback])
                end = start + len(value)
            mask[row, start:end] = True
        return mask

    def _action_surface(
        self, labels: torch.Tensor, samples: list[Any]
    ) -> ActionSurface:
        specs: list[Any | None] = []
        for sample in samples:
            metadata = sample.sc_esr_metadata
            if not metadata["gold_action"] or not metadata["legal_actions"]:
                specs.append(None)
                continue
            try:
                spec = action_token_spec(
                    self.tokenizer,
                    sample.response,
                    metadata["legal_actions"],
                )
            except ValueError:
                spec = None
            specs.append(spec)
        width = max(
            [len(spec.legal_token_ids) for spec in specs if spec is not None]
            or [1]
        )
        positions = torch.zeros(len(samples), dtype=torch.long)
        token_ids = torch.zeros((len(samples), width), dtype=torch.long)
        token_mask = torch.zeros((len(samples), width), dtype=torch.bool)
        gold = torch.zeros(len(samples), dtype=torch.long)
        valid = torch.zeros(len(samples), dtype=torch.bool)
        for row, (sample, spec) in enumerate(zip(samples, specs)):
            if spec is None:
                continue
            active = labels[row].ne(-100).nonzero(as_tuple=True)[0]
            response_start = int(active[0])
            label_position = response_start + spec.response_token_offset
            gold_id = spec.legal_token_ids[spec.gold_index]
            if (label_position >= labels.shape[1]
                    or int(labels[row, label_position]) != gold_id):
                # Some legacy replay rows have a different chat-template
                # prefix under InternVL. Align the action to the actual
                # response labels, but accept only a unique contextual match.
                response_ids = self.tokenizer(
                    sample.response, add_special_tokens=False
                )["input_ids"]
                active_values = labels[row, active].tolist()
                offset = spec.response_token_offset
                matches: list[tuple[int, int]] = []
                for index, token_id in enumerate(active_values):
                    if token_id != gold_id:
                        continue
                    score = 0
                    for distance in range(1, 9):
                        if (index - distance < 0 or offset - distance < 0
                                or active_values[index - distance]
                                != response_ids[offset - distance]):
                            break
                        score += 1
                    for distance in range(1, 9):
                        if (index + distance >= len(active_values)
                                or offset + distance >= len(response_ids)
                                or active_values[index + distance]
                                != response_ids[offset + distance]):
                            break
                        score += 1
                    matches.append((score, index))
                if not matches:
                    continue
                best_score = max(score for score, _ in matches)
                best = [index for score, index in matches if score == best_score]
                if best_score < 2 or len(best) != 1:
                    continue
                label_position = int(active[best[0]])
            logit_position = label_position - 1
            count = len(spec.legal_token_ids)
            positions[row] = logit_position
            token_ids[row, :count] = torch.tensor(spec.legal_token_ids)
            token_mask[row, :count] = True
            gold[row] = spec.gold_index
            valid[row] = True
        return ActionSurface(positions, token_ids, token_mask, gold, valid)

    @staticmethod
    def _gain_mask(samples: list[Any]) -> torch.Tensor:
        result = torch.zeros(len(samples), dtype=torch.bool)
        if len(samples) < 2:
            return result
        metadata = [sample.sc_esr_metadata for sample in samples]
        for index, current in enumerate(metadata):
            donor = metadata[(index - 1) % len(metadata)]
            shared = any(
                left and right
                for left, right in zip(
                    current["ability_targets"], donor["ability_targets"]
                )
            )
            result[index] = bool(
                current["gain_valid"]
                and donor["gain_valid"]
                and current["goal"] == donor["goal"]
                and current["state_label"] == donor["state_label"]
                and len(current["legal_actions"]) == len(donor["legal_actions"])
                and current["image_path"] != donor["image_path"]
                and current["gold_action"] != donor["gold_action"]
                and shared
            )
        return result

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        batch = self.base(samples)
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        image_mask = input_ids.eq(self.image_token_id)
        if not bool(image_mask.any(1).all()):
            raise ValueError("every SC-ESR row must retain current RGB tokens")
        action_mask = self._action_token_mask(labels, samples)
        first_label = torch.stack(
            [
                labels[row].ne(-100).nonzero(as_tuple=True)[0][0]
                for row in range(labels.shape[0])
            ]
        )
        route_positions = first_label - 1
        metadata = [sample.sc_esr_metadata for sample in samples]
        # Decoder hidden position t predicts label t+1. Runtime cached
        # generation activates SC-ESR at the prompt terminal and every later
        # generated token. Reproduce that path for action rows; auxiliary LM
        # replay deliberately receives no spatial residual.
        decision_hidden_mask = torch.zeros_like(action_mask)
        if self.runtime_aligned_injection:
            for row, item in enumerate(metadata):
                if not item["gold_action"]:
                    continue
                active = labels[row].ne(-100).nonzero(as_tuple=True)[0]
                start = int(active[0]) - 1
                stop = int(active[-1])
                if start < 0 or stop <= start:
                    raise ValueError(
                        f"invalid runtime injection span: {item['sample_id']}"
                    )
                decision_hidden_mask[row, start:stop] = True
        else:
            decision_hidden_mask[:, :-1] = action_mask[:, 1:]
        candidate_positions, candidate_mask = candidate_terminal_positions(
            input_ids,
            batch["attention_mask"],
            self.tokenizer,
            [item["legal_actions"] for item in metadata],
            prefix_end_positions=route_positions,
        )
        candidate_image_mask: torch.Tensor | None = None
        candidate_local_valid = torch.zeros_like(candidate_mask)
        detected_forward = torch.zeros(len(samples), dtype=torch.long)
        if self.candidate_local_masks:
            overlay_images: list[Image.Image] = []
            for item in metadata:
                with Image.open(item["image_path"]) as image:
                    overlay_images.append(image.convert("RGB").copy())
            (
                candidate_image_mask,
                candidate_local_valid,
                detections,
            ) = candidate_overlay_token_masks(
                images=overlay_images,
                legal_actions=[
                    item["legal_actions"] for item in metadata
                ],
                input_ids=input_ids,
                image_grid_thw=batch["image_grid_thw"],
                image_token_id=self.image_token_id,
                candidate_width=candidate_positions.shape[1],
            )
            detected_forward = torch.tensor(
                [item.forward_components for item in detections],
                dtype=torch.long,
            )
        sample_ids = [item["sample_id"] for item in metadata]
        evidence_targets, evidence_present = self.evidence.batch(sample_ids)
        if self.candidate_relation is None:
            relation_targets = torch.zeros(
                len(sample_ids),
                self.evidence.hidden_size,
                dtype=torch.bfloat16,
            )
            relation_present = torch.zeros(
                len(sample_ids), dtype=torch.bool
            )
        else:
            relation_targets, relation_present = (
                self.candidate_relation.batch(sample_ids)
            )
        if self.all_candidate_relation is None:
            all_relation_targets = torch.zeros(
                len(sample_ids),
                candidate_positions.shape[1],
                len(ABILITIES),
                self.evidence.hidden_size,
                dtype=torch.bfloat16,
            )
            all_relation_active = torch.zeros(
                len(sample_ids),
                candidate_positions.shape[1],
                len(ABILITIES),
                dtype=torch.bool,
            )
        else:
            all_relation_targets, all_relation_active = (
                self.all_candidate_relation.batch(
                    sample_ids, candidate_positions.shape[1]
                )
            )
        ability_targets = torch.tensor(
            [item["ability_targets"] for item in metadata],
            dtype=torch.float32,
        )
        evidence_active = (
            evidence_present
            & ability_targets.bool()
            & torch.tensor(
                [item["evidence_valid"] for item in metadata]
            )[:, None]
        )
        all_relation_active = (
            all_relation_active
            & candidate_mask[:, :, None]
            & ability_targets[:, None, :].bool()
        )
        batch.update(
            {
                "sc_esr_image_mask": image_mask,
                "sc_esr_decision_hidden_mask": decision_hidden_mask,
                "sc_esr_action_token_mask": action_mask,
                "sc_esr_route_positions": route_positions,
                "sc_esr_candidate_positions": candidate_positions,
                "sc_esr_candidate_mask": candidate_mask,
                "sc_esr_candidate_image_mask": candidate_image_mask,
                "sc_esr_candidate_local_valid": candidate_local_valid,
                "sc_esr_detected_forward_candidates": detected_forward,
                "sc_esr_state_targets": torch.tensor(
                    [item["state_label"] for item in metadata], dtype=torch.long
                ),
                "sc_esr_ability_targets": ability_targets,
                "sc_esr_route_valid": torch.tensor(
                    [item["route_valid"] for item in metadata], dtype=torch.bool
                ),
                "sc_esr_state_valid": torch.tensor(
                    [item["state_valid"] for item in metadata], dtype=torch.bool
                ),
                "sc_esr_ability_valid": torch.tensor(
                    [item["ability_valid"] for item in metadata],
                    dtype=torch.bool,
                ),
                "sc_esr_evidence_targets": evidence_targets,
                "sc_esr_evidence_active": evidence_active,
                "sc_esr_candidate_relation_targets": relation_targets,
                "sc_esr_candidate_relation_present": relation_present,
                "sc_esr_all_candidate_relation_targets": all_relation_targets,
                "sc_esr_all_candidate_relation_active": all_relation_active,
                "sc_esr_closure_valid": torch.tensor(
                    [item["gain_valid"] for item in metadata],
                    dtype=torch.bool,
                ),
                "sc_esr_gain_valid": self._gain_mask(samples),
                "sc_esr_action_surface": self._action_surface(labels, samples),
                "sc_esr_sample_ids": sample_ids,
                "sc_esr_row_loss_weight": torch.tensor(
                    [item["row_loss_weight"] for item in metadata],
                    dtype=torch.float32,
                ),
                "sc_esr_decision_loss_weight": torch.tensor(
                    [item["decision_loss_weight"] for item in metadata],
                    dtype=torch.float32,
                ),
                "sc_esr_action_field_loss_weight": torch.tensor(
                    [item["action_field_loss_weight"] for item in metadata],
                    dtype=torch.float32,
                ),
            }
        )
        return batch


class EvidenceBucketBatchSampler(Sampler[list[int]]):
    """Keep hard counterfactuals local while retaining every Move anchor."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int,
        world: int,
        rank: int,
        seed: int,
    ) -> None:
        if batch_size < 2 or world <= 0 or not 0 <= rank < world:
            raise ValueError("invalid SC-ESR batch sampler configuration")
        self.rows = rows
        self.batch_size = batch_size
        self.world = world
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self._rank_batches = self._build(epoch=0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rank_batches = self._build(epoch=self.epoch)

    @staticmethod
    def _interleave_actions(
        indices: list[int], rows: list[dict[str, Any]], rng: random.Random
    ) -> list[int]:
        by_action: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_action[str(rows[index]["gold_action"])].append(index)
        for values in by_action.values():
            rng.shuffle(values)
        result: list[int] = []
        last = ""
        while any(by_action.values()):
            choices = sorted(
                (
                    (len(values), action)
                    for action, values in by_action.items()
                    if values and action != last
                ),
                reverse=True,
            )
            if not choices:
                choices = sorted(
                    (
                        (len(values), action)
                        for action, values in by_action.items()
                        if values
                    ),
                    reverse=True,
                )
            action = choices[0][1]
            result.append(by_action[action].pop())
            last = action
        return result

    def _build(self, *, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            if row.get("spatial_gain_supervision_valid"):
                key = (
                    "spatial",
                    row["goal"],
                    int(row["state_label"]),
                    len(row["legal_actions"]),
                )
            else:
                key = ("anchor",)
            groups[key].append(index)
        batches: list[list[int]] = []
        for key, indices in sorted(groups.items(), key=lambda item: str(item[0])):
            if key[0] == "spatial":
                ordered = self._interleave_actions(indices, self.rows, rng)
            else:
                ordered = list(indices)
                rng.shuffle(ordered)
            for start in range(0, len(ordered), self.batch_size):
                batch = ordered[start : start + self.batch_size]
                if len(batch) < self.batch_size:
                    pool = ordered or indices
                    batch.extend(
                        pool[index % len(pool)]
                        for index in range(self.batch_size - len(batch))
                    )
                batches.append(batch)
        rng.shuffle(batches)
        padded = math.ceil(len(batches) / self.world) * self.world
        if len(batches) < padded:
            original = list(batches)
            batches.extend(
                original[index % len(original)]
                for index in range(padded - len(batches))
            )
        return batches[self.rank :: self.world]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._rank_batches)

    def __len__(self) -> int:
        return len(self._rank_batches)


class ExactStableCourseBatchSampler(Sampler[list[int]]):
    """Exact one-pass 40k sampler with local action-counterfactual structure."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int,
        world: int,
        rank: int,
        seed: int,
    ) -> None:
        if len(rows) != 40000:
            raise ValueError("exact stable course requires exactly 40,000 rows")
        if len(rows) % (batch_size * world):
            raise ValueError("course must divide evenly across global batch")
        self.rows = rows
        self.batch_size = int(batch_size)
        self.world = int(world)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self._rank_batches = self._build(epoch=0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rank_batches = self._build(epoch=self.epoch)

    def _build(self, *, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        remainder: list[int] = []
        for index, row in enumerate(self.rows):
            if row.get("spatial_gain_supervision_valid"):
                groups[
                    (
                        row["goal"],
                        int(row["state_label"]),
                        len(row["legal_actions"]),
                    )
                ].append(index)
            else:
                remainder.append(index)
        local_batches: list[list[int]] = []
        for _key, indices in sorted(
            groups.items(), key=lambda item: str(item[0])
        ):
            ordered = EvidenceBucketBatchSampler._interleave_actions(
                indices, self.rows, rng
            )
            full = len(ordered) // self.batch_size * self.batch_size
            local_batches.extend(
                ordered[start : start + self.batch_size]
                for start in range(0, full, self.batch_size)
            )
            remainder.extend(ordered[full:])
        rng.shuffle(remainder)
        if len(remainder) % self.batch_size:
            raise RuntimeError("exact stable course remainder is not divisible")
        local_batches.extend(
            remainder[start : start + self.batch_size]
            for start in range(0, len(remainder), self.batch_size)
        )
        if len(local_batches) * self.batch_size != len(self.rows):
            raise RuntimeError("exact stable course dropped or duplicated rows")
        rng.shuffle(local_batches)
        if len(local_batches) % self.world:
            raise RuntimeError("local batch count does not divide across ranks")
        rank_batches = local_batches[self.rank :: self.world]
        flat = [index for batch in local_batches for index in batch]
        if len(flat) != len(set(flat)) or sorted(flat) != list(range(len(self.rows))):
            raise RuntimeError("exact stable course is not a permutation")
        return rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._rank_batches)

    def __len__(self) -> int:
        return len(self._rank_batches)


def native_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in MODEL_INPUT_KEYS and torch.is_tensor(value)
    }


def metadata_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in tuple(batch):
        if key.startswith("sc_esr_"):
            value = batch.pop(key)
            name = key.removeprefix("sc_esr_")
            if isinstance(value, ActionSurface):
                value = ActionSurface(
                    value.positions.to(device),
                    value.token_ids.to(device),
                    value.token_mask.to(device),
                    value.gold_indices.to(device),
                    value.valid.to(device),
                )
            elif torch.is_tensor(value):
                value = value.to(device, non_blocking=True)
            result[name] = value
    return result


def mean_native_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    action_mask: torch.Tensor,
    action_weight: float,
    row_weight: torch.Tensor | None = None,
    decision_weight: torch.Tensor | None = None,
    action_field_weight: torch.Tensor | None = None,
    weight_semantics: str = "legacy_normalized",
) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    active = shift_labels.ne(-100)
    safe = shift_labels.masked_fill(~active, 0)
    token_loss = F.cross_entropy(
        shift_logits.transpose(1, 2), safe, reduction="none"
    )
    weights = active.float()
    if decision_weight is not None:
        weights = weights * decision_weight[:, None].float()
    if action_field_weight is None:
        field_weight = torch.full(
            (labels.shape[0],),
            float(action_weight),
            dtype=torch.float32,
            device=labels.device,
        )
    else:
        field_weight = action_field_weight.float()
    weights = weights * (
        1.0
        + action_mask[:, 1:].float() * (field_weight[:, None] - 1.0)
    )
    if weight_semantics == "legacy_normalized":
        denominator = weights.sum(1)
    elif weight_semantics == "historical_sft":
        # Match train_cold_start_vision_sft.py: decision/action weights scale
        # gradients and therefore must not cancel out in the denominator.
        denominator = active.float().sum(1)
    else:
        raise ValueError(f"unknown native LM weight semantics: {weight_semantics}")
    per_row = (token_loss * weights).sum(1) / denominator.clamp_min(1)
    valid = active.any(1)
    if row_weight is not None:
        per_row = per_row * row_weight.float()
    return per_row[valid].mean()


def route_loss(
    route: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    ability_pos_weight: torch.Tensor,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.20,
    class_balanced: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    state_valid = metadata.get(
        "state_valid", metadata["route_valid"]
    ).bool()
    ability_valid = metadata.get(
        "ability_valid", metadata["route_valid"]
    ).bool()
    zero = route["state_logits"].sum() * 0.0
    terms: list[torch.Tensor] = []
    state = zero
    ability = zero
    count = zero
    ranking = zero
    if state_valid.any():
        state = F.cross_entropy(
            route["state_logits"].float()[state_valid],
            metadata["state_targets"][state_valid],
        )
        terms.append(state)
    if ability_valid.any():
        supervised_logits = route["ability_logits"].float()[ability_valid]
        supervised_targets = metadata["ability_targets"].float()[ability_valid]
        if class_balanced:
            per_class: list[torch.Tensor] = []
            raw = F.binary_cross_entropy_with_logits(
                supervised_logits, supervised_targets, reduction="none"
            )
            for ability_index in range(supervised_logits.shape[1]):
                positive = supervised_targets[:, ability_index].bool()
                negative = ~positive
                terms_for_class: list[torch.Tensor] = []
                if positive.any():
                    terms_for_class.append(raw[positive, ability_index].mean())
                if negative.any():
                    terms_for_class.append(raw[negative, ability_index].mean())
                if terms_for_class:
                    per_class.append(torch.stack(terms_for_class).mean())
            ability = torch.stack(per_class).mean() if per_class else zero
        else:
            ability = F.binary_cross_entropy_with_logits(
                supervised_logits,
                supervised_targets,
                pos_weight=ability_pos_weight,
            )
        count_target = (
            metadata["ability_targets"][ability_valid]
            .sum(-1)
            .round()
            .long()
            .clamp(1, 3)
            - 1
        )
        count = F.cross_entropy(
            route["count_logits"].float()[ability_valid], count_target
        )
        terms.extend((ability, count))
        positive_logits = supervised_logits.unsqueeze(2)
        negative_logits = supervised_logits.unsqueeze(1)
        ranking_mask = (
            supervised_targets.unsqueeze(2)
            * (1.0 - supervised_targets.unsqueeze(1))
        )
        ranking_hinges = F.relu(
            float(ranking_margin) - positive_logits + negative_logits
        )
        ranking = (
            (ranking_hinges * ranking_mask).sum()
            / ranking_mask.sum().clamp_min(1.0)
        )
    if not terms:
        return zero, {
            "state": 0.0,
            "ability": 0.0,
            "count": 0.0,
            "ranking": 0.0,
        }
    total = torch.stack(terms).mean() + float(ranking_weight) * ranking
    return total, {
        "state": float(state.detach()),
        "ability": float(ability.detach()),
        "count": float(count.detach()),
        "ranking": float(ranking.detach()),
    }


def surface_logits(
    logits: torch.Tensor, surface: ActionSurface
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = legal_action_logits(
        logits,
        surface.positions,
        surface.token_ids,
        surface.token_mask,
    )
    return gathered, surface.valid


def residual_action_loss(
    action_logits: torch.Tensor, surface: ActionSurface
) -> torch.Tensor:
    valid = surface.valid.bool()
    if not valid.any():
        # Invalid surface entries are intentionally torch.finfo.min.  Summing
        # a full padded surface can overflow to -inf before multiplication by
        # zero.  Use one finite-normalised scalar to retain an autograd edge
        # without any reduction over sentinel values.
        probe = torch.nan_to_num(
            action_logits.reshape(-1)[0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return probe * 0.0
    return F.cross_entropy(
        action_logits.float()[valid], surface.gold_indices[valid]
    )


def candidate_relation_representations(
    controller: SCESRController,
    route: dict[str, torch.Tensor],
    ability_targets: torch.Tensor,
) -> torch.Tensor:
    """Select the representation that actually enters the decision path."""

    if controller.residual.config.candidate_memory_injection:
        return controller.residual.candidate_memory_vectors(route)
    weights = ability_targets.float().clamp_min(0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    return (
        route["candidate_slots"].float()
        * weights[:, None, :, None]
    ).sum(2)


RELATION_DELTA_CONTRAST_VARIANT = (
    "within_image_ability_candidate_contrast_v1"
)


def all_candidate_relation_supervision_loss(
    controller: SCESRController,
    route: dict[str, torch.Tensor],
    targets: torch.Tensor,
    active: torch.Tensor,
    *,
    target_variant: str,
    semantic_target_projection: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise the representation named by the cache contract.

    Full relation-text targets describe the complete candidate relation slot.
    Candidate-contrast targets deliberately describe only the learned
    candidate-varying relation increment.  Binding the latter to
    ``candidate_relation_delta_low`` prevents the objective from rewriting or
    taking credit for the preserved candidate-local visual base.
    """

    if target_variant == RELATION_DELTA_CONTRAST_VARIANT:
        if controller.residual.config.relation_semantic_decision_separation:
            relation_low = route.get("candidate_relation_semantic_low")
            representation_name = "relation semantic"
        else:
            relation_low = route.get("candidate_relation_delta_low")
            representation_name = "relation delta"
        if relation_low is None:
            raise RuntimeError(
                f"{representation_name} contrast requires latent spatial rollout"
            )
        if semantic_target_projection is not None:
            if not controller.residual.config.relation_semantic_decision_separation:
                raise RuntimeError(
                    "fixed low-rank targets require semantic/decision separation"
                )
            if semantic_target_projection.shape != (
                relation_low.shape[-1],
                targets.shape[-1],
            ):
                raise RuntimeError(
                    "fixed semantic projection shape does not match targets"
                )
            lowrank_targets = F.linear(
                targets.float(), semantic_target_projection.float()
            )
            return relation_delta_contrast_loss(
                relation_low,
                lowrank_targets,
                active,
            )
        # The semantic/decision-separated path freezes only this projection
        # for the semantic loss.  The action path still trains slot_to_hidden
        # normally, while relation semantics receive the contrast gradient.
        projection_weight = controller.residual.slot_to_hidden.weight
        if controller.residual.config.relation_semantic_decision_separation:
            projection_weight = projection_weight.detach()
        relation_hidden = F.linear(relation_low, projection_weight)
        return relation_delta_contrast_loss(
            relation_hidden,
            targets,
            active,
        )
    if target_variant in {"", "full_relation_text_v1"}:
        return all_candidate_relation_alignment_loss(
            route["candidate_slots"],
            targets,
            active,
        )
    raise ValueError(
        f"unsupported all-candidate relation target variant: {target_variant}"
    )


def counterfactual_neutrality_loss(
    wrong_margin: torch.Tensor,
    off_margin: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Prevent shuffled candidate evidence from helping the gold action.

    Unlike the ordinary utility loss, this term is differentiated through the
    shuffled-candidate SC-ESR path.  The residual-off native policy is a
    detached reference, so the counterfactual gradient cannot modify the B0
    policy or reward a generic action prior.
    """

    valid = valid.bool()
    if not valid.any():
        return wrong_margin.sum() * 0.0
    return F.relu(
        wrong_margin[valid] - off_margin[valid].detach() + float(margin)
    ).mean()


def adaptive_need_loss(
    residual_audit: dict[str, torch.Tensor],
    surface: ActionSurface,
    *,
    closure_valid: torch.Tensor,
    off_action_logits: torch.Tensor,
    off_margin: torch.Tensor,
    margin_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teach the residual to activate only when native evidence is insufficient.

    Targets are derived from the residual-off native LM on train-only closure
    rows: an incorrect or low-margin native action needs a spatial correction;
    an already-confident native action does not.  The target is never available
    to runtime, where the learned gate uses only current hidden/evidence states.
    """

    need_logits = residual_audit["need_logits"]
    positions = surface.positions.clamp(0, need_logits.shape[1] - 1)
    rows = torch.arange(need_logits.shape[0], device=need_logits.device)
    selected = need_logits[rows, positions]
    valid = surface.valid.bool()
    if not valid.any():
        return selected.sum() * 0.0, torch.zeros_like(valid)
    native_wrong = off_action_logits.argmax(-1).ne(surface.gold_indices)
    target = (
        closure_valid.bool()
        & valid
        & (native_wrong | off_margin.lt(float(margin_threshold)))
    )
    return (
        F.binary_cross_entropy_with_logits(
            selected[valid].float(), target[valid].float()
        ),
        target,
    )


def candidate_evidence_coupling_loss(
    residual_audit: dict[str, torch.Tensor],
    surface: ActionSurface,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bind the native action-token state to its gold visual candidate slots.

    The supervision acts on SC-ESR cross-attention, not on an independent
    classifier.  Slot values remain functions only of current image tokens;
    candidate labels condition where each query looks in the overlay image.
    """
    attention = residual_audit["slot_attention"]
    if attention.ndim != 4:
        raise ValueError(
            "candidate coupling requires [batch, token, candidate, ability]"
        )
    batch, tokens, candidates, _abilities = attention.shape
    if candidate_mask.shape != (batch, candidates):
        raise ValueError("candidate coupling mask shape drift")
    valid = surface.valid.bool()
    valid = (
        valid
        & surface.positions.ge(0)
        & surface.positions.lt(tokens)
        & surface.gold_indices.ge(0)
        & surface.gold_indices.lt(candidates)
    )
    if not valid.any():
        return attention.sum() * 0.0, valid
    rows = torch.arange(batch, device=attention.device)
    safe_positions = surface.positions.clamp(0, tokens - 1)
    candidate_probability = attention[
        rows, safe_positions
    ].sum(-1).float()
    legal = candidate_mask.bool()
    candidate_probability = candidate_probability * legal.float()
    candidate_probability = candidate_probability / candidate_probability.sum(
        -1, keepdim=True
    ).clamp_min(1e-8)
    gold_probability = candidate_probability[
        rows, surface.gold_indices.clamp(0, candidates - 1)
    ]
    loss = -gold_probability[valid].clamp_min(1e-8).log().mean()
    return loss, valid


def synchronize_gradients(
    parameters: list[torch.nn.Parameter], world: int
) -> None:
    if world <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture CPU and local-CUDA dropout state for paired native forwards."""
    return torch.random.get_rng_state(), torch.cuda.get_rng_state(device)


def restore_rng(
    state: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> None:
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], device)


@torch.no_grad()
def diagnostic(
    model: torch.nn.Module,
    controller: SCESRController,
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int,
    all_candidate_relation_target_variant: str,
    relation_semantic_target_projection: torch.Tensor | None,
) -> dict[str, Any]:
    model.eval()
    total = correct = state_total = state_correct = 0
    ability_tp = ability_fp = ability_fn = 0
    ability_tp_by_class = torch.zeros(len(ABILITIES), dtype=torch.long)
    ability_fp_by_class = torch.zeros(len(ABILITIES), dtype=torch.long)
    ability_fn_by_class = torch.zeros(len(ABILITIES), dtype=torch.long)
    ability_predicted_by_class = torch.zeros(len(ABILITIES), dtype=torch.long)
    ability_target_by_class = torch.zeros(len(ABILITIES), dtype=torch.long)
    ability_weight_sum = torch.zeros(len(ABILITIES), dtype=torch.float64)
    ability_weight_min = torch.full(
        (len(ABILITIES),), float("inf"), dtype=torch.float64
    )
    ability_weight_max = torch.zeros(len(ABILITIES), dtype=torch.float64)
    ability_weight_rows = 0
    predicted_count_distribution: Counter[int] = Counter()
    residual_ratios: list[float] = []
    gain_count = gain_positive = 0
    gain_nonzero = 0
    gain_delta_sum = 0.0
    gain_full_correct = gain_off_correct = gain_shuffle_correct = 0
    off_gain_count = off_gain_positive = off_gain_nonzero = 0
    off_gain_delta_sum = 0.0
    candidate_coupling_total = candidate_coupling_correct = 0
    local_action_rows = local_gold_rows = 0
    relation_cosine_sum = 0.0
    relation_cosine_rows = 0
    all_relation_cosine_sum = 0.0
    all_relation_items = 0
    for raw in loader:
        metadata = metadata_to_device(raw, device)
        native = native_batch(raw, device)
        with controller.context(
            image_mask=metadata["image_mask"],
            token_mask=metadata["decision_hidden_mask"],
            route_positions=metadata["route_positions"],
            candidate_positions=metadata["candidate_positions"],
            candidate_mask=metadata["candidate_mask"],
            candidate_image_mask=metadata["candidate_image_mask"],
            candidate_local_valid=metadata["candidate_local_valid"],
            ablation="full",
        ):
            outputs = model(**native)
            route = controller.last_route
            audit = controller.last_residual_audit
        action, valid = surface_logits(
            outputs.logits, metadata["action_surface"]
        )
        total += int(valid.sum())
        correct += int(
            action.argmax(-1)[valid]
            .eq(metadata["action_surface"].gold_indices[valid])
            .sum()
        )
        state_valid = metadata.get(
            "state_valid", metadata["route_valid"]
        ).bool()
        ability_valid = metadata.get(
            "ability_valid", metadata["route_valid"]
        ).bool()
        state_total += int(state_valid.sum())
        state_correct += int(
            route["state_logits"][state_valid]
            .argmax(-1)
            .eq(metadata["state_targets"][state_valid])
            .sum()
        )
        predicted = route["ability_mask"][ability_valid]
        target = metadata["ability_targets"][ability_valid].bool()
        route_weights = route["ability_weights"][ability_valid].float().cpu()
        if route_weights.numel():
            ability_weight_sum += route_weights.double().sum(0)
            ability_weight_min = torch.minimum(
                ability_weight_min, route_weights.double().amin(0)
            )
            ability_weight_max = torch.maximum(
                ability_weight_max, route_weights.double().amax(0)
            )
            ability_weight_rows += int(route_weights.shape[0])
        ability_tp += int((predicted & target).sum())
        ability_fp += int((predicted & ~target).sum())
        ability_fn += int((~predicted & target).sum())
        ability_tp_by_class += (predicted & target).sum(0).cpu()
        ability_fp_by_class += (predicted & ~target).sum(0).cpu()
        ability_fn_by_class += (~predicted & target).sum(0).cpu()
        ability_predicted_by_class += predicted.sum(0).cpu()
        ability_target_by_class += target.sum(0).cpu()
        predicted_count_distribution.update(
            int(value) for value in predicted.sum(-1).cpu().tolist()
        )
        residual_ratios.append(
            float(audit["residual_to_hidden_norm"].max().float())
        )
        if (
            controller.residual.config.candidate_conditioned
            and "slot_attention" in audit
        ):
            attention = audit["slot_attention"]
            rows = torch.arange(attention.shape[0], device=attention.device)
            positions = metadata["action_surface"].positions.clamp(
                0, attention.shape[1] - 1
            )
            candidate_score = attention[rows, positions].sum(-1)
            candidate_score = candidate_score.masked_fill(
                ~metadata["candidate_mask"].bool(),
                torch.finfo(candidate_score.dtype).min,
            )
            coupling_valid = valid.bool()
            candidate_coupling_total += int(coupling_valid.sum())
            candidate_coupling_correct += int(
                candidate_score.argmax(-1)[coupling_valid]
                .eq(
                    metadata["action_surface"]
                    .gold_indices[coupling_valid]
                )
                .sum()
            )
        if controller.residual.config.candidate_local_masks:
            local_action_rows += int(valid.sum())
            gain = local_gold_candidate_mask(
                metadata["candidate_local_valid"],
                metadata["action_surface"].gold_indices,
                valid,
            )
            local_gold_rows += int(gain.sum())
        else:
            gain = metadata["gain_valid"].bool() & valid
        candidate_relation_repr = candidate_relation_representations(
            controller, route, metadata["ability_targets"]
        )
        candidate_relation_target = metadata["candidate_relation_targets"]
        if candidate_relation_repr.shape[-1] == candidate_relation_target.shape[-1]:
            _relation_loss, relation_cosine, relation_valid = (
                candidate_relation_alignment_loss(
                    candidate_relation_repr,
                    candidate_relation_target,
                    metadata["action_surface"].gold_indices,
                    metadata["candidate_relation_present"].bool() & gain,
                )
            )
        else:
            relation_valid = torch.zeros(
                metadata["candidate_relation_present"].shape,
                dtype=torch.bool,
                device=device,
            )
            relation_cosine = torch.zeros(
                relation_valid.shape, dtype=torch.float32, device=device
            )
        if relation_valid.any():
            relation_cosine_sum += float(
                relation_cosine[relation_valid].sum()
            )
            relation_cosine_rows += int(relation_valid.sum())
        all_relation_target = metadata["all_candidate_relation_targets"]
        if all_relation_target.shape[-1] == route["candidate_slots"].shape[-1]:
            _, all_relation_cosine, all_relation_valid = (
                all_candidate_relation_supervision_loss(
                    controller,
                    route,
                    all_relation_target,
                    metadata["all_candidate_relation_active"].bool(),
                    target_variant=all_candidate_relation_target_variant,
                    semantic_target_projection=(
                        relation_semantic_target_projection
                    ),
                )
            )
        else:
            all_relation_valid = torch.zeros(
                metadata["all_candidate_relation_active"].shape,
                dtype=torch.bool,
                device=device,
            )
            all_relation_cosine = torch.zeros(
                all_relation_valid.shape, dtype=torch.float32, device=device
            )
        if all_relation_valid.any():
            all_relation_cosine_sum += float(
                all_relation_cosine[all_relation_valid].sum()
            )
            all_relation_items += int(all_relation_valid.sum())
        if gain.any():
            with controller.context(
                image_mask=metadata["image_mask"],
                token_mask=metadata["decision_hidden_mask"],
                route_positions=metadata["route_positions"],
                candidate_positions=metadata["candidate_positions"],
                candidate_mask=metadata["candidate_mask"],
                candidate_image_mask=metadata["candidate_image_mask"],
                candidate_local_valid=metadata["candidate_local_valid"],
                ablation="off",
            ):
                off_outputs = model(**native)
            off_action, _ = surface_logits(
                off_outputs.logits, metadata["action_surface"]
            )
            with controller.context(
                image_mask=metadata["image_mask"],
                token_mask=metadata["decision_hidden_mask"],
                route_positions=metadata["route_positions"],
                candidate_positions=metadata["candidate_positions"],
                candidate_mask=metadata["candidate_mask"],
                candidate_image_mask=metadata["candidate_image_mask"],
                candidate_local_valid=metadata["candidate_local_valid"],
                ablation=(
                    "shuffle_relation"
                    if controller.residual.config.latent_spatial_rollout
                    else (
                        "shuffle_candidate"
                        if controller.residual.config.candidate_local_masks
                        else "shuffle_image"
                    )
                ),
            ):
                shuffled_outputs = model(**native)
            shuffled_action, _ = surface_logits(
                shuffled_outputs.logits, metadata["action_surface"]
            )
            full_margin = gold_margin(
                action, metadata["action_surface"].gold_indices
            )
            off_margin = gold_margin(
                off_action, metadata["action_surface"].gold_indices
            )
            shuffled_margin = gold_margin(
                shuffled_action, metadata["action_surface"].gold_indices
            )
            delta = full_margin[gain] - shuffled_margin[gain]
            gain_count += int(delta.numel())
            gold_indices = metadata["action_surface"].gold_indices[gain]
            gain_full_correct += int(
                action.argmax(-1)[gain].eq(gold_indices).sum()
            )
            gain_off_correct += int(
                off_action.argmax(-1)[gain].eq(gold_indices).sum()
            )
            gain_shuffle_correct += int(
                shuffled_action.argmax(-1)[gain].eq(gold_indices).sum()
            )
            gain_positive += int(delta.gt(0).sum())
            gain_nonzero += int(delta.ne(0).sum())
            gain_delta_sum += float(delta.sum())
            off_delta = full_margin[gain] - off_margin[gain]
            off_gain_count += int(off_delta.numel())
            off_gain_positive += int(off_delta.gt(0).sum())
            off_gain_nonzero += int(off_delta.ne(0).sum())
            off_gain_delta_sum += float(off_delta.sum())
            del off_outputs, shuffled_outputs
        if total >= max_samples:
            break
    model.train()
    return {
        "rows": total,
        "action_accuracy": correct / max(1, total),
        "state_accuracy": state_correct / max(1, state_total),
        "ability_micro_f1": (
            2 * ability_tp
            / max(1, 2 * ability_tp + ability_fp + ability_fn)
        ),
        "ability_predicted_counts": {
            ability: int(ability_predicted_by_class[index])
            for index, ability in enumerate(ABILITIES)
        },
        "ability_target_counts": {
            ability: int(ability_target_by_class[index])
            for index, ability in enumerate(ABILITIES)
        },
        "ability_weight_mean": {
            ability: float(
                ability_weight_sum[index] / max(1, ability_weight_rows)
            )
            for index, ability in enumerate(ABILITIES)
        },
        "ability_weight_min": {
            ability: (
                float(ability_weight_min[index])
                if ability_weight_rows
                else 0.0
            )
            for index, ability in enumerate(ABILITIES)
        },
        "ability_weight_max": {
            ability: float(ability_weight_max[index])
            for index, ability in enumerate(ABILITIES)
        },
        "ability_per_class_precision": {
            ability: (
                int(ability_tp_by_class[index])
                / max(
                    1,
                    int(
                        ability_tp_by_class[index]
                        + ability_fp_by_class[index]
                    ),
                )
            )
            for index, ability in enumerate(ABILITIES)
        },
        "ability_per_class_recall": {
            ability: (
                int(ability_tp_by_class[index])
                / max(
                    1,
                    int(
                        ability_tp_by_class[index]
                        + ability_fn_by_class[index]
                    ),
                )
            )
            for index, ability in enumerate(ABILITIES)
        },
        "predicted_ability_count_distribution": dict(
            sorted(predicted_count_distribution.items())
        ),
        "max_residual_ratio": max(residual_ratios or [0.0]),
        "gain_rows": gain_count,
        "full_minus_shuffle_margin_mean": (
            gain_delta_sum / max(1, gain_count)
        ),
        "gain_rows_full_action_accuracy": (
            gain_full_correct / max(1, gain_count)
        ),
        "gain_rows_off_action_accuracy": (
            gain_off_correct / max(1, gain_count)
        ),
        "gain_rows_shuffle_action_accuracy": (
            gain_shuffle_correct / max(1, gain_count)
        ),
        "full_beats_shuffle_fraction": (
            gain_positive / max(1, gain_count)
        ),
        "full_differs_from_shuffle_fraction": (
            gain_nonzero / max(1, gain_count)
        ),
        "full_minus_off_margin_mean": (
            off_gain_delta_sum / max(1, off_gain_count)
        ),
        "full_beats_off_fraction": (
            off_gain_positive / max(1, off_gain_count)
        ),
        "full_differs_from_off_fraction": (
            off_gain_nonzero / max(1, off_gain_count)
        ),
        "candidate_evidence_attention_accuracy": (
            candidate_coupling_correct / max(1, candidate_coupling_total)
        ),
        "candidate_evidence_attention_rows": candidate_coupling_total,
        "candidate_local_action_rows": local_action_rows,
        "candidate_local_gold_rows": local_gold_rows,
        "candidate_local_gold_coverage": (
            local_gold_rows / max(1, local_action_rows)
        ),
        "candidate_relation_cosine_mean": (
            relation_cosine_sum / max(1, relation_cosine_rows)
        ),
        "candidate_relation_rows": relation_cosine_rows,
        "all_candidate_relation_cosine_mean": (
            all_relation_cosine_sum / max(1, all_relation_items)
        ),
        "all_candidate_relation_items": all_relation_items,
    }


def main() -> None:
    args = arguments()
    if not 0 <= args.gate_warmup_ratio <= 1:
        raise ValueError("gate warmup ratio must be in [0,1]")
    if args.candidate_coupling_weight < 0:
        raise ValueError("candidate coupling weight must be non-negative")
    if args.candidate_relation_weight < 0:
        raise ValueError("candidate relation weight must be non-negative")
    if args.all_candidate_relation_weight < 0:
        raise ValueError("all-candidate relation weight must be non-negative")
    if (
        args.candidate_relation_weight > 0
        and not args.candidate_relation_cache
    ):
        raise ValueError(
            "candidate relation weight requires --candidate-relation-cache"
        )
    if (
        args.candidate_relation_weight > 0
        and not args.candidate_local_masks
    ):
        raise ValueError(
            "candidate relation supervision requires candidate-local masks"
        )
    if (
        args.all_candidate_relation_weight > 0
        and not args.all_candidate_relation_cache
    ):
        raise ValueError(
            "all-candidate relation weight requires its training cache"
        )
    if args.all_candidate_relation_weight > 0 and not args.latent_spatial_rollout:
        raise ValueError(
            "all-candidate relation supervision requires latent spatial rollout"
        )
    if (
        args.relation_semantic_decision_separation
        and not args.latent_spatial_rollout
    ):
        raise ValueError(
            "relation semantic/decision separation requires latent rollout"
        )
    if args.relation_decision_learning_rate < 0:
        raise ValueError("relation decision learning rate must be non-negative")
    if (
        args.relation_decision_learning_rate > 0
        and not args.relation_semantic_decision_separation
    ):
        raise ValueError(
            "relation decision LR requires semantic/decision separation"
        )
    if (
        args.relation_semantic_target_space == "fixed_lowrank"
        and not args.relation_semantic_decision_separation
    ):
        raise ValueError(
            "fixed low-rank relation targets require "
            "--relation-semantic-decision-separation"
        )
    if args.candidate_coupling_weight > 0 and not args.candidate_conditioned:
        raise ValueError(
            "candidate coupling requires --candidate-conditioned"
        )
    if (
        args.counterfactual_gradient_weight < 0
        or args.counterfactual_neutrality_margin < 0
    ):
        raise ValueError("counterfactual gradient settings must be non-negative")
    if (
        args.counterfactual_gradient_weight > 0
        and not args.candidate_local_masks
    ):
        raise ValueError(
            "counterfactual gradients require --candidate-local-masks"
        )
    if args.candidate_local_masks and not args.candidate_conditioned:
        raise ValueError(
            "candidate-local masks require --candidate-conditioned"
        )
    if (
        args.global_conditioned_local_slots
        and not args.candidate_local_masks
    ):
        raise ValueError(
            "global-conditioned local slots require --candidate-local-masks"
        )
    if (
        args.candidate_value_interaction
        and not args.candidate_local_masks
    ):
        raise ValueError(
            "candidate-value interaction requires --candidate-local-masks"
        )
    if (
        args.candidate_memory_injection
        and not args.candidate_local_masks
    ):
        raise ValueError(
            "candidate-memory injection requires --candidate-local-masks"
        )
    if (
        args.candidate_memory_injection
        and args.candidate_value_interaction
    ):
        raise ValueError(
            "candidate-memory and candidate-value paths are exclusive"
        )
    if args.candidate_memory_injection and args.adaptive_need_gate:
        raise ValueError(
            "candidate-memory injection cannot use --adaptive-need-gate"
        )
    if args.candidate_memory_injection and args.candidate_coupling_weight > 0:
        raise ValueError(
            "candidate-memory injection does not use attention coupling"
        )
    if args.latent_spatial_rollout and not (
        args.candidate_conditioned
        and args.candidate_local_masks
        and args.global_conditioned_local_slots
    ):
        raise ValueError(
            "latent spatial rollout requires candidate-conditioned local masks, "
            "and global conditioning"
        )
    if args.exact_stable_course and args.historical_distributed_course:
        raise ValueError("choose exactly one formal course sampler")
    if args.require_sc_lsr_v1_contract and args.require_sc_lsr_closure_contract:
        raise ValueError("choose one SC-LSR formal course contract")
    if (
        args.residual_action_weight < 0
        or args.need_weight < 0
        or args.need_margin_threshold < 0
    ):
        raise ValueError("V4 residual/need weights must be non-negative")
    if args.need_weight > 0 and not args.adaptive_need_gate:
        raise ValueError("need loss requires --adaptive-need-gate")
    if args.gradient_separated_native_lm and (
        args.auxiliary_gradient_scope != "sc_esr"
    ):
        raise ValueError(
            "gradient-separated native LM requires SC-ESR-only auxiliary gradients"
        )
    device, _local_rank, rank, world = distributed_device()
    is_main = rank == 0
    # Model and SC-ESR parameters must initialize identically on every rank.
    set_seed(args.seed)

    output = Path(args.output_dir).resolve()
    if is_main:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    log_handle = None
    if is_main and args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_handle = Path(args.log_file).open("a", encoding="utf-8")

    def log(payload: dict[str, Any]) -> None:
        if not is_main:
            return
        line = json.dumps(
            {**payload, "time": time.time()},
            ensure_ascii=False,
            sort_keys=True,
        )
        print(line, flush=True)
        if log_handle is not None:
            log_handle.write(line + "\n")
            log_handle.flush()

    data = Path(args.data_dir).resolve()
    audit = json.loads((data / "dataset_audit.json").read_text())
    if not audit.get("passed"):
        raise RuntimeError("SC-ESR data audit did not pass")
    train_rows = read_jsonl(data / "train.jsonl")
    dev_rows = read_jsonl(data / "dev.jsonl")
    if (
        len(train_rows) != args.expected_train_rows
        or len(dev_rows) != args.expected_dev_rows
    ):
        raise RuntimeError("SC-ESR row count drift")
    if args.max_train_samples > 0:
        lsr_smoke_ids: set[str] | None = None
        if args.latent_spatial_rollout:
            index_path = (
                Path(args.all_candidate_relation_cache) / "item_index.jsonl"
            )
            if not index_path.is_file():
                raise RuntimeError(
                    "SC-LSR smoke requires an all-candidate cache index"
                )
            lsr_smoke_ids = {
                str(item["sample_id"]) for item in read_jsonl(index_path)
            }
        ranked = sorted(
            train_rows,
            key=lambda row: hashlib.sha256(
                f"{args.seed}\0{row['sample_id']}".encode()
            ).hexdigest(),
        )
        causal = [
            row
            for row in ranked
            if row["spatial_gain_supervision_valid"]
            and (
                lsr_smoke_ids is None
                or str(row["sample_id"]) in lsr_smoke_ids
            )
        ]
        by_bucket: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in causal:
            by_bucket[
                (
                    row["goal"],
                    int(row["state_label"]),
                    len(row["legal_actions"]),
                )
            ].append(row)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        # First reserve genuine closure rows for every ability.
        for ability_index in range(len(ABILITIES)):
            candidates = [
                row
                for row in causal
                if row["ability_targets"][ability_index]
            ]
            for row in candidates[:2]:
                if row["sample_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["sample_id"])
        # Then prefer dense matched buckets so shuffled-RGB supervision is
        # actually exercised rather than padded with duplicate singletons.
        def matchable_rows(values: list[dict[str, Any]]) -> int:
            action_counts = Counter(row["gold_action"] for row in values)
            majority = max(action_counts.values())
            return min(len(values), 2 * (len(values) - majority))

        for _key, values in sorted(
            by_bucket.items(),
            key=lambda item: (matchable_rows(item[1]), len(item[1])),
            reverse=True,
        ):
            action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in values:
                action_groups[row["gold_action"]].append(row)
            balanced: list[dict[str, Any]] = []
            last_action = ""
            while any(action_groups.values()):
                choices = sorted(
                    (
                        (len(group), action)
                        for action, group in action_groups.items()
                        if group and action != last_action
                    ),
                    reverse=True,
                )
                if not choices:
                    choices = sorted(
                        (
                            (len(group), action)
                            for action, group in action_groups.items()
                            if group
                        ),
                        reverse=True,
                    )
                last_action = choices[0][1]
                balanced.append(action_groups[last_action].pop())
            for row in balanced:
                if row["sample_id"] in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row["sample_id"])
                if len(selected) == args.max_train_samples:
                    break
            if len(selected) == args.max_train_samples:
                break
        if len(selected) < args.max_train_samples:
            raise RuntimeError("not enough audited causal rows for overfit smoke")
        selected = selected[: args.max_train_samples]
        # Ensure every ability receives supervised gradients in overfit smoke.
        covered = {
            ability
            for row in selected
            if row["route_supervision_valid"]
            for ability, active in zip(ABILITIES, row["ability_targets"])
            if active
        }
        for ability_index, ability in enumerate(ABILITIES):
            if ability in covered:
                continue
            raise RuntimeError(f"causal smoke lacks ability: {ability}")
        train_rows = selected
    if args.max_dev_samples > 0:
        dev_rows = dev_rows[: args.max_dev_samples]

    expected_b0_sha256 = str(args.expected_b0_sha256).strip().lower()
    if (
        len(expected_b0_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_b0_sha256)
    ):
        raise RuntimeError("invalid --expected-b0-sha256")
    b0 = Path(args.b0_adapter).resolve()
    if sha256(b0 / "adapter_model.safetensors") != expected_b0_sha256:
        raise RuntimeError("fixed B0 adapter hash drift")
    adapter_config = json.loads((b0 / "adapter_config.json").read_text())
    if int(adapter_config.get("r", -1)) != 64:
        raise RuntimeError("fixed B0 is not the audited r64 adapter")
    merged_b0_manifest: dict[str, Any] | None = None
    if args.fresh_move_delta_on_merged_b0:
        if not args.merged_b0_manifest:
            raise RuntimeError(
                "fresh Move delta requires --merged-b0-manifest"
            )
        merged_manifest_path = Path(args.merged_b0_manifest).resolve()
        merged_b0_manifest = json.loads(merged_manifest_path.read_text())
        merged_checks = {
            "schema": (
                merged_b0_manifest.get("schema")
                == "audited_b0_merged_base_v1"
            ),
            "passed": merged_b0_manifest.get("passed") is True,
            "source_b0": (
                merged_b0_manifest.get("source_b0_adapter_sha256")
                == expected_b0_sha256
            ),
            "output_path": (
                Path(merged_b0_manifest.get("output_dir", "")).resolve()
                == Path(args.model_path).resolve()
            ),
        }
        if not all(merged_checks.values()):
            raise RuntimeError(
                f"merged B0 manifest contract failed: {merged_checks}"
            )
    elif args.merged_b0_manifest:
        raise RuntimeError(
            "--merged-b0-manifest requires "
            "--fresh-move-delta-on-merged-b0"
        )

    evidence = EvidenceCache(Path(args.evidence_cache).resolve(), data=data)
    candidate_relation = (
        CandidateRelationCache(
            Path(args.candidate_relation_cache).resolve(), data=data
        )
        if args.candidate_relation_cache
        else None
    )
    all_candidate_relation = (
        AllCandidateRelationCache(
            Path(args.all_candidate_relation_cache).resolve(), data=data
        )
        if args.all_candidate_relation_cache
        else None
    )
    all_candidate_relation_target_variant = (
        str(
            all_candidate_relation.manifest.get(
                "relation_target_variant", "full_relation_text_v1"
            )
        )
        if all_candidate_relation is not None
        else ""
    )
    if all_candidate_relation_target_variant not in {
        "",
        "full_relation_text_v1",
        RELATION_DELTA_CONTRAST_VARIANT,
    }:
        raise RuntimeError(
            "unsupported all-candidate relation target variant: "
            f"{all_candidate_relation_target_variant}"
        )
    if (
        candidate_relation is not None
        and candidate_relation.hidden_size != evidence.hidden_size
    ):
        raise RuntimeError("candidate/evidence target hidden sizes differ")
    if (
        candidate_relation is not None
        and args.candidate_memory_injection
        and int(candidate_relation.manifest["source_layer"])
        != args.injection_layer
    ):
        raise RuntimeError(
            "candidate relation target layer must equal memory injection layer"
        )
    if (
        all_candidate_relation is not None
        and all_candidate_relation.hidden_size != evidence.hidden_size
    ):
        raise RuntimeError(
            "all-candidate/evidence target hidden sizes differ"
        )
    processor = load_processor(args.model_path)
    train_dataset = SCESRDataset(train_rows)
    dev_dataset = SCESRDataset(dev_rows)
    collator = SCESRCollator(
        processor,
        evidence,
        candidate_relation,
        all_candidate_relation,
        max_length=args.max_length,
        image_token_id=args.image_token_id,
        runtime_aligned_injection=args.runtime_aligned_injection,
        candidate_local_masks=args.candidate_local_masks,
    )
    if args.historical_distributed_course:
        # Match train_cold_start_vision_sft.py exactly: DistributedSampler uses
        # its default seed=0 and set_epoch(epoch), independent of model seed.
        course_sampler: Any = DistributedSampler(
            train_dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_batch_size,
            sampler=course_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
    else:
        sampler_type = (
            ExactStableCourseBatchSampler
            if args.exact_stable_course
            else EvidenceBucketBatchSampler
        )
        course_sampler = sampler_type(
            train_rows,
            batch_size=args.per_device_batch_size,
            world=world,
            rank=rank,
            seed=args.seed,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=course_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
    diagnostic_rows = (
        train_rows if args.diagnostic_split == "train" else dev_rows
    )
    if args.diagnostic_action_only:
        diagnostic_rows = [
            row
            for row in diagnostic_rows
            if row.get("gold_action") and row.get("legal_actions")
        ]
    if not diagnostic_rows:
        raise RuntimeError("SC-ESR diagnostic selection is empty")
    diagnostic_dataset = SCESRDataset(diagnostic_rows)
    diagnostic_batch_sampler = EvidenceBucketBatchSampler(
        diagnostic_rows,
        batch_size=max(2, args.per_device_batch_size),
        world=1,
        rank=0,
        seed=args.seed + 997,
    )
    diagnostic_loader = DataLoader(
        diagnostic_dataset,
        batch_sampler=diagnostic_batch_sampler,
        num_workers=0,
        collate_fn=collator,
    )

    base, loader_name = load_model(args.model_path, device)
    base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    if hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base.gradient_checkpointing_enable()
    if args.fresh_move_delta_on_merged_b0:
        target_modules = adapter_config.get("target_modules")
        if not isinstance(target_modules, list) or not target_modules:
            raise RuntimeError("fixed B0 target_modules are missing")
        model = get_peft_model(
            base,
            LoraConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=float(adapter_config.get("lora_dropout", 0.05)),
                bias="none",
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                init_lora_weights=True,
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            base, str(b0), is_trainable=True
        )
    vision_lora = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" in name
        and any(
            marker in name.lower()
            for marker in ("vision_tower", "vision_model", ".visual.", ".visual_")
        )
    ]
    if vision_lora:
        raise RuntimeError(
            "vision LoRA is forbidden for the InternVL Move retrain; found "
            f"{len(vision_lora)} trainable tensors ({', '.join(vision_lora[:8])})"
        )
    hidden_size = hidden_size_from_config(model.config)
    residual = StateConditionedEvidenceSlotResidual(
        SCESRConfig(
            hidden_size=hidden_size,
            slot_rank=args.slot_rank,
            source_layer=args.source_layer,
            injection_layer=args.injection_layer,
            max_residual_scale=args.max_residual_scale,
            state_dropout=args.state_dropout,
            candidate_conditioned=args.candidate_conditioned,
            independent_ability_gates=args.independent_ability_gates,
            adaptive_need_gate=args.adaptive_need_gate,
            candidate_local_masks=args.candidate_local_masks,
            global_conditioned_local_slots=(
                args.global_conditioned_local_slots
            ),
            candidate_value_interaction=(
                args.candidate_value_interaction
            ),
            candidate_memory_injection=(
                args.candidate_memory_injection
            ),
            latent_spatial_rollout=args.latent_spatial_rollout,
            relation_semantic_decision_separation=(
                args.relation_semantic_decision_separation
            ),
            straight_through_sparse_route=(
                args.straight_through_sparse_route
            ),
            soft_ability_route=args.soft_ability_route,
            local_evidence_control=args.local_evidence_control,
        )
    ).to(device=device, dtype=torch.bfloat16)
    relation_semantic_target_projection: torch.Tensor | None = None
    relation_semantic_projection_sha256: str | None = None
    if args.relation_semantic_target_space == "fixed_lowrank":
        projection_cpu = fixed_orthonormal_projection(
            hidden_size,
            args.slot_rank,
            seed=args.relation_semantic_projection_seed,
        )
        relation_semantic_projection_sha256 = hashlib.sha256(
            projection_cpu.numpy().tobytes()
        ).hexdigest()
        relation_semantic_target_projection = projection_cpu.to(device)
    model.add_module("_sc_esr_residual", residual)
    controller = SCESRController(model, residual, route_seed=args.seed + rank)

    residual_ids = {id(parameter) for parameter in residual.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in residual_ids
    ]
    output_gate = (
        residual.candidate_memory_gate
        if args.candidate_memory_injection
        else residual.residual_gate
    )
    gate_parameters = [output_gate]
    if args.global_conditioned_local_slots:
        gate_parameters.append(residual.global_local_gate)
    if args.candidate_value_interaction:
        gate_parameters.append(residual.candidate_value_gate)
    # output_gate already is candidate_memory_gate on the memory path.
    gate_parameters = list(dict.fromkeys(gate_parameters))
    relation_decision_parameters = (
        [
            parameter
            for adapter in residual.relation_decision_adapters
            for parameter in adapter.parameters()
        ]
        if args.relation_semantic_decision_separation
        else []
    )
    relation_decision_ids = {
        id(parameter) for parameter in relation_decision_parameters
    }
    residual_parameters = [
        parameter
        for parameter in residual.parameters()
        if (
            all(parameter is not gate for gate in gate_parameters)
            and id(parameter) not in relation_decision_ids
        )
    ]
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and id(parameter) not in residual_ids
        and "lora_" not in name
    ]
    if unexpected or not backbone_parameters or not residual_parameters:
        raise RuntimeError(
            f"invalid trainable parameter contract unexpected={unexpected[:8]}"
        )
    parameters = [
        *backbone_parameters,
        *residual_parameters,
        *relation_decision_parameters,
        *gate_parameters,
    ]
    auxiliary_parameters = [
        *residual_parameters,
        *relation_decision_parameters,
        *gate_parameters,
    ]
    relation_decision_lr = (
        args.relation_decision_learning_rate
        if args.relation_decision_learning_rate > 0
        else args.residual_learning_rate
    )
    optimizer_groups = [
        {
            "params": backbone_parameters,
            "lr": args.backbone_learning_rate,
        },
        {
            "params": residual_parameters,
            "lr": args.residual_learning_rate,
        },
    ]
    if relation_decision_parameters:
        optimizer_groups.append(
            {
                "params": relation_decision_parameters,
                "lr": relation_decision_lr,
            }
        )
    optimizer_groups.append(
        {
            "params": gate_parameters,
            "lr": args.gate_learning_rate,
            "weight_decay": 0.0,
        }
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    # ``micro`` is intentionally continuous across epoch boundaries below.
    # Match scheduler/provenance to the loop that actually runs: 625
    # batches/epoch * 2 epochs with grad_accum=2 is exactly 625 updates, not
    # ceil(625 / 2) * 2 == 626.
    total_micro_batches = len(loader) * args.epochs
    total_steps = total_micro_batches // args.grad_accum_steps
    if total_steps <= 0:
        raise RuntimeError(
            "training course has fewer micro-batches than grad_accum_steps"
        )
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    if args.require_historical_course_contract:
        historical_checks = {
            "candidate_conditioned": args.candidate_conditioned,
            "exact_stable_course": args.exact_stable_course,
            "declared_closure_world_size": (
                args.expected_closure_world_size >= 1
                and world == args.expected_closure_world_size
            ),
            "per_device_batch_4": args.per_device_batch_size == 4,
            "grad_accum_1": args.grad_accum_steps == 1,
            "epochs_1": args.epochs == 1,
            "optimizer_steps_1250": total_steps == 1250,
            "backbone_lr_contract": math.isclose(
                args.backbone_learning_rate, 1e-5
            ) if not args.open_capacity_exploration_contract else (
                2e-6 <= args.backbone_learning_rate <= 1e-5
            ),
            "seed_20260713": args.seed == 20260713,
        }
        if not all(historical_checks.values()):
            raise RuntimeError(
                f"historical course contract failed: {historical_checks}"
            )
    gate_warmup_steps = int(total_steps * args.gate_warmup_ratio)
    state_rows = [
        row
        for row in train_rows
        if row.get(
            "state_supervision_valid",
            row.get("route_supervision_valid", False),
        )
    ]
    ability_rows = [
        row
        for row in train_rows
        if row.get(
            "ability_supervision_valid",
            row.get("route_supervision_valid", False),
        )
    ]
    if args.require_v3_course_contract:
        v3_checks = {
            "historical_course_contract": args.require_historical_course_contract,
            "historical_sft_weight_semantics": (
                args.native_lm_weight_semantics == "historical_sft"
            ),
            "sc_esr_auxiliary_gradient_scope": (
                args.auxiliary_gradient_scope == "sc_esr"
            ),
            "runtime_aligned_injection": args.runtime_aligned_injection,
            "candidate_conditioned": args.candidate_conditioned,
            "navigation_route_rows_only": len(ability_rows) == 9049,
            "no_auxiliary_route_labels": all(
                not row.get("ability_supervision_valid", False)
                for row in train_rows
                if row.get("mix_role") == "training_only_spatial_auxiliary"
            ),
        }
        if not all(v3_checks.values()):
            raise RuntimeError(f"V3 course contract failed: {v3_checks}")
    if args.require_v4_course_contract:
        v4_checks = {
            "order_preserving_data": (
                audit.get("schema")
                == "sc_esr_candidate_move_course_v4_order_preserving"
                and audit.get("checks", {}).get(
                    "historical_row_order_preserved"
                )
            ),
            "historical_distributed_course": args.historical_distributed_course,
            "not_closure_grouped_course": not args.exact_stable_course,
            "world_size_8": world == 8,
            "per_device_batch_4": args.per_device_batch_size == 4,
            "grad_accum_1": args.grad_accum_steps == 1,
            "epochs_1": args.epochs == 1,
            "optimizer_steps_1250": total_steps == 1250,
            "backbone_lr_contract": math.isclose(
                args.backbone_learning_rate, 1e-5
            ) if not args.open_capacity_exploration_contract else (
                2e-6 <= args.backbone_learning_rate <= 1e-5
            ),
            "historical_sft_weight_semantics": (
                args.native_lm_weight_semantics == "historical_sft"
            ),
            "gradient_separated_native_lm": args.gradient_separated_native_lm,
            "sc_esr_auxiliary_gradient_scope": (
                args.auxiliary_gradient_scope == "sc_esr"
            ),
            "runtime_aligned_injection": args.runtime_aligned_injection,
            "candidate_conditioned": args.candidate_conditioned,
            "independent_ability_gates": args.independent_ability_gates,
            "class_balanced_route": args.route_class_balanced,
            "need_gate_contract": (
                (
                    not args.adaptive_need_gate
                    and args.need_weight == 0
                )
                if args.open_capacity_exploration_contract
                else args.adaptive_need_gate
            ),
            "native_lm_action_coupling": args.residual_action_weight > 0,
            "open_capacity_contract": (
                (
                    args.soft_ability_route
                    and not args.straight_through_sparse_route
                    and 0.10 <= args.max_residual_scale <= 0.20
                    and args.counterfactual_gradient_weight > 0
                )
                if args.open_capacity_exploration_contract
                else not args.soft_ability_route
            ),
            "need_supervision_contract": (
                args.need_weight == 0
                if args.open_capacity_exploration_contract
                else args.need_weight > 0
            ),
            "navigation_route_rows_only": len(ability_rows) == 9049,
            "no_auxiliary_route_labels": all(
                not row.get("ability_supervision_valid", False)
                for row in train_rows
                if row.get("mix_role") == "training_only_spatial_auxiliary"
            ),
        }
        if not all(v4_checks.values()):
            raise RuntimeError(f"V4 course contract failed: {v4_checks}")
    if args.require_sc_lsr_v1_contract:
        lsr_checks = {
            "order_preserving_v4_data": (
                audit.get("schema")
                == "sc_esr_candidate_move_course_v4_order_preserving"
                and audit.get("checks", {}).get(
                    "historical_row_order_preserved"
                )
            ),
            "world_size_8": world == 8,
            "global_batch_32": (
                world
                * args.per_device_batch_size
                * args.grad_accum_steps
                == 32
            ),
            "epochs_1": args.epochs == 1,
            "optimizer_steps_1250": total_steps == 1250,
            "backbone_lr_1e_5": math.isclose(
                args.backbone_learning_rate, 1e-5
            ),
            "historical_distributed_course": args.historical_distributed_course,
            "historical_sft_weight_semantics": (
                args.native_lm_weight_semantics == "historical_sft"
            ),
            "gradient_separated_native_lm": args.gradient_separated_native_lm,
            "sc_esr_auxiliary_gradient_scope": (
                args.auxiliary_gradient_scope == "sc_esr"
            ),
            "runtime_aligned_injection": args.runtime_aligned_injection,
            "candidate_conditioned": args.candidate_conditioned,
            "candidate_local_masks": args.candidate_local_masks,
            "global_conditioned": args.global_conditioned_local_slots,
            "v4_native_decision_residual": (
                not args.candidate_memory_injection
                and args.adaptive_need_gate
                and args.candidate_coupling_weight > 0
            ),
            "latent_spatial_rollout": args.latent_spatial_rollout,
            "all_candidate_relation_supervision": (
                args.all_candidate_relation_weight > 0
                and all_candidate_relation is not None
                and int(
                    all_candidate_relation.manifest["accepted_rows"]
                )
                >= 2500
            ),
            "independent_ability_gates": args.independent_ability_gates,
            "class_balanced_route": args.route_class_balanced,
            "native_lm_action_coupling": args.residual_action_weight > 0,
            "single_stage_from_fixed_b0": True,
        }
        if not all(lsr_checks.values()):
            raise RuntimeError(f"SC-LSR V1 contract failed: {lsr_checks}")
    if args.require_sc_lsr_closure_contract:
        train_summary = (audit.get("splits") or {}).get("train") or {}
        data_checks = audit.get("checks") or {}
        closure_checks = {
            "closure_course_schema": (
                audit.get("schema") == "sc_lsr_move_closure_course_v1"
            ),
            "data_audit_passed": audit.get("passed") is True,
            "no_validation_training_rows": (
                data_checks.get("official_validation_training_rows") == 0
            ),
            "all_rows_direct_move": (
                train_summary.get("rows") == 10_000
                and train_summary.get("direct_move_rows") == 10_000
                and all(
                    row.get("mix_role") == "direct_move_only_policy"
                    for row in train_rows
                )
            ),
            "no_standalone_spatial_replay": (
                train_summary.get("standalone_spatial_replay_rows") == 0
                and data_checks.get("standalone_spatial_replay_removed") is True
            ),
            "same_row_relation_closure": (
                audit.get("relation_supervision_rows") == 2_979
                and data_checks.get(
                    "all_relation_rows_attach_to_direct_move"
                )
                is True
                and all_candidate_relation is not None
                and int(all_candidate_relation.manifest["accepted_rows"])
                == 2_979
            ),
            "route_supervision_preserved": len(ability_rows) == 9_049,
            "declared_closure_world_size": (
                args.expected_closure_world_size >= 1
                and world == args.expected_closure_world_size
            ),
            "global_batch_32": (
                world
                * args.per_device_batch_size
                * args.grad_accum_steps
                == 32
            ),
            "declared_closure_epoch_budget": (
                args.expected_closure_epochs >= 1
                and args.epochs == args.expected_closure_epochs
            ),
            "continuous_accumulation_optimizer_steps": (
                total_steps
                == (
                    len(loader)
                    * args.expected_closure_epochs
                    // args.grad_accum_steps
                )
            ),
            "backbone_lr_contract": math.isclose(
                args.backbone_learning_rate, 1e-5
            ) if not args.open_capacity_exploration_contract else (
                2e-6 <= args.backbone_learning_rate <= 1e-5
            ),
            "historical_distributed_sampler": (
                args.historical_distributed_course
            ),
            "historical_sft_weight_semantics": (
                args.native_lm_weight_semantics == "historical_sft"
            ),
            "gradient_separated_native_lm": (
                args.gradient_separated_native_lm
            ),
            "sc_esr_auxiliary_gradient_scope": (
                args.auxiliary_gradient_scope == "sc_esr"
            ),
            "runtime_aligned_injection": args.runtime_aligned_injection,
            "candidate_conditioned": args.candidate_conditioned,
            "candidate_local_masks": args.candidate_local_masks,
            "global_conditioned": args.global_conditioned_local_slots,
            "latent_spatial_rollout": args.latent_spatial_rollout,
            "independent_ability_gates": args.independent_ability_gates,
            "need_gate_contract": (
                (
                    not args.adaptive_need_gate
                    and args.need_weight == 0
                )
                if args.open_capacity_exploration_contract
                else args.adaptive_need_gate
            ),
            "ability_route_contract": (
                (
                    args.soft_ability_route
                    and not args.straight_through_sparse_route
                )
                if args.open_capacity_exploration_contract
                else not args.soft_ability_route
            ),
            "residual_scale_contract": (
                0.10 <= args.max_residual_scale <= 0.20
                if args.open_capacity_exploration_contract
                else math.isclose(args.max_residual_scale, 0.02)
            ),
            "counterfactual_gradient_contract": (
                args.counterfactual_gradient_weight > 0
                if args.open_capacity_exploration_contract
                else True
            ),
            "semantic_binding_contract": (
                (
                    args.relation_semantic_decision_separation
                    and args.relation_semantic_target_space
                    == "fixed_lowrank"
                )
                if args.open_capacity_exploration_contract
                else True
            ),
            "native_lm_action_coupling": args.residual_action_weight > 0,
        }
        if not all(closure_checks.values()):
            raise RuntimeError(
                f"SC-LSR closure contract failed: {closure_checks}"
            )
    ability_count = torch.tensor(
        [row["ability_targets"] for row in ability_rows]
    ).sum(0)
    balanced_ability_pos_weight = (
        (len(ability_rows) - ability_count) / ability_count.clamp_min(1)
    ).clamp(1.0, 4.0).to(device)
    if args.ability_pos_weight_mode == "balanced":
        ability_pos_weight = balanced_ability_pos_weight
    elif args.ability_pos_weight_mode == "sqrt_balanced":
        ability_pos_weight = balanced_ability_pos_weight.sqrt()
    else:
        ability_pos_weight = torch.ones_like(
            balanced_ability_pos_weight, device=device
        )
    run_config = {
        "schema": (
            (
                "sc_lsr_candidate_memory_run_v1"
                if args.candidate_memory_injection
                else "sc_lsr_candidate_move_run_v1"
            )
            if args.latent_spatial_rollout
            else (
            "sc_esr_candidate_move_run_v11_candidate_memory"
            if args.candidate_memory_injection
            else (
            "sc_esr_candidate_move_run_v10_candidate_value"
            if args.candidate_value_interaction
            else (
            "sc_esr_candidate_move_run_v6_global_local"
            if args.global_conditioned_local_slots
            else (
            "sc_esr_candidate_move_run_v5_local"
            if args.candidate_local_masks
            else (
            "sc_esr_candidate_move_run_v4"
            if args.gradient_separated_native_lm
            else (
            "sc_esr_candidate_move_run_v3"
            if (
                args.candidate_conditioned
                and args.native_lm_weight_semantics == "historical_sft"
                and args.auxiliary_gradient_scope == "sc_esr"
                and args.runtime_aligned_injection
            )
            else (
                "sc_esr_candidate_move_run_v2"
                if args.candidate_conditioned
                else "sc_esr_native_move_run_v1"
            )
            )
            )
            )
            )
            )
            )
        ),
        "args": vars(args),
        "world_size": world,
        "global_batch_size": (
            world * args.per_device_batch_size * args.grad_accum_steps
        ),
        "total_micro_batches": total_micro_batches,
        "discarded_tail_micro_batches": (
            total_micro_batches % args.grad_accum_steps
        ),
        "expected_optimizer_steps": total_steps,
        "fixed_b0_sha256": expected_b0_sha256,
        "b0_semantics": (
            "audited merged Base+B0 frozen; fresh r64 Move delta"
            if args.fresh_move_delta_on_merged_b0
            else "audited start; continued r64 inside independent Move branch"
        ),
        "b0_adapter_frozen": args.fresh_move_delta_on_merged_b0,
        "fresh_move_delta_lora": args.fresh_move_delta_on_merged_b0,
        "merged_b0_manifest": (
            str(Path(args.merged_b0_manifest).resolve())
            if args.merged_b0_manifest
            else ""
        ),
        "merged_b0_manifest_sha256": (
            sha256(Path(args.merged_b0_manifest))
            if args.merged_b0_manifest
            else ""
        ),
        "data_sha256": sha256(data / "train.jsonl"),
        "evidence_manifest_sha256": sha256(
            Path(args.evidence_cache) / "manifest.json"
        ),
        "candidate_relation_manifest_sha256": (
            sha256(Path(args.candidate_relation_cache) / "manifest.json")
            if args.candidate_relation_cache
            else ""
        ),
        "all_candidate_relation_manifest_sha256": (
            sha256(
                Path(args.all_candidate_relation_cache) / "manifest.json"
            )
            if args.all_candidate_relation_cache
            else ""
        ),
        "all_candidate_relation_target_variant": (
            all_candidate_relation_target_variant
        ),
        "all_candidate_relation_supervised_representation": (
            (
                "candidate_relation_semantic_hidden"
                if args.relation_semantic_decision_separation
                else "candidate_relation_delta_hidden"
            )
            if (
                all_candidate_relation_target_variant
                == RELATION_DELTA_CONTRAST_VARIANT
            )
            else (
                "candidate_relation_full_slot_hidden"
                if all_candidate_relation is not None
                else ""
            )
        ),
        "all_candidate_relation_active_fraction": (
            float(
                all_candidate_relation.manifest.get(
                    "active_fraction", 1.0
                )
            )
            if all_candidate_relation is not None
            else 0.0
        ),
        "model_loader": loader_name,
        "native_visual_encoder_transformer_lm_head": True,
        "independent_decision_head": False,
        "visible_cot_at_runtime": False,
        "single_native_forward_at_runtime": True,
        "injection_points": 1,
        "abilities": list(ABILITIES),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "state_supervision_rows": len(state_rows),
        "ability_supervision_rows": len(ability_rows),
        "official_validation_training_rows": 0,
        "exact_stable_course": args.exact_stable_course,
        "historical_distributed_course": args.historical_distributed_course,
        "candidate_conditioned": args.candidate_conditioned,
        "independent_ability_gates": args.independent_ability_gates,
        "adaptive_need_gate": args.adaptive_need_gate,
        "candidate_local_masks": args.candidate_local_masks,
        "global_conditioned_local_slots": (
            args.global_conditioned_local_slots
        ),
        "candidate_value_interaction": (
            args.candidate_value_interaction
        ),
        "candidate_memory_injection": (
            args.candidate_memory_injection
        ),
        "latent_spatial_rollout": args.latent_spatial_rollout,
        "relation_semantic_decision_separation": (
            args.relation_semantic_decision_separation
        ),
        "local_evidence_control": args.local_evidence_control,
        "relation_semantic_target_space": (
            args.relation_semantic_target_space
        ),
        "relation_semantic_projection_seed": (
            args.relation_semantic_projection_seed
            if relation_semantic_target_projection is not None
            else None
        ),
        "relation_semantic_projection_sha256": (
            relation_semantic_projection_sha256
        ),
        "relation_semantic_gradient_path": (
            "action_blind_contrast_only"
            if args.relation_semantic_decision_separation
            else "shared_with_decision_delta"
        ),
        "relation_decision_gradient_path": (
            "native_lm_action_losses_only"
            if args.relation_semantic_decision_separation
            else "shared_with_relation_semantics"
        ),
        "relation_decision_learning_rate": (
            relation_decision_lr
            if relation_decision_parameters
            else 0.0
        ),
        "candidate_relation_weight": args.candidate_relation_weight,
        "all_candidate_relation_weight": (
            args.all_candidate_relation_weight
        ),
        "counterfactual_gradient_weight": (
            args.counterfactual_gradient_weight
        ),
        "counterfactual_neutrality_margin": (
            args.counterfactual_neutrality_margin
        ),
        "counterfactual_ablation": (
            "shuffle_relation"
            if args.latent_spatial_rollout
            else (
            "shuffle_candidate"
            if args.candidate_local_masks
            else "shuffle_image"
            )
        ),
        "gradient_separated_native_lm": args.gradient_separated_native_lm,
        "native_lm_residual_path": (
            "off" if args.gradient_separated_native_lm else "full"
        ),
        "native_lm_weight_semantics": args.native_lm_weight_semantics,
        "auxiliary_gradient_scope": args.auxiliary_gradient_scope,
        "diagnostic_split": args.diagnostic_split,
        "diagnostic_action_only": args.diagnostic_action_only,
        "diagnostic_rows": len(diagnostic_rows),
        "ability_pos_weight": ability_pos_weight.float().cpu().tolist(),
        "runtime_aligned_injection": args.runtime_aligned_injection,
        "ability_route_forward_semantics": (
            "dense_soft_exploration"
            if args.soft_ability_route
            else (
                "runtime_exact_hard_sparse_straight_through"
                if args.straight_through_sparse_route
                else "legacy_train_90pct_hard_10pct_dense"
            )
        ),
        "open_capacity_exploration_contract": (
            args.open_capacity_exploration_contract
        ),
        "trainer_source_sha256": sha256(Path(__file__).resolve()),
        "sc_esr_source_sha256": sha256(
            Path(__file__).resolve().parents[1]
            / "src"
            / "embodied_nav_cot"
            / "sc_esr.py"
        ),
    }
    if args.require_sc_lsr_closure_contract:
        run_config["schema"] = "sc_lsr_candidate_move_run_v1_closure_only"
        run_config["move_stage_semantics"] = (
            "direct action SFT plus same-row state/ability/relation-to-action "
            "closure; no standalone spatial-language replay"
        )
    if is_main:
        (output / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
    log(
        {
            "event": "start",
            "steps": total_steps,
            "batches_per_epoch_per_rank": len(loader),
            "train_rows": len(train_rows),
            "state_supervision_rows": len(state_rows),
            "ability_supervision_rows": len(ability_rows),
            "trainable_lora": sum(p.numel() for p in backbone_parameters),
            "trainable_sc_esr": sum(
                p.numel()
                for p in [
                    *residual_parameters,
                    *relation_decision_parameters,
                    *gate_parameters,
                ]
            ),
        }
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro = 0
    gradient_by_ability = torch.zeros(len(ABILITIES), device=device)
    relation_gradient_by_ability = torch.zeros(
        len(ABILITIES), device=device
    )
    relation_decision_gradient_by_ability = torch.zeros(
        len(ABILITIES), device=device
    )
    route_generator = torch.Generator(device=device)
    route_generator.manual_seed(args.seed + 1009 * rank)
    stop = False
    for epoch in range(args.epochs):
        course_sampler.set_epoch(epoch)
        for raw in loader:
            metadata = metadata_to_device(raw, device)
            native = native_batch(raw, device)
            do_preservation = (
                args.preservation_weight > 0
                and (global_step + 1) % args.preservation_every == 0
            )
            do_gain_vs_off = (
                args.gain_vs_off_weight > 0
                and (global_step + 1) % args.preservation_every == 0
            )
            do_counterfactual = (
                args.utility_weight > 0
                and (global_step + 1) % args.counterfactual_every == 0
            )
            off_margin: torch.Tensor | None = None
            off_action: torch.Tensor | None = None
            off_lm: torch.Tensor | None = None
            # Off/full/shuffled passes must differ only in SC-ESR evidence,
            # never in LoRA or Transformer dropout masks.
            paired_rng = (
                capture_rng(device)
                if (
                    args.gradient_separated_native_lm
                    or do_preservation
                    or do_gain_vs_off
                    or do_counterfactual
                )
                else None
            )
            route_dropout_mask = (
                torch.rand(
                    metadata["image_mask"].shape[0],
                    1,
                    device=device,
                    generator=route_generator,
                )
                < args.state_dropout
            )
            if args.gradient_separated_native_lm:
                # This is the only path allowed to update the continued B0
                # LoRA.  With the order-preserving data and historical
                # DistributedSampler it follows the proven native SFT course
                # while SC-ESR is structurally absent.
                with controller.context(
                    image_mask=metadata["image_mask"],
                    token_mask=metadata["decision_hidden_mask"],
                    route_positions=metadata["route_positions"],
                    candidate_positions=metadata["candidate_positions"],
                    candidate_mask=metadata["candidate_mask"],
                    candidate_image_mask=metadata["candidate_image_mask"],
                    candidate_local_valid=metadata["candidate_local_valid"],
                    enabled=False,
                    ablation="off",
                ):
                    off_outputs = model(**native)
                    off_lm = mean_native_lm_loss(
                        off_outputs.logits,
                        native["labels"],
                        action_mask=metadata["action_token_mask"],
                        action_weight=args.action_token_weight,
                        row_weight=metadata["row_loss_weight"],
                        decision_weight=metadata["decision_loss_weight"],
                        action_field_weight=metadata[
                            "action_field_loss_weight"
                        ],
                        weight_semantics=args.native_lm_weight_semantics,
                    )
                    off_action_graph, _ = surface_logits(
                        off_outputs.logits, metadata["action_surface"]
                    )
                    off_margin = gold_margin(
                        off_action_graph,
                        metadata["action_surface"].gold_indices,
                    ).detach()
                    off_action = off_action_graph.detach()
                    (off_lm / args.grad_accum_steps).backward()
                del off_action_graph, off_outputs
                restore_rng(paired_rng, device)
            elif do_preservation or do_gain_vs_off:
                with torch.no_grad(), controller.context(
                    image_mask=metadata["image_mask"],
                    token_mask=metadata["decision_hidden_mask"],
                    route_positions=metadata["route_positions"],
                    candidate_positions=metadata["candidate_positions"],
                    candidate_mask=metadata["candidate_mask"],
                    candidate_image_mask=metadata["candidate_image_mask"],
                    candidate_local_valid=metadata["candidate_local_valid"],
                    enabled=False,
                    ablation="off",
                ):
                    off_outputs = model(**native)
                    off_action, surface_valid = surface_logits(
                        off_outputs.logits, metadata["action_surface"]
                    )
                    off_margin = gold_margin(
                        off_action, metadata["action_surface"].gold_indices
                    )
                del off_outputs
                restore_rng(paired_rng, device)

            surface_valid = metadata["action_surface"].valid.bool()
            if args.candidate_local_masks:
                gain_rows = local_gold_candidate_mask(
                    metadata["candidate_local_valid"],
                    metadata["action_surface"].gold_indices,
                    surface_valid,
                )
                closure_rows = gain_rows
            else:
                gain_rows = metadata["gain_valid"].bool() & surface_valid
                closure_rows = (
                    metadata["closure_valid"].bool() & surface_valid
                )
            wrong_margin: torch.Tensor | None = None
            wrong_neutrality = torch.zeros((), device=device)
            wrong_auxiliary_gradients: tuple[
                torch.Tensor | None, ...
            ] | None = None
            if do_counterfactual:
                wrong_requires_gradient = (
                    args.counterfactual_gradient_weight > 0
                    and off_margin is not None
                    and gain_rows.any()
                )
                gradient_context = (
                    torch.enable_grad()
                    if wrong_requires_gradient
                    else torch.no_grad()
                )
                with gradient_context, controller.context(
                        image_mask=metadata["image_mask"],
                        token_mask=metadata["decision_hidden_mask"],
                        route_positions=metadata["route_positions"],
                        candidate_positions=metadata["candidate_positions"],
                        candidate_mask=metadata["candidate_mask"],
                        candidate_image_mask=metadata["candidate_image_mask"],
                        candidate_local_valid=metadata[
                            "candidate_local_valid"
                        ],
                        state_dropout_mask=route_dropout_mask,
                        ablation=(
                            "shuffle_relation"
                            if args.latent_spatial_rollout
                            else (
                            "shuffle_candidate"
                            if args.candidate_local_masks
                            else "shuffle_image"
                            )
                        ),
                    ):
                    wrong_outputs = model(**native)
                    wrong_action, _ = surface_logits(
                        wrong_outputs.logits, metadata["action_surface"]
                    )
                    wrong_margin_graph = gold_margin(
                        wrong_action,
                        metadata["action_surface"].gold_indices,
                    )
                    wrong_margin = wrong_margin_graph.detach()
                    if wrong_requires_gradient:
                        wrong_neutrality = counterfactual_neutrality_loss(
                            wrong_margin_graph,
                            off_margin,
                            gain_rows,
                            margin=args.counterfactual_neutrality_margin,
                        )
                        wrong_auxiliary_gradients = torch.autograd.grad(
                            (
                                args.counterfactual_gradient_weight
                                * wrong_neutrality
                                / args.grad_accum_steps
                            ),
                            auxiliary_parameters,
                            allow_unused=True,
                        )
                del wrong_outputs
                restore_rng(paired_rng, device)

            with controller.context(
                image_mask=metadata["image_mask"],
                token_mask=metadata["decision_hidden_mask"],
                route_positions=metadata["route_positions"],
                candidate_positions=metadata["candidate_positions"],
                candidate_mask=metadata["candidate_mask"],
                candidate_image_mask=metadata["candidate_image_mask"],
                candidate_local_valid=metadata["candidate_local_valid"],
                state_dropout_mask=route_dropout_mask,
                ablation="full",
            ):
                outputs = model(**native)
                route = controller.last_route
                residual_audit = controller.last_residual_audit
                if args.gradient_separated_native_lm:
                    if off_lm is None:
                        raise RuntimeError("gradient-separated off LM is missing")
                    lm = off_lm.detach()
                else:
                    lm = mean_native_lm_loss(
                        outputs.logits,
                        native["labels"],
                        action_mask=metadata["action_token_mask"],
                        action_weight=args.action_token_weight,
                        row_weight=metadata["row_loss_weight"],
                        decision_weight=metadata["decision_loss_weight"],
                        action_field_weight=metadata[
                            "action_field_loss_weight"
                        ],
                        weight_semantics=args.native_lm_weight_semantics,
                    )
                routing, route_terms = route_loss(
                    route,
                    metadata,
                    ability_pos_weight=ability_pos_weight,
                    ranking_weight=args.route_ranking_weight,
                    ranking_margin=args.route_ranking_margin,
                    class_balanced=args.route_class_balanced,
                )
                # Do not materialize disabled auxiliary losses.  In
                # particular, InternVL's text hidden size differs from the
                # Qwen evidence-cache hidden size; a zero-weight loss must
                # not validate or combine those tensors at all.
                auxiliary_zero = lm * 0.0
                if args.evidence_weight > 0:
                    evidence_loss = evidence_slot_contrastive_loss(
                        route["slots"],
                        metadata["evidence_targets"],
                        metadata["evidence_active"],
                    )
                else:
                    evidence_loss = auxiliary_zero
                candidate_relation_valid = torch.zeros(
                    metadata["candidate_relation_present"].shape,
                    dtype=torch.bool,
                    device=device,
                )
                candidate_relation_cosine = torch.zeros(
                    candidate_relation_valid.shape,
                    dtype=lm.dtype,
                    device=device,
                )
                if args.candidate_relation_weight > 0:
                    (
                        candidate_relation_loss,
                        candidate_relation_cosine,
                        candidate_relation_valid,
                    ) = candidate_relation_alignment_loss(
                        candidate_relation_representations(
                            controller, route, metadata["ability_targets"]
                        ),
                        metadata["candidate_relation_targets"],
                        metadata["action_surface"].gold_indices,
                        metadata["candidate_relation_present"].bool()
                        & closure_rows,
                    )
                else:
                    candidate_relation_loss = auxiliary_zero
                all_candidate_relation_valid = torch.zeros(
                    metadata["all_candidate_relation_active"].shape,
                    dtype=torch.bool,
                    device=device,
                )
                all_candidate_relation_cosine = torch.zeros(
                    all_candidate_relation_valid.shape,
                    dtype=lm.dtype,
                    device=device,
                )
                if args.all_candidate_relation_weight > 0:
                    (
                        all_candidate_relation_loss,
                        all_candidate_relation_cosine,
                        all_candidate_relation_valid,
                    ) = all_candidate_relation_supervision_loss(
                        controller,
                        route,
                        metadata["all_candidate_relation_targets"],
                        metadata["all_candidate_relation_active"].bool(),
                        target_variant=all_candidate_relation_target_variant,
                        semantic_target_projection=(
                            relation_semantic_target_projection
                        ),
                    )
                else:
                    all_candidate_relation_loss = auxiliary_zero
                action, surface_valid = surface_logits(
                    outputs.logits, metadata["action_surface"]
                )
                correct_margin = gold_margin(
                    action, metadata["action_surface"].gold_indices
                )
                residual_action = residual_action_loss(
                    action, metadata["action_surface"]
                )
                candidate_coupling = lm * 0.0
                coupling_valid = surface_valid & False
                if args.candidate_coupling_weight > 0:
                    candidate_coupling, coupling_valid = (
                        candidate_evidence_coupling_loss(
                            residual_audit,
                            metadata["action_surface"],
                            metadata["candidate_mask"],
                        )
                    )
                preservation = lm * 0.0
                if off_margin is not None and surface_valid.any():
                    preservation = residual_preservation_loss(
                        correct_margin[surface_valid],
                        off_margin[surface_valid],
                        tolerance=args.preservation_tolerance,
                    )
                gain_vs_off = lm * 0.0
                if off_margin is not None and closure_rows.any():
                    gain_vs_off = residual_utility_margin_loss(
                        correct_margin[closure_rows],
                        off_margin[closure_rows],
                        margin=args.gain_vs_off_margin,
                    )
                utility = lm * 0.0
                if wrong_margin is not None and gain_rows.any():
                    utility = residual_utility_margin_loss(
                        correct_margin[gain_rows],
                        wrong_margin[gain_rows],
                        margin=args.utility_margin,
                    )
                need = lm * 0.0
                need_targets = torch.zeros_like(surface_valid)
                if (
                    args.need_weight > 0
                    and off_action is not None
                    and off_margin is not None
                ):
                    need, need_targets = adaptive_need_loss(
                        residual_audit,
                        metadata["action_surface"],
                        closure_valid=metadata["closure_valid"],
                        off_action_logits=off_action,
                        off_margin=off_margin,
                        margin_threshold=args.need_margin_threshold,
                    )
                auxiliary_loss = (
                    args.route_weight * routing
                    + args.evidence_weight * evidence_loss
                    + args.candidate_relation_weight
                    * candidate_relation_loss
                    + args.all_candidate_relation_weight
                    * all_candidate_relation_loss
                    + args.residual_action_weight * residual_action
                    + args.utility_weight * utility
                    + args.gain_vs_off_weight * gain_vs_off
                    + args.preservation_weight * preservation
                    + args.candidate_coupling_weight * candidate_coupling
                    + args.need_weight * need
                )
                # Hooks are stateful and decoder layers are gradient
                # checkpointed.  Backward must remain inside this context so
                # recomputation sees the identical image/decision masks.
                if args.auxiliary_gradient_scope == "all":
                    full_loss = (
                        lm + auxiliary_loss
                    ) / args.grad_accum_steps
                    full_loss.backward()
                else:
                    # Route/evidence losses describe the incremental spatial
                    # mechanism, not a new objective for the general B0 LoRA.
                    # Keep their gradients inside SC-ESR so the 8B backbone is
                    # still governed by the proven native LM course.
                    scaled_auxiliary = auxiliary_loss / args.grad_accum_steps
                    auxiliary_gradients = torch.autograd.grad(
                        scaled_auxiliary,
                        auxiliary_parameters,
                        retain_graph=not args.gradient_separated_native_lm,
                        allow_unused=True,
                    )
                    if not args.gradient_separated_native_lm:
                        (lm / args.grad_accum_steps).backward()
                    for parameter, gradient in zip(
                        auxiliary_parameters,
                        auxiliary_gradients,
                    ):
                        if gradient is None:
                            continue
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
                if wrong_auxiliary_gradients is not None:
                    for parameter, gradient in zip(
                        auxiliary_parameters,
                        wrong_auxiliary_gradients,
                    ):
                        if gradient is None:
                            continue
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
                ratio_max = float(
                    residual_audit["residual_to_hidden_norm"].max().detach()
                )
            del outputs

            micro += 1
            if micro % args.grad_accum_steps != 0:
                continue
            if global_step < gate_warmup_steps:
                if output_gate.grad is not None:
                    output_gate.grad.zero_()
                with torch.no_grad():
                    output_gate.zero_()
            synchronize_gradients(parameters, world)
            if residual.ability_queries.grad is not None:
                gradient_by_ability += (
                    residual.ability_queries.grad.float().norm(dim=-1).detach()
                )
            if residual.config.latent_spatial_rollout:
                for ability_index, operator in enumerate(
                    residual.relation_operators
                ):
                    relation_gradient_by_ability[ability_index] += sum(
                        (
                            parameter.grad.float().norm()
                            if parameter.grad is not None
                            else torch.zeros((), device=device)
                        )
                        for parameter in operator.parameters()
                    ).detach()
            if residual.config.relation_semantic_decision_separation:
                for ability_index, adapter in enumerate(
                    residual.relation_decision_adapters
                ):
                    relation_decision_gradient_by_ability[
                        ability_index
                    ] += sum(
                        (
                            parameter.grad.float().norm()
                            if parameter.grad is not None
                            else torch.zeros((), device=device)
                        )
                        for parameter in adapter.parameters()
                    ).detach()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.log_every_steps == 0 or global_step == 1:
                action_correct = int(
                    action.argmax(-1)[surface_valid]
                    .eq(metadata["action_surface"].gold_indices[surface_valid])
                    .sum()
                )
                log(
                    {
                        "event": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss_lm": float(lm),
                        "loss_route": float(routing.detach()),
                        "loss_evidence": float(evidence_loss.detach()),
                        "loss_candidate_relation": float(
                            candidate_relation_loss.detach()
                        ),
                        "candidate_relation_cosine": (
                            float(
                                candidate_relation_cosine[
                                    candidate_relation_valid
                                ].mean()
                            )
                            if candidate_relation_valid.any()
                            else 0.0
                        ),
                        "candidate_relation_rows": int(
                            candidate_relation_valid.sum()
                        ),
                        "loss_all_candidate_relation": float(
                            all_candidate_relation_loss.detach()
                        ),
                        "all_candidate_relation_cosine": (
                            float(
                                all_candidate_relation_cosine[
                                    all_candidate_relation_valid
                                ].mean()
                            )
                            if all_candidate_relation_valid.any()
                            else 0.0
                        ),
                        "all_candidate_relation_items": int(
                            all_candidate_relation_valid.sum()
                        ),
                        "loss_residual_action": float(
                            residual_action.detach()
                        ),
                        "loss_utility": float(utility.detach()),
                        "loss_gain_vs_off": float(gain_vs_off.detach()),
                        "loss_preservation": float(preservation.detach()),
                        "loss_need": float(need.detach()),
                        "loss_candidate_coupling": float(
                            candidate_coupling.detach()
                        ),
                        "loss_counterfactual_neutrality": float(
                            wrong_neutrality.detach()
                        ),
                        "route_terms": route_terms,
                        "batch_action_accuracy": (
                            action_correct / max(1, int(surface_valid.sum()))
                        ),
                        "gain_rows": int(gain_rows.sum()),
                        "closure_rows": int(closure_rows.sum()),
                        "need_positive_rows": int(need_targets.sum()),
                        "candidate_coupling_rows": int(
                            coupling_valid.sum()
                        ),
                        "residual_gate": float(
                            residual.residual_gate.detach().float()
                        ),
                        "residual_ratio_max": ratio_max,
                        "global_local_scale": float(
                            residual.current_global_local_scale().detach()
                        ),
                        "candidate_value_scale": float(
                            residual.current_candidate_value_scale().detach()
                        ),
                        "candidate_memory_scale": float(
                            residual.current_candidate_memory_scale().detach()
                        ),
                        "lr": scheduler.get_last_lr(),
                    }
                )
            if global_step >= total_steps:
                stop = True
                break
        if stop:
            break

    if micro % args.grad_accum_steps:
        optimizer.zero_grad(set_to_none=True)
    if world > 1:
        dist.barrier()
        # All optimizer work is complete and gradients have already been
        # synchronized explicitly.  Do not leave non-main ranks waiting in an
        # NCCL process group while rank zero performs the potentially long
        # read-only diagnostic.  The old ordering hit the NCCL watchdog after
        # a 512-row rank-zero diagnostic and discarded an otherwise completed
        # 625-step run before any weights were saved.
        dist.destroy_process_group()
    if not is_main:
        controller.close()
        if log_handle is not None:
            log_handle.close()
        return

    # Persist the trained state before any read-only post-training diagnostic.
    # A diagnostic failure must never destroy a completed optimizer course.
    final_adapter = output / "final_adapter"
    model.save_pretrained(final_adapter, safe_serialization=True)
    processor.save_pretrained(final_adapter)
    residual.save_pretrained(final_adapter / "sc_esr")

    metrics: dict[str, Any] = diagnostic(
        model,
        controller,
        diagnostic_loader,
        device,
        max_samples=args.diagnostic_samples,
        all_candidate_relation_target_variant=(
            all_candidate_relation_target_variant
        ),
        relation_semantic_target_projection=(
            relation_semantic_target_projection
        ),
    )
    metrics["ability_query_gradient_sum"] = (
        gradient_by_ability.detach().cpu().tolist()
    )
    metrics["all_five_abilities_received_gradient"] = bool(
        (gradient_by_ability > 0).all()
    )
    metrics["relation_operator_gradient_sum"] = (
        relation_gradient_by_ability.detach().cpu().tolist()
    )
    metrics["all_five_relation_operators_received_gradient"] = bool(
        residual.config.latent_spatial_rollout
        and (relation_gradient_by_ability > 0).all()
    )
    metrics["relation_decision_adapter_gradient_sum"] = (
        relation_decision_gradient_by_ability.detach().cpu().tolist()
    )
    metrics["all_five_relation_decision_adapters_received_gradient"] = bool(
        residual.config.relation_semantic_decision_separation
        and (relation_decision_gradient_by_ability > 0).all()
    )
    metrics.update(
        {
            "global_step": global_step,
            "adapter_sha256": sha256(
                final_adapter / "adapter_model.safetensors"
            ),
            "sc_esr_sha256": sha256(
                final_adapter / "sc_esr" / "sc_esr.pt"
            ),
            "residual_gate": float(
                residual.residual_gate.detach().float()
            ),
            "global_local_scale": float(
                residual.current_global_local_scale().detach()
            ),
            "candidate_value_scale": float(
                residual.current_candidate_value_scale().detach()
            ),
            "candidate_memory_scale": float(
                residual.current_candidate_memory_scale().detach()
            ),
            "passed_runtime_contract": True,
        }
    )
    (output / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    log({"event": "complete", **metrics})
    controller.close()
    if log_handle is not None:
        log_handle.close()


if __name__ == "__main__":
    main()
