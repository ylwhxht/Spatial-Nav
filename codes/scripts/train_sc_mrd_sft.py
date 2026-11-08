#!/usr/bin/env python3
"""Single-stage native-LM SC-MRD/OPD training from a navigation adapter.

By default the fixed B0 LoRA remains frozen and one fresh additive delta-LoRA
is optimized.  ``--continue-b0-adapter-directly`` instead updates B0 itself and
emits one continued adapter, avoiding an anchor-plus-delta stack.  ``opd`` uses
cached frozen privileged-teacher legal-action logits and trains only the direct
Student view. Runtime uses only the direct prompt.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from train_cold_start_vision_sft import (  # noqa: E402
    VisionSFTCollator,
    VisionSample,
    load_model,
    load_processor,
    set_seed,
)
from peft import (  # noqa: E402
    LoraConfig,
    TaskType,
    get_peft_model_state_dict,
)
from safetensors.torch import save_file as save_safetensors  # noqa: E402

from embodied_nav_cot.peft_stack import (  # noqa: E402
    ANCHOR_ADAPTER,
    DELTA_ADAPTER,
    adapter_parameter_counts,
    load_frozen_anchor_with_delta,
    load_trainable_adapter_continuation,
    named_adapter_parameter_counts,
)

from embodied_nav_cot.sc_mrd import (  # noqa: E402
    PADDED_LEGAL_LOGIT,
    action_marginal_residual_targets,
    action_token_spec,
    apply_action_marginal_residual_effects,
    counterfactual_margin_loss,
    gold_margin,
    legal_action_logits,
    masked_distillation_loss,
    masked_bridge_increment_representation_distillation_loss,
    masked_action_projected_bridge_increment_loss,
    masked_representation_residual_distillation_loss,
    masked_residual_logit_distillation_loss,
    ema_sol_match_weights,
    online_advantage_kd_weights,
    online_pairwise_adaptive_targets,
    online_safe_full_distribution_weights,
    privileged_bridge_correction_margin_targets,
    privileged_correction_margin_targets,
    privileged_margin_rank_weights,
    reliable_teacher_mask,
    selective_pairwise_margin_loss,
    selective_pairwise_huber_margin_loss,
)


EMA_TEACHER_ADAPTER = "ema_teacher"
HYBRID_CACHED_UNIFORM_EMA_PAIRWISE = (
    "hybrid_cached_uniform_ema_pairwise_adaptive"
)
ONLINE_EMA_KD_MODES = {
    "online_ema_uniform_full_distribution",
    "online_ema_sol_match_full_distribution",
    "online_ema_pairwise_adaptive",
    HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
}


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
    "mm_token_type_ids",
    "labels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--b0-adapter", required=True)
    parser.add_argument(
        "--expected-b0-sha256",
        default=B0_SHA256,
        help="Expected frozen anchor adapter weight hash.",
    )
    parser.add_argument("--expected-anchor-r", type=int, default=64)
    parser.add_argument("--expected-anchor-alpha", type=int, default=128)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--phase2-data-dir",
        default="",
        help=(
            "Optional second course trained in the same process on the same "
            "cumulative adapter.  Its optimizer and schedule are reset at the "
            "phase boundary; the base model and adapter are never reloaded."
        ),
    )
    parser.add_argument(
        "--phase1-historical-move-input-contract",
        action="store_true",
        help=(
            "Reproduce the proven Move40k chat contract in phase 1: use the "
            "navigation-spatial system prompt and place the image before the "
            "row prompt. Phase 2 keeps the runtime-aligned OPD contract."
        ),
    )
    parser.add_argument(
        "--phase1-historical-distributed-sampler",
        action="store_true",
        help=(
            "Reproduce the proven Move40k DistributedSampler(seed=0) order "
            "in phase 1. This is valid only when phase 1 has no paired rows."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--objective",
        choices=("direct", "aux", "sc_mrd", "opd"),
        default="sc_mrd",
    )
    parser.add_argument(
        "--decision-kind",
        choices=("move", "stop"),
        default="move",
    )
    parser.add_argument("--expected-train-rows", type=int, default=7997)
    parser.add_argument("--expected-dev-rows", type=int, default=1728)
    parser.add_argument("--expected-phase2-train-rows", type=int, default=0)
    parser.add_argument("--expected-phase2-dev-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--phase2-learning-rate", type=float, default=0.0)
    parser.add_argument("--phase2-per-device-batch-size", type=int, default=0)
    parser.add_argument("--phase2-grad-accum-steps", type=int, default=0)
    parser.add_argument("--phase2-max-steps", type=int, default=0)
    parser.add_argument("--phase2-seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--phase2-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--kd-warmup-ratio", type=float, default=0.20)
    parser.add_argument(
        "--representation-cache",
        default="",
        help="Train-only cached Strong-Move Direct/Bridge decision representations.",
    )
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument(
        "--ema-teacher-max-length",
        type=int,
        default=4096,
        help=(
            "Independent context limit for the privileged EMA Teacher view; "
            "the runtime/direct Student keeps --max-length."
        ),
    )
    parser.add_argument(
        "--phase2-max-length",
        type=int,
        default=0,
        help="Optional phase-2 sequence length; zero reuses --max-length.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--continue-b0-adapter-directly",
        action="store_true",
        help=(
            "Train the supplied B0 adapter itself and save one continued "
            "adapter, instead of freezing B0 and adding a fresh delta LoRA."
        ),
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--lora-exclude-modules",
        default="",
        help="Comma-separated module-name patterns excluded from delta LoRA injection.",
    )
    parser.add_argument(
        "--forbid-vision-lora",
        action="store_true",
        help="Fail closed if a trainable LoRA tensor is attached below a vision module.",
    )
    parser.add_argument("--direct-weight", type=float, default=1.0)
    parser.add_argument(
        "--direct-positive-weight",
        type=float,
        default=1.0,
        help="Per-row Direct CE multiplier for done=1 in Stop OPD.",
    )
    parser.add_argument(
        "--direct-negative-weight",
        type=float,
        default=1.0,
        help="Per-row Direct CE multiplier for done=0 in Stop OPD.",
    )
    parser.add_argument("--reason-weight", type=float, default=0.10)
    parser.add_argument("--reasoned-action-weight", type=float, default=0.50)
    parser.add_argument("--kd-weight", type=float, default=0.30)
    parser.add_argument(
        "--anchor-retention-weight",
        type=float,
        default=0.0,
        help=(
            "KL coefficient from cached frozen-anchor Direct legal-action "
            "logits to the current Student. Zero preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--kd-mode",
        choices=(
            "uniform",
            "online_advantage_tiered",
            "online_pairwise_adaptive",
            "online_safe_full_distribution",
            "online_ema_uniform_full_distribution",
            "online_ema_sol_match_full_distribution",
            "online_ema_pairwise_adaptive",
            HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
            "online_action_debiased_residual",
            "online_bridge_representation_residual",
            "online_bridge_increment_representation_residual",
            "online_action_projected_bridge_increment",
            "privileged_margin_adaptive",
            "privileged_correction_margin",
            "privileged_bridge_correction_margin",
        ),
        default="uniform",
        help=(
            "Uniform applies one coefficient to every reliable Teacher row. "
            "online_advantage_tiered gives full weight to current Student "
            "repairs, lower weight to genuine confidence gains, and disables "
            "KD where the current Student is already at least as strong. "
            "online_pairwise_adaptive recomputes the Student competitor, "
            "repair/confidence gate, continuous weight, and local margin "
            "target on every forward pass. "
            "online_safe_full_distribution first requires the privileged "
            "Teacher to improve over the frozen Direct anchor, then weights "
            "full-distribution KL by its current gold-log-probability gain. "
            "online_ema_uniform_full_distribution runs the privileged Bridge "
            "view through an EMA copy of the current Student and transfers "
            "its full legal-candidate distribution uniformly on every paired "
            "row, without Sol-match weighting or hard filtering. "
            "online_ema_sol_match_full_distribution runs the privileged "
            "Bridge view through an EMA copy of the current Student and "
            "weights full-distribution KL continuously by its Sol-action "
            "margin; it never hard-filters Sol CE rows. "
            "online_ema_pairwise_adaptive recomputes both the privileged EMA "
            "Teacher and Direct Student on every batch, dynamically retains "
            "only Teacher-correct Sol rows with a positive current margin "
            "advantage, and transfers a capped local pairwise margin. It "
            "requires no precomputed Teacher logits or fixed KD subset. "
            "hybrid_cached_uniform_ema_pairwise_adaptive preserves cached "
            "full-distribution KD on historical rows while using the same "
            "online EMA pairwise gate on newly added rows. Row-level "
            "teacher_runtime_mode chooses cached or ema without stacking "
            "another LoRA stage. "
            "online_action_debiased_residual uses the same causal and online "
            "gate but transfers only the two-way-centered privileged logit "
            "residual, removing global candidate-index effects. "
            "online_bridge_representation_residual uses the same causal and "
            "on-policy action gate but transfers only the Bridge-specific "
            "final decision-state representation residual; Direct CE remains "
            "the sole action-label objective. "
            "online_bridge_increment_representation_residual transfers only "
            "Bridge minus the action-blind spatial-only view, preserving the "
            "B2 spatial prior. "
            "online_action_projected_bridge_increment additionally projects "
            "that causal increment into the current legal-action LM-head span "
            "and retains the Direct anchor in its orthogonal complement. "
            "privileged_margin_adaptive selects correct privileged-view "
            "decisions and weights their margins by within-action rank. "
            "privileged_correction_margin only repairs a frozen Direct "
            "top-1 error that the privileged view corrects, using a capped "
            "gold-vs-error pairwise margin instead of full-distribution KL. "
            "privileged_bridge_correction_margin additionally requires the "
            "spatial-without-Bridge ablation to remain wrong and uses a "
            "one-sided Huber margin loss."
        ),
    )
    parser.add_argument(
        "--kd-margin-weight-min",
        type=float,
        default=0.5,
        help="Lowest within-action privileged-margin KD multiplier.",
    )
    parser.add_argument(
        "--kd-margin-weight-max",
        type=float,
        default=1.5,
        help="Highest within-action privileged-margin KD multiplier.",
    )
    parser.add_argument(
        "--kd-min-reasoning-gain",
        type=float,
        default=0.25,
        help=(
            "Minimum privileged-minus-frozen-Direct pairwise margin gain for "
            "repair-only correction KD."
        ),
    )
    parser.add_argument(
        "--kd-pair-target-margin-cap",
        type=float,
        default=1.0,
        help=(
            "Maximum privileged gold-vs-Direct-error margin transferred by "
            "repair-only pairwise KD."
        ),
    )
    parser.add_argument(
        "--kd-min-bridge-gain",
        type=float,
        default=0.5,
        help=(
            "Minimum spatial-plus-Bridge minus spatial-without-Bridge "
            "gold-vs-Direct-error margin gain."
        ),
    )
    parser.add_argument(
        "--kd-pair-huber-beta",
        type=float,
        default=0.5,
        help="Quadratic-to-linear transition for Bridge pairwise margin KD.",
    )
    parser.add_argument(
        "--kd-min-online-margin-advantage",
        type=float,
        default=0.5,
        help="Teacher-minus-Student gold-margin gain for confidence shaping.",
    )
    parser.add_argument(
        "--kd-online-margin-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Teacher-minus-current-Student pair-margin gain that reaches "
            "full continuous on-policy KD weight."
        ),
    )
    parser.add_argument(
        "--kd-online-confidence-increment-cap",
        type=float,
        default=0.25,
        help=(
            "Maximum per-forward target-margin increment for a currently "
            "correct Student in on-policy pairwise KD."
        ),
    )
    parser.add_argument(
        "--kd-min-static-gold-logp-gain",
        type=float,
        default=0.0,
        help=(
            "Minimum T-scaled privileged-minus-frozen-Direct gold-action "
            "log-probability gain for causal-safe full-distribution KD."
        ),
    )
    parser.add_argument(
        "--kd-dynamic-gold-logp-gap-scale",
        type=float,
        default=0.5,
        help=(
            "Privileged-minus-current-Student gold log-probability gap that "
            "reaches full online safe-KD example weight."
        ),
    )
    parser.add_argument(
        "--kd-active-weight-floor",
        type=float,
        default=0.25,
        help="Minimum weight for an active causal-safe full-KL example.",
    )
    parser.add_argument(
        "--ema-teacher-decay",
        type=float,
        default=0.95,
        help=(
            "Parameter EMA decay for the online privileged Teacher. "
            "The frozen anchor is shared; only the fresh delta is tracked."
        ),
    )
    parser.add_argument(
        "--ema-sol-match-temperature",
        type=float,
        default=0.5,
        help="Temperature for the EMA-Teacher Sol-action margin sigmoid.",
    )
    parser.add_argument(
        "--ema-sol-match-margin-center",
        type=float,
        default=0.0,
        help=(
            "Sol-action Teacher margin mapped to adaptive weight 0.5. "
            "Zero preserves the original uncentered sigmoid."
        ),
    )
    parser.add_argument(
        "--ema-sol-match-weight-floor",
        type=float,
        default=0.0,
        help="Minimum full-KL weight after EMA/Sol matching (in [0,1)).",
    )
    parser.add_argument(
        "--kd-repair-tier-weight",
        type=float,
        default=1.0,
        help="Multiplier inside --kd-weight for wrong-Student repairs.",
    )
    parser.add_argument(
        "--kd-confidence-tier-weight",
        type=float,
        default=0.5,
        help="Multiplier inside --kd-weight for correct-Student confidence gains.",
    )
    parser.add_argument(
        "--phase2-kd-weight",
        type=float,
        default=None,
        help="KD coefficient after the in-process phase boundary.",
    )
    parser.add_argument(
        "--phase2-kd-warmup-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--kd-positive-weight",
        type=float,
        default=1.0,
        help=(
            "Per-example KD multiplier for gold action 1 in the Stop branch. "
            "Move KD is unchanged and requires the default value."
        ),
    )
    parser.add_argument(
        "--kd-negative-weight",
        type=float,
        default=1.0,
        help=(
            "Per-example KD multiplier for gold action 0 in the Stop branch. "
            "Move KD is unchanged and requires the default value."
        ),
    )
    parser.add_argument(
        "--kd-require-teacher-correct",
        action="store_true",
        help=(
            "Distill only rows whose reason-conditioned native-LM action "
            "argmax matches the gold action."
        ),
    )
    parser.add_argument(
        "--kd-min-teacher-gold-margin",
        type=float,
        default=None,
        help=(
            "Optional minimum teacher gold-vs-best-wrong legal-action logit "
            "margin. Unset preserves the original all-row KD behavior."
        ),
    )
    parser.add_argument("--counterfactual-weight", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.10)
    parser.add_argument("--direct-action-token-weight", type=float, default=4.0)
    parser.add_argument(
        "--native-lm-weight-semantics",
        choices=("legacy_normalized", "historical_sft"),
        default="legacy_normalized",
    )
    parser.add_argument(
        "--phase2-native-lm-weight-semantics",
        choices=("legacy_normalized", "historical_sft"),
        default="legacy_normalized",
    )
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--log-file", default="")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Deterministic debug subset. Zero keeps the formal dataset.",
    )
    parser.add_argument(
        "--max-phase2-train-samples",
        type=int,
        default=0,
        help="Deterministic debug subset for phase 2. Zero keeps all rows.",
    )
    parser.add_argument(
        "--max-dev-samples",
        type=int,
        default=512,
        help="Bound final teacher-forced diagnostic cost. Zero uses all dev rows.",
    )
    parser.add_argument(
        "--max-train-diagnostic-samples",
        type=int,
        default=64,
        help="Bound the final train-set overfit diagnostic.",
    )
    parser.add_argument(
        "--diagnostic-batch-size",
        type=int,
        default=4,
        help="Rank-0 teacher-forced diagnostic batch size.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL row: {path}")
                if "image_path" in value and not Path(value["image_path"]).is_absolute():
                    value["image_path"] = str((path.parent / value["image_path"]).resolve())
                rows.append(value)
    return rows


def read_representation_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load train-only Direct/Bridge decision-state representations."""
    records: dict[str, dict[str, Any]] = {}
    shards = sorted(path.glob("hidden.rank*.pt"))
    if not shards:
        raise ValueError(f"representation cache has no hidden.rank*.pt: {path}")
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if payload.get("schema") != "move_bridge_final_hidden_cache_v1":
            raise ValueError(f"unexpected representation cache schema: {shard}")
        for item in payload.get("records", []):
            sample_id = str(item["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate representation cache row: {sample_id}")
            direct = item["direct_final_hidden"]
            bridge = item["bridge_final_hidden"]
            if not isinstance(direct, torch.Tensor) or not isinstance(
                bridge, torch.Tensor
            ):
                raise ValueError(f"invalid representation tensors: {sample_id}")
            if direct.ndim != 1 or bridge.shape != direct.shape:
                raise ValueError(f"representation shape mismatch: {sample_id}")
            records[sample_id] = {
                "direct_final_hidden": direct.float(),
                "bridge_final_hidden": bridge.float(),
                "teacher_reliable": bool(item.get("teacher_reliable", False)),
            }
            no_bridge = item.get("no_bridge_final_hidden")
            if no_bridge is not None:
                if not isinstance(no_bridge, torch.Tensor) or no_bridge.shape != direct.shape:
                    raise ValueError(f"invalid no-Bridge representation tensor: {sample_id}")
                records[sample_id]["no_bridge_final_hidden"] = no_bridge.float()
    return records


def attach_privileged_margin_adaptive_weights(
    rows: list[dict[str, Any]],
    *,
    minimum_margin: float,
    minimum_weight: float,
    maximum_weight: float,
) -> dict[str, Any]:
    """Calibrate frozen privileged-view margins on the formal train course."""
    paired = [row for row in rows if bool(row.get("paired"))]
    if not paired:
        raise ValueError("privileged-margin adaptive KD requires paired rows")
    max_actions = max(len(row["legal_actions"]) for row in paired)
    teacher_logits = torch.full(
        (len(paired), max_actions),
        PADDED_LEGAL_LOGIT,
        dtype=torch.float32,
    )
    gold_indices = torch.empty((len(paired),), dtype=torch.long)
    action_to_group: dict[str, int] = {}
    group_indices = torch.empty((len(paired),), dtype=torch.long)
    for index, row in enumerate(paired):
        legal = [str(value) for value in row["legal_actions"]]
        values = [float(value) for value in row.get("teacher_legal_scores", [])]
        gold_action = str(row["gold_action"])
        if len(values) != len(legal) or gold_action not in legal:
            raise ValueError(
                f"invalid cached privileged surface: {row.get('sample_id')}"
            )
        teacher_logits[index, : len(values)] = torch.tensor(values)
        gold_indices[index] = legal.index(gold_action)
        group_indices[index] = action_to_group.setdefault(
            gold_action, len(action_to_group)
        )

    weights, eligible, margins = privileged_margin_rank_weights(
        teacher_logits,
        gold_indices,
        group_indices,
        minimum_margin=minimum_margin,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )
    correct = teacher_logits.argmax(dim=-1).eq(gold_indices)
    for index, row in enumerate(paired):
        row["privileged_margin_kd_weight"] = float(weights[index])
        row["privileged_margin_kd_eligible"] = bool(eligible[index])
        row["privileged_teacher_correct"] = bool(correct[index])
        row["privileged_teacher_gold_margin"] = float(margins[index])

    by_action: dict[str, Any] = {}
    for action, group in sorted(action_to_group.items()):
        mask = group_indices.eq(group)
        active = mask & eligible
        group_margins = margins[mask]
        group_weights = weights[active]
        by_action[action] = {
            "rows": int(mask.sum()),
            "teacher_correct": int((mask & correct).sum()),
            "kd_eligible": int(active.sum()),
            "margin_min": float(group_margins.min()),
            "margin_mean": float(group_margins.mean()),
            "margin_max": float(group_margins.max()),
            "active_weight_min": (
                float(group_weights.min()) if group_weights.numel() else 0.0
            ),
            "active_weight_mean": (
                float(group_weights.mean()) if group_weights.numel() else 0.0
            ),
            "active_weight_max": (
                float(group_weights.max()) if group_weights.numel() else 0.0
            ),
            "active_weight_sum": float(group_weights.sum()),
        }
    return {
        "schema": "privileged_margin_adaptive_kd_calibration_v1",
        "rows": len(rows),
        "paired_rows": len(paired),
        "teacher_correct": int(correct.sum()),
        "teacher_correct_fraction": float(correct.float().mean()),
        "kd_eligible": int(eligible.sum()),
        "kd_eligible_fraction": float(eligible.float().mean()),
        "minimum_margin": float(minimum_margin),
        "minimum_weight": float(minimum_weight),
        "maximum_weight": float(maximum_weight),
        "active_weight_sum": float(weights.sum()),
        "active_weight_mean": float(weights[eligible].mean()),
        "by_gold_action": by_action,
    }


def attach_privileged_correction_margin_targets(
    rows: list[dict[str, Any]],
    *,
    minimum_teacher_margin: float,
    minimum_reasoning_gain: float,
    target_margin_cap: float,
    minimum_weight: float,
    maximum_weight: float,
) -> dict[str, Any]:
    """Attach frozen Direct-error competitors and privileged repair targets."""
    paired = [row for row in rows if bool(row.get("paired"))]
    if not paired:
        raise ValueError("privileged correction-margin KD requires paired rows")
    max_actions = max(len(row["legal_actions"]) for row in paired)
    teacher_logits = torch.full(
        (len(paired), max_actions), PADDED_LEGAL_LOGIT, dtype=torch.float32
    )
    direct_logits = torch.full_like(teacher_logits, PADDED_LEGAL_LOGIT)
    gold_indices = torch.empty((len(paired),), dtype=torch.long)
    group_indices = torch.empty((len(paired),), dtype=torch.long)
    action_to_group: dict[str, int] = {}
    for index, row in enumerate(paired):
        legal = [str(value) for value in row["legal_actions"]]
        teacher_values = [
            float(value) for value in row.get("teacher_legal_scores", [])
        ]
        direct_values = [
            float(value)
            for value in row.get("anchor_direct_legal_scores", [])
        ]
        gold_action = str(row["gold_action"])
        if (
            len(teacher_values) != len(legal)
            or len(direct_values) != len(legal)
            or gold_action not in legal
        ):
            raise ValueError(
                f"invalid cached correction surfaces: {row.get('sample_id')}"
            )
        teacher_logits[index, : len(legal)] = torch.tensor(teacher_values)
        direct_logits[index, : len(legal)] = torch.tensor(direct_values)
        gold_indices[index] = legal.index(gold_action)
        group_indices[index] = action_to_group.setdefault(
            gold_action, len(action_to_group)
        )

    (
        weights,
        eligible,
        competitor_indices,
        target_margins,
        margin_gains,
        teacher_margins,
    ) = privileged_correction_margin_targets(
        teacher_logits,
        direct_logits,
        gold_indices,
        group_indices,
        minimum_teacher_margin=minimum_teacher_margin,
        minimum_reasoning_gain=minimum_reasoning_gain,
        target_margin_cap=target_margin_cap,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )
    teacher_correct = teacher_logits.argmax(dim=-1).eq(gold_indices)
    direct_correct = direct_logits.argmax(dim=-1).eq(gold_indices)
    for index, row in enumerate(paired):
        row["privileged_correction_kd_weight"] = float(weights[index])
        row["privileged_correction_kd_eligible"] = bool(eligible[index])
        row["privileged_correction_competitor_index"] = int(
            competitor_indices[index]
        )
        row["privileged_correction_target_margin"] = float(
            target_margins[index]
        )
        row["privileged_correction_margin_gain"] = float(margin_gains[index])

    by_action: dict[str, Any] = {}
    for action, group in sorted(action_to_group.items()):
        mask = group_indices.eq(group)
        active = mask & eligible
        active_weights = weights[active]
        active_targets = target_margins[active]
        active_gains = margin_gains[active]
        by_action[action] = {
            "rows": int(mask.sum()),
            "frozen_direct_correct": int((mask & direct_correct).sum()),
            "privileged_teacher_correct": int((mask & teacher_correct).sum()),
            "correction_kd_eligible": int(active.sum()),
            "active_weight_sum": float(active_weights.sum()),
            "active_weight_mean": (
                float(active_weights.mean()) if active_weights.numel() else 0.0
            ),
            "target_margin_mean": (
                float(active_targets.mean()) if active_targets.numel() else 0.0
            ),
            "reasoning_gain_mean": (
                float(active_gains.mean()) if active_gains.numel() else 0.0
            ),
        }
    return {
        "schema": "privileged_correction_margin_kd_calibration_v1",
        "rows": len(rows),
        "paired_rows": len(paired),
        "frozen_direct_correct": int(direct_correct.sum()),
        "frozen_direct_correct_fraction": float(direct_correct.float().mean()),
        "privileged_teacher_correct": int(teacher_correct.sum()),
        "privileged_teacher_correct_fraction": float(
            teacher_correct.float().mean()
        ),
        "correction_kd_eligible": int(eligible.sum()),
        "correction_kd_eligible_fraction": float(eligible.float().mean()),
        "minimum_teacher_margin": float(minimum_teacher_margin),
        "minimum_reasoning_gain": float(minimum_reasoning_gain),
        "target_margin_cap": float(target_margin_cap),
        "minimum_weight": float(minimum_weight),
        "maximum_weight": float(maximum_weight),
        "active_weight_sum": float(weights.sum()),
        "active_weight_mean": float(weights[eligible].mean()),
        "target_margin_mean": float(target_margins[eligible].mean()),
        "reasoning_gain_mean": float(margin_gains[eligible].mean()),
        "by_gold_action": by_action,
    }


def attach_privileged_bridge_correction_margin_targets(
    rows: list[dict[str, Any]],
    *,
    minimum_teacher_margin: float,
    minimum_bridge_gain: float,
    target_margin_cap: float,
    minimum_weight: float,
    maximum_weight: float,
) -> dict[str, Any]:
    """Attach only correction targets caused specifically by the Bridge."""
    paired = [row for row in rows if bool(row.get("paired"))]
    if not paired:
        raise ValueError("privileged Bridge correction KD requires paired rows")
    max_actions = max(len(row["legal_actions"]) for row in paired)
    teacher_logits = torch.full(
        (len(paired), max_actions), PADDED_LEGAL_LOGIT, dtype=torch.float32
    )
    spatial_logits = torch.full_like(teacher_logits, PADDED_LEGAL_LOGIT)
    direct_logits = torch.full_like(teacher_logits, PADDED_LEGAL_LOGIT)
    gold_indices = torch.empty((len(paired),), dtype=torch.long)
    group_indices = torch.empty((len(paired),), dtype=torch.long)
    action_to_group: dict[str, int] = {}
    for index, row in enumerate(paired):
        legal = [str(value) for value in row["legal_actions"]]
        teacher_values = [
            float(value) for value in row.get("teacher_legal_scores", [])
        ]
        spatial_values = [
            float(value)
            for value in row.get("spatial_without_bridge_legal_scores", [])
        ]
        direct_values = [
            float(value)
            for value in row.get("anchor_direct_legal_scores", [])
        ]
        gold_action = str(row["gold_action"])
        if (
            len(teacher_values) != len(legal)
            or len(spatial_values) != len(legal)
            or len(direct_values) != len(legal)
            or gold_action not in legal
        ):
            raise ValueError(
                f"invalid cached Bridge-ablation surfaces: {row.get('sample_id')}"
            )
        teacher_logits[index, : len(legal)] = torch.tensor(teacher_values)
        spatial_logits[index, : len(legal)] = torch.tensor(spatial_values)
        direct_logits[index, : len(legal)] = torch.tensor(direct_values)
        gold_indices[index] = legal.index(gold_action)
        group_indices[index] = action_to_group.setdefault(
            gold_action, len(action_to_group)
        )

    (
        weights,
        eligible,
        competitor_indices,
        target_margins,
        bridge_gains,
        teacher_margins,
    ) = privileged_bridge_correction_margin_targets(
        teacher_logits,
        spatial_logits,
        direct_logits,
        gold_indices,
        group_indices,
        minimum_teacher_margin=minimum_teacher_margin,
        minimum_bridge_gain=minimum_bridge_gain,
        target_margin_cap=target_margin_cap,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )
    teacher_correct = teacher_logits.argmax(dim=-1).eq(gold_indices)
    spatial_correct = spatial_logits.argmax(dim=-1).eq(gold_indices)
    direct_correct = direct_logits.argmax(dim=-1).eq(gold_indices)
    for index, row in enumerate(paired):
        row["privileged_correction_kd_weight"] = float(weights[index])
        row["privileged_correction_kd_eligible"] = bool(eligible[index])
        row["privileged_correction_competitor_index"] = int(
            competitor_indices[index]
        )
        row["privileged_correction_target_margin"] = float(target_margins[index])
        row["privileged_correction_margin_gain"] = float(bridge_gains[index])

    by_action: dict[str, Any] = {}
    for action, group in sorted(action_to_group.items()):
        mask = group_indices.eq(group)
        active = mask & eligible
        active_weights = weights[active]
        active_targets = target_margins[active]
        active_gains = bridge_gains[active]
        by_action[action] = {
            "rows": int(mask.sum()),
            "frozen_direct_correct": int((mask & direct_correct).sum()),
            "spatial_without_bridge_correct": int((mask & spatial_correct).sum()),
            "privileged_teacher_correct": int((mask & teacher_correct).sum()),
            "bridge_correction_kd_eligible": int(active.sum()),
            "active_weight_sum": float(active_weights.sum()),
            "active_weight_mean": (
                float(active_weights.mean()) if active_weights.numel() else 0.0
            ),
            "target_margin_mean": (
                float(active_targets.mean()) if active_targets.numel() else 0.0
            ),
            "bridge_gain_mean": (
                float(active_gains.mean()) if active_gains.numel() else 0.0
            ),
        }
    return {
        "schema": "privileged_bridge_correction_margin_kd_calibration_v1",
        "rows": len(rows),
        "paired_rows": len(paired),
        "frozen_direct_correct": int(direct_correct.sum()),
        "frozen_direct_correct_fraction": float(direct_correct.float().mean()),
        "spatial_without_bridge_correct": int(spatial_correct.sum()),
        "spatial_without_bridge_correct_fraction": float(
            spatial_correct.float().mean()
        ),
        "privileged_teacher_correct": int(teacher_correct.sum()),
        "privileged_teacher_correct_fraction": float(teacher_correct.float().mean()),
        "bridge_correction_kd_eligible": int(eligible.sum()),
        "bridge_correction_kd_eligible_fraction": float(eligible.float().mean()),
        "minimum_teacher_margin": float(minimum_teacher_margin),
        "minimum_bridge_gain": float(minimum_bridge_gain),
        "target_margin_cap": float(target_margin_cap),
        "minimum_weight": float(minimum_weight),
        "maximum_weight": float(maximum_weight),
        "active_weight_sum": float(weights.sum()),
        "active_weight_mean": float(weights[eligible].mean()),
        "target_margin_mean": float(target_margins[eligible].mean()),
        "bridge_gain_mean": float(bridge_gains[eligible].mean()),
        "teacher_margin_mean": float(teacher_margins[eligible].mean()),
        "by_gold_action": by_action,
    }


def attach_action_debiased_residual_targets(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    *,
    temperature: float,
    minimum_teacher_margin: float,
    minimum_static_gold_logp_gain: float,
    dynamic_gold_logp_gap_scale: float,
    active_weight_floor: float,
) -> dict[str, Any]:
    """Fit train-only candidate effects and attach sample-specific residuals."""

    def surfaces(rows: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        paired = [row for row in rows if bool(row.get("paired"))]
        if not paired:
            raise ValueError("action-debiased residual KD requires paired rows")
        max_actions = max(len(row["legal_actions"]) for row in paired)
        teacher = torch.full(
            (len(paired), max_actions), PADDED_LEGAL_LOGIT, dtype=torch.float32
        )
        anchor = torch.full_like(teacher, PADDED_LEGAL_LOGIT)
        groups = torch.full_like(teacher, -1, dtype=torch.long)
        mask = torch.zeros_like(teacher, dtype=torch.bool)
        gold = torch.empty(len(paired), dtype=torch.long)
        for index, row in enumerate(paired):
            legal = [str(value) for value in row["legal_actions"]]
            teacher_values = [
                float(value) for value in row.get("teacher_legal_scores", [])
            ]
            anchor_values = [
                float(value)
                for value in row.get("anchor_direct_legal_scores", [])
            ]
            if len(teacher_values) != len(legal) or len(anchor_values) != len(legal):
                raise ValueError(
                    f"invalid action-debiased surfaces: {row.get('sample_id')}"
                )
            action_groups: list[int] = []
            for action in legal:
                prefix = "candidate_"
                if not action.startswith(prefix) or not action[len(prefix) :].isdigit():
                    raise ValueError(f"invalid Move action group: {action}")
                action_groups.append(int(action[len(prefix) :]))
            count = len(legal)
            teacher[index, :count] = torch.tensor(teacher_values)
            anchor[index, :count] = torch.tensor(anchor_values)
            groups[index, :count] = torch.tensor(action_groups)
            mask[index, :count] = True
            gold[index] = legal.index(str(row["gold_action"]))
        return paired, teacher, anchor, groups, mask, gold

    (
        paired_train,
        teacher_train,
        anchor_train,
        groups_train,
        mask_train,
        gold_train,
    ) = surfaces(train_rows)
    reliable, _, _ = reliable_teacher_mask(
        teacher_train,
        gold_train,
        require_correct=True,
        min_gold_margin=minimum_teacher_margin,
    )
    (
        initial_weights,
        static_valid,
        active,
        static_gain,
        _,
    ) = online_safe_full_distribution_weights(
        teacher_train,
        anchor_train,
        anchor_train,
        gold_train,
        temperature=temperature,
        minimum_static_gold_logp_gain=minimum_static_gold_logp_gain,
        dynamic_gold_logp_gap_scale=dynamic_gold_logp_gap_scale,
        active_weight_floor=active_weight_floor,
    )
    initial_weights = initial_weights * reliable.float()
    active = active & reliable
    residual_train, action_effects, row_effects = action_marginal_residual_targets(
        teacher_train,
        anchor_train,
        groups_train,
        mask_train,
        initial_weights,
    )
    for index, row in enumerate(paired_train):
        count = len(row["legal_actions"])
        row["action_debiased_teacher_residual"] = [
            float(value) for value in residual_train[index, :count]
        ]

    paired_dev, teacher_dev, anchor_dev, groups_dev, mask_dev, _ = surfaces(dev_rows)
    maximum_dev_group = int(groups_dev[mask_dev].max())
    unseen_dev_groups: list[int] = []
    if maximum_dev_group >= action_effects.numel():
        unseen_dev_groups = list(range(action_effects.numel(), maximum_dev_group + 1))
        action_effects = F.pad(
            action_effects,
            (0, maximum_dev_group + 1 - action_effects.numel()),
        )
    residual_dev, _ = apply_action_marginal_residual_effects(
        teacher_dev,
        anchor_dev,
        groups_dev,
        mask_dev,
        action_effects,
    )
    for index, row in enumerate(paired_dev):
        count = len(row["legal_actions"])
        row["action_debiased_teacher_residual"] = [
            float(value) for value in residual_dev[index, :count]
        ]

    by_action: dict[str, Any] = {}
    for action_index in sorted(set(int(value) for value in groups_train[mask_train])):
        cells = mask_train & groups_train.eq(action_index)
        active_cells = cells & active[:, None]
        per_cell_weights = initial_weights / mask_train.sum(dim=-1).clamp_min(1)
        cell_weights = per_cell_weights[:, None].expand_as(residual_train)[
            active_cells
        ]
        values = residual_train[active_cells]
        weighted_mean = (
            float((values * cell_weights).sum() / cell_weights.sum())
            if cell_weights.numel() and float(cell_weights.sum()) > 0
            else 0.0
        )
        by_action[f"candidate_{action_index}"] = {
            "action_effect": float(action_effects[action_index]),
            "active_occurrences": int(active_cells.sum()),
            "weighted_residual_mean": weighted_mean,
        }
    legal_residual = residual_train[mask_train]
    active_residual_square = (
        residual_train.square() * mask_train
    ).sum(dim=-1) / mask_train.sum(dim=-1).clamp_min(1)
    return {
        "schema": "action_marginal_debiased_residual_calibration_v1",
        "train_paired_rows": len(paired_train),
        "dev_paired_rows": len(paired_dev),
        "teacher_reliable_rows": int(reliable.sum()),
        "static_valid_rows": int((static_valid & reliable).sum()),
        "initial_active_rows": int(active.sum()),
        "initial_weight_sum": float(initial_weights.sum()),
        "static_gold_logp_gain_sum": float(static_gain[active].sum()),
        "residual_absolute_mean": float(legal_residual.abs().mean()),
        "initial_weighted_residual_mse": float(
            (active_residual_square * initial_weights).mean()
        ),
        "maximum_absolute_row_residual_mean": float(
            (
                (residual_train * mask_train).sum(dim=-1)
                / mask_train.sum(dim=-1).clamp_min(1)
            ).abs().max()
        ),
        "fitted_row_effect_mean": float(row_effects.mean()),
        "unseen_dev_action_groups_assigned_zero_effect": unseen_dev_groups,
        "by_action": by_action,
    }


def distributed_device() -> tuple[torch.device, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        return torch.device(f"cuda:{local_rank}"), local_rank, rank, world
    if not torch.cuda.is_available():
        raise RuntimeError("SC-MRD training requires CUDA")
    return torch.device("cuda:0"), 0, 0, 1


class GroupDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class TypeAlignedDistributedSampler(Sampler[int]):
    """Give every DDP rank the same view type at each micro-step.

    SC-MRD paired rows execute extra native-model forwards.  Different forward
    counts across ranks can desynchronize DDP buffer collectives even when the
    backward passes use ``no_sync``.  Blocks here contain either only paired or
    only direct-only rows, and each rank receives one member of the same block.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        world: int,
        rank: int,
        seed: int,
        multiple: int,
    ) -> None:
        if world <= 1 or not 0 <= rank < world or multiple <= 0:
            raise ValueError("type-aligned sampler requires valid distributed ranks")
        if not rows:
            raise ValueError("type-aligned sampler requires nonempty rows")
        self.rows = rows
        self.world = world
        self.rank = rank
        self.seed = seed
        self.multiple = multiple
        self.epoch = 0
        paired_count = sum(bool(row["paired"]) for row in rows)
        unpaired_count = len(rows) - paired_count
        paired_blocks = math.ceil(paired_count / world)
        raw_length = paired_blocks + math.ceil(unpaired_count / world)
        self._length = math.ceil(raw_length / multiple) * multiple
        # Tiny/debug courses can contain one block of the minority type.  In
        # that case the first accumulation boundary may be too short to retain
        # the source paired fraction even if every padding block is paired.
        # Extend to the next complete boundary until the contract is feasible.
        while math.ceil(self._length * paired_count / len(rows)) > (
            paired_blocks + self._length - raw_length
        ):
            self._length += multiple

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        blocks: list[list[int]] = []
        blocks_by_type: dict[bool, list[list[int]]] = {False: [], True: []}
        for paired in (False, True):
            indices = [
                index
                for index, row in enumerate(self.rows)
                if bool(row["paired"]) == paired
            ]
            if not indices:
                continue
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[index] for index in order]
            padded = math.ceil(len(indices) / self.world) * self.world
            original = list(indices)
            indices.extend(
                original[index % len(original)]
                for index in range(padded - len(indices))
            )
            typed_blocks = [
                indices[start : start + self.world]
                for start in range(0, len(indices), self.world)
            ]
            blocks_by_type[paired].extend(typed_blocks)
            blocks.extend(typed_blocks)
        block_order = torch.randperm(
            len(blocks), generator=generator
        ).tolist()
        ordered = [blocks[index] for index in block_order]
        padding = self._length - len(ordered)
        if padding:
            # Closing a gradient-accumulation boundary must not silently
            # dilute selective KD.  Preserve at least the source paired-row
            # fraction when choosing the few repeated DDP blocks.
            source_paired = sum(bool(row["paired"]) for row in self.rows)
            target_paired_blocks = math.ceil(
                self._length * source_paired / len(self.rows)
            )
            paired_padding = min(
                padding,
                max(0, target_paired_blocks - len(blocks_by_type[True])),
            )
            unpaired_padding = padding - paired_padding
            requested = ((True, paired_padding), (False, unpaired_padding))
            for paired, count in requested:
                pool = blocks_by_type[paired]
                if count and not pool:
                    raise RuntimeError(
                        "type-aligned sampler cannot preserve paired exposure"
                    )
                ordered.extend(pool[index % len(pool)] for index in range(count))
            final_order = torch.randperm(
                len(ordered), generator=generator
            ).tolist()
            ordered = [ordered[index] for index in final_order]
        return iter(block[self.rank] for block in ordered)


class SequentialPhaseLoader:
    """Stream complete phases without mixing their batches.

    Each yielded batch records its phase and phase-local micro-step.  This
    lets the training loop reset AdamW, the LR schedule, KD warmup, and RNG at
    the boundary while retaining one in-memory model and one cumulative LoRA.
    """

    def __init__(
        self,
        phases: list[tuple[DataLoader, int]],
    ) -> None:
        if not phases or any(limit <= 0 for _loader, limit in phases):
            raise ValueError("sequential phases require positive micro-step limits")
        self.phases = phases

    def __len__(self) -> int:
        return sum(limit for _loader, limit in self.phases)

    def __iter__(self):
        for phase_index, (loader, limit) in enumerate(self.phases):
            emitted = 0
            for phase_micro_step, batch in enumerate(loader, start=1):
                if phase_micro_step > limit:
                    break
                batch["_training_phase"] = phase_index
                batch["_phase_micro_step"] = phase_micro_step
                emitted += 1
                yield batch
            if emitted != limit:
                raise RuntimeError(
                    f"phase {phase_index + 1} emitted {emitted}/{limit} micro-batches"
                )


def vision_sample(
    row: dict[str, Any],
    *,
    prompt_field: str,
    response_field: str,
    suffix: str,
) -> VisionSample:
    image = str(row["image_path"])
    return VisionSample(
        sample_id=f"{row['sample_id']}:{suffix}",
        image_path=image,
        image_paths=[image],
        prompt=str(row[prompt_field]).strip(),
        response=str(row[response_field]).strip(),
    )


@dataclass
class ActionSurface:
    positions: torch.Tensor
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    gold_indices: torch.Tensor
    legal_scoring_sequences: tuple[tuple[tuple[int, ...], ...], ...]


class GroupCollator:
    def __init__(
        self,
        processor: Any,
        *,
        max_length: int,
        objective: str,
        system_prompt: str = "",
        text_before_image: bool = True,
        online_ema_teacher: bool = False,
        ema_teacher_max_length: int | None = None,
    ) -> None:
        self.objective = objective
        self.online_ema_teacher = bool(online_ema_teacher)
        self.base = VisionSFTCollator(
            processor,
            max_length=max_length,
            system_prompt=system_prompt,
            text_before_image=text_before_image,
        )
        self.tokenizer = self.base.tokenizer
        self.ema_teacher_base = None
        if self.online_ema_teacher:
            teacher_length = int(ema_teacher_max_length or max_length)
            self.ema_teacher_base = VisionSFTCollator(
                processor,
                max_length=teacher_length,
                system_prompt=system_prompt,
                text_before_image=text_before_image,
            )

    def _response_start(
        self, labels: torch.Tensor, response: str, sample_id: str
    ) -> int:
        response_ids = self.tokenizer(
            response, add_special_tokens=False
        ).get("input_ids", [])
        if not response_ids:
            raise ValueError(f"empty assistant response: {sample_id}")
        active = labels.ne(-100).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            raise ValueError(f"truncated assistant response: {sample_id}")
        first = int(active[0])
        last_start = int(active[-1]) - len(response_ids) + 1
        matches = [
            start
            for start in range(first, last_start + 1)
            if labels[start : start + len(response_ids)].tolist() == response_ids
        ]
        if len(matches) != 1:
            raise ValueError(
                f"assistant response alignment failed: {sample_id}; matches={matches}"
            )
        return matches[0]

    def _collate(
        self,
        rows: list[dict[str, Any]],
        *,
        prompt_field: str,
        response_field: str,
        suffix: str,
    ) -> dict[str, torch.Tensor]:
        samples = [
            vision_sample(
                row,
                prompt_field=prompt_field,
                response_field=response_field,
                suffix=suffix,
            )
            for row in rows
        ]
        return self.base(samples)

    def _collate_ema_teacher(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if self.ema_teacher_base is None:
            raise RuntimeError("EMA Teacher collator is not configured")
        samples = [
            vision_sample(
                row,
                prompt_field="privileged_prompt",
                response_field="privileged_response",
                suffix="privileged",
            )
            for row in rows
        ]
        return self.ema_teacher_base(samples)

    def _action_token_mask(
        self,
        batch: dict[str, torch.Tensor],
        rows: list[dict[str, Any]],
        *,
        response_field: str,
    ) -> torch.Tensor:
        labels = batch["labels"]
        result = torch.zeros_like(labels, dtype=torch.bool)
        for row_index, row in enumerate(rows):
            # Unified one-stage courses may carry action-blind spatial SFT
            # replay beside Direct decision rows.  Those responses have no
            # action field by design and therefore receive ordinary token CE
            # only; action-aware rows retain the historical action-token boost.
            if not bool(row.get("action_supervision", True)):
                continue
            response = str(row[response_field])
            match = re_action_value(response)
            prefix_ids = self.tokenizer(
                response[: match[0]], add_special_tokens=False
            ).get("input_ids", [])
            value_ids = self.tokenizer(
                response[match[0] : match[1]], add_special_tokens=False
            ).get("input_ids", [])
            response_start = self._response_start(
                labels[row_index], response, str(row["sample_id"])
            )
            start = response_start + len(prefix_ids)
            end = start + len(value_ids)
            if end > labels.shape[1]:
                raise ValueError(f"truncated action value: {row['sample_id']}")
            if labels[row_index, start:end].tolist() != value_ids:
                raise ValueError(f"action token alignment failed: {row['sample_id']}")
            result[row_index, start:end] = True
        return result

    def _action_surface(
        self,
        batch: dict[str, torch.Tensor],
        rows: list[dict[str, Any]],
        *,
        response_field: str,
        batch_indices: list[int] | None = None,
    ) -> ActionSurface:
        labels = batch["labels"]
        source_indices = batch_indices or list(range(len(rows)))
        if len(source_indices) != len(rows):
            raise ValueError("action surface rows and batch indices disagree")
        specs = [
            action_token_spec(
                self.tokenizer,
                str(row[response_field]),
                row["legal_actions"],
            )
            for row in rows
        ]
        max_actions = max(len(spec.legal_token_ids) for spec in specs)
        token_ids = torch.zeros((len(rows), max_actions), dtype=torch.long)
        token_mask = torch.zeros((len(rows), max_actions), dtype=torch.bool)
        positions = torch.zeros((len(rows),), dtype=torch.long)
        gold = torch.zeros((len(rows),), dtype=torch.long)
        for index, (row, spec, source_index) in enumerate(
            zip(rows, specs, source_indices)
        ):
            response_start = self._response_start(
                labels[source_index], str(row[response_field]), str(row["sample_id"])
            )
            label_position = response_start + spec.response_token_offset
            logit_position = label_position - 1
            if logit_position < 0:
                raise ValueError("candidate token has no autoregressive context")
            if int(labels[source_index, label_position]) != spec.legal_token_ids[
                spec.gold_index
            ]:
                raise ValueError(
                    f"legal-action position mismatch: {row['sample_id']}"
                )
            count = len(spec.legal_token_ids)
            token_ids[index, :count] = torch.tensor(spec.legal_token_ids)
            token_mask[index, :count] = True
            positions[index] = logit_position
            gold[index] = spec.gold_index
        return ActionSurface(
            positions,
            token_ids,
            token_mask,
            gold,
            tuple(spec.legal_scoring_sequences for spec in specs),
        )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        direct = self._collate(
            rows,
            prompt_field="direct_prompt",
            response_field="direct_response",
            suffix="direct",
        )
        direct["sc_mrd_action_token_mask"] = self._action_token_mask(
            direct, rows, response_field="direct_response"
        )
        paired_indices = [index for index, row in enumerate(rows) if row["paired"]]
        result: dict[str, Any] = {
            "direct": direct,
            "paired_indices": torch.tensor(paired_indices, dtype=torch.long),
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "gold_actions": [str(row["gold_action"]) for row in rows],
            "direct_example_weights": torch.tensor(
                [float(row.get("direct_loss_weight", 1.0)) for row in rows],
                dtype=torch.float32,
            ),
            "direct_decision_weights": torch.tensor(
                [float(row.get("decision_loss_weight", 1.0)) for row in rows],
                dtype=torch.float32,
            ),
            "direct_action_field_weights": torch.tensor(
                [
                    float(row.get("action_field_loss_weight", 4.0))
                    for row in rows
                ],
                dtype=torch.float32,
            ),
        }
        if self.objective == "direct" or not paired_indices:
            return result

        paired = [rows[index] for index in paired_indices]
        if self.objective == "opd":
            result["direct_paired_surface"] = self._action_surface(
                direct,
                paired,
                response_field="direct_response",
                batch_indices=paired_indices,
            )
            cached_flags = [bool(row.get("teacher_legal_scores")) for row in paired]
            has_cached_teacher = all(cached_flags)
            if not has_cached_teacher and not self.online_ema_teacher:
                raise ValueError("OPD paired row lacks cached teacher legal scores")
            max_actions = max(len(row["legal_actions"]) for row in paired)
            teacher_scores = torch.full(
                (len(paired), max_actions),
                PADDED_LEGAL_LOGIT,
                dtype=torch.float32,
            )
            teacher_mask = torch.zeros(
                (len(paired), max_actions), dtype=torch.bool
            )
            for index, row in enumerate(paired):
                count = len(row["legal_actions"])
                teacher_mask[index, :count] = True
                if cached_flags[index]:
                    values = [
                        float(value) for value in row["teacher_legal_scores"]
                    ]
                    if len(values) != count:
                        raise ValueError("cached teacher surface length mismatch")
                    teacher_scores[index, :count] = torch.tensor(values)
                else:
                    # Online EMA modes replace these values before every loss
                    # computation.  Keep only the legal-action surface mask;
                    # no static Teacher decision or gate enters the course.
                    teacher_scores[index, :count] = 0.0
            result["cached_teacher_logits"] = teacher_scores
            result["cached_teacher_mask"] = teacher_mask
            teacher_modes = [
                str(row.get("teacher_runtime_mode", "ema"))
                for row in paired
            ]
            if any(mode not in {"cached", "ema"} for mode in teacher_modes):
                raise ValueError(
                    "teacher_runtime_mode must be cached or ema"
                )
            result["online_teacher_flags"] = torch.tensor(
                [mode == "ema" for mode in teacher_modes],
                dtype=torch.bool,
            )
            result["cached_teacher_available"] = torch.tensor(
                [bool(row.get("teacher_legal_scores")) for row in paired],
                dtype=torch.bool,
            )
            if self.online_ema_teacher:
                required = {"privileged_prompt", "privileged_response"}
                if any(not required.issubset(row) for row in paired):
                    raise ValueError(
                        "online EMA Teacher rows must carry privileged prompt/response"
                    )
                privileged = self._collate_ema_teacher(paired)
                result["reasoned"] = privileged
                result["reasoned_surface"] = self._action_surface(
                    privileged,
                    paired,
                    response_field="privileged_response",
                )
                direct_surface = result["direct_paired_surface"]
                reasoned_surface = result["reasoned_surface"]
                if not (
                    torch.equal(
                        direct_surface.token_ids,
                        reasoned_surface.token_ids,
                    )
                    and torch.equal(
                        direct_surface.token_mask,
                        reasoned_surface.token_mask,
                    )
                    and torch.equal(
                        direct_surface.gold_indices,
                        reasoned_surface.gold_indices,
                    )
                ):
                    raise ValueError(
                        "Direct and privileged online-KD action surfaces disagree"
                    )
            adaptive_rows = [
                row.get("privileged_margin_kd_weight") for row in paired
            ]
            if any(value is not None for value in adaptive_rows):
                if not all(value is not None for value in adaptive_rows):
                    raise ValueError(
                        "privileged-margin KD weights must cover every paired row"
                    )
                result["privileged_margin_kd_weights"] = torch.tensor(
                    [float(value) for value in adaptive_rows],
                    dtype=torch.float32,
                )
                result["privileged_margin_kd_eligible"] = torch.tensor(
                    [
                        bool(row.get("privileged_margin_kd_eligible"))
                        for row in paired
                    ],
                    dtype=torch.bool,
                )
            correction_rows = [
                row.get("privileged_correction_kd_weight") for row in paired
            ]
            if any(value is not None for value in correction_rows):
                if not all(value is not None for value in correction_rows):
                    raise ValueError(
                        "privileged correction targets must cover every paired row"
                    )
                result["privileged_correction_kd_weights"] = torch.tensor(
                    [float(value) for value in correction_rows],
                    dtype=torch.float32,
                )
                result["privileged_correction_kd_eligible"] = torch.tensor(
                    [
                        bool(row.get("privileged_correction_kd_eligible"))
                        for row in paired
                    ],
                    dtype=torch.bool,
                )
                result["privileged_correction_competitor_indices"] = torch.tensor(
                    [
                        int(row["privileged_correction_competitor_index"])
                        for row in paired
                    ],
                    dtype=torch.long,
                )
                result["privileged_correction_target_margins"] = torch.tensor(
                    [
                        float(row["privileged_correction_target_margin"])
                        for row in paired
                    ],
                    dtype=torch.float32,
                )
                result["privileged_correction_margin_gains"] = torch.tensor(
                    [
                        float(row["privileged_correction_margin_gain"])
                        for row in paired
                    ],
                    dtype=torch.float32,
                )
            anchor_rows = [
                row.get("anchor_direct_legal_scores") for row in paired
            ]
            if any(values is not None for values in anchor_rows):
                if not all(values is not None for values in anchor_rows):
                    raise ValueError(
                        "anchor Direct score cache must cover every paired row"
                    )
                anchor_scores = torch.full(
                    (len(paired), max_actions),
                    PADDED_LEGAL_LOGIT,
                    dtype=torch.float32,
                )
                for index, (row, raw_values) in enumerate(
                    zip(paired, anchor_rows)
                ):
                    values = [float(value) for value in raw_values]
                    if len(values) != len(row["legal_actions"]):
                        raise ValueError(
                            "cached anchor Direct surface length mismatch"
                        )
                    anchor_scores[index, : len(values)] = torch.tensor(values)
                result["cached_anchor_direct_logits"] = anchor_scores
                result["cached_anchor_direct_mask"] = teacher_mask.clone()
            no_bridge_score_rows = [
                row.get("spatial_without_bridge_legal_scores") for row in paired
            ]
            if any(values is not None for values in no_bridge_score_rows):
                if not all(values is not None for values in no_bridge_score_rows):
                    raise ValueError(
                        "spatial-without-Bridge score cache must cover every paired row"
                    )
                no_bridge_scores = torch.full(
                    (len(paired), max_actions),
                    PADDED_LEGAL_LOGIT,
                    dtype=torch.float32,
                )
                for index, (row, raw_values) in enumerate(
                    zip(paired, no_bridge_score_rows)
                ):
                    values = [float(value) for value in raw_values]
                    if len(values) != len(row["legal_actions"]):
                        raise ValueError(
                            "cached spatial-without-Bridge score surface mismatch"
                        )
                    no_bridge_scores[index, : len(values)] = torch.tensor(values)
                result["cached_no_bridge_logits"] = no_bridge_scores
            residual_rows = [
                row.get("action_debiased_teacher_residual") for row in paired
            ]
            if any(values is not None for values in residual_rows):
                if not all(values is not None for values in residual_rows):
                    raise ValueError(
                        "action-debiased residual targets must cover every paired row"
                    )
                residual_scores = torch.zeros(
                    (len(paired), max_actions), dtype=torch.float32
                )
                for index, (row, raw_values) in enumerate(
                    zip(paired, residual_rows)
                ):
                    values = [float(value) for value in raw_values]
                    if len(values) != len(row["legal_actions"]):
                        raise ValueError(
                            "action-debiased residual target length mismatch"
                        )
                    residual_scores[index, : len(values)] = torch.tensor(values)
                result["cached_action_debiased_teacher_residual"] = residual_scores
            representation_rows = [row.get("bridge_final_hidden") for row in paired]
            anchor_representation_rows = [row.get("anchor_final_hidden") for row in paired]
            if any(value is not None for value in representation_rows):
                if not all(
                    value is not None for value in representation_rows
                ) or not all(
                    value is not None for value in anchor_representation_rows
                ):
                    raise ValueError(
                        "representation cache must cover every paired row"
                    )
                result["cached_bridge_final_hidden"] = torch.stack(
                    [value.float() for value in representation_rows]
                )
                result["cached_anchor_final_hidden"] = torch.stack(
                    [value.float() for value in anchor_representation_rows]
                )
            no_bridge_rows = [row.get("no_bridge_final_hidden") for row in paired]
            if any(value is not None for value in no_bridge_rows):
                if not all(value is not None for value in no_bridge_rows):
                    raise ValueError(
                        "no-Bridge representation cache must cover every paired row"
                    )
                result["cached_no_bridge_final_hidden"] = torch.stack(
                    [value.float() for value in no_bridge_rows]
                )
            if not self.online_ema_teacher:
                result["reasoned_surface"] = result["direct_paired_surface"]
            return result
        reason = self._collate(
            paired,
            prompt_field="reasoning_prompt",
            response_field="reasoning_response",
            suffix="reason",
        )
        result["reason"] = reason
        if self.objective == "aux":
            return result

        reasoned = self._collate(
            paired,
            prompt_field="reasoned_action_prompt",
            response_field="reasoned_action_response",
            suffix="reasoned",
        )
        result["reasoned"] = reasoned
        result["reasoned_surface"] = self._action_surface(
            reasoned, paired, response_field="reasoned_action_response"
        )
        result["direct_paired_surface"] = self._action_surface(
            direct,
            paired,
            response_field="direct_response",
            batch_indices=paired_indices,
        )

        negative_positions = [
            index
            for index, row in enumerate(paired)
            if row.get("counterfactual_active")
            and row.get("counterfactual_action_prompt")
        ]
        result["negative_pair_positions"] = torch.tensor(
            negative_positions, dtype=torch.long
        )
        if negative_positions:
            negative_rows = [paired[index] for index in negative_positions]
            negative = self._collate(
                negative_rows,
                prompt_field="counterfactual_action_prompt",
                response_field="reasoned_action_response",
                suffix="counterfactual",
            )
            result["negative"] = negative
            result["negative_surface"] = self._action_surface(
                negative,
                negative_rows,
                response_field="reasoned_action_response",
            )
        return result


def re_action_value(response: str) -> tuple[int, int]:
    import re

    match = re.search(
        r'"action"\s*:\s*"(?P<value>candidate_[0-9]+|Continue|Stop)"',
        response,
    )
    if match is None:
        match = re.search(r'"done"\s*:\s*(?P<value>[01])', response)
    if match is None:
        raise ValueError(f"response has no SC-MRD action: {response!r}")
    return match.span("value")


def native_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in MODEL_INPUT_KEYS and torch.is_tensor(value)
    }


def pop_action_token_mask(
    batch: dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor | None:
    value = batch.pop("sc_mrd_action_token_mask", None)
    return value.to(device) if value is not None else None


def surface_to_device(surface: ActionSurface, device: torch.device) -> ActionSurface:
    return ActionSurface(
        surface.positions.to(device),
        surface.token_ids.to(device),
        surface.token_mask.to(device),
        surface.gold_indices.to(device),
        surface.legal_scoring_sequences,
    )


def mean_row_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    action_token_mask: torch.Tensor | None = None,
    action_token_weight: float = 1.0,
    example_weights: torch.Tensor | None = None,
    decision_weights: torch.Tensor | None = None,
    action_field_weights: torch.Tensor | None = None,
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
    if decision_weights is not None:
        if decision_weights.ndim != 1 or decision_weights.shape[0] != labels.shape[0]:
            raise ValueError("decision weights do not align with batch rows")
        weights = weights * decision_weights[:, None].float()
    if action_token_mask is not None:
        if action_token_mask.shape != labels.shape:
            raise ValueError("action-token mask does not align with labels")
        boost = action_token_mask[:, 1:].float()
        if action_field_weights is None:
            field_weights = torch.full(
                (labels.shape[0],),
                float(action_token_weight),
                dtype=torch.float32,
                device=labels.device,
            )
        else:
            if (
                action_field_weights.ndim != 1
                or action_field_weights.shape[0] != labels.shape[0]
            ):
                raise ValueError("action-field weights do not align with batch rows")
            field_weights = action_field_weights.float()
        weights = weights * (1.0 + boost * (field_weights[:, None] - 1.0))
    if weight_semantics == "legacy_normalized":
        denominator = weights.sum(1)
    elif weight_semantics == "historical_sft":
        denominator = active.float().sum(1)
    else:
        raise ValueError(f"unknown native LM weight semantics: {weight_semantics}")
    per_row = (token_loss * weights).sum(1) / denominator.clamp_min(1.0)
    valid = active.any(1)
    if not valid.any():
        raise ValueError("batch has no supervised assistant tokens")
    if example_weights is None:
        return per_row[valid].mean()
    if example_weights.ndim != 1 or example_weights.shape[0] != per_row.shape[0]:
        raise ValueError("Direct CE example weights do not align with batch rows")
    selected_weights = example_weights[valid].to(per_row.dtype)
    if not torch.isfinite(selected_weights).all() or (selected_weights < 0).any():
        raise ValueError("Direct CE example weights must be finite and nonnegative")
    return (per_row[valid] * selected_weights).mean()


def action_logits(
    output_logits: torch.Tensor,
    surface: ActionSurface,
) -> torch.Tensor:
    if any(
        len(sequence) > 1
        for row_sequences in surface.legal_scoring_sequences
        for sequence in row_sequences
    ):
        raise ValueError(
            "multi-token actions require the OPD sequence-aware action scorer"
        )
    return legal_action_logits(
        output_logits,
        surface.positions,
        surface.token_ids,
        surface.token_mask,
    )


def opd_student_action_logits(
    model: torch.nn.Module,
    output_logits: torch.Tensor,
    surface: ActionSurface,
    direct_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Score every legal OPD action, including multi-digit candidate ids.

    The common case remains one native-LM gather.  For the small fraction of
    rows containing candidate_10+, the legal decimal strings form a depth-two
    prefix tree.  We reuse the Direct prefix logits and make one differentiable
    continuation forward per ambiguous first digit, rather than dropping rows
    or introducing an unrelated classification head.
    """

    needs_sequence = any(
        len(sequence) > 1
        for row_sequences in surface.legal_scoring_sequences
        for sequence in row_sequences
    )
    if not needs_sequence:
        return legal_action_logits(
            output_logits,
            surface.positions,
            surface.token_ids,
            surface.token_mask,
        )
    if (
        output_logits.shape[0] != 1
        or surface.positions.numel() != 1
        or direct_inputs["input_ids"].shape[0] != 1
    ):
        raise ValueError(
            "multi-token OPD action scoring requires per-device batch size 1"
        )
    sequences = surface.legal_scoring_sequences[0]
    if any(not sequence or len(sequence) > 2 for sequence in sequences):
        raise ValueError("formal Move action codes must contain one or two tokens")
    position = int(surface.positions[0])
    label_position = position + 1
    if label_position <= 0 or label_position > direct_inputs["input_ids"].shape[1]:
        raise ValueError("multi-token OPD action position is out of bounds")
    base_logits = output_logits[0, position].float()
    continuation_logits: dict[int, torch.Tensor] = {}
    for first_token in sorted(
        {sequence[0] for sequence in sequences if len(sequence) == 2}
    ):
        prefix_ids = direct_inputs["input_ids"][:, :label_position]
        branch_ids = torch.cat(
            (
                prefix_ids,
                torch.tensor(
                    [[first_token]],
                    dtype=prefix_ids.dtype,
                    device=prefix_ids.device,
                ),
            ),
            dim=1,
        )
        branch: dict[str, torch.Tensor] = {
            key: value
            for key, value in direct_inputs.items()
            if key
            not in {
                "input_ids",
                "attention_mask",
                "labels",
                "position_ids",
            }
        }
        branch["input_ids"] = branch_ids
        branch["attention_mask"] = torch.ones_like(branch_ids)
        branch_output = model(**branch, logits_to_keep=1, use_cache=False)
        continuation_logits[first_token] = F.log_softmax(
            branch_output.logits[0, -1].float(), dim=-1
        )
    scores: list[torch.Tensor] = []
    for sequence in sequences:
        score = base_logits[sequence[0]]
        if len(sequence) == 2:
            score = score + continuation_logits[sequence[0]][sequence[1]]
        scores.append(score)
    return torch.stack(scores).unsqueeze(0)


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


def named_lora_parameters(
    model: torch.nn.Module, adapter_name: str
) -> dict[str, torch.nn.Parameter]:
    """Return the LoRA A/B parameters for one named adapter."""
    marker = f".{adapter_name}."
    result = {
        name: parameter
        for name, parameter in model.named_parameters()
        if marker in name and ("lora_A" in name or "lora_B" in name)
    }
    if not result:
        raise RuntimeError(f"adapter {adapter_name!r} has no LoRA parameters")
    return result


def restore_student_adapter_training_contract(model: torch.nn.Module) -> None:
    conditional = model.module if isinstance(model, DDP) else model
    for name, parameter in conditional.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        parameter.requires_grad_(f".{DELTA_ADAPTER}." in name)


@torch.no_grad()
def initialize_ema_teacher_adapter(model: torch.nn.Module) -> None:
    """Copy the trainable delta into an inference-only EMA adapter."""
    conditional = model.module if isinstance(model, DDP) else model
    config = conditional.peft_config[DELTA_ADAPTER]
    conditional.add_adapter(EMA_TEACHER_ADAPTER, config)
    student = named_lora_parameters(conditional, DELTA_ADAPTER)
    teacher = named_lora_parameters(conditional, EMA_TEACHER_ADAPTER)
    expected = {
        name.replace(f".{DELTA_ADAPTER}.", f".{EMA_TEACHER_ADAPTER}.")
        for name in student
    }
    if expected != set(teacher):
        raise RuntimeError("Student/EMA Teacher LoRA parameter surfaces disagree")
    for student_name, student_parameter in student.items():
        teacher_name = student_name.replace(
            f".{DELTA_ADAPTER}.", f".{EMA_TEACHER_ADAPTER}."
        )
        teacher_parameter = teacher[teacher_name]
        teacher_parameter.copy_(student_parameter)
        teacher_parameter.requires_grad_(False)
    conditional.set_adapter([ANCHOR_ADAPTER, DELTA_ADAPTER])
    restore_student_adapter_training_contract(conditional)


@torch.no_grad()
def update_ema_teacher_adapter(model: torch.nn.Module, decay: float) -> None:
    conditional = model.module if isinstance(model, DDP) else model
    student = named_lora_parameters(conditional, DELTA_ADAPTER)
    teacher = named_lora_parameters(conditional, EMA_TEACHER_ADAPTER)
    for student_name, student_parameter in student.items():
        teacher_name = student_name.replace(
            f".{DELTA_ADAPTER}.", f".{EMA_TEACHER_ADAPTER}."
        )
        teacher[teacher_name].mul_(decay).add_(
            student_parameter.detach(), alpha=1.0 - decay
        )


@torch.no_grad()
def ema_teacher_action_logits(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    surface: ActionSurface,
) -> torch.Tensor:
    """Run the privileged view through anchor plus the EMA delta."""
    conditional = model.module if isinstance(model, DDP) else model
    was_training = conditional.training
    conditional.set_adapter([ANCHOR_ADAPTER, EMA_TEACHER_ADAPTER])
    conditional.eval()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            outputs = conditional(**inputs)
        logits = opd_student_action_logits(
            conditional, outputs.logits, surface, inputs
        ).detach()
        del outputs
        return logits
    finally:
        conditional.set_adapter([ANCHOR_ADAPTER, DELTA_ADAPTER])
        restore_student_adapter_training_contract(conditional)
        conditional.train(was_training)


def save_training_adapter(
    model: torch.nn.Module,
    processor: Any,
    output: Path,
) -> None:
    """Save the sole trainable adapter as a normal PEFT adapter.

    The model may be either the fresh delta in an anchor-plus-delta training
    stack or a directly continued B0 adapter.  Both use the canonical
    ``default`` adapter name and produce one independently loadable artifact.
    """
    output.mkdir(parents=True, exist_ok=False)
    state = get_peft_model_state_dict(
        model,
        adapter_name=DELTA_ADAPTER,
    )
    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
    }
    if not cpu_state or not all("lora_" in key for key in cpu_state):
        raise RuntimeError("adapter extraction did not produce a pure LoRA state")
    save_safetensors(cpu_state, output / "adapter_model.safetensors")
    config = model.peft_config[DELTA_ADAPTER]
    config.save_pretrained(output)
    processor.save_pretrained(output)


@torch.no_grad()
def evaluate_paired(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int,
    kd_require_teacher_correct: bool,
    kd_min_teacher_gold_margin: float | None,
    cached_teacher: bool = False,
    online_ema_teacher: bool = False,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    reasoned_correct = 0
    count = 0
    losses: list[float] = []
    counterfactual_pass = 0
    counterfactual_count = 0
    kd_eligible = 0
    teacher_margin_sum = 0.0
    for grouped in loader:
        indices = grouped["paired_indices"]
        if indices.numel() == 0:
            continue
        direct_raw = grouped["direct"]
        pop_action_token_mask(direct_raw, device)
        direct = native_batch(direct_raw, device)
        outputs = model(**direct)
        surface = surface_to_device(grouped["direct_paired_surface"], device)
        # The direct batch may include unpaired rows; action positions are
        # absolute in that batch while the legal rows follow paired order.
        paired = indices.to(device)
        logits = opd_student_action_logits(
            model,
            outputs.logits[paired],
            surface,
            direct,
        )
        correct += int(logits.argmax(-1).eq(surface.gold_indices).sum())
        count += logits.shape[0]
        losses.append(
            float(F.cross_entropy(logits, surface.gold_indices).detach())
        )

        reasoned_surface = surface_to_device(grouped["reasoned_surface"], device)
        reasoned_outputs = None
        if online_ema_teacher:
            online_flags = grouped.get(
                "online_teacher_flags",
                torch.ones(
                    reasoned_surface.gold_indices.shape[0], dtype=torch.bool
                ),
            ).to(device)
            cached_available = grouped.get(
                "cached_teacher_available",
                torch.zeros(
                    reasoned_surface.gold_indices.shape[0], dtype=torch.bool
                ),
            ).to(device)
            if bool((~online_flags & ~cached_available).any()):
                raise RuntimeError(
                    "diagnostic mixed Teacher row lacks a cached surface"
                )
            cached_logits = grouped["cached_teacher_logits"].to(device)
            if bool(online_flags.any()):
                reasoned_raw = grouped["reasoned"]
                reasoned = native_batch(reasoned_raw, device)
                ema_logits = ema_teacher_action_logits(
                    model, reasoned, reasoned_surface
                )
                teacher_logits = torch.where(
                    online_flags[:, None], ema_logits, cached_logits
                )
            else:
                teacher_logits = cached_logits
        elif cached_teacher:
            teacher_mask = grouped["cached_teacher_mask"].to(device)
            if not torch.equal(teacher_mask, reasoned_surface.token_mask):
                raise ValueError("cached teacher and legal-action masks disagree")
            teacher_logits = grouped["cached_teacher_logits"].to(device)
        else:
            reasoned_raw = grouped["reasoned"]
            reasoned = native_batch(reasoned_raw, device)
            reasoned_outputs = model(**reasoned)
            teacher_logits = action_logits(
                reasoned_outputs.logits, reasoned_surface
            )
        reliable, teacher_correct, teacher_margin = reliable_teacher_mask(
            teacher_logits,
            reasoned_surface.gold_indices,
            require_correct=kd_require_teacher_correct,
            min_gold_margin=kd_min_teacher_gold_margin,
        )
        reasoned_correct += int(teacher_correct.sum())
        teacher_margin_sum += float(teacher_margin.sum())
        kd_eligible += int(reliable.sum())
        negative_positions = (
            torch.empty(0, dtype=torch.long, device=device)
            if cached_teacher or online_ema_teacher
            else grouped["negative_pair_positions"].to(device)
        )
        if negative_positions.numel():
            negative_raw = grouped["negative"]
            negative = native_batch(negative_raw, device)
            negative_outputs = model(**negative)
            negative_surface = surface_to_device(
                grouped["negative_surface"], device
            )
            negative_logits = action_logits(
                negative_outputs.logits, negative_surface
            )
            positive_margin = gold_margin(
                teacher_logits[negative_positions],
                reasoned_surface.gold_indices[negative_positions],
            )
            negative_margin = gold_margin(
                negative_logits, negative_surface.gold_indices
            )
            passed = positive_margin - negative_margin >= 0.1
            counterfactual_pass += int(passed.sum())
            counterfactual_count += int(passed.numel())
            del negative_outputs
        if reasoned_outputs is not None:
            del reasoned_outputs
        if max_samples > 0 and count >= max_samples:
            break
    model.train()
    return {
        "paired_rows": count,
        "direct_action_accuracy": correct / max(1, count),
        "reasoned_action_accuracy": reasoned_correct / max(1, count),
        "teacher_gold_margin_mean": teacher_margin_sum / max(1, count),
        "kd_eligible_rows": kd_eligible,
        "kd_eligible_coverage": kd_eligible / max(1, count),
        "direct_action_ce": sum(losses) / max(1, len(losses)),
        "counterfactual_rows": counterfactual_count,
        "counterfactual_margin_pass_rate": (
            counterfactual_pass / max(1, counterfactual_count)
        ),
    }


def main() -> None:
    args = parse_args()
    phase2_enabled = bool(args.phase2_data_dir)
    if args.temperature <= 0 or args.counterfactual_margin <= 0:
        raise ValueError("temperature and counterfactual margin must be positive")
    if args.ema_teacher_max_length < args.max_length:
        raise ValueError("--ema-teacher-max-length cannot be below --max-length")
    if (
        not math.isfinite(args.anchor_retention_weight)
        or args.anchor_retention_weight < 0
    ):
        raise ValueError("--anchor-retention-weight must be finite and nonnegative")
    if args.anchor_retention_weight > 0 and args.objective != "opd":
        raise ValueError("anchor retention is supported only for cached OPD")
    if (
        args.kd_min_teacher_gold_margin is not None
        and not math.isfinite(args.kd_min_teacher_gold_margin)
    ):
        raise ValueError("--kd-min-teacher-gold-margin must be finite")
    adaptive_kd_values = (
        args.kd_min_online_margin_advantage,
        args.kd_online_margin_weight_scale,
        args.kd_online_confidence_increment_cap,
        args.kd_repair_tier_weight,
        args.kd_confidence_tier_weight,
    )
    if any(not math.isfinite(value) for value in adaptive_kd_values):
        raise ValueError("adaptive KD parameters must be finite")
    if args.kd_min_online_margin_advantage < 0:
        raise ValueError("online KD margin advantage must be nonnegative")
    if (
        args.kd_online_margin_weight_scale <= 0
        or args.kd_online_confidence_increment_cap <= 0
    ):
        raise ValueError("online pairwise KD scale/cap must be positive")
    if args.kd_repair_tier_weight < 0 or args.kd_confidence_tier_weight < 0:
        raise ValueError("online KD tier weights must be nonnegative")
    if args.kd_mode in {
        "online_advantage_tiered",
        "online_pairwise_adaptive",
        "online_ema_pairwise_adaptive",
        HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
        "online_safe_full_distribution",
        "online_action_debiased_residual",
        "online_bridge_representation_residual",
        "online_bridge_increment_representation_residual",
        "online_action_projected_bridge_increment",
    }:
        if not args.kd_require_teacher_correct:
            raise ValueError(
                "online advantage KD requires --kd-require-teacher-correct"
            )
        if not any(
            value > 0
            for value in (
                args.kd_repair_tier_weight,
                args.kd_confidence_tier_weight,
            )
        ):
            raise ValueError("online advantage KD requires a positive tier weight")
    ema_values = (
        args.ema_teacher_decay,
        args.ema_sol_match_temperature,
        args.ema_sol_match_margin_center,
        args.ema_sol_match_weight_floor,
    )
    if any(not math.isfinite(value) for value in ema_values):
        raise ValueError("EMA/Sol-match parameters must be finite")
    if not 0.0 < args.ema_teacher_decay < 1.0:
        raise ValueError("--ema-teacher-decay must be in (0,1)")
    if args.ema_sol_match_temperature <= 0:
        raise ValueError("--ema-sol-match-temperature must be positive")
    if not 0.0 <= args.ema_sol_match_weight_floor < 1.0:
        raise ValueError("--ema-sol-match-weight-floor must be in [0,1)")
    if args.kd_mode in ONLINE_EMA_KD_MODES:
        if args.objective != "opd" or args.decision_kind != "move":
            raise ValueError(
                "online EMA KD requires cached Move OPD"
            )
        if (
            args.kd_mode not in {
                "online_ema_pairwise_adaptive",
                HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
            }
            and (
                args.kd_require_teacher_correct
                or args.kd_min_teacher_gold_margin is not None
            )
        ):
            raise ValueError(
                "online EMA KD must not hard-filter by cached Teacher correctness"
            )
    if args.kd_mode in {
        "online_pairwise_adaptive",
        "online_ema_pairwise_adaptive",
        HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
    }:
        if args.objective != "opd" or args.decision_kind != "move":
            raise ValueError("online pairwise adaptive KD requires Move OPD")
        if args.kd_min_teacher_gold_margin is None:
            raise ValueError(
                "online pairwise adaptive KD requires an explicit Teacher margin"
            )
    safe_full_kd_values = (
        args.kd_min_static_gold_logp_gain,
        args.kd_dynamic_gold_logp_gap_scale,
        args.kd_active_weight_floor,
    )
    if any(not math.isfinite(value) for value in safe_full_kd_values):
        raise ValueError("safe full-distribution KD parameters must be finite")
    if args.kd_dynamic_gold_logp_gap_scale <= 0:
        raise ValueError("safe full-distribution KD gap scale must be positive")
    if not 0 <= args.kd_active_weight_floor <= 1:
        raise ValueError("safe full-distribution KD weight floor must be in [0, 1]")
    if args.kd_mode in {
        "online_safe_full_distribution",
        "online_action_debiased_residual",
        "online_bridge_representation_residual",
        "online_bridge_increment_representation_residual",
        "online_action_projected_bridge_increment",
    }:
        if args.objective != "opd" or args.decision_kind != "move":
            raise ValueError(
                "online causal-safe distribution/residual/representation KD requires cached Move OPD"
            )
        if not args.kd_require_teacher_correct:
            raise ValueError(
                "online causal-safe distribution/residual/representation KD requires a correct Teacher"
            )
        if args.kd_min_teacher_gold_margin is None:
            raise ValueError(
                "online causal-safe distribution/residual/representation KD requires a Teacher margin"
            )
    if args.kd_mode in {
        "online_bridge_representation_residual",
        "online_bridge_increment_representation_residual",
        "online_action_projected_bridge_increment",
    }:
        if not args.representation_cache:
            raise ValueError(
                "representation residual KD requires --representation-cache"
            )
    margin_weight_bounds = (
        args.kd_margin_weight_min,
        args.kd_margin_weight_max,
    )
    if any(not math.isfinite(value) for value in margin_weight_bounds):
        raise ValueError("privileged-margin KD weights must be finite")
    if (
        args.kd_margin_weight_min < 0
        or args.kd_margin_weight_max < args.kd_margin_weight_min
    ):
        raise ValueError("privileged-margin KD weight bounds are invalid")
    if args.kd_mode == "privileged_margin_adaptive":
        if args.objective != "opd":
            raise ValueError("privileged-margin adaptive KD requires cached OPD")
        if not args.kd_require_teacher_correct:
            raise ValueError(
                "privileged-margin adaptive KD requires "
                "--kd-require-teacher-correct"
            )
        if args.kd_min_teacher_gold_margin is None:
            raise ValueError(
                "privileged-margin adaptive KD requires an explicit minimum margin"
            )
        if not math.isclose(
            args.kd_margin_weight_min + args.kd_margin_weight_max,
            2.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "privileged-margin bounds must be symmetric around one to "
                "preserve nominal KD mass"
            )
    correction_kd_values = (
        args.kd_min_reasoning_gain,
        args.kd_pair_target_margin_cap,
        args.kd_min_bridge_gain,
        args.kd_pair_huber_beta,
    )
    if any(not math.isfinite(value) for value in correction_kd_values):
        raise ValueError("correction-margin KD parameters must be finite")
    if args.kd_min_reasoning_gain < 0:
        raise ValueError("correction-margin reasoning gain must be nonnegative")
    if args.kd_pair_target_margin_cap <= 0:
        raise ValueError("correction-margin target cap must be positive")
    if args.kd_min_bridge_gain < 0:
        raise ValueError("Bridge correction gain must be nonnegative")
    if args.kd_pair_huber_beta <= 0:
        raise ValueError("Bridge correction Huber beta must be positive")
    if args.kd_mode in {
        "privileged_correction_margin",
        "privileged_bridge_correction_margin",
    }:
        if args.objective != "opd" or args.decision_kind != "move":
            raise ValueError(
                "privileged correction-margin KD requires cached Move OPD"
            )
        if not args.kd_require_teacher_correct:
            raise ValueError(
                "privileged correction-margin KD requires "
                "--kd-require-teacher-correct"
            )
        if args.kd_min_teacher_gold_margin is None:
            raise ValueError(
                "privileged correction-margin KD requires an explicit "
                "minimum Teacher margin"
            )
        if not math.isclose(
            args.kd_margin_weight_min + args.kd_margin_weight_max,
            2.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "correction-margin bounds must be symmetric around one"
            )
    kd_class_weights = (args.kd_negative_weight, args.kd_positive_weight)
    if any(not math.isfinite(value) or value < 0 for value in kd_class_weights):
        raise ValueError("KD class weights must be finite and nonnegative")
    if not any(value > 0 for value in kd_class_weights):
        raise ValueError("at least one KD class weight must be positive")
    if args.decision_kind != "stop" and kd_class_weights != (1.0, 1.0):
        raise ValueError("KD class weights are supported only for Stop")
    direct_class_weights = (
        args.direct_negative_weight,
        args.direct_positive_weight,
    )
    if any(
        not math.isfinite(value) or value < 0 for value in direct_class_weights
    ):
        raise ValueError("Direct CE class weights must be finite and nonnegative")
    if not any(value > 0 for value in direct_class_weights):
        raise ValueError("at least one Direct CE class weight must be positive")
    if (
        (args.objective != "opd" or args.decision_kind != "stop")
        and direct_class_weights != (1.0, 1.0)
    ):
        raise ValueError("Direct CE class weights are supported only for Stop OPD")
    if args.diagnostic_batch_size <= 0:
        raise ValueError("--diagnostic-batch-size must be positive")
    if not 0.0 <= args.kd_warmup_ratio <= 1.0:
        raise ValueError("--kd-warmup-ratio must be in [0,1]")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be in [0,1]")
    if phase2_enabled:
        if args.epochs != 1:
            raise ValueError("in-process two-phase training requires --epochs 1")
        if args.phase2_learning_rate <= 0:
            raise ValueError("--phase2-learning-rate must be positive")
        if args.phase2_per_device_batch_size <= 0:
            raise ValueError("--phase2-per-device-batch-size must be positive")
        if args.phase2_grad_accum_steps <= 0:
            raise ValueError("--phase2-grad-accum-steps must be positive")
        if (
            args.expected_phase2_train_rows <= 0
            or args.expected_phase2_dev_rows <= 0
        ):
            raise ValueError("phase-2 row contracts must be positive")
        if args.phase2_kd_weight is None or args.phase2_kd_weight < 0:
            raise ValueError("--phase2-kd-weight must be nonnegative")
        if args.phase2_kd_warmup_ratio is None:
            args.phase2_kd_warmup_ratio = args.kd_warmup_ratio
        if not 0.0 <= args.phase2_kd_warmup_ratio <= 1.0:
            raise ValueError("--phase2-kd-warmup-ratio must be in [0,1]")
        if not 0.0 <= args.phase2_warmup_ratio <= 1.0:
            raise ValueError("--phase2-warmup-ratio must be in [0,1]")
        if args.phase2_seed <= 0:
            args.phase2_seed = args.seed
        if args.phase2_max_length <= 0:
            args.phase2_max_length = args.max_length
    device, local_rank, rank, world = distributed_device()
    is_main = rank == 0
    set_seed(args.seed + rank)

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
    audit_path = data / "dataset_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("SC-MRD dataset audit did not pass")
    train_rows = read_jsonl(data / "train.jsonl")
    dev_rows = read_jsonl(data / "dev.jsonl")
    if args.anchor_retention_weight > 0:
        missing_anchor_cache = sum(
            bool(row.get("paired"))
            and not isinstance(row.get("anchor_direct_legal_scores"), list)
            for row in [*train_rows, *dev_rows]
        )
        if missing_anchor_cache:
            raise RuntimeError(
                f"anchor retention cache missing on {missing_anchor_cache} paired rows"
            )
    source_train_rows_actual = len(train_rows)
    source_dev_rows_actual = len(dev_rows)
    formal_row_contract = args.max_train_samples <= 0
    if formal_row_contract and (
        len(train_rows) != args.expected_train_rows
        or len(dev_rows) != args.expected_dev_rows
    ):
        raise RuntimeError(
            f"SC-MRD row drift: {len(train_rows)}/{len(dev_rows)} "
            f"expected {args.expected_train_rows}/{args.expected_dev_rows}"
        )
    if args.max_train_samples > 0:
        train_rows = sorted(
            train_rows,
            key=lambda row: hashlib.sha256(
                f"{args.seed}\0{row['sample_id']}".encode()
            ).hexdigest(),
        )[: args.max_train_samples]
        # The overfit smoke must contain enough paired rows to exercise every
        # SC-MRD loss.  Replace tail direct-only rows deterministically.
        paired_all = [row for row in read_jsonl(data / "train.jsonl") if row["paired"]]
        required_pairs = min(max(16, args.max_train_samples // 2), args.max_train_samples)
        present = {row["sample_id"] for row in train_rows if row["paired"]}
        additions = [row for row in paired_all if row["sample_id"] not in present]
        while sum(row["paired"] for row in train_rows) < required_pairs and additions:
            replace = next(
                (index for index in range(len(train_rows) - 1, -1, -1) if not train_rows[index]["paired"]),
                None,
            )
            if replace is None:
                break
            train_rows[replace] = additions.pop(0)
        if args.kd_mode == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE:
            for required_mode in ("cached", "ema"):
                if any(
                    bool(row.get("paired"))
                    and row.get("teacher_runtime_mode") == required_mode
                    for row in train_rows
                ):
                    continue
                replacement_row = next(
                    row
                    for row in paired_all
                    if row.get("teacher_runtime_mode") == required_mode
                )
                replace = next(
                    (
                        index
                        for index in range(len(train_rows) - 1, -1, -1)
                        if not bool(train_rows[index].get("paired"))
                    ),
                    len(train_rows) - 1,
                )
                train_rows[replace] = replacement_row
    if args.max_dev_samples > 0:
        dev_rows = dev_rows[: args.max_dev_samples]
        if not any(row["paired"] for row in dev_rows):
            replacement = next(
                (
                    row
                    for row in read_jsonl(data / "dev.jsonl")
                    if row["paired"]
                ),
                None,
            )
            if replacement is not None:
                dev_rows[-1] = replacement
    phase2_data: Path | None = None
    phase2_audit_path: Path | None = None
    phase2_train_rows: list[dict[str, Any]] = []
    phase2_dev_rows: list[dict[str, Any]] = []
    source_phase2_train_rows_actual = 0
    source_phase2_dev_rows_actual = 0
    if phase2_enabled:
        phase2_data = Path(args.phase2_data_dir).resolve()
        phase2_audit_path = phase2_data / "dataset_audit.json"
        phase2_audit = json.loads(
            phase2_audit_path.read_text(encoding="utf-8")
        )
        if not phase2_audit.get("passed"):
            raise RuntimeError("phase-2 SC-MRD dataset audit did not pass")
        phase2_train_rows = read_jsonl(phase2_data / "train.jsonl")
        phase2_dev_rows = read_jsonl(phase2_data / "dev.jsonl")
        source_phase2_train_rows_actual = len(phase2_train_rows)
        source_phase2_dev_rows_actual = len(phase2_dev_rows)
        if args.max_phase2_train_samples <= 0 and (
            len(phase2_train_rows) != args.expected_phase2_train_rows
            or len(phase2_dev_rows) != args.expected_phase2_dev_rows
        ):
            raise RuntimeError(
                "phase-2 SC-MRD row drift: "
                f"{len(phase2_train_rows)}/{len(phase2_dev_rows)} expected "
                f"{args.expected_phase2_train_rows}/"
                f"{args.expected_phase2_dev_rows}"
            )
        if args.max_phase2_train_samples > 0:
            phase2_train_rows = sorted(
                phase2_train_rows,
                key=lambda row: hashlib.sha256(
                    f"{args.phase2_seed}\0{row['sample_id']}".encode()
                ).hexdigest(),
            )[: args.max_phase2_train_samples]
            paired_all = [
                row
                for row in read_jsonl(phase2_data / "train.jsonl")
                if row["paired"]
            ]
            required_pairs = min(
                max(16, args.max_phase2_train_samples // 2),
                args.max_phase2_train_samples,
            )
            present = {
                row["sample_id"] for row in phase2_train_rows if row["paired"]
            }
            additions = [
                row for row in paired_all if row["sample_id"] not in present
            ]
            while (
                sum(row["paired"] for row in phase2_train_rows) < required_pairs
                and additions
            ):
                replace = next(
                    (
                        index
                        for index in range(len(phase2_train_rows) - 1, -1, -1)
                        if not phase2_train_rows[index]["paired"]
                    ),
                    None,
                )
                if replace is None:
                    break
                phase2_train_rows[replace] = additions.pop(0)
        if args.max_dev_samples > 0:
            phase2_dev_rows = phase2_dev_rows[: args.max_dev_samples]
            if not any(row["paired"] for row in phase2_dev_rows):
                replacement = next(
                    (
                        row
                        for row in read_jsonl(phase2_data / "dev.jsonl")
                        if row["paired"]
                    ),
                    None,
                )
                if replacement is not None:
                    phase2_dev_rows[-1] = replacement
    observed_kinds = {
        str(row.get("decision_kind") or "move")
        for row in [
            *train_rows,
            *dev_rows,
            *phase2_train_rows,
            *phase2_dev_rows,
        ]
    }
    if observed_kinds != {args.decision_kind}:
        raise RuntimeError(
            f"data/--decision-kind mismatch: expected {args.decision_kind}, "
            f"saw {sorted(observed_kinds)}"
        )

    hybrid_teacher_counts = {"cached": 0, "ema": 0}
    if args.kd_mode == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE:
        for row in [
            *train_rows,
            *dev_rows,
            *phase2_train_rows,
            *phase2_dev_rows,
        ]:
            if not bool(row.get("paired")):
                continue
            mode = str(row.get("teacher_runtime_mode", ""))
            if mode not in hybrid_teacher_counts:
                raise RuntimeError(
                    "hybrid KD paired rows require explicit "
                    "teacher_runtime_mode=cached|ema"
                )
            scores = row.get("teacher_legal_scores")
            if mode == "cached":
                if not isinstance(scores, list) or len(scores) != len(
                    row["legal_actions"]
                ):
                    raise RuntimeError(
                        "hybrid cached Teacher row lacks a complete legal surface"
                    )
            elif scores is not None:
                raise RuntimeError(
                    "hybrid EMA Teacher row must not carry cached Teacher logits"
                )
            if not {"privileged_prompt", "privileged_response"}.issubset(row):
                raise RuntimeError(
                    "hybrid KD paired row lacks its privileged Teacher view"
                )
            hybrid_teacher_counts[mode] += 1
        train_modes = {
            str(row.get("teacher_runtime_mode"))
            for row in train_rows
            if bool(row.get("paired"))
        }
        if train_modes != {"cached", "ema"}:
            raise RuntimeError(
                f"formal hybrid training requires both Teacher routes, got {train_modes}"
            )

    representation_cache: dict[str, dict[str, Any]] = {}
    if args.representation_cache:
        representation_cache = read_representation_cache(
            Path(args.representation_cache).resolve()
        )
        for row in [*train_rows, *dev_rows, *phase2_train_rows, *phase2_dev_rows]:
            if not bool(row.get("paired")):
                continue
            cached = representation_cache.get(str(row["sample_id"]))
            if cached is None:
                raise RuntimeError(
                    f"representation cache missing paired row {row['sample_id']}"
                )
            row["bridge_final_hidden"] = cached["bridge_final_hidden"]
            row["anchor_final_hidden"] = cached["direct_final_hidden"]
            if "no_bridge_final_hidden" in cached:
                row["no_bridge_final_hidden"] = cached["no_bridge_final_hidden"]
        if args.kd_mode in {
            "online_bridge_representation_residual",
            "online_bridge_increment_representation_residual",
            "online_action_projected_bridge_increment",
        } and not representation_cache:
            raise RuntimeError("empty representation cache")

    privileged_margin_calibrations: list[dict[str, Any]] = []
    if args.kd_mode == "privileged_margin_adaptive":
        minimum_margin = float(args.kd_min_teacher_gold_margin)
        privileged_margin_calibrations.append(
            {
                "phase": 1,
                **attach_privileged_margin_adaptive_weights(
                    train_rows,
                    minimum_margin=minimum_margin,
                    minimum_weight=args.kd_margin_weight_min,
                    maximum_weight=args.kd_margin_weight_max,
                ),
            }
        )
        if phase2_enabled:
            privileged_margin_calibrations.append(
                {
                    "phase": 2,
                    **attach_privileged_margin_adaptive_weights(
                        phase2_train_rows,
                        minimum_margin=minimum_margin,
                        minimum_weight=args.kd_margin_weight_min,
                        maximum_weight=args.kd_margin_weight_max,
                    ),
                }
            )
    if args.kd_mode == "privileged_correction_margin":
        minimum_margin = float(args.kd_min_teacher_gold_margin)
        privileged_margin_calibrations.append(
            {
                "phase": 1,
                **attach_privileged_correction_margin_targets(
                    train_rows,
                    minimum_teacher_margin=minimum_margin,
                    minimum_reasoning_gain=args.kd_min_reasoning_gain,
                    target_margin_cap=args.kd_pair_target_margin_cap,
                    minimum_weight=args.kd_margin_weight_min,
                    maximum_weight=args.kd_margin_weight_max,
                ),
            }
        )
        if phase2_enabled:
            privileged_margin_calibrations.append(
                {
                    "phase": 2,
                    **attach_privileged_correction_margin_targets(
                        phase2_train_rows,
                        minimum_teacher_margin=minimum_margin,
                        minimum_reasoning_gain=args.kd_min_reasoning_gain,
                        target_margin_cap=args.kd_pair_target_margin_cap,
                        minimum_weight=args.kd_margin_weight_min,
                        maximum_weight=args.kd_margin_weight_max,
                    ),
                }
            )
    if args.kd_mode == "privileged_bridge_correction_margin":
        minimum_margin = float(args.kd_min_teacher_gold_margin)
        privileged_margin_calibrations.append(
            {
                "phase": 1,
                **attach_privileged_bridge_correction_margin_targets(
                    train_rows,
                    minimum_teacher_margin=minimum_margin,
                    minimum_bridge_gain=args.kd_min_bridge_gain,
                    target_margin_cap=args.kd_pair_target_margin_cap,
                    minimum_weight=args.kd_margin_weight_min,
                    maximum_weight=args.kd_margin_weight_max,
                ),
            }
        )
        if phase2_enabled:
            privileged_margin_calibrations.append(
                {
                    "phase": 2,
                    **attach_privileged_bridge_correction_margin_targets(
                        phase2_train_rows,
                        minimum_teacher_margin=minimum_margin,
                        minimum_bridge_gain=args.kd_min_bridge_gain,
                        target_margin_cap=args.kd_pair_target_margin_cap,
                        minimum_weight=args.kd_margin_weight_min,
                        maximum_weight=args.kd_margin_weight_max,
                    ),
                }
            )
    if args.kd_mode == "online_action_debiased_residual":
        minimum_margin = float(args.kd_min_teacher_gold_margin)
        privileged_margin_calibrations.append(
            {
                "phase": 1,
                **attach_action_debiased_residual_targets(
                    train_rows,
                    dev_rows,
                    temperature=args.temperature,
                    minimum_teacher_margin=minimum_margin,
                    minimum_static_gold_logp_gain=(
                        args.kd_min_static_gold_logp_gain
                    ),
                    dynamic_gold_logp_gap_scale=(
                        args.kd_dynamic_gold_logp_gap_scale
                    ),
                    active_weight_floor=args.kd_active_weight_floor,
                ),
            }
        )
        if phase2_enabled:
            privileged_margin_calibrations.append(
                {
                    "phase": 2,
                    **attach_action_debiased_residual_targets(
                        phase2_train_rows,
                        phase2_dev_rows,
                        temperature=args.temperature,
                        minimum_teacher_margin=minimum_margin,
                        minimum_static_gold_logp_gain=(
                            args.kd_min_static_gold_logp_gain
                        ),
                        dynamic_gold_logp_gap_scale=(
                            args.kd_dynamic_gold_logp_gap_scale
                        ),
                        active_weight_floor=args.kd_active_weight_floor,
                    ),
                }
            )

    b0 = Path(args.b0_adapter).resolve()
    observed_b0_sha = (
        sha256(b0 / "adapter_model.safetensors") if is_main else ""
    )
    if world > 1:
        shared_b0_sha = [observed_b0_sha]
        dist.broadcast_object_list(shared_b0_sha, src=0)
        observed_b0_sha = str(shared_b0_sha[0])
    if observed_b0_sha != args.expected_b0_sha256:
        raise RuntimeError("fixed B0 adapter hash drift")
    b0_config = json.loads(
        (b0 / "adapter_config.json").read_text(encoding="utf-8")
    )
    if (
        int(b0_config.get("r", -1)) != args.expected_anchor_r
        or int(b0_config.get("lora_alpha", -1)) != args.expected_anchor_alpha
    ):
        raise RuntimeError(
            "frozen anchor rank/scaling drift: "
            f"got r{b0_config.get('r')}/alpha{b0_config.get('lora_alpha')} "
            f"expected r{args.expected_anchor_r}/alpha{args.expected_anchor_alpha}"
        )

    processor = load_processor(args.model_path)
    collator = GroupCollator(
        processor,
        max_length=args.max_length,
        objective=args.objective,
        system_prompt=(
            "You are a navigation spatial reasoning assistant."
            if args.phase1_historical_move_input_contract
            else ""
        ),
        text_before_image=(
            False if args.phase1_historical_move_input_contract else True
        ),
        online_ema_teacher=(
            args.kd_mode in ONLINE_EMA_KD_MODES
        ),
        ema_teacher_max_length=args.ema_teacher_max_length,
    )
    dataset = GroupDataset(train_rows)
    if world > 1 and args.per_device_batch_size <= 0:
        raise ValueError("distributed SC-MRD requires a positive local batch size")
    if args.phase1_historical_distributed_sampler:
        if any(bool(row["paired"]) for row in train_rows):
            raise ValueError(
                "historical phase-1 sampler requires an all-unpaired course"
            )
        sampler: Sampler[int] | None = DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False,
        )
    else:
        sampler = (
            TypeAlignedDistributedSampler(
                train_rows,
                world=world,
                rank=rank,
                seed=args.seed,
                multiple=args.grad_accum_steps * args.per_device_batch_size,
            )
            if world > 1
            else None
        )
    phase1_loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    phase1_updates_per_epoch = math.ceil(
        len(phase1_loader) / args.grad_accum_steps
    )
    phase1_updates = phase1_updates_per_epoch * args.epochs
    if args.max_steps > 0:
        phase1_updates = min(phase1_updates, args.max_steps)
    phase1_micro_limit = (
        min(len(phase1_loader), phase1_updates * args.grad_accum_steps)
        if phase2_enabled
        else len(phase1_loader)
    )
    phase_specs: list[dict[str, Any]] = [
        {
            "name": "historical_move_sft",
            "loader": phase1_loader,
            "sampler": sampler,
            "micro_limit": phase1_micro_limit,
            "optimizer_steps": phase1_updates,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "grad_accum_steps": args.grad_accum_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "global_batch_size": (
                world * args.per_device_batch_size * args.grad_accum_steps
            ),
            "kd_weight": args.kd_weight,
            "kd_warmup_ratio": args.kd_warmup_ratio,
            "weight_semantics": args.native_lm_weight_semantics,
            "max_length": args.max_length,
            "seed": args.seed,
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "paired_rows": sum(bool(row["paired"]) for row in train_rows),
        }
    ]
    phase2_sampler = None
    if phase2_enabled:
        phase2_sampler = (
            TypeAlignedDistributedSampler(
                phase2_train_rows,
                world=world,
                rank=rank,
                seed=args.phase2_seed,
                multiple=(
                    args.phase2_grad_accum_steps
                    * args.phase2_per_device_batch_size
                ),
            )
            if world > 1
            else None
        )
        phase2_collator = GroupCollator(
            processor,
            max_length=args.phase2_max_length,
            objective=args.objective,
            online_ema_teacher=(
                args.kd_mode in ONLINE_EMA_KD_MODES
            ),
            ema_teacher_max_length=args.ema_teacher_max_length,
        )
        phase2_loader = DataLoader(
            GroupDataset(phase2_train_rows),
            batch_size=args.phase2_per_device_batch_size,
            shuffle=phase2_sampler is None,
            sampler=phase2_sampler,
            collate_fn=phase2_collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        phase2_updates = math.ceil(
            len(phase2_loader) / args.phase2_grad_accum_steps
        )
        if args.phase2_max_steps > 0:
            phase2_updates = min(phase2_updates, args.phase2_max_steps)
        phase2_micro_limit = min(
            len(phase2_loader),
            phase2_updates * args.phase2_grad_accum_steps,
        )
        phase_specs.append(
            {
                "name": "selective_move_opd",
                "loader": phase2_loader,
                "sampler": phase2_sampler,
                "micro_limit": phase2_micro_limit,
                "optimizer_steps": phase2_updates,
                "learning_rate": args.phase2_learning_rate,
                "warmup_ratio": args.phase2_warmup_ratio,
                "grad_accum_steps": args.phase2_grad_accum_steps,
                "per_device_batch_size": args.phase2_per_device_batch_size,
                "global_batch_size": (
                    world
                    * args.phase2_per_device_batch_size
                    * args.phase2_grad_accum_steps
                ),
                "kd_weight": args.phase2_kd_weight,
                "kd_warmup_ratio": args.phase2_kd_warmup_ratio,
                "weight_semantics": args.phase2_native_lm_weight_semantics,
                "max_length": args.phase2_max_length,
                "seed": args.phase2_seed,
                "train_rows": len(phase2_train_rows),
                "dev_rows": len(phase2_dev_rows),
                "paired_rows": sum(
                    bool(row["paired"]) for row in phase2_train_rows
                ),
            }
        )
    loader = SequentialPhaseLoader(
        [
            (phase["loader"], int(phase["micro_limit"]))
            for phase in phase_specs
        ]
    )

    base, loader_name = load_model(args.model_path, device)
    base.config.use_cache = False
    training_mode = (
        "direct_b0_adapter_continuation"
        if args.continue_b0_adapter_directly
        else "frozen_b0_plus_fresh_delta"
    )
    if args.continue_b0_adapter_directly:
        model = load_trainable_adapter_continuation(base, b0)
    else:
        target_modules = [
            value.strip()
            for value in args.lora_target_modules.split(",")
            if value.strip()
        ]
        raw_exclude = args.lora_exclude_modules.strip()
        if raw_exclude and any(ch in raw_exclude for ch in "|*()[]\\"):
            exclude_modules = raw_exclude
        else:
            exclude_modules = [x.strip() for x in raw_exclude.split(",") if x.strip()]
        delta_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            exclude_modules=exclude_modules or None,
        )
        model = load_frozen_anchor_with_delta(
            base,
            b0,
            delta_config=delta_config,
            train_delta=True,
        )
    if args.kd_mode in ONLINE_EMA_KD_MODES:
        initialize_ema_teacher_adapter(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    counts = (
        named_adapter_parameter_counts(model, [DELTA_ADAPTER])
        if args.continue_b0_adapter_directly
        else adapter_parameter_counts(model)
    )
    if args.continue_b0_adapter_directly:
        if counts["default_trainable"] <= 0:
            raise RuntimeError(
                f"invalid direct-continuation parameter contract: {counts}"
            )
    elif counts["anchor_trainable"] != 0 or counts["default_trainable"] <= 0:
        raise RuntimeError(f"invalid frozen-B0/delta parameter contract: {counts}")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if args.forbid_vision_lora:
        offenders = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name and any(
                marker in name.lower()
                for marker in ("vision_tower", "vision_model", ".visual.", ".visual_")
            )
        ]
        if offenders:
            raise RuntimeError(
                "vision LoRA is forbidden for this retrain; found "
                f"{len(offenders)} trainable tensors ({', '.join(offenders[:8])})"
            )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (f".{DELTA_ADAPTER}." not in name or "lora_" not in name)
    ]
    if unexpected:
        raise RuntimeError(f"non-delta trainable parameters: {unexpected[:8]}")

    if world > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    model.train()
    total_steps = sum(int(phase["optimizer_steps"]) for phase in phase_specs)
    if total_steps <= 0:
        raise ValueError("training course has no optimizer steps")

    def phase_optimizer_and_scheduler(phase: dict[str, Any]):
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(phase["learning_rate"]),
            weight_decay=args.weight_decay,
        )
        phase_steps = int(phase["optimizer_steps"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(
                1, int(phase_steps * float(phase["warmup_ratio"]))
            ),
            num_training_steps=phase_steps,
        )
        return optimizer, scheduler

    active_phase_index = 0
    active_phase = phase_specs[active_phase_index]
    optimizer, scheduler = phase_optimizer_and_scheduler(active_phase)
    phase_boundaries = []
    cumulative_steps = 0
    for phase in phase_specs:
        cumulative_steps += int(phase["optimizer_steps"])
        phase_boundaries.append(cumulative_steps)
    serializable_phases = [
        {
            key: value
            for key, value in phase.items()
            if key not in {"loader", "sampler"}
        }
        for phase in phase_specs
    ]
    run_config = {
        "schema": "sc_mrd_native_lm_run_v1",
        "objective": args.objective,
        "decision_kind": args.decision_kind,
        "args": vars(args),
        "world_size": world,
        "global_batch_size": world * args.per_device_batch_size * args.grad_accum_steps,
        "expected_optimizer_steps": total_steps,
        "in_process_training_phases": serializable_phases,
        "phase_boundaries_optimizer_steps": phase_boundaries,
        "one_model_load": True,
        "optimizer_reset_at_phase_boundary": phase2_enabled,
        "rng_reset_at_phase_boundary": phase2_enabled,
        "phase1_historical_move_input_contract": (
            args.phase1_historical_move_input_contract
        ),
        "phase1_historical_distributed_sampler": (
            args.phase1_historical_distributed_sampler
        ),
        "model_loader": loader_name,
        "adapter_counts": counts,
        "adapter_training_mode": training_mode,
        "single_adapter_output": True,
        "fixed_b0_sha256": args.expected_b0_sha256,
        "data_audit_sha256": sha256(audit_path),
        "phase2_data_audit_sha256": (
            sha256(phase2_audit_path) if phase2_audit_path is not None else ""
        ),
        "train_rows_actual": len(train_rows) + len(phase2_train_rows),
        "dev_rows_actual": len(dev_rows) + len(phase2_dev_rows),
        "source_train_rows_actual": source_train_rows_actual,
        "source_dev_rows_actual": source_dev_rows_actual,
        "source_phase2_train_rows_actual": source_phase2_train_rows_actual,
        "source_phase2_dev_rows_actual": source_phase2_dev_rows_actual,
        "diagnostic_dev_rows_actual": len(
            phase2_dev_rows if phase2_enabled else dev_rows
        ),
        "native_visual_encoder_transformer_lm_head": True,
        "independent_decision_head": False,
        "runtime_view": "direct_only",
        "runtime_max_new_tokens": 24,
        "training_views": {
            "direct": True,
            "reason": args.objective in {"aux", "sc_mrd"},
            "reasoned_action": args.objective in {"sc_mrd", "opd"},
            "counterfactual": args.objective == "sc_mrd",
            "cached_privileged_teacher": (
                args.objective == "opd"
                and (
                    args.kd_mode not in ONLINE_EMA_KD_MODES
                    or args.kd_mode
                    == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE
                )
            ),
            "online_ema_privileged_teacher": (
                args.kd_mode in ONLINE_EMA_KD_MODES
            ),
            "teacher_gate_recomputed_each_batch": (
                args.kd_mode in ONLINE_EMA_KD_MODES
            ),
            "cached_anchor_direct_retention": args.anchor_retention_weight > 0,
        },
        "anchor_direct_retention": {
            "weight": args.anchor_retention_weight,
            "temperature": args.temperature,
            "fresh_delta_initial_kl_max": 0.05,
        },
        "reliable_teacher_kd": {
            "mode": args.kd_mode,
            "teacher_source": (
                "cached_historical_plus_ema_sol_privileged_evidence"
                if args.kd_mode == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE
                else (
                    "anchor_plus_ema_delta_privileged_original_evidence"
                    if args.kd_mode in ONLINE_EMA_KD_MODES
                    else "cached_privileged_logits"
                )
            ),
            "selection_timing": (
                "cached_historical_surface_plus_recomputed_sol_gate_each_batch"
                if args.kd_mode == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE
                else (
                    "recomputed_each_training_batch"
                    if args.kd_mode in ONLINE_EMA_KD_MODES
                    else "cached_course_surface"
                )
            ),
            "hybrid_teacher_rows_loaded": hybrid_teacher_counts,
            "require_teacher_correct": args.kd_require_teacher_correct,
            "min_teacher_gold_margin": args.kd_min_teacher_gold_margin,
            "negative_example_weight": args.kd_negative_weight,
            "positive_example_weight": args.kd_positive_weight,
            "privileged_margin_weight_min": args.kd_margin_weight_min,
            "privileged_margin_weight_max": args.kd_margin_weight_max,
            "minimum_reasoning_gain": args.kd_min_reasoning_gain,
            "minimum_bridge_gain": args.kd_min_bridge_gain,
            "minimum_online_margin_advantage": (
                args.kd_min_online_margin_advantage
            ),
            "online_margin_weight_scale": args.kd_online_margin_weight_scale,
            "online_confidence_increment_cap": (
                args.kd_online_confidence_increment_cap
            ),
            "minimum_static_gold_logp_gain": (
                args.kd_min_static_gold_logp_gain
            ),
            "dynamic_gold_logp_gap_scale": (
                args.kd_dynamic_gold_logp_gap_scale
            ),
            "active_weight_floor": args.kd_active_weight_floor,
            "online_repair_weight": args.kd_repair_tier_weight,
            "online_confidence_weight": args.kd_confidence_tier_weight,
            "pair_target_margin_cap": args.kd_pair_target_margin_cap,
            "pair_huber_beta": args.kd_pair_huber_beta,
            "loss_surface": (
                "gold_vs_frozen_direct_error_pairwise_hinge"
                if args.kd_mode == "privileged_correction_margin"
                else (
                    "bridge_specific_gold_vs_direct_error_pairwise_huber"
                    if args.kd_mode == "privileged_bridge_correction_margin"
                    else (
                        "online_student_competitor_pairwise_huber"
                        if args.kd_mode in {
                            "online_pairwise_adaptive",
                            "online_ema_pairwise_adaptive",
                            HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
                        }
                        else (
                            "two_way_centered_full_legal_action_logit_residual_mse"
                            if args.kd_mode == "online_action_debiased_residual"
                            else (
                                "action_projected_bridge_increment_representation_residual_mse"
                                if args.kd_mode
                                == "online_action_projected_bridge_increment"
                                else (
                                    "bridge_increment_final_decision_representation_residual_mse"
                                    if args.kd_mode
                                    == "online_bridge_increment_representation_residual"
                                    else (
                                        "bridge_specific_final_decision_representation_residual_mse"
                                        if args.kd_mode
                                        == "online_bridge_representation_residual"
                                        else "full_legal_action_distribution_kl"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        },
        "privileged_margin_adaptive_calibrations": (
            privileged_margin_calibrations
        ),
        "direct_ce_class_weights": {
            "negative_example_weight": args.direct_negative_weight,
            "positive_example_weight": args.direct_positive_weight,
        },
    }
    if is_main:
        (output / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    log(
        {
            "event": "start",
            "objective": args.objective,
            "steps": total_steps,
            "train_rows": len(train_rows) + len(phase2_train_rows),
            "paired_rows": sum(
                row["paired"] for row in [*train_rows, *phase2_train_rows]
            ),
            "counterfactual_rows": sum(
                bool(row.get("counterfactual_active")) for row in train_rows
            ),
            "trainable_adapter_parameters": sum(p.numel() for p in trainable),
            "adapter_training_mode": training_mode,
            "phases": serializable_phases,
        }
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    phase_step = 0
    stop = False
    rolling: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        if phase2_sampler is not None:
            phase2_sampler.set_epoch(epoch)
        for micro_step, grouped in enumerate(loader, start=1):
            if global_step >= total_steps:
                stop = True
                break
            batch_phase_index = int(grouped.pop("_training_phase"))
            phase_micro_step = int(grouped.pop("_phase_micro_step"))
            if batch_phase_index != active_phase_index:
                if batch_phase_index != active_phase_index + 1:
                    raise RuntimeError("training phases were not sequential")
                if global_step != phase_boundaries[active_phase_index]:
                    raise RuntimeError(
                        "phase boundary reached with incomplete optimizer steps"
                    )
                active_phase_index = batch_phase_index
                active_phase = phase_specs[active_phase_index]
                set_seed(int(active_phase["seed"]) + rank)
                optimizer, scheduler = phase_optimizer_and_scheduler(active_phase)
                optimizer.zero_grad(set_to_none=True)
                phase_step = 0
                rolling.clear()
                log(
                    {
                        "event": "phase_transition",
                        "phase": active_phase_index + 1,
                        "phase_name": active_phase["name"],
                        "global_step": global_step,
                        "learning_rate": active_phase["learning_rate"],
                        "optimizer_reset": True,
                        "rng_reset_seed": int(active_phase["seed"]),
                    }
                )
            active_grad_accum = int(active_phase["grad_accum_steps"])
            paired_indices = grouped["paired_indices"].to(device)
            sync_context = (
                model.no_sync() if isinstance(model, DDP) else nullcontext()
            )
            values = {
                "direct": 0.0,
                "reason": 0.0,
                "reasoned_action": 0.0,
                "kd": 0.0,
                "anchor_retention": 0.0,
                "counterfactual": 0.0,
                "paired": float(paired_indices.numel()),
                "teacher_correct": 0.0,
                "teacher_count": 0.0,
                "teacher_gold_margin_sum": 0.0,
                "direct_correct": 0.0,
                "direct_count": 0.0,
                "kd_eligible": 0.0,
                "kd_candidates": 0.0,
                "kd_active": 0.0,
                "kd_eligible_online": 0.0,
                "kd_candidates_online": 0.0,
                "kd_active_online": 0.0,
                "kd_eligible_cached": 0.0,
                "kd_candidates_cached": 0.0,
                "kd_active_cached": 0.0,
                "kd_positive": 0.0,
                "kd_negative": 0.0,
                "kd_repair": 0.0,
                "kd_confidence": 0.0,
                "kd_example_weight_sum": 0.0,
                "kd_margin_gain_sum": 0.0,
                "kd_gold_logp_gain_sum": 0.0,
            }
            with sync_context:
                if args.objective in {"aux", "sc_mrd"} and paired_indices.numel():
                    reason_raw = grouped["reason"]
                    reason = native_batch(reason_raw, device)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                        reason_outputs = model(**reason)
                        reason_loss = mean_row_lm_loss(
                            reason_outputs.logits, reason["labels"]
                        )
                    (
                        args.reason_weight
                        * reason_loss
                        / active_grad_accum
                    ).backward()
                    values["reason"] = float(reason_loss.detach())
                    del reason_outputs

                teacher_logits_detached = None
                teacher_reliable_mask = None
                if args.objective == "opd" and paired_indices.numel():
                    reasoned_surface = surface_to_device(
                        grouped["reasoned_surface"], device
                    )
                    teacher_mask = grouped["cached_teacher_mask"].to(device)
                    if not torch.equal(
                        teacher_mask, reasoned_surface.token_mask
                    ):
                        raise ValueError(
                            "cached teacher and legal-action masks disagree"
                        )
                    if args.kd_mode in ONLINE_EMA_KD_MODES:
                        online_flags = grouped.get(
                            "online_teacher_flags",
                            torch.ones(
                                reasoned_surface.gold_indices.shape[0],
                                dtype=torch.bool,
                            ),
                        ).to(device)
                        cached_available = grouped.get(
                            "cached_teacher_available",
                            torch.zeros(
                                reasoned_surface.gold_indices.shape[0],
                                dtype=torch.bool,
                            ),
                        ).to(device)
                        if bool((~online_flags & ~cached_available).any()):
                            raise RuntimeError(
                                "mixed Teacher batch has no cached surface for a "
                                "non-EMA row"
                            )
                        cached_logits = grouped["cached_teacher_logits"].to(device)
                        if bool(online_flags.any()):
                            privileged_raw = grouped["reasoned"]
                            privileged = native_batch(privileged_raw, device)
                            ema_logits = ema_teacher_action_logits(
                                model,
                                privileged,
                                reasoned_surface,
                            )
                            teacher_logits = torch.where(
                                online_flags[:, None], ema_logits, cached_logits
                            )
                        else:
                            teacher_logits = cached_logits
                    else:
                        teacher_logits = grouped["cached_teacher_logits"].to(device)
                    teacher_logits_detached = teacher_logits.detach()
                    (
                        teacher_reliable_mask,
                        teacher_correct,
                        teacher_margin,
                    ) = reliable_teacher_mask(
                        teacher_logits,
                        reasoned_surface.gold_indices,
                        require_correct=args.kd_require_teacher_correct,
                        min_gold_margin=args.kd_min_teacher_gold_margin,
                    )
                    reasoned_action = F.cross_entropy(
                        teacher_logits, reasoned_surface.gold_indices
                    )
                    values["reasoned_action"] = float(reasoned_action)
                    values["teacher_correct"] = float(teacher_correct.sum())
                    values["teacher_count"] = float(teacher_correct.numel())
                    values["teacher_gold_margin_sum"] = float(
                        teacher_margin.sum()
                    )
                    values["kd_eligible"] = float(teacher_reliable_mask.sum())
                    values["kd_candidates"] = float(
                        teacher_reliable_mask.numel()
                    )
                    if args.kd_mode == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE:
                        route_online = grouped["online_teacher_flags"].to(device)
                        route_cached = ~route_online
                        values["kd_candidates_online"] = float(
                            route_online.sum()
                        )
                        values["kd_eligible_online"] = float(
                            (teacher_reliable_mask & route_online).sum()
                        )
                        values["kd_candidates_cached"] = float(
                            route_cached.sum()
                        )
                        values["kd_eligible_cached"] = float(
                            (teacher_reliable_mask & route_cached).sum()
                        )
                if args.objective == "sc_mrd" and paired_indices.numel():
                    negative_logits = None
                    negative_positions = grouped["negative_pair_positions"].to(
                        device
                    )
                    if negative_positions.numel():
                        negative_raw = grouped["negative"]
                        negative = native_batch(negative_raw, device)
                        negative_surface = surface_to_device(
                            grouped["negative_surface"], device
                        )
                        with torch.no_grad(), torch.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=True,
                        ):
                            negative_outputs = model(**negative)
                            negative_logits = action_logits(
                                negative_outputs.logits, negative_surface
                            )
                        del negative_outputs

                    reasoned_raw = grouped["reasoned"]
                    reasoned = native_batch(reasoned_raw, device)
                    reasoned_surface = surface_to_device(
                        grouped["reasoned_surface"], device
                    )
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                        reasoned_outputs = model(**reasoned)
                        teacher_logits = action_logits(
                            reasoned_outputs.logits, reasoned_surface
                        )
                        reasoned_action = F.cross_entropy(
                            teacher_logits, reasoned_surface.gold_indices
                        )
                        counterfactual = teacher_logits.sum() * 0.0
                        if negative_logits is not None:
                            counterfactual = counterfactual_margin_loss(
                                teacher_logits[negative_positions],
                                negative_logits,
                                reasoned_surface.gold_indices[
                                    negative_positions
                                ],
                                margin=args.counterfactual_margin,
                            )
                        teacher_objective = (
                            args.reasoned_action_weight * reasoned_action
                            + args.counterfactual_weight * counterfactual
                        )
                    (
                        teacher_objective / active_grad_accum
                    ).backward()
                    teacher_logits_detached = teacher_logits.detach()
                    (
                        teacher_reliable_mask,
                        teacher_correct,
                        teacher_margin,
                    ) = reliable_teacher_mask(
                        teacher_logits,
                        reasoned_surface.gold_indices,
                        require_correct=args.kd_require_teacher_correct,
                        min_gold_margin=args.kd_min_teacher_gold_margin,
                    )
                    values["reasoned_action"] = float(reasoned_action.detach())
                    values["counterfactual"] = float(counterfactual.detach())
                    values["teacher_correct"] = float(teacher_correct.sum())
                    values["teacher_count"] = float(teacher_correct.numel())
                    values["teacher_gold_margin_sum"] = float(
                        teacher_margin.sum()
                    )
                    values["kd_eligible"] = float(
                        teacher_reliable_mask.sum()
                    )
                    values["kd_candidates"] = float(
                        teacher_reliable_mask.numel()
                    )
                    del reasoned_outputs

                direct_raw = grouped["direct"]
                direct_action_mask = pop_action_token_mask(direct_raw, device)
                direct = native_batch(direct_raw, device)
                direct_example_weights = grouped[
                    "direct_example_weights"
                ].to(device)
                direct_decision_weights = grouped[
                    "direct_decision_weights"
                ].to(device)
                direct_action_field_weights = grouped[
                    "direct_action_field_weights"
                ].to(device)
                if args.objective == "opd" and args.decision_kind == "stop":
                    direct_class_weights = torch.tensor(
                        [
                            args.direct_positive_weight
                            if gold_action == "1"
                            else args.direct_negative_weight
                            for gold_action in grouped["gold_actions"]
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                    direct_example_weights = (
                        direct_example_weights * direct_class_weights
                    )
                representation_capture: dict[str, torch.Tensor] = {}
                representation_hook = None
                representation_surface = None
                if (
                    args.kd_mode in {
                        "online_bridge_representation_residual",
                        "online_bridge_increment_representation_residual",
                        "online_action_projected_bridge_increment",
                    }
                    and paired_indices.numel()
                ):
                    representation_surface = surface_to_device(
                        grouped["direct_paired_surface"], device
                    )
                    hook_model = model.module if isinstance(model, DDP) else model
                    if hasattr(hook_model, "get_base_model"):
                        hook_model = hook_model.get_base_model()
                    final_norm = hook_model.model.language_model.norm

                    def capture_representation(
                        _module: torch.nn.Module,
                        _inputs: tuple[torch.Tensor, ...],
                        output: Any,
                    ) -> None:
                        hidden = output[0] if isinstance(output, tuple) else output
                        representation_capture["hidden"] = hidden[
                            paired_indices, representation_surface.positions
                        ]

                    representation_hook = final_norm.register_forward_hook(
                        capture_representation
                    )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    direct_outputs = model(**direct)
                    direct_loss = mean_row_lm_loss(
                        direct_outputs.logits,
                        direct["labels"],
                        action_token_mask=direct_action_mask,
                        action_token_weight=args.direct_action_token_weight,
                        example_weights=direct_example_weights,
                        decision_weights=direct_decision_weights,
                        action_field_weights=direct_action_field_weights,
                        weight_semantics=str(active_phase["weight_semantics"]),
                    )
                if representation_hook is not None:
                    representation_hook.remove()
                    representation_hook = None
                    if "hidden" not in representation_capture:
                        raise RuntimeError(
                            "representation KD failed to capture final decision state"
                        )
                    kd = direct_loss * 0.0
                    anchor_retention = direct_loss * 0.0
                    kd_scale = 0.0
                    if (
                        args.objective in {"sc_mrd", "opd"}
                        and teacher_logits_detached is not None
                    ):
                        direct_surface = surface_to_device(
                            grouped["direct_paired_surface"], device
                        )
                        if args.objective == "opd":
                            student_logits = opd_student_action_logits(
                                model,
                                direct_outputs.logits[paired_indices],
                                direct_surface,
                                direct,
                            )
                        else:
                            student_logits = legal_action_logits(
                                direct_outputs.logits[paired_indices],
                                direct_surface.positions,
                                direct_surface.token_ids,
                                direct_surface.token_mask,
                            )
                        if args.anchor_retention_weight > 0:
                            if "cached_anchor_direct_logits" not in grouped:
                                raise RuntimeError(
                                    "positive anchor retention weight lacks cached logits"
                                )
                            anchor_mask = grouped[
                                "cached_anchor_direct_mask"
                            ].to(device)
                            if not torch.equal(
                                anchor_mask, direct_surface.token_mask
                            ):
                                raise ValueError(
                                    "cached anchor and legal-action masks disagree"
                                )
                            anchor_logits = grouped[
                                "cached_anchor_direct_logits"
                            ].to(device)
                            anchor_retention = masked_distillation_loss(
                                anchor_logits,
                                student_logits,
                                temperature=args.temperature,
                            )
                            if (
                                global_step == 0
                                and float(anchor_retention.detach()) > 0.05
                            ):
                                raise RuntimeError(
                                    "fresh delta does not reproduce cached anchor "
                                    "Direct logits at initialization"
                                )
                        direct_correct = student_logits.argmax(-1).eq(
                            direct_surface.gold_indices
                        )
                        values["direct_correct"] = float(
                            direct_correct.sum()
                        )
                        values["direct_count"] = float(
                            direct_correct.numel()
                        )
                        if args.kd_mode in {
                            "privileged_correction_margin",
                            "privileged_bridge_correction_margin",
                        }:
                            required_fields = {
                                "privileged_correction_kd_weights",
                                "privileged_correction_kd_eligible",
                                "privileged_correction_competitor_indices",
                                "privileged_correction_target_margins",
                                "privileged_correction_margin_gains",
                            }
                            missing_fields = sorted(required_fields - grouped.keys())
                            if missing_fields:
                                raise RuntimeError(
                                    "correction-margin KD batch lacks calibrated "
                                    f"fields: {missing_fields}"
                                )
                            calibrated_eligible = grouped[
                                "privileged_correction_kd_eligible"
                            ].to(device)
                            if teacher_reliable_mask is None or bool(
                                (calibrated_eligible & ~teacher_reliable_mask).any()
                            ):
                                raise RuntimeError(
                                    "offline correction calibration violates the "
                                    "runtime reliable-Teacher gate"
                                )
                            correction_weights = grouped[
                                "privileged_correction_kd_weights"
                            ].to(device)
                            competitor_indices = grouped[
                                "privileged_correction_competitor_indices"
                            ].to(device)
                            target_margins = grouped[
                                "privileged_correction_target_margins"
                            ].to(device)
                            correction_gains = grouped[
                                "privileged_correction_margin_gains"
                            ].to(device)
                            if args.kd_mode == "privileged_bridge_correction_margin":
                                kd = selective_pairwise_huber_margin_loss(
                                    student_logits,
                                    direct_surface.gold_indices,
                                    competitor_indices,
                                    target_margins,
                                    example_weights=correction_weights,
                                    beta=args.kd_pair_huber_beta,
                                )
                            else:
                                kd = selective_pairwise_margin_loss(
                                    student_logits,
                                    direct_surface.gold_indices,
                                    competitor_indices,
                                    target_margins,
                                    example_weights=correction_weights,
                                )
                            active_corrections = correction_weights.gt(0)
                            values["kd_active"] = float(
                                active_corrections.sum()
                            )
                            values["kd_repair"] = float(
                                active_corrections.sum()
                            )
                            values["kd_example_weight_sum"] = float(
                                correction_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                correction_gains[active_corrections].sum()
                            )
                        elif (
                            args.kd_mode in {
                                "online_safe_full_distribution",
                                "online_action_debiased_residual",
                                "online_bridge_representation_residual",
                                "online_bridge_increment_representation_residual",
                                "online_action_projected_bridge_increment",
                            }
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            if "cached_anchor_direct_logits" not in grouped:
                                raise RuntimeError(
                                    "safe full-distribution KD lacks frozen "
                                    "Direct-anchor logits"
                                )
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_anchor = grouped[
                                "cached_anchor_direct_logits"
                            ].to(device)[teacher_reliable_mask]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            (
                                online_weights,
                                static_valid,
                                active_online,
                                static_gold_logp_gain,
                                dynamic_gold_logp_gap,
                            ) = online_safe_full_distribution_weights(
                                selected_teacher,
                                selected_anchor,
                                selected_student,
                                selected_gold,
                                temperature=args.temperature,
                                minimum_static_gold_logp_gain=(
                                    args.kd_min_static_gold_logp_gain
                                ),
                                dynamic_gold_logp_gap_scale=(
                                    args.kd_dynamic_gold_logp_gap_scale
                                ),
                                active_weight_floor=args.kd_active_weight_floor,
                            )
                            if args.kd_mode == "online_action_projected_bridge_increment":
                                if "cached_no_bridge_logits" not in grouped:
                                    raise RuntimeError(
                                        "action-projected Bridge KD lacks no-Bridge logits"
                                    )
                                selected_no_bridge = grouped[
                                    "cached_no_bridge_logits"
                                ].to(device)[teacher_reliable_mask]
                                no_bridge_gold_logp = torch.log_softmax(
                                    selected_no_bridge.float() / args.temperature,
                                    dim=-1,
                                ).gather(1, selected_gold[:, None]).squeeze(1)
                                teacher_gold_logp = torch.log_softmax(
                                    selected_teacher.float() / args.temperature,
                                    dim=-1,
                                ).gather(1, selected_gold[:, None]).squeeze(1)
                                bridge_increment_valid = (
                                    teacher_gold_logp - no_bridge_gold_logp
                                ).gt(args.kd_min_static_gold_logp_gain)
                                single_token_supported_all = torch.tensor(
                                    [
                                        all(len(sequence) == 1 for sequence in sequences)
                                        and len({sequence[0] for sequence in sequences})
                                        == len(sequences)
                                        for sequences in direct_surface.legal_scoring_sequences
                                    ],
                                    dtype=torch.bool,
                                    device=device,
                                )
                                single_token_supported = single_token_supported_all[
                                    teacher_reliable_mask
                                ]
                                online_weights = online_weights * bridge_increment_valid.float()
                                online_weights = online_weights * single_token_supported.float()
                                active_online = online_weights.gt(0)
                            if args.kd_mode == "online_action_debiased_residual":
                                if (
                                    "cached_action_debiased_teacher_residual"
                                    not in grouped
                                ):
                                    raise RuntimeError(
                                        "action-debiased KD lacks calibrated residual targets"
                                    )
                                selected_target_residual = grouped[
                                    "cached_action_debiased_teacher_residual"
                                ].to(device)[teacher_reliable_mask]
                                selected_legal_mask = direct_surface.token_mask[
                                    teacher_reliable_mask
                                ]
                                kd = masked_residual_logit_distillation_loss(
                                    selected_target_residual,
                                    selected_student,
                                    selected_anchor,
                                    selected_legal_mask,
                                    example_weights=online_weights,
                                )
                            elif args.kd_mode in {
                                "online_bridge_representation_residual",
                                "online_bridge_increment_representation_residual",
                                "online_action_projected_bridge_increment",
                            }:
                                if (
                                    "cached_bridge_final_hidden" not in grouped
                                    or "cached_anchor_final_hidden" not in grouped
                                ):
                                    raise RuntimeError(
                                        "representation KD lacks cached Direct/Bridge hidden states"
                                    )
                                if "hidden" not in representation_capture:
                                    raise RuntimeError(
                                        "representation KD has no Student decision state"
                                    )
                                student_hidden = representation_capture[
                                    "hidden"
                                ][teacher_reliable_mask]
                                target_hidden = grouped[
                                    "cached_bridge_final_hidden"
                                ].to(device)[teacher_reliable_mask]
                                anchor_hidden = grouped[
                                    "cached_anchor_final_hidden"
                                ].to(device)[teacher_reliable_mask]
                                if args.kd_mode == "online_bridge_representation_residual":
                                    kd = masked_representation_residual_distillation_loss(
                                        target_hidden,
                                        student_hidden,
                                        anchor_hidden,
                                        example_weights=online_weights,
                                    )
                                elif args.kd_mode == "online_bridge_increment_representation_residual":
                                    if "cached_no_bridge_final_hidden" not in grouped:
                                        raise RuntimeError(
                                            "Bridge-increment KD lacks cached no-Bridge hidden states"
                                        )
                                    no_bridge_hidden = grouped[
                                        "cached_no_bridge_final_hidden"
                                    ].to(device)[teacher_reliable_mask]
                                    kd = masked_bridge_increment_representation_distillation_loss(
                                        target_hidden,
                                        no_bridge_hidden,
                                        student_hidden,
                                        anchor_hidden,
                                        example_weights=online_weights,
                                    )
                                else:
                                    if "cached_no_bridge_final_hidden" not in grouped:
                                        raise RuntimeError(
                                            "action-projected Bridge KD lacks cached no-Bridge hidden states"
                                        )
                                    no_bridge_hidden = grouped[
                                        "cached_no_bridge_final_hidden"
                                    ].to(device)[teacher_reliable_mask]
                                    conditional = model.module if isinstance(model, DDP) else model
                                    if hasattr(conditional, "get_base_model"):
                                        conditional = conditional.get_base_model()
                                    head_weight = conditional.lm_head.weight
                                    legal_ids = direct_surface.token_ids[
                                        teacher_reliable_mask
                                    ]
                                    legal_mask = direct_surface.token_mask[
                                        teacher_reliable_mask
                                    ]
                                    legal_heads = head_weight[legal_ids]
                                    kd = masked_action_projected_bridge_increment_loss(
                                        target_hidden,
                                        no_bridge_hidden,
                                        student_hidden,
                                        anchor_hidden,
                                        legal_heads,
                                        legal_mask,
                                        example_weights=online_weights,
                                    )
                            else:
                                kd = masked_distillation_loss(
                                    selected_teacher,
                                    selected_student,
                                    temperature=args.temperature,
                                    example_weights=online_weights,
                                )
                            values["kd_active"] = float(active_online.sum())
                            values["kd_confidence"] = float(static_valid.sum())
                            values["kd_example_weight_sum"] = float(
                                online_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                static_gold_logp_gain[static_valid].sum()
                            )
                            values["kd_gold_logp_gain_sum"] = float(
                                dynamic_gold_logp_gap[active_online].sum()
                            )
                        elif (
                            args.kd_mode
                            == "online_ema_sol_match_full_distribution"
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            selected_legal_mask = direct_surface.token_mask[
                                teacher_reliable_mask
                            ]
                            (
                                kd_example_weights,
                                sol_match,
                                sol_margin,
                            ) = ema_sol_match_weights(
                                selected_teacher,
                                selected_gold,
                                selected_legal_mask,
                                temperature=args.ema_sol_match_temperature,
                                margin_center=args.ema_sol_match_margin_center,
                                weight_floor=args.ema_sol_match_weight_floor,
                            )
                            kd = masked_distillation_loss(
                                selected_teacher,
                                selected_student,
                                temperature=args.temperature,
                                example_weights=kd_example_weights,
                            )
                            active_online = kd_example_weights.gt(0)
                            values["kd_active"] = float(active_online.sum())
                            values["kd_confidence"] = float(
                                sol_margin.gt(0).sum()
                            )
                            values["kd_example_weight_sum"] = float(
                                kd_example_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                sol_margin[active_online].sum()
                            )
                            values["kd_gold_logp_gain_sum"] = float(
                                sol_match[active_online].sum()
                            )
                        elif (
                            args.kd_mode in {
                                "online_pairwise_adaptive",
                                "online_ema_pairwise_adaptive",
                            }
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            (
                                online_weights,
                                repair_mask,
                                confidence_mask,
                                competitor_indices,
                                target_margins,
                                margin_advantages,
                            ) = online_pairwise_adaptive_targets(
                                selected_teacher,
                                selected_student,
                                selected_gold,
                                minimum_margin_advantage=(
                                    args.kd_min_online_margin_advantage
                                ),
                                repair_target_margin_cap=(
                                    args.kd_pair_target_margin_cap
                                ),
                                confidence_increment_cap=(
                                    args.kd_online_confidence_increment_cap
                                ),
                                margin_weight_scale=(
                                    args.kd_online_margin_weight_scale
                                ),
                                repair_weight=args.kd_repair_tier_weight,
                                confidence_weight=(
                                    args.kd_confidence_tier_weight
                                ),
                            )
                            kd = selective_pairwise_huber_margin_loss(
                                selected_student,
                                selected_gold,
                                competitor_indices,
                                target_margins,
                                example_weights=online_weights,
                                beta=args.kd_pair_huber_beta,
                            )
                            active_online = online_weights.gt(0)
                            values["kd_active"] = float(active_online.sum())
                            values["kd_repair"] = float(repair_mask.sum())
                            values["kd_confidence"] = float(confidence_mask.sum())
                            values["kd_example_weight_sum"] = float(
                                online_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                margin_advantages[active_online].sum()
                            )
                        elif (
                            teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            kd_example_weights = None
                            if args.kd_mode == "privileged_margin_adaptive":
                                if (
                                    "privileged_margin_kd_weights" not in grouped
                                    or "privileged_margin_kd_eligible" not in grouped
                                ):
                                    raise RuntimeError(
                                        "adaptive KD batch lacks calibrated weights"
                                    )
                                calibrated_eligible = grouped[
                                    "privileged_margin_kd_eligible"
                                ].to(device)
                                if not torch.equal(
                                    calibrated_eligible,
                                    teacher_reliable_mask,
                                ):
                                    raise RuntimeError(
                                        "offline privileged-margin calibration "
                                        "disagrees with the runtime teacher gate"
                                    )
                                kd_example_weights = grouped[
                                    "privileged_margin_kd_weights"
                                ].to(device)[teacher_reliable_mask]
                                values["kd_active"] = float(
                                    kd_example_weights.gt(0).sum()
                                )
                            elif args.kd_mode == "online_advantage_tiered":
                                (
                                    kd_example_weights,
                                    repair_mask,
                                    confidence_mask,
                                    margin_gain,
                                    gold_logp_gain,
                                ) = online_advantage_kd_weights(
                                    selected_teacher,
                                    selected_student,
                                    selected_gold,
                                    minimum_margin_advantage=(
                                        args.kd_min_online_margin_advantage
                                    ),
                                    repair_weight=args.kd_repair_tier_weight,
                                    confidence_weight=(
                                        args.kd_confidence_tier_weight
                                    ),
                                )
                                values["kd_active"] = float(
                                    kd_example_weights.gt(0).sum()
                                )
                                values["kd_repair"] = float(repair_mask.sum())
                                values["kd_confidence"] = float(
                                    confidence_mask.sum()
                                )
                                values["kd_margin_gain_sum"] = float(
                                    margin_gain.sum()
                                )
                                values["kd_gold_logp_gain_sum"] = float(
                                    gold_logp_gain.sum()
                                )
                            else:
                                values["kd_active"] = float(
                                    teacher_reliable_mask.sum()
                                )
                            if args.decision_kind == "stop":
                                values["kd_positive"] = float(
                                    selected_gold.eq(1).sum()
                                )
                                values["kd_negative"] = float(
                                    selected_gold.eq(0).sum()
                                )
                                kd_class_example_weights = torch.where(
                                    selected_gold.eq(1),
                                    torch.full_like(
                                        selected_gold,
                                        args.kd_positive_weight,
                                        dtype=student_logits.dtype,
                                    ),
                                    torch.full_like(
                                        selected_gold,
                                        args.kd_negative_weight,
                                        dtype=student_logits.dtype,
                                    ),
                                )
                                kd_example_weights = (
                                    kd_class_example_weights
                                    if kd_example_weights is None
                                    else kd_example_weights
                                    * kd_class_example_weights
                                )
                            values["kd_example_weight_sum"] = float(
                                teacher_reliable_mask.sum()
                                if kd_example_weights is None
                                else kd_example_weights.sum()
                            )
                            kd = masked_distillation_loss(
                                selected_teacher,
                                selected_student,
                                temperature=args.temperature,
                                example_weights=kd_example_weights,
                            )
                        else:
                            kd = student_logits.sum() * 0.0
                        warmup_steps = max(
                            1,
                            int(
                                int(active_phase["optimizer_steps"])
                                * float(active_phase["kd_warmup_ratio"])
                            ),
                        )
                        kd_scale = min(
                            1.0, (phase_step + 1) / float(warmup_steps)
                        )
                        student_objective = (
                            args.direct_weight * direct_loss
                            + float(active_phase["kd_weight"]) * kd_scale * kd
                            + args.anchor_retention_weight * anchor_retention
                        )
                        if not bool(torch.isfinite(kd).all()):
                            raise FloatingPointError(
                                "non-finite KD loss; refusing to update adapter"
                            )
                if representation_hook is None:
                    # The historical implementation placed the OPD objective
                    # inside the representation-hook branch.  Native full-
                    # distribution EMA KD does not need a hidden-state hook,
                    # so keep its action-surface objective explicit here.
                    kd = direct_loss * 0.0
                    anchor_retention = direct_loss * 0.0
                    kd_scale = 0.0
                    if (
                        args.objective == "opd"
                        and teacher_logits_detached is not None
                        and paired_indices.numel()
                    ):
                        direct_surface = surface_to_device(
                            grouped["direct_paired_surface"], device
                        )
                        student_logits = opd_student_action_logits(
                            model,
                            direct_outputs.logits[paired_indices],
                            direct_surface,
                            direct,
                        )
                        # Keep the per-step diagnostic consistent with the
                        # representation-hook path.  The native EMA KD path
                        # intentionally has no hook, but its Student logits
                        # are still the direct-policy logits we want to audit.
                        direct_correct = student_logits.argmax(-1).eq(
                            direct_surface.gold_indices
                        )
                        values["direct_correct"] = float(
                            direct_correct.sum()
                        )
                        values["direct_count"] = float(
                            direct_correct.numel()
                        )
                        if args.kd_mode in {
                            "privileged_correction_margin",
                            "privileged_bridge_correction_margin",
                        }:
                            required_fields = {
                                "privileged_correction_kd_weights",
                                "privileged_correction_kd_eligible",
                                "privileged_correction_competitor_indices",
                                "privileged_correction_target_margins",
                                "privileged_correction_margin_gains",
                            }
                            missing_fields = sorted(required_fields - grouped.keys())
                            if missing_fields:
                                raise RuntimeError(
                                    "correction-margin KD batch lacks calibrated "
                                    f"fields: {missing_fields}"
                                )
                            calibrated_eligible = grouped[
                                "privileged_correction_kd_eligible"
                            ].to(device)
                            if teacher_reliable_mask is None or bool(
                                (calibrated_eligible & ~teacher_reliable_mask).any()
                            ):
                                raise RuntimeError(
                                    "offline correction calibration violates the "
                                    "runtime reliable-Teacher gate"
                                )
                            correction_weights = grouped[
                                "privileged_correction_kd_weights"
                            ].to(device)
                            competitor_indices = grouped[
                                "privileged_correction_competitor_indices"
                            ].to(device)
                            target_margins = grouped[
                                "privileged_correction_target_margins"
                            ].to(device)
                            correction_gains = grouped[
                                "privileged_correction_margin_gains"
                            ].to(device)
                            if args.kd_mode == "privileged_bridge_correction_margin":
                                kd = selective_pairwise_huber_margin_loss(
                                    student_logits,
                                    direct_surface.gold_indices,
                                    competitor_indices,
                                    target_margins,
                                    example_weights=correction_weights,
                                    beta=args.kd_pair_huber_beta,
                                )
                            else:
                                kd = selective_pairwise_margin_loss(
                                    student_logits,
                                    direct_surface.gold_indices,
                                    competitor_indices,
                                    target_margins,
                                    example_weights=correction_weights,
                                )
                            active_corrections = correction_weights.gt(0)
                            values["kd_active"] = float(active_corrections.sum())
                            values["kd_repair"] = float(active_corrections.sum())
                            values["kd_example_weight_sum"] = float(
                                correction_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                correction_gains[active_corrections].sum()
                            )
                        elif (
                            args.kd_mode == "online_ema_sol_match_full_distribution"
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            selected_legal_mask = direct_surface.token_mask[
                                teacher_reliable_mask
                            ]
                            (
                                kd_example_weights,
                                sol_match,
                                sol_margin,
                            ) = ema_sol_match_weights(
                                selected_teacher,
                                selected_gold,
                                selected_legal_mask,
                                temperature=args.ema_sol_match_temperature,
                                margin_center=args.ema_sol_match_margin_center,
                                weight_floor=args.ema_sol_match_weight_floor,
                            )
                            kd = masked_distillation_loss(
                                selected_teacher,
                                selected_student,
                                temperature=args.temperature,
                                example_weights=kd_example_weights,
                            )
                            active_online = kd_example_weights.gt(0)
                            values["kd_active"] = float(active_online.sum())
                            values["kd_confidence"] = float(
                                sol_margin.gt(0).sum()
                            )
                            values["kd_example_weight_sum"] = float(
                                kd_example_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                sol_margin[active_online].sum()
                            )
                            values["kd_gold_logp_gain_sum"] = float(
                                sol_match[active_online].sum()
                            )
                        elif (
                            args.kd_mode == "online_advantage_tiered"
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            (
                                kd_example_weights,
                                repair_mask,
                                confidence_mask,
                                margin_gain,
                                gold_logp_gain,
                            ) = online_advantage_kd_weights(
                                selected_teacher,
                                selected_student,
                                selected_gold,
                                minimum_margin_advantage=(
                                    args.kd_min_online_margin_advantage
                                ),
                                repair_weight=args.kd_repair_tier_weight,
                                confidence_weight=(
                                    args.kd_confidence_tier_weight
                                ),
                            )
                            kd = masked_distillation_loss(
                                selected_teacher,
                                selected_student,
                                temperature=args.temperature,
                                example_weights=kd_example_weights,
                            )
                            active_online = kd_example_weights.gt(0)
                            values["kd_active"] = float(active_online.sum())
                            values["kd_repair"] = float(repair_mask.sum())
                            values["kd_confidence"] = float(
                                confidence_mask.sum()
                            )
                            values["kd_example_weight_sum"] = float(
                                kd_example_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                margin_gain.sum()
                            )
                            values["kd_gold_logp_gain_sum"] = float(
                                gold_logp_gain.sum()
                            )
                        elif (
                            args.kd_mode in {
                                "online_pairwise_adaptive",
                                "online_ema_pairwise_adaptive",
                                HYBRID_CACHED_UNIFORM_EMA_PAIRWISE,
                            }
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected_teacher = teacher_logits_detached[
                                teacher_reliable_mask
                            ]
                            selected_student = student_logits[
                                teacher_reliable_mask
                            ]
                            selected_gold = direct_surface.gold_indices[
                                teacher_reliable_mask
                            ]
                            (
                                online_weights,
                                repair_mask,
                                confidence_mask,
                                competitor_indices,
                                target_margins,
                                margin_advantages,
                            ) = online_pairwise_adaptive_targets(
                                selected_teacher,
                                selected_student,
                                selected_gold,
                                minimum_margin_advantage=(
                                    args.kd_min_online_margin_advantage
                                ),
                                repair_target_margin_cap=(
                                    args.kd_pair_target_margin_cap
                                ),
                                confidence_increment_cap=(
                                    args.kd_online_confidence_increment_cap
                                ),
                                margin_weight_scale=(
                                    args.kd_online_margin_weight_scale
                                ),
                                repair_weight=args.kd_repair_tier_weight,
                                confidence_weight=(
                                    args.kd_confidence_tier_weight
                                ),
                            )
                            cached_uniform_weights = torch.zeros_like(
                                online_weights
                            )
                            selected_online = torch.ones_like(
                                online_weights,
                                dtype=torch.bool,
                            )
                            if (
                                args.kd_mode
                                == HYBRID_CACHED_UNIFORM_EMA_PAIRWISE
                            ):
                                selected_online = grouped[
                                    "online_teacher_flags"
                                ].to(device)[teacher_reliable_mask]
                                cached_uniform_weights = (
                                    ~selected_online
                                ).to(online_weights.dtype)
                                online_weights = (
                                    online_weights
                                    * selected_online.to(online_weights.dtype)
                                )
                            pairwise_kd = selective_pairwise_huber_margin_loss(
                                selected_student,
                                selected_gold,
                                competitor_indices,
                                target_margins,
                                example_weights=online_weights,
                                beta=args.kd_pair_huber_beta,
                            )
                            cached_uniform_kd = masked_distillation_loss(
                                selected_teacher,
                                selected_student,
                                temperature=args.temperature,
                                example_weights=cached_uniform_weights,
                            )
                            kd = pairwise_kd + cached_uniform_kd
                            active_online = online_weights.gt(0)
                            active_cached = cached_uniform_weights.gt(0)
                            values["kd_active"] = float(
                                active_online.sum() + active_cached.sum()
                            )
                            values["kd_active_online"] = float(
                                active_online.sum()
                            )
                            values["kd_active_cached"] = float(
                                active_cached.sum()
                            )
                            values["kd_repair"] = float(
                                (repair_mask & selected_online).sum()
                            )
                            values["kd_confidence"] = float(
                                (confidence_mask & selected_online).sum()
                            )
                            values["kd_example_weight_sum"] = float(
                                online_weights.sum()
                                + cached_uniform_weights.sum()
                            )
                            values["kd_margin_gain_sum"] = float(
                                margin_advantages[active_online].sum()
                            )
                        elif (
                            args.kd_mode
                            in {
                                "uniform",
                                "online_ema_uniform_full_distribution",
                            }
                            and teacher_reliable_mask is not None
                            and teacher_reliable_mask.any()
                        ):
                            selected = teacher_reliable_mask
                            selected_gold = direct_surface.gold_indices[selected]
                            kd_example_weights = None
                            if args.decision_kind == "stop":
                                values["kd_positive"] = float(
                                    selected_gold.eq(1).sum()
                                )
                                values["kd_negative"] = float(
                                    selected_gold.eq(0).sum()
                                )
                                kd_example_weights = torch.where(
                                    selected_gold.eq(1),
                                    torch.full_like(
                                        selected_gold,
                                        args.kd_positive_weight,
                                        dtype=student_logits.dtype,
                                    ),
                                    torch.full_like(
                                        selected_gold,
                                        args.kd_negative_weight,
                                        dtype=student_logits.dtype,
                                    ),
                                )
                            kd = masked_distillation_loss(
                                teacher_logits_detached[selected],
                                student_logits[selected],
                                temperature=args.temperature,
                                example_weights=kd_example_weights,
                            )
                            values["kd_active"] = float(selected.sum())
                            values["kd_example_weight_sum"] = float(
                                selected.sum()
                                if kd_example_weights is None
                                else kd_example_weights.sum()
                            )
                        warmup_steps = max(
                            1,
                            int(
                                int(active_phase["optimizer_steps"])
                                * float(active_phase["kd_warmup_ratio"])
                            ),
                        )
                        kd_scale = min(1.0, (phase_step + 1) / float(warmup_steps))
                    student_objective = (
                        args.direct_weight * direct_loss
                        + float(active_phase["kd_weight"]) * kd_scale * kd
                        + args.anchor_retention_weight * anchor_retention
                    )
                    if not bool(torch.isfinite(kd).all()):
                        raise FloatingPointError(
                            "non-finite KD loss; refusing to update adapter"
                        )
                # Direct-only SFT has no paired privileged branch and therefore
                # never installs a representation hook.  Keep its objective
                # explicit instead of relying on OPD/KD-local initialization.
                if args.objective == "direct":
                    kd = direct_loss * 0.0
                    anchor_retention = direct_loss * 0.0
                    kd_scale = 0.0
                    student_objective = args.direct_weight * direct_loss
                (
                    student_objective / active_grad_accum
                ).backward()
                values["direct"] = float(direct_loss.detach())
                values["kd"] = float(kd.detach())
                values["anchor_retention"] = float(anchor_retention.detach())
                values["kd_scale"] = float(kd_scale)
                del direct_outputs

            rolling.append(values)
            if phase_micro_step % active_grad_accum != 0:
                continue
            synchronize_gradients(trainable, world)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if args.kd_mode in ONLINE_EMA_KD_MODES:
                update_ema_teacher_adapter(model, args.ema_teacher_decay)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            phase_step += 1
            if global_step % args.log_every_steps == 0 or global_step == 1:
                mean = {
                    key: sum(row.get(key, 0.0) for row in rolling) / len(rolling)
                    for key in rolling[0]
                }
                count_keys = (
                    "teacher_correct",
                    "teacher_count",
                    "teacher_gold_margin_sum",
                    "direct_correct",
                    "direct_count",
                    "kd_eligible",
                    "kd_candidates",
                    "kd_active",
                    "kd_eligible_online",
                    "kd_candidates_online",
                    "kd_active_online",
                    "kd_eligible_cached",
                    "kd_candidates_cached",
                    "kd_active_cached",
                    "kd_positive",
                    "kd_negative",
                    "kd_repair",
                    "kd_confidence",
                    "kd_example_weight_sum",
                    "kd_margin_gain_sum",
                    "kd_gold_logp_gain_sum",
                )
                counts = torch.tensor(
                    [
                        sum(row.get(key, 0.0) for row in rolling)
                        for key in count_keys
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if world > 1:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                count_values = dict(zip(count_keys, counts.tolist()))
                teacher_count = count_values["teacher_count"]
                direct_count = count_values["direct_count"]
                kd_candidates = count_values["kd_candidates"]
                mean["teacher_accuracy"] = (
                    count_values["teacher_correct"] / teacher_count
                    if teacher_count
                    else 0.0
                )
                mean["teacher_gold_margin"] = (
                    count_values["teacher_gold_margin_sum"] / teacher_count
                    if teacher_count
                    else 0.0
                )
                mean["direct_action_accuracy"] = (
                    count_values["direct_correct"] / direct_count
                    if direct_count
                    else 0.0
                )
                mean["kd_eligible_coverage"] = (
                    count_values["kd_eligible"] / kd_candidates
                    if kd_candidates
                    else 0.0
                )
                kd_eligible = count_values["kd_eligible"]
                mean["kd_active_fraction_of_eligible"] = (
                    count_values["kd_active"] / kd_eligible
                    if kd_eligible
                    else 0.0
                )
                online_eligible = count_values["kd_eligible_online"]
                cached_eligible = count_values["kd_eligible_cached"]
                mean["kd_online_eligible_coverage"] = (
                    online_eligible / count_values["kd_candidates_online"]
                    if count_values["kd_candidates_online"]
                    else 0.0
                )
                mean["kd_online_active_fraction_of_eligible"] = (
                    count_values["kd_active_online"] / online_eligible
                    if online_eligible
                    else 0.0
                )
                mean["kd_cached_eligible_coverage"] = (
                    cached_eligible / count_values["kd_candidates_cached"]
                    if count_values["kd_candidates_cached"]
                    else 0.0
                )
                mean["kd_cached_active_fraction_of_eligible"] = (
                    count_values["kd_active_cached"] / cached_eligible
                    if cached_eligible
                    else 0.0
                )
                mean["kd_repair_fraction_of_eligible"] = (
                    count_values["kd_repair"] / kd_eligible
                    if kd_eligible
                    else 0.0
                )
                mean["kd_confidence_fraction_of_eligible"] = (
                    count_values["kd_confidence"] / kd_eligible
                    if kd_eligible
                    else 0.0
                )
                mean["kd_mean_example_weight"] = (
                    count_values["kd_example_weight_sum"] / kd_eligible
                    if kd_eligible
                    else 0.0
                )
                mean["kd_mean_teacher_student_margin_gain"] = (
                    count_values["kd_margin_gain_sum"] / kd_eligible
                    if kd_eligible
                    else 0.0
                )
                mean["kd_mean_teacher_student_gold_logp_gain"] = (
                    count_values["kd_gold_logp_gain_sum"] / kd_eligible
                    if kd_eligible
                    else 0.0
                )
                mean["teacher_examples"] = int(teacher_count)
                mean["kd_eligible_examples"] = int(
                    count_values["kd_eligible"]
                )
                mean["kd_active_examples"] = int(count_values["kd_active"])
                mean["kd_positive_examples"] = int(
                    count_values["kd_positive"]
                )
                mean["kd_negative_examples"] = int(
                    count_values["kd_negative"]
                )
                for key in count_keys:
                    mean.pop(key, None)
                log(
                    {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "phase": active_phase_index + 1,
                        "phase_name": active_phase["name"],
                        "phase_step": phase_step,
                        "lr": scheduler.get_last_lr()[0],
                        **mean,
                    }
                )
                rolling.clear()
            if global_step >= total_steps:
                stop = True
                break

    if world > 1:
        dist.barrier()
    raw_model = model.module if isinstance(model, DDP) else model
    if is_main:
        final_adapter = output / "final_adapter"
        save_training_adapter(raw_model, processor, final_adapter)
    if world > 1:
        dist.barrier()

    # Run a bounded, teacher-forced direct diagnostic on rank 0 only.  It
    # never gates navigation automatically.
    diagnostic = None
    if is_main:
        diagnostic_collator = GroupCollator(
            processor,
            max_length=(
                args.phase2_max_length if phase2_enabled else args.max_length
            ),
            objective=("opd" if args.objective == "opd" else "sc_mrd"),
            online_ema_teacher=(
                args.kd_mode in ONLINE_EMA_KD_MODES
            ),
            ema_teacher_max_length=args.ema_teacher_max_length,
        )
        diagnostic_dev_source = phase2_dev_rows if phase2_enabled else dev_rows
        diagnostic_train_source = (
            phase2_train_rows if phase2_enabled else train_rows
        )
        dev_paired_rows = [
            row for row in diagnostic_dev_source if row["paired"]
        ]
        train_paired_rows = [
            row for row in diagnostic_train_source if row["paired"]
        ]
        if args.max_dev_samples > 0:
            dev_paired_rows = dev_paired_rows[: args.max_dev_samples]
        if args.max_train_diagnostic_samples > 0:
            train_paired_rows = train_paired_rows[
                : args.max_train_diagnostic_samples
            ]
        # Sequence-aware scoring of candidate_10+ needs an individual
        # autoregressive prefix.  This affects only the bounded post-training
        # diagnostic; formal training already uses per-device batch size one.
        diagnostic_batch_size = (
            1 if args.decision_kind == "move" else args.diagnostic_batch_size
        )
        diagnostic_loader = DataLoader(
            GroupDataset(dev_paired_rows),
            batch_size=diagnostic_batch_size,
            shuffle=False,
            collate_fn=diagnostic_collator,
            num_workers=args.num_workers,
        )
        diagnostic = evaluate_paired(
            raw_model,
            diagnostic_loader,
            device,
            max_samples=args.max_dev_samples,
            kd_require_teacher_correct=args.kd_require_teacher_correct,
            kd_min_teacher_gold_margin=args.kd_min_teacher_gold_margin,
            cached_teacher=(args.kd_mode not in ONLINE_EMA_KD_MODES),
            online_ema_teacher=(args.kd_mode in ONLINE_EMA_KD_MODES),
        )
        train_diagnostic_loader = DataLoader(
            GroupDataset(train_paired_rows),
            batch_size=diagnostic_batch_size,
            shuffle=False,
            collate_fn=diagnostic_collator,
            num_workers=args.num_workers,
        )
        train_diagnostic = evaluate_paired(
            raw_model,
            train_diagnostic_loader,
            device,
            max_samples=args.max_train_diagnostic_samples,
            kd_require_teacher_correct=args.kd_require_teacher_correct,
            kd_min_teacher_gold_margin=args.kd_min_teacher_gold_margin,
            cached_teacher=(args.kd_mode not in ONLINE_EMA_KD_MODES),
            online_ema_teacher=(args.kd_mode in ONLINE_EMA_KD_MODES),
        )
        final_adapter = output / "final_adapter"
        config_path = final_adapter / "adapter_config.json"
        weights_path = final_adapter / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise RuntimeError("training adapter save did not produce root PEFT files")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_saved_r = (
            args.expected_anchor_r
            if args.continue_b0_adapter_directly
            else args.lora_r
        )
        expected_saved_alpha = (
            args.expected_anchor_alpha
            if args.continue_b0_adapter_directly
            else args.lora_alpha
        )
        complete = {
            "schema": "sc_mrd_native_lm_complete_v1",
            "passed": (
                global_step == total_steps
                and int(saved_config.get("r", -1)) == expected_saved_r
                and int(saved_config.get("lora_alpha", -1))
                == expected_saved_alpha
            ),
            "objective": args.objective,
            "decision_kind": args.decision_kind,
            "global_step": global_step,
            "expected_optimizer_steps": total_steps,
            "delta_adapter_sha256": sha256(weights_path),
            "trained_adapter_sha256": sha256(weights_path),
            "adapter_training_mode": training_mode,
            "single_adapter_output": True,
            "one_model_load": True,
            "in_process_training_phases": serializable_phases,
            "phase_boundaries_optimizer_steps": phase_boundaries,
            "optimizer_reset_at_phase_boundary": phase2_enabled,
            "continued_b0_adapter_directly": bool(
                args.continue_b0_adapter_directly
            ),
            "fixed_b0_sha256": args.expected_b0_sha256,
            "native_generation_chain_preserved": True,
            "independent_decision_head": False,
            "runtime_view": "direct_only",
            "runtime_max_new_tokens": 24,
            "diagnostic": diagnostic,
            "train_diagnostic": train_diagnostic,
        }
        (output / "pipeline_complete.json").write_text(
            json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        log({"event": "done", **complete})
    if log_handle is not None:
        log_handle.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
