#!/usr/bin/env python3
"""Vision-conditioned structured navigation CoT SFT.

Supports LoRA SFT by default and full-parameter SFT through PyTorch FSDP.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
except Exception:  # pragma: no cover
    FSDP = None
    FullStateDictConfig = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None
    size_based_auto_wrap_policy = None

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError("peft is required for this script. Please install peft in the training env.") from exc

# wandb upload disabled permanently: even with WANDB_MODE=disabled, wandb's
# Artifact.add_dir() still copies the entire output_dir (incl. multi-GB
# adapter safetensors) into ~/.local/share/wandb/artifacts/staging/, which
# has previously filled the local disk and crashed the whole DDP job
# (ChildFailedError) right after a successful training run finished. Force
# wandb=None unconditionally so wandb.init()/wandb.Artifact() are never
# called, instead of relying on an environment flag that doesn't actually
# suppress the artifact upload.
wandb = None


SYSTEM_PROMPT = "You are a navigation spatial reasoning assistant."


if not hasattr(torch, "compiler"):
    class _TorchCompilerShim:
        @staticmethod
        def is_compiling() -> bool:
            return False

    torch.compiler = _TorchCompilerShim()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exact-image cold-start vision SFT.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-adapter-path", default="", help="Optional LoRA adapter to continue training from.")
    parser.add_argument(
        "--require-fresh-base-model-path",
        default="",
        help=(
            "When set, require --model-path to resolve exactly to this path and reject "
            "adapter initialization or nonzero continuation counters. Defaults off."
        ),
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finetune-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer update steps. 0 means use epochs.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--eval-steps", default="", help="Comma-separated exact optimizer steps for validation eval. Overrides periodic eval when set.")
    parser.add_argument(
        "--distributed-eval",
        action="store_true",
        help=(
            "Shard validation across all distributed ranks and all-reduce loss statistics. "
            "Defaults off for exact legacy behavior."
        ),
    )
    parser.add_argument("--disable-eval", action="store_true", help="Disable periodic and final validation evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only score the val split with the loaded adapter and write val_metrics.json, then exit.")
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument(
        "--save-epoch-adapters",
        default="",
        help="Comma-separated 1-indexed epochs to save as epoch{N}_adapter/model after epoch_end.",
    )
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--train-loss-every-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable activation checkpointing (default for legacy runs).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation checkpointing when GPU memory permits faster training.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--response-only-logits",
        action="store_true",
        help=(
            "Compute LM-head logits only from the position immediately before the first "
            "supervised response token. This preserves the masked causal-LM loss while "
            "avoiding vocabulary projection over the image/prompt prefix."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help=(
            "Comma-separated LoRA target module names. Defaults to the legacy "
            "attention+MLP setting used by previous runs."
        ),
    )
    parser.add_argument(
        "--lora-exclude-modules",
        default="",
        help=(
            "Comma-separated module-name patterns excluded from LoRA injection. "
            "InternVL runs must exclude vision_tower so training and vLLM serving "
            "have the same language-only adapter surface."
        ),
    )
    parser.add_argument(
        "--forbid-vision-lora",
        action="store_true",
        help="Fail closed if a LoRA parameter is attached below a vision module.",
    )
    parser.add_argument("--fsdp-min-num-params", type=int, default=10_000_000)
    parser.add_argument("--save-best-on-eval", action="store_true", help="Save best checkpoint during periodic eval. Defaults off for full mode to avoid repeated huge full-state saves.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "embodied-nav-cot"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-file", default="", help="Path to write JSON log lines (flushed immediately).")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT, help="Optional chat system prompt. Use an empty value to match runtime prompts without a system message.")
    parser.add_argument("--no-system-prompt", dest="system_prompt", action="store_const", const="", help="Do not prepend a system message.")
    parser.add_argument(
        "--text-before-image",
        action="store_true",
        help=(
            "Place user prompt text before the image in multimodal chat content. "
            "Use this to match the VLMNav OpenAI runtime message order; the legacy "
            "training order remains image then text by default."
        ),
    )
    parser.add_argument("--use-row-loss-weight", action="store_true", help="Multiply each sample loss by row['action_loss_weight']. Defaults off for exact legacy behavior.")
    parser.add_argument(
        "--balance-task-family-loss",
        action="store_true",
        help=(
            "For rows carrying unified_task_family=move/stop, compute the "
            "length-normalized, row-weighted mean within each represented task "
            "family and average the family means. This prevents dataset size or "
            "response length from making Move or Stop dominate a unified batch. "
            "The option is inert for legacy rows without unified_task_family."
        ),
    )
    parser.add_argument(
        "--normalize-loss-per-row",
        action="store_true",
        help=(
            "Average assistant-token loss within each sample before averaging the batch. "
            "This prevents long responses from receiving more total gradient solely because "
            "they contain more supervised tokens. Defaults off for exact legacy behavior."
        ),
    )
    parser.add_argument(
        "--loss-length-normalization-power",
        type=float,
        default=0.0,
        help=(
            "Partially normalize each sample's total loss by its supervised-token count. "
            "0.0 is legacy token averaging and 1.0 is per-row averaging; values between "
            "0 and 1 interpolate the two. --normalize-loss-per-row is an alias for 1.0."
        ),
    )
    parser.add_argument(
        "--arrival-certified-value-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Opt-in multiplier applied only to the uniquely located 0/1 value token of "
            "an arrival_certified response. 1.0 preserves the legacy loss exactly. "
            "When enabled, malformed, ambiguous, multi-token, or truncated targets fail closed."
        ),
    )
    parser.add_argument("--use-decision-loss-weight", action="store_true", help="Apply row['decision_loss_weight'] to assistant tokens from the JSON decision block onward.")
    parser.add_argument("--use-action-field-loss-weight", action="store_true", help="Apply row['action_field_loss_weight'] only to the JSON action value tokens.")
    parser.add_argument(
        "--use-decision-basis-field-loss-weight",
        action="store_true",
        help=(
            "Apply row['decision_basis_field_loss_weight'] only to the compact "
            "JSON decision_basis value. This gives an explicit spatial-to-decision "
            "bridge intermediate supervision without adding a runtime module."
        ),
    )
    parser.add_argument(
        "--use-routing-field-loss-weight",
        action="store_true",
        help=(
            "Apply row state/ability/spatial-evidence field weights to the "
            "corresponding JSON values."
        ),
    )
    parser.add_argument(
        "--use-action-unlikelihood",
        action="store_true",
        help=(
            "Apply row['action_unlikelihood_weight'] to discourage row['unlikelihood_actions'] "
            "at the JSON decision.action value position. This is a light-weight action-prior "
            "preservation objective for direct action rows."
        ),
    )
    parser.add_argument(
        "--use-stop-unlikelihood",
        action="store_true",
        help=(
            "Apply row['stop_unlikelihood_weight'] to discourage emitting true at the JSON "
            "stop/should_stop field when the supervised response contains false. This is "
            "intended for stop hard negatives where ordinary SFT does not sufficiently "
            "suppress the visible-target -> stop=true failure mode."
        ),
    )
    parser.add_argument(
        "--teacher-logprob-jsonl",
        default="",
        help=(
            "Optional JSONL cache produced by precompute_base_direct_action_logprobs.py. "
            "Only rows with matching sample_id and response-aligned base_content contribute "
            "to the teacher KL / top-k CE objective."
        ),
    )
    parser.add_argument("--use-teacher-kl", action="store_true", help="Add top-k teacher-distribution CE on cached direct-action rows.")
    parser.add_argument("--teacher-kl-weight", type=float, default=0.0, help="Weight for --use-teacher-kl.")
    parser.add_argument("--teacher-kl-max-topk", type=int, default=8)
    parser.add_argument(
        "--use-action-head-kl",
        action="store_true",
        help=(
            "Add an auxiliary training-only action classifier on the hidden state "
            "that predicts the JSON decision.action value. Rows can set "
            "action_head_weight for supervised CE and action_head_kl_weight plus "
            "action_head_base_label/base_action for base-action preservation CE."
        ),
    )
    parser.add_argument("--action-head-max-id", type=int, default=32, help="Largest candidate id modeled by the auxiliary action head; -1 is included.")
    parser.add_argument("--action-head-loss-weight", type=float, default=1.0, help="Global multiplier for row action_head_weight.")
    parser.add_argument("--action-head-base-kl-weight", type=float, default=1.0, help="Global multiplier for row action_head_kl_weight.")
    # SGRC: state-gated reasoning activation + reasoning->decision coupling (train-time heads)
    parser.add_argument("--use-sgrc", action="store_true", help="Enable SGRC-Implicit gated-reasoning module on the decision path.")
    parser.add_argument("--sgrc-num-states", type=int, default=4)
    parser.add_argument("--sgrc-num-abilities", type=int, default=9)
    parser.add_argument("--sgrc-expert-dim", type=int, default=512, help="Per-ability expert bottleneck dim.")
    parser.add_argument("--sgrc-decision-lambda", type=float, default=1.0, help="Weight of decision-logit CE recomputed through h'=h+P(r).")
    parser.add_argument("--sgrc-state-lambda", type=float, default=0.3, help="Weight of state-gate CE (ablation: 0).")
    parser.add_argument("--sgrc-activation-lambda", type=float, default=0.3, help="Weight of ability-activation BCE (ablation: 0).")
    parser.add_argument("--sgrc-gate-state", type=int, default=1, help="1=alpha conditioned on state s; 0=ablation.")
    parser.add_argument("--sgrc-gate-activation", type=int, default=1, help="1=alpha gating; 0=uniform (ablation).")
    parser.add_argument("--sgrc-fuse", type=int, default=1, help="1=h'=h+P(r) fusion on decision path; 0=bypass (ablation).")
    parser.add_argument(
        "--sgrc-couple-lambda", type=float, default=0.0,
        help=(
            "Weight of the reasoning->decision coupling stop loss: a class-balanced BCE on "
            "couple_stop_head(r) at the reasoning-terminal position (token just before the "
            "decision block), supervised by row['sgrc_couple_stop']. This is the causal "
            "bottleneck test from docs/SGRC_implicit_design.md; unlike the token-level l_dec "
            "CE (which is dominated by JSON syntax tokens and heavily class-imbalanced), this "
            "loss directly targets stop discrimination from r. Set 0 to disable (prior behavior)."
        ),
    )
    parser.add_argument(
        "--sgrc-couple-pos-weight", type=float, default=0.0,
        help=(
            "pos_weight for the coupling stop BCE, to counter should_stop=true label scarcity "
            "(~6%% positive in the action mix). 0 = auto-compute per-batch from the label ratio."
        ),
    )
    parser.add_argument(
        "--sgrc-dec-use-stop-field-weight", action="store_true",
        help=(
            "Multiply the per-token l_dec CE by loss_token_weights (so should_stop value tokens "
            "get the same row['stop_field_loss_weight'] boost on the SGRC decision path as they "
            "already get on the raw backbone path)."
        ),
    )
    parser.add_argument(
        "--action-head-base-label-smoothing",
        type=float,
        default=0.02,
        help="Smoothing for the action-head base-action CE target; CE equals KL to this fixed target up to a constant.",
    )
    parser.add_argument("--start-global-step", type=int, default=0, help="Initial global_step for adapter-only resume.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch index to start from for adapter-only resume.")
    parser.add_argument("--skip-train-steps", type=int, default=0, help="Skip this many optimizer update steps before training.")
    return parser.parse_args()


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_device(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    if is_dist():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank, rank, world_size

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0, 0, 1


def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    if FSDP is not None and isinstance(model, FSDP):
        return model.module
    return model


def is_fsdp_model(model) -> bool:
    return FSDP is not None and isinstance(model, FSDP)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                for key in ("image_path",):
                    if key in row and not Path(row[key]).is_absolute():
                        row[key] = str((path.parent / row[key]).resolve())
                if isinstance(row.get("image_paths"), list):
                    row["image_paths"] = [
                        str((path.parent / p).resolve()) if not Path(p).is_absolute() else p
                        for p in row["image_paths"]
                    ]
                rows.append(row)
    return rows


def load_model(model_path: str, device: torch.device):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_5":
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except Exception as exc:
            print(
                f"[load_model] Qwen3_5ForConditionalGeneration unavailable: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            return model.to(device), "Qwen3_5ForConditionalGeneration"

    if model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception as exc:
            print(f"[load_model] Qwen3VLForConditionalGeneration unavailable: {exc}", file=sys.stderr, flush=True)
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            return model.to(device), "Qwen3VLForConditionalGeneration"

    if model_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            return model.to(device), "Qwen2_5_VLForConditionalGeneration"
        except Exception as exc:
            print(f"[load_model] Qwen2_5_VLForConditionalGeneration failed: {exc}", file=sys.stderr, flush=True)

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model.to(device), "AutoModelForImageTextToText"
    except Exception as exc:
        print(f"[load_model] AutoModelForImageTextToText failed: {exc}", file=sys.stderr, flush=True)

    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model.to(device), "AutoModelForVision2Seq"
    except Exception as exc:
        print(f"[load_model] AutoModelForVision2Seq failed: {exc}", file=sys.stderr, flush=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model.to(device), "AutoModelForCausalLM"


def load_processor(model_path: str):
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


def assert_no_vision_lora(model, *, enabled: bool) -> None:
    """Guard the InternVL train/serve contract against unsupported vision LoRA."""
    if not enabled:
        return
    offenders = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name or not parameter.requires_grad:
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in ("vision_tower", "vision_model", ".visual.", ".visual_")):
            offenders.append(name)
    if offenders:
        preview = ", ".join(offenders[:8])
        raise RuntimeError(
            "vision LoRA is forbidden for this retrain because the serving backend "
            f"does not apply it; found {len(offenders)} trainable tensors ({preview})"
        )


@dataclass
class VisionSample:
    sample_id: str
    image_path: str
    image_paths: list[str]
    prompt: str
    response: str
    # Optional per-row chat-presentation overrides.  These are needed by
    # curriculum-compression ablations that mix rows which were originally
    # trained with different (but runtime-relevant) message layouts.  ``None``
    # preserves the command-line/global collator setting exactly.
    presentation_system_prompt: str | None = None
    presentation_text_before_image: bool | None = None
    task_family_id: int = -1
    loss_weight: float = 1.0
    decision_loss_weight: float = 1.0
    decision_basis_field_loss_weight: float = 1.0
    action_field_loss_weight: float = 1.0
    stop_field_loss_weight: float = 1.0
    state_field_loss_weight: float = 1.0
    ability_field_loss_weight: float = 1.0
    evidence_field_loss_weight: float = 1.0
    gate_field_loss_weight: float = 1.0
    action_unlikelihood_weight: float = 0.0
    stop_unlikelihood_weight: float = 0.0
    unlikelihood_actions: list[str] | None = None
    teacher_logprobs: list[dict] | None = None
    action_head_label: int | None = None
    action_head_base_label: int | None = None
    action_head_weight: float = 0.0
    action_head_kl_weight: float = 0.0
    # SGRC train-time supervision (state-gated reasoning + reasoning->decision coupling)
    sgrc_state_id: int = -1
    sgrc_ability_activation: list[float] | None = None
    sgrc_couple_stop: int = -1
    sgrc_couple_move: int | None = None
    sgrc_couple_active: int = 0
    state_head_loss_weight: float = 0.0
    activation_head_loss_weight: float = 0.0
    couple_head_loss_weight: float = 0.0


def parse_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        except Exception:
            pass
    return [x.strip() for x in text.split(",") if x.strip()]


def action_id_from_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"candidate_(-?\d+)", text)
    if match:
        return int(match.group(1))
    try:
        return int(text)
    except Exception:
        return None


def response_action_id(response: str) -> int | None:
    try:
        obj = json.loads(response)
    except Exception:
        return None
    decision = obj.get("decision") if isinstance(obj, dict) else None
    if not isinstance(decision, dict):
        return None
    return action_id_from_label(decision.get("action"))


def row_action_head_label(row: dict, response: str) -> int | None:
    for key in ("action_head_label", "gpt_action", "supervised_action", "gold_action"):
        label = action_id_from_label(row.get(key))
        if label is not None:
            return label
    return response_action_id(response)


def row_action_head_base_label(row: dict) -> int | None:
    for key in ("action_head_base_label", "base_action", "baseline_action", "base_decision_action"):
        label = action_id_from_label(row.get(key))
        if label is not None:
            return label
    content = row.get("base_content")
    if isinstance(content, str) and content.strip():
        return response_action_id(content)
    return None


def load_teacher_logprob_cache(path: str) -> dict[str, dict]:
    if not path:
        return {}
    cache: dict[str, dict] = {}
    for row in read_jsonl(Path(path)):
        if row.get("status") != "ok":
            continue
        for key in ("sample_id", "cache_key", "source_sample_id", "prompt_sha1"):
            sid = str(row.get(key) or "")
            if sid:
                cache[sid] = row
    return cache


def float_field(row: dict, key: str, default: float) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def optional_bool_field(row: dict, key: str) -> bool | None:
    """Parse an optional, explicitly typed boolean dataset field."""
    if key not in row or row[key] is None:
        return None
    value = row[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"row field {key!r} must be an optional boolean, got {value!r}")


def rescale_incomplete_accumulation(
    parameters: list[torch.nn.Parameter],
    *,
    expected_microbatches: int,
    actual_microbatches: int,
) -> float:
    """Restore a mean gradient for an epoch-final partial accumulation.

    Every microbatch loss is divided by ``expected_microbatches``.  If an
    epoch ends early, the accumulated gradients therefore need multiplication
    by ``expected/actual`` before clipping and stepping.  The old code scaled
    only the number written to the log, leaving the real optimizer gradient
    systematically too small.
    """

    if expected_microbatches <= 0:
        raise ValueError("expected_microbatches must be positive")
    if not 0 < actual_microbatches <= expected_microbatches:
        raise ValueError("actual_microbatches must be in [1, expected]")
    scale = expected_microbatches / actual_microbatches
    if scale != 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return scale


def find_jsonish_scalar_value_span(
    text: str,
    *,
    field_names: tuple[str, ...],
    value_pattern: str,
) -> tuple[int, int] | None:
    """Locate a scalar value in the JSON-like formats used by VLMNav prompts."""
    spans = find_jsonish_scalar_value_spans(
        text,
        field_names=field_names,
        value_pattern=value_pattern,
    )
    return spans[0] if spans else None


def find_jsonish_scalar_value_spans(
    text: str,
    *,
    field_names: tuple[str, ...],
    value_pattern: str,
) -> list[tuple[int, int]]:
    """Locate every matching scalar value without changing legacy first-match behavior."""
    fields = "|".join(re.escape(name) for name in field_names)
    matches = re.finditer(
        rf"(?:[\"'](?:{fields})[\"']|(?:{fields}))\s*:\s*"
        rf"(?P<quote>[\"']?)(?P<value>{value_pattern})(?P=quote)",
        text,
        flags=re.IGNORECASE,
    )
    return [match.span("value") for match in matches]


def arrival_certified_value_token_spec(tokenizer, response: str) -> dict[str, Any]:
    """Return the exact binary target token and fail closed on an ambiguous surface."""
    spans = find_jsonish_scalar_value_spans(
        response,
        field_names=("arrival_certified",),
        value_pattern=r"[01](?![0-9])",
    )
    if len(spans) != 1:
        raise ValueError(
            "arrival_certified must have exactly one binary scalar value; "
            f"found {len(spans)} in {response!r}"
        )
    value_start, value_end = spans[0]
    value_text = response[value_start:value_end]
    prefix_ids = tokenizer(
        response[:value_start], add_special_tokens=False
    ).get("input_ids", [])
    value_ids = tokenizer(value_text, add_special_tokens=False).get("input_ids", [])
    suffix_ids = tokenizer(
        response[value_end:], add_special_tokens=False
    ).get("input_ids", [])
    response_ids = tokenizer(response, add_special_tokens=False).get("input_ids", [])
    if len(value_ids) != 1:
        raise ValueError(
            "arrival_certified value must tokenize to exactly one token; "
            f"value={value_text!r} token_ids={value_ids!r}"
        )
    if response_ids != [*prefix_ids, *value_ids, *suffix_ids]:
        raise ValueError(
            "arrival_certified value is not an independently aligned tokenizer span"
        )
    return {
        "label": int(value_text),
        "value_token_id": int(value_ids[0]),
        "response_token_ids": [int(token_id) for token_id in response_ids],
        "value_token_offset": len(prefix_ids),
    }


def find_json_field_value_span(
    text: str,
    *,
    field_name: str,
) -> tuple[int, int] | None:
    """Locate a complete JSON value, including strings or arrays."""
    match = re.search(
        rf"[\"']{re.escape(field_name)}[\"']\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    value_start = match.end()
    try:
        _, value_length = json.JSONDecoder().raw_decode(text[value_start:])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value_start, value_start + value_length


def find_all_json_field_value_spans(
    text: str,
    *,
    field_name: str,
) -> list[tuple[int, int]]:
    """Locate every complete value for a repeated JSON field name.

    B3 intentionally places a categorical ``conclusion`` inside each active
    spatial ability.  Weighting only the first occurrence would supervise
    target identity but miss the decisive proximity conclusion.
    """
    spans: list[tuple[int, int]] = []
    for match in re.finditer(
        rf"[\"']{re.escape(field_name)}[\"']\s*:\s*",
        text,
        flags=re.IGNORECASE,
    ):
        value_start = match.end()
        try:
            _, value_length = json.JSONDecoder().raw_decode(text[value_start:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        spans.append((value_start, value_start + value_length))
    return spans


def hidden_size_from_config(config: Any) -> int:
    candidates = [
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(config, "llm_config", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        for key in ("hidden_size", "d_model", "n_embd"):
            value = getattr(obj, key, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError("Could not infer hidden size for --use-action-head-kl from model config.")


class VisionPromptResponseDataset(Dataset):
    def __init__(self, rows: List[dict], teacher_cache: dict[str, dict] | None = None) -> None:
        teacher_cache = teacher_cache or {}
        self.samples = [
            VisionSample(
                sample_id=row["sample_id"],
                image_path=row["image_path"],
                image_paths=[
                    str(path)
                    for path in (
                        row.get("image_paths")
                        if isinstance(row.get("image_paths"), list) and row.get("image_paths")
                        else [row["image_path"]]
                    )
                ],
                prompt=row["prompt"].strip(),
                response=row["response"].strip(),
                presentation_system_prompt=(
                    str(row["presentation_system_prompt"])
                    if "presentation_system_prompt" in row
                    and row["presentation_system_prompt"] is not None
                    else None
                ),
                presentation_text_before_image=optional_bool_field(
                    row, "presentation_text_before_image"
                ),
                task_family_id={
                    "move": 0,
                    "stop": 1,
                }.get(str(row.get("unified_task_family", "")).strip().lower(), -1),
                loss_weight=float_field(row, "action_loss_weight", 1.0),
                decision_loss_weight=float_field(row, "decision_loss_weight", 1.0),
                decision_basis_field_loss_weight=float_field(row, "decision_basis_field_loss_weight", 1.0),
                action_field_loss_weight=float_field(row, "action_field_loss_weight", 1.0),
                stop_field_loss_weight=float_field(row, "stop_field_loss_weight", 1.0),
                state_field_loss_weight=float_field(row, "state_field_loss_weight", 1.0),
                ability_field_loss_weight=float_field(row, "ability_field_loss_weight", 1.0),
                evidence_field_loss_weight=float_field(row, "evidence_field_loss_weight", 1.0),
                gate_field_loss_weight=float_field(row, "gate_field_loss_weight", 1.0),
                action_unlikelihood_weight=float_field(row, "action_unlikelihood_weight", 0.0),
                stop_unlikelihood_weight=float_field(row, "stop_unlikelihood_weight", 0.0),
                unlikelihood_actions=parse_str_list(row.get("unlikelihood_actions")),
                teacher_logprobs=self._teacher_for_row(row, teacher_cache),
                action_head_label=row_action_head_label(row, str(row.get("response", "")).strip()),
                action_head_base_label=row_action_head_base_label(row),
                action_head_weight=float_field(row, "action_head_weight", 0.0),
                action_head_kl_weight=float_field(row, "action_head_kl_weight", 0.0),
                sgrc_state_id=int(row.get("sgrc_state_id", -1)) if row.get("sgrc_state_id") is not None else -1,
                sgrc_ability_activation=row.get("sgrc_ability_activation"),
                sgrc_couple_stop=int(row.get("sgrc_couple_stop", -1)) if row.get("sgrc_couple_stop") is not None else -1,
                sgrc_couple_move=action_id_from_label(row.get("sgrc_couple_move")),
                sgrc_couple_active=int(row.get("sgrc_couple_active", 0) or 0),
                state_head_loss_weight=float_field(row, "state_head_loss_weight", 0.0),
                activation_head_loss_weight=float_field(row, "activation_head_loss_weight", 0.0),
                couple_head_loss_weight=float_field(row, "couple_head_loss_weight", 0.0),
            )
            for row in rows
        ]

    @staticmethod
    def _teacher_for_row(row: dict, teacher_cache: dict[str, dict]) -> list[dict] | None:
        teacher = None
        for key in ("sample_id", "source_sample_id", "prompt_sha1"):
            value = str(row.get(key) or "")
            if value and value in teacher_cache:
                teacher = teacher_cache[value]
                break
        if not teacher:
            return None
        # Keep this objective conservative: only align teacher distributions to
        # the row target when the base generated the same compact response.
        if str(teacher.get("base_content", "")).strip() != str(row.get("response", "")).strip():
            return None
        content = teacher.get("base_logprobs")
        if isinstance(content, list) and content:
            return content
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VisionSample:
        return self.samples[idx]


class VisionSFTCollator:
    def __init__(
        self,
        processor,
        max_length: int,
        system_prompt: str = SYSTEM_PROMPT,
        teacher_kl_max_topk: int = 8,
        text_before_image: bool = False,
        use_routing_field_loss_weight: bool = False,
        use_decision_basis_field_loss_weight: bool = False,
        arrival_certified_value_loss_weight: float = 1.0,
        include_arrival_classification_metadata: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt.strip()
        self.teacher_kl_max_topk = max(1, int(teacher_kl_max_topk))
        self.text_before_image = bool(text_before_image)
        self.use_routing_field_loss_weight = bool(use_routing_field_loss_weight)
        self.use_decision_basis_field_loss_weight = bool(
            use_decision_basis_field_loss_weight
        )
        self.arrival_certified_value_loss_weight = float(
            arrival_certified_value_loss_weight
        )
        self.include_arrival_classification_metadata = bool(
            include_arrival_classification_metadata
        )
        if (
            not math.isfinite(self.arrival_certified_value_loss_weight)
            or self.arrival_certified_value_loss_weight <= 0
        ):
            raise ValueError(
                "arrival_certified_value_loss_weight must be finite and positive"
            )

    def _build_messages(
        self,
        prompt: str,
        response: str | None = None,
        image_count: int = 1,
        *,
        system_prompt: str | None = None,
        text_before_image: bool | None = None,
    ) -> list:
        messages = []
        effective_system_prompt = (
            self.system_prompt if system_prompt is None else system_prompt.strip()
        )
        effective_text_before_image = (
            self.text_before_image
            if text_before_image is None
            else bool(text_before_image)
        )
        if effective_system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": effective_system_prompt}]})
        image_contents = [{"type": "image"} for _ in range(max(1, int(image_count)))]
        text_content = {"type": "text", "text": prompt}
        user_content = (
            [text_content, *image_contents]
            if effective_text_before_image
            else [*image_contents, text_content]
        )
        messages.append({"role": "user", "content": user_content})
        if response is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        return messages

    def __call__(self, batch: List[VisionSample]) -> dict:
        images = [
            [Image.open(path).convert("RGB") for path in sample.image_paths]
            for sample in batch
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._build_messages(
                    sample.prompt,
                    image_count=len(sample.image_paths),
                    system_prompt=sample.presentation_system_prompt,
                    text_before_image=sample.presentation_text_before_image,
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in batch
        ]
        full_texts = [
            self.processor.apply_chat_template(
                self._build_messages(
                    sample.prompt,
                    sample.response,
                    image_count=len(sample.image_paths),
                    system_prompt=sample.presentation_system_prompt,
                    text_before_image=sample.presentation_text_before_image,
                ),
                tokenize=False,
                add_generation_prompt=False,
            )
            for sample in batch
        ]

        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        full_batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_sizes = prompt_batch["attention_mask"].sum(dim=1).tolist()
        prompt_lens: list[int] = []
        for idx, prompt_size in enumerate(prompt_sizes):
            active = full_batch["attention_mask"][idx].nonzero(as_tuple=True)[0]
            if active.numel() < int(prompt_size):
                raise ValueError(
                    f"full sequence is shorter than prompt for sample_id={batch[idx].sample_id}"
                )
            # `prompt_size` is a token count, not an absolute column. InternVL
            # left-pads mixed-length batches, so include the leading pad offset
            # before locating the assistant response in the full batch.
            response_start = int(active[0]) + int(prompt_size)
            labels[idx, :response_start] = -100
            prompt_lens.append(response_start)

        loss_token_weights = torch.zeros_like(labels, dtype=torch.float32)
        loss_token_weights[labels != -100] = 1.0
        arrival_value_positions = torch.full(
            (len(batch),), -1, dtype=torch.long
        )
        arrival_value_labels = torch.full((len(batch),), -1, dtype=torch.long)
        if (
            self.arrival_certified_value_loss_weight != 1.0
            or self.include_arrival_classification_metadata
        ):
            for idx, sample in enumerate(batch):
                try:
                    spec = arrival_certified_value_token_spec(
                        self.tokenizer, sample.response
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"invalid arrival target for sample_id={sample.sample_id}: {exc}"
                    ) from exc
                prompt_len = int(prompt_lens[idx])
                response_ids = spec["response_token_ids"]
                response_end = prompt_len + len(response_ids)
                if response_end > labels.shape[1]:
                    raise ValueError(
                        "arrival target was truncated by max_length for "
                        f"sample_id={sample.sample_id}: prompt_len={prompt_len} "
                        f"response_tokens={len(response_ids)} max_length={self.max_length}"
                    )
                observed_response = labels[idx, prompt_len:response_end].tolist()
                if observed_response != response_ids:
                    raise ValueError(
                        "arrival assistant target is incomplete or misaligned for "
                        f"sample_id={sample.sample_id}: expected={response_ids!r} "
                        f"observed={observed_response!r}"
                    )
                value_position = prompt_len + int(spec["value_token_offset"])
                if (
                    value_position < 1
                    or value_position >= labels.shape[1]
                    or int(labels[idx, value_position])
                    != int(spec["value_token_id"])
                ):
                    raise ValueError(
                        "arrival value token is missing, truncated, or not uniquely aligned for "
                        f"sample_id={sample.sample_id}"
                    )
                loss_token_weights[idx, value_position] = (
                    self.arrival_certified_value_loss_weight
                )
                arrival_value_positions[idx] = value_position
                arrival_value_labels[idx] = int(spec["label"])
        if self.use_routing_field_loss_weight:
            for idx, sample in enumerate(batch):
                routing_fields = [("navigation_state", sample.state_field_loss_weight)]
                # B8 represents its activated routing set as a top-level
                # ``abilities`` array.  Its per-reasoning-item ``ability``
                # keys must not be mistaken for that router target.  Preserve
                # the legacy singular field for historical datasets.
                if '"abilities"' in sample.response:
                    routing_fields.append(("abilities", sample.ability_field_loss_weight))
                else:
                    routing_fields.append(("ability", sample.ability_field_loss_weight))
                # F-v2's spatial evidence is the substantive visual-reasoning
                # target.  It must not be silently left at unit weight while
                # only the compact state/ability labels are emphasized.
                if '"spatial_evidence"' in sample.response:
                    routing_fields.append(("spatial_evidence", sample.evidence_field_loss_weight))
                for field_name, raw_weight in routing_fields:
                    weight = max(0.0, float(raw_weight))
                    if weight == 1.0:
                        continue
                    value_span = find_json_field_value_span(
                        sample.response,
                        field_name=field_name,
                    )
                    if value_span is None:
                        continue
                    value_start, value_end = value_span
                    prefix_ids = self.tokenizer(
                        sample.response[:value_start],
                        add_special_tokens=False,
                    ).get("input_ids", [])
                    value_ids = self.tokenizer(
                        sample.response[value_start:value_end],
                        add_special_tokens=False,
                    ).get("input_ids", [])
                    token_start = int(prompt_lens[idx]) + len(prefix_ids)
                    for tok_idx in range(len(value_ids)):
                        pos = token_start + tok_idx
                        if 0 <= pos < labels.shape[1] and labels[idx, pos] != -100:
                            loss_token_weights[idx, pos] = weight
        if self.use_decision_basis_field_loss_weight:
            for idx, sample in enumerate(batch):
                weight = max(0.0, float(sample.decision_basis_field_loss_weight))
                if weight == 1.0:
                    continue
                value_span = find_json_field_value_span(
                    sample.response,
                    field_name="decision_basis",
                )
                if value_span is None:
                    continue
                value_start, value_end = value_span
                prefix_ids = self.tokenizer(
                    sample.response[:value_start], add_special_tokens=False
                ).get("input_ids", [])
                value_ids = self.tokenizer(
                    sample.response[value_start:value_end], add_special_tokens=False
                ).get("input_ids", [])
                token_start = int(prompt_lens[idx]) + len(prefix_ids)
                for tok_idx in range(len(value_ids)):
                    pos = token_start + tok_idx
                    if 0 <= pos < labels.shape[1] and labels[idx, pos] != -100:
                        loss_token_weights[idx, pos] = weight
        for idx, sample in enumerate(batch):
            weight = max(0.0, float(sample.decision_loss_weight))
            if weight == 1.0:
                continue
            decision_char = sample.response.find('"decision"')
            if decision_char < 0:
                decision_char = sample.response.find("decision")
            if decision_char < 0:
                continue
            prefix = sample.response[:decision_char]
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
            decision_start = int(prompt_lens[idx]) + len(prefix_ids)
            if decision_start < labels.shape[1]:
                active = labels[idx, decision_start:] != -100
                loss_token_weights[idx, decision_start:][active] = weight

        for idx, sample in enumerate(batch):
            weight = max(0.0, float(sample.stop_field_loss_weight))
            if weight == 1.0:
                continue
            value_span = find_jsonish_scalar_value_span(
                sample.response,
                field_names=("done", "stop", "should_stop"),
                value_pattern=r"true|false|0|1",
            )
            if value_span is None:
                continue
            value_start, value_end = value_span
            value_text = sample.response[value_start:value_end]
            prefix = sample.response[:value_start]
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
            value_ids = self.tokenizer(value_text, add_special_tokens=False).get("input_ids", [])
            stop_start = int(prompt_lens[idx]) + len(prefix_ids)
            for tok_idx in range(len(value_ids)):
                pos = stop_start + tok_idx
                if 0 <= pos < labels.shape[1] and labels[idx, pos] != -100:
                    loss_token_weights[idx, pos] = weight

        for idx, sample in enumerate(batch):
            weight = max(0.0, float(sample.action_field_loss_weight))
            if weight == 1.0:
                continue
            value_span = find_jsonish_scalar_value_span(
                sample.response,
                field_names=("action",),
                value_pattern=r"candidate_-?\d+|-?\d+",
            )
            if value_span is None:
                continue
            value_start, value_end = value_span
            value_text = sample.response[value_start:value_end]
            prefix = sample.response[:value_start]
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
            value_ids = self.tokenizer(value_text, add_special_tokens=False).get("input_ids", [])
            action_start = int(prompt_lens[idx]) + len(prefix_ids)
            for tok_idx in range(len(value_ids)):
                pos = action_start + tok_idx
                if 0 <= pos < labels.shape[1] and labels[idx, pos] != -100:
                    loss_token_weights[idx, pos] = weight

        for idx, sample in enumerate(batch):
            weight = max(0.0, float(sample.gate_field_loss_weight))
            if weight == 1.0:
                continue
            for field_name in (
                "stop_evidence",
                "identity_evidence",
                "proximity_evidence",
                "temporal_relation",
                "conclusion",
            ):
                value_spans = find_all_json_field_value_spans(
                    sample.response, field_name=field_name
                )
                for value_start, value_end in value_spans:
                    prefix_ids = self.tokenizer(
                        sample.response[:value_start], add_special_tokens=False
                    ).get("input_ids", [])
                    value_ids = self.tokenizer(
                        sample.response[value_start:value_end], add_special_tokens=False
                    ).get("input_ids", [])
                    gate_start = int(prompt_lens[idx]) + len(prefix_ids)
                    for tok_idx in range(len(value_ids)):
                        pos = gate_start + tok_idx
                        if 0 <= pos < labels.shape[1] and labels[idx, pos] != -100:
                            loss_token_weights[idx, pos] = weight

        batch_out = {"labels": labels}
        for key, value in full_batch.items():
            if torch.is_tensor(value):
                batch_out[key] = value
        batch_out["loss_weights"] = torch.tensor(
            [max(0.0, float(sample.loss_weight)) for sample in batch],
            dtype=torch.float32,
        )
        batch_out["task_family_ids"] = torch.tensor(
            [int(sample.task_family_id) for sample in batch],
            dtype=torch.long,
        )
        batch_out["loss_token_weights"] = loss_token_weights
        if self.include_arrival_classification_metadata:
            batch_out["arrival_value_positions"] = arrival_value_positions
            batch_out["arrival_value_labels"] = arrival_value_labels
        unlikelihood_positions: list[list[int]] = []
        unlikelihood_token_ids: list[list[int]] = []
        unlikelihood_weights: list[list[float]] = []
        for idx, sample in enumerate(batch):
            row_positions: list[int] = []
            row_token_ids: list[int] = []
            row_weights: list[float] = []
            weight = max(0.0, float(sample.action_unlikelihood_weight))
            actions = list(sample.unlikelihood_actions or [])
            if weight > 0 and actions:
                match = re.search(r'"action"\s*:\s*"([^"]+)"', sample.response)
                if match is not None:
                    correct_action = match.group(1)
                    prefix = sample.response[: match.start(1)]
                    suffix = sample.response[match.end(1) :]
                    correct_ids = self.tokenizer(correct_action, add_special_tokens=False).get("input_ids", [])
                    suffix_ids = self.tokenizer(suffix, add_special_tokens=False).get("input_ids", [])
                    action_start = int(prompt_lens[idx]) + len(
                        self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
                    )
                    for action in actions:
                        forbidden_ids = self.tokenizer(action, add_special_tokens=False).get("input_ids", [])
                        if not forbidden_ids:
                            continue
                        prefix_len = 0
                        for lhs, rhs in zip(correct_ids, forbidden_ids):
                            if lhs != rhs:
                                break
                            prefix_len += 1
                        # Penalize only the action-specific suffix. Penalizing the shared
                        # "candidate" prefix would suppress every candidate action.
                        if prefix_len >= len(forbidden_ids):
                            # If the forbidden action is a prefix of the correct action
                            # (e.g. candidate_1 vs candidate_14), discourage ending the
                            # JSON string immediately after the forbidden prefix.
                            if suffix_ids:
                                shifted_pos = action_start + len(forbidden_ids) - 1
                                if 0 <= shifted_pos < labels.shape[1] - 1:
                                    row_positions.append(int(shifted_pos))
                                    row_token_ids.append(int(suffix_ids[0]))
                                    row_weights.append(weight)
                        else:
                            for tok_idx in range(prefix_len, len(forbidden_ids)):
                                shifted_pos = action_start + tok_idx - 1
                                if 0 <= shifted_pos < labels.shape[1] - 1:
                                    row_positions.append(int(shifted_pos))
                                    row_token_ids.append(int(forbidden_ids[tok_idx]))
                                    row_weights.append(weight)
            stop_weight = max(0.0, float(sample.stop_unlikelihood_weight))
            if stop_weight > 0:
                match = re.search(r'"(?:stop|should_stop)"\s*:\s*false', sample.response, flags=re.IGNORECASE)
                if match is not None:
                    value_match = re.search(r'false', sample.response[match.start() : match.end()], flags=re.IGNORECASE)
                    if value_match is not None:
                        value_start = match.start() + value_match.start()
                        prefix = sample.response[:value_start]
                        correct_ids = self.tokenizer("false", add_special_tokens=False).get("input_ids", [])
                        forbidden_ids = self.tokenizer("true", add_special_tokens=False).get("input_ids", [])
                        stop_start = int(prompt_lens[idx]) + len(
                            self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
                        )
                        prefix_len = 0
                        for lhs, rhs in zip(correct_ids, forbidden_ids):
                            if lhs != rhs:
                                break
                            prefix_len += 1
                        for tok_idx in range(prefix_len, len(forbidden_ids)):
                            shifted_pos = stop_start + tok_idx - 1
                            if 0 <= shifted_pos < labels.shape[1] - 1:
                                row_positions.append(int(shifted_pos))
                                row_token_ids.append(int(forbidden_ids[tok_idx]))
                                row_weights.append(stop_weight)
            unlikelihood_positions.append(row_positions)
            unlikelihood_token_ids.append(row_token_ids)
            unlikelihood_weights.append(row_weights)
        max_ul = max([len(x) for x in unlikelihood_positions] + [0])
        if max_ul > 0:
            pos_tensor = torch.full((len(batch), max_ul), -1, dtype=torch.long)
            tok_tensor = torch.full((len(batch), max_ul), -1, dtype=torch.long)
            weight_tensor = torch.zeros((len(batch), max_ul), dtype=torch.float32)
            for idx, positions in enumerate(unlikelihood_positions):
                for jdx, pos in enumerate(positions):
                    pos_tensor[idx, jdx] = pos
                    tok_tensor[idx, jdx] = unlikelihood_token_ids[idx][jdx]
                    weight_tensor[idx, jdx] = unlikelihood_weights[idx][jdx]
            batch_out["unlikelihood_positions"] = pos_tensor
            batch_out["unlikelihood_token_ids"] = tok_tensor
            batch_out["unlikelihood_weights"] = weight_tensor
        teacher_positions: list[list[int]] = []
        teacher_token_ids: list[list[list[int]]] = []
        teacher_probs: list[list[list[float]]] = []
        for idx, sample in enumerate(batch):
            row_positions: list[int] = []
            row_token_ids: list[list[int]] = []
            row_probs: list[list[float]] = []
            teacher_content = sample.teacher_logprobs or []
            if teacher_content:
                response_ids = self.tokenizer(sample.response, add_special_tokens=False).get("input_ids", [])
                for tok_idx, item in enumerate(teacher_content[: len(response_ids)]):
                    top = item.get("top_logprobs") if isinstance(item, dict) else None
                    token = item.get("token") if isinstance(item, dict) else None
                    if not top or token is None:
                        continue
                    token_id = self.tokenizer.convert_tokens_to_ids(str(token))
                    if token_id != response_ids[tok_idx]:
                        continue
                    ids: list[int] = []
                    lps: list[float] = []
                    for cand in top[: self.teacher_kl_max_topk]:
                        cand_token = cand.get("token") if isinstance(cand, dict) else None
                        if cand_token is None:
                            continue
                        cand_id = self.tokenizer.convert_tokens_to_ids(str(cand_token))
                        if not isinstance(cand_id, int) or cand_id < 0:
                            continue
                        try:
                            lp = float(cand.get("logprob"))
                        except Exception:
                            continue
                        ids.append(int(cand_id))
                        lps.append(lp)
                    if not ids:
                        continue
                    # Normalize the stored top-k distribution. This is a
                    # top-k teacher CE rather than exact full-vocab KL.
                    max_lp = max(lps)
                    exps = [math.exp(lp - max_lp) for lp in lps]
                    denom = sum(exps)
                    if denom <= 0:
                        continue
                    probs = [x / denom for x in exps]
                    shifted_pos = int(prompt_lens[idx]) + tok_idx - 1
                    if 0 <= shifted_pos < labels.shape[1] - 1:
                        row_positions.append(int(shifted_pos))
                        row_token_ids.append(ids)
                        row_probs.append(probs)
            teacher_positions.append(row_positions)
            teacher_token_ids.append(row_token_ids)
            teacher_probs.append(row_probs)
        max_teacher_pos = max([len(x) for x in teacher_positions] + [0])
        if max_teacher_pos > 0:
            max_teacher_topk = max([len(ids) for row in teacher_token_ids for ids in row] + [0])
            pos_tensor = torch.full((len(batch), max_teacher_pos), -1, dtype=torch.long)
            tok_tensor = torch.full((len(batch), max_teacher_pos, max_teacher_topk), -1, dtype=torch.long)
            prob_tensor = torch.zeros((len(batch), max_teacher_pos, max_teacher_topk), dtype=torch.float32)
            mask_tensor = torch.zeros((len(batch), max_teacher_pos, max_teacher_topk), dtype=torch.bool)
            for idx, positions in enumerate(teacher_positions):
                for jdx, pos in enumerate(positions):
                    pos_tensor[idx, jdx] = pos
                    ids = teacher_token_ids[idx][jdx]
                    probs = teacher_probs[idx][jdx]
                    for kdx, tok_id in enumerate(ids):
                        tok_tensor[idx, jdx, kdx] = tok_id
                        prob_tensor[idx, jdx, kdx] = probs[kdx]
                        mask_tensor[idx, jdx, kdx] = True
            batch_out["teacher_positions"] = pos_tensor
            batch_out["teacher_token_ids"] = tok_tensor
            batch_out["teacher_probs"] = prob_tensor
            batch_out["teacher_mask"] = mask_tensor
        action_head_positions = torch.full((len(batch),), -1, dtype=torch.long)
        action_head_labels = torch.full((len(batch),), -1, dtype=torch.long)
        action_head_base_labels = torch.full((len(batch),), -1, dtype=torch.long)
        action_head_weights = torch.zeros((len(batch),), dtype=torch.float32)
        action_head_kl_weights = torch.zeros((len(batch),), dtype=torch.float32)
        for idx, sample in enumerate(batch):
            match = re.search(r'"action"\s*:\s*"([^"]+)"', sample.response)
            if match is not None:
                prefix = sample.response[: match.start(1)]
                prefix_ids = self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
                action_pos = int(prompt_lens[idx]) + len(prefix_ids) - 1
            elif sample.action_head_label is not None or sample.action_head_base_label is not None:
                action_pos = int(prompt_lens[idx]) - 1
            else:
                continue
            if action_pos < 0 or action_pos >= labels.shape[1] - 1:
                continue
            action_head_positions[idx] = int(action_pos)
            if sample.action_head_label is not None:
                action_head_labels[idx] = int(sample.action_head_label)
            if sample.action_head_base_label is not None:
                action_head_base_labels[idx] = int(sample.action_head_base_label)
            action_head_weights[idx] = max(0.0, float(sample.action_head_weight))
            action_head_kl_weights[idx] = max(0.0, float(sample.action_head_kl_weight))
        batch_out["action_head_positions"] = action_head_positions
        batch_out["action_head_labels"] = action_head_labels
        batch_out["action_head_base_labels"] = action_head_base_labels
        batch_out["action_head_weights"] = action_head_weights
        batch_out["action_head_kl_weights"] = action_head_kl_weights

        # ---- SGRC train-time supervision tensors ----------------------------
        # state head reads the prompt-terminal hidden (state known before generation).
        # coupling head reads a reasoning-terminal hidden: the token right before the
        # decision block, so z has attended the full spatial_reasoning span -> the
        # decision is forced to be derivable from the activated reasoning (bottleneck).
        n = len(batch)
        num_abil = len(batch[0].sgrc_ability_activation) if (batch and batch[0].sgrc_ability_activation) else 9
        sgrc_state_pos = torch.full((n,), -1, dtype=torch.long)
        sgrc_couple_pos = torch.full((n,), -1, dtype=torch.long)
        sgrc_state_labels = torch.full((n,), -1, dtype=torch.long)
        sgrc_activation = torch.zeros((n, num_abil), dtype=torch.float32)
        sgrc_activation_valid = torch.zeros((n,), dtype=torch.bool)
        sgrc_couple_stop = torch.full((n,), -1, dtype=torch.long)
        sgrc_couple_move = torch.full((n,), -1, dtype=torch.long)
        sgrc_couple_active = torch.zeros((n,), dtype=torch.bool)
        sgrc_state_w = torch.zeros((n,), dtype=torch.float32)
        sgrc_activation_w = torch.zeros((n,), dtype=torch.float32)
        sgrc_couple_w = torch.zeros((n,), dtype=torch.float32)
        for idx, sample in enumerate(batch):
            plen = int(prompt_lens[idx])
            state_pos = plen - 1  # prompt-terminal hidden
            if 0 <= state_pos < labels.shape[1] - 1:
                sgrc_state_pos[idx] = state_pos
            # reasoning-terminal position = token just before the decision block
            dec_char = sample.response.find('"decision"')
            if dec_char < 0:
                dec_char = sample.response.find('"done"')
            if dec_char >= 0:
                prefix_ids = self.tokenizer(sample.response[:dec_char], add_special_tokens=False).get("input_ids", [])
                cpos = plen + max(0, len(prefix_ids) - 1)
            else:
                cpos = labels.shape[1] - 1  # stop verifier rows: use sequence end
            if 0 <= cpos < labels.shape[1] - 1:
                sgrc_couple_pos[idx] = cpos
            sgrc_state_labels[idx] = int(sample.sgrc_state_id)
            if sample.sgrc_ability_activation and any(sample.sgrc_ability_activation):
                sgrc_activation[idx] = torch.tensor(sample.sgrc_ability_activation[:num_abil], dtype=torch.float32)
                sgrc_activation_valid[idx] = True
            sgrc_couple_stop[idx] = int(sample.sgrc_couple_stop)
            if sample.sgrc_couple_move is not None:
                sgrc_couple_move[idx] = int(sample.sgrc_couple_move)
            sgrc_couple_active[idx] = bool(sample.sgrc_couple_active)
            sgrc_state_w[idx] = max(0.0, float(sample.state_head_loss_weight))
            sgrc_activation_w[idx] = max(0.0, float(sample.activation_head_loss_weight))
            sgrc_couple_w[idx] = max(0.0, float(sample.couple_head_loss_weight))
        # image-token mask (Qwen3-VL image_pad id=151655): SGRC heads pool over
        # visual tokens where navigation state lives, not a single text token.
        sgrc_image_mask = (full_batch["input_ids"] == 151655)
        batch_out["sgrc_image_mask"] = sgrc_image_mask
        batch_out["sgrc_state_pos"] = sgrc_state_pos
        batch_out["sgrc_couple_pos"] = sgrc_couple_pos
        batch_out["sgrc_state_labels"] = sgrc_state_labels
        batch_out["sgrc_activation"] = sgrc_activation
        batch_out["sgrc_activation_valid"] = sgrc_activation_valid
        batch_out["sgrc_couple_stop"] = sgrc_couple_stop
        batch_out["sgrc_couple_move"] = sgrc_couple_move
        batch_out["sgrc_couple_active"] = sgrc_couple_active
        batch_out["sgrc_state_w"] = sgrc_state_w
        batch_out["sgrc_activation_w"] = sgrc_activation_w
        batch_out["sgrc_couple_w"] = sgrc_couple_w
        return batch_out


@torch.no_grad()
def evaluate(model, loader, device, use_bf16: bool, distributed: bool = False) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("loss_weights", None)
        batch.pop("task_family_ids", None)
        batch.pop("loss_token_weights", None)
        batch.pop("unlikelihood_positions", None)
        batch.pop("unlikelihood_token_ids", None)
        batch.pop("unlikelihood_weights", None)
        batch.pop("teacher_positions", None)
        batch.pop("teacher_token_ids", None)
        batch.pop("teacher_probs", None)
        batch.pop("teacher_mask", None)
        batch.pop("action_head_positions", None)
        batch.pop("action_head_labels", None)
        batch.pop("action_head_base_labels", None)
        batch.pop("action_head_weights", None)
        batch.pop("action_head_kl_weights", None)
        for _k in ("sgrc_image_mask", "sgrc_state_pos", "sgrc_couple_pos", "sgrc_state_labels",
                   "sgrc_activation", "sgrc_activation_valid", "sgrc_couple_stop",
                   "sgrc_couple_move", "sgrc_couple_active", "sgrc_state_w",
                   "sgrc_activation_w", "sgrc_couple_w"):
            batch.pop(_k, None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**batch)
        loss_sum += out.loss.detach().float().item()
        count += 1
    model.train()
    if distributed and is_dist():
        stats = torch.tensor([loss_sum, float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum = float(stats[0].item())
        count = int(stats[1].item())
    return loss_sum / max(1, count)


def save_model_checkpoint(model, processor, path: Path, rank: int, full_state: bool,
                          sgrc_heads: dict | None = None, sgrc_meta: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if full_state and is_fsdp_model(model):
        if FullStateDictConfig is None or StateDictType is None:
            raise RuntimeError("FSDP full-state save is unavailable in this PyTorch build.")
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        if rank == 0:
            unwrap_model(model).save_pretrained(path, state_dict=state_dict, safe_serialization=True)
            processor.save_pretrained(path)
        return

    if rank == 0:
        unwrap_model(model).save_pretrained(path)
        processor.save_pretrained(path)
        # SGRC auxiliary heads: saved so the coupling stop head can drive the
        # runtime stop decision (reasoning-bottleneck -> stop), not just shape LoRA.
        if sgrc_heads:
            payload = {name: unwrap_model(h).state_dict() for name, h in sgrc_heads.items() if h is not None}
            if payload:
                torch.save({"heads": payload, "meta": (sgrc_meta or {})}, path / "sgrc_heads.pt")


def main() -> None:
    args = parse_args()
    if args.require_fresh_base_model_path:
        actual_model_path = Path(args.model_path).expanduser().resolve()
        required_model_path = Path(args.require_fresh_base_model_path).expanduser().resolve()
        if actual_model_path != required_model_path:
            raise ValueError(
                "fresh-base contract violation: --model-path resolved to "
                f"{actual_model_path}, expected {required_model_path}"
            )
        if args.init_adapter_path:
            raise ValueError(
                "fresh-base contract violation: --init-adapter-path must be empty"
            )
        continuation_values = {
            "start_global_step": int(args.start_global_step),
            "start_epoch": int(args.start_epoch),
            "skip_train_steps": int(args.skip_train_steps),
        }
        nonzero_values = {key: value for key, value in continuation_values.items() if value != 0}
        if nonzero_values:
            raise ValueError(
                "fresh-base contract violation: continuation counters must all be zero; "
                f"got {nonzero_values}"
            )
    if not 0.0 <= float(args.loss_length_normalization_power) <= 1.0:
        raise ValueError("--loss-length-normalization-power must be between 0.0 and 1.0")
    if (
        not math.isfinite(float(args.arrival_certified_value_loss_weight))
        or float(args.arrival_certified_value_loss_weight) <= 0
    ):
        raise ValueError(
            "--arrival-certified-value-loss-weight must be finite and positive"
        )
    normalization_power = (
        1.0 if args.normalize_loss_per_row else float(args.loss_length_normalization_power)
    )
    set_seed(args.seed)

    device, local_rank, rank, world_size = setup_device(args)
    is_main = rank == 0
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this training script.")
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_dist():
        dist.barrier()

    log_fh = open(args.log_file, "a") if args.log_file and is_main else None

    def log_line(msg: str) -> None:
        if not is_main:
            return
        print(msg, flush=True)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    run = None
    if wandb is not None and is_main:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=args.wandb_run_name,
            config=vars(args),
            dir=str(output_dir),
        )

    processor = load_processor(args.model_path)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model, model_loader_name = load_model(args.model_path, device)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if args.finetune_mode == "full" and args.init_adapter_path:
        raise ValueError("--init-adapter-path is only supported with --finetune-mode lora")
    if args.finetune_mode == "lora" and args.init_adapter_path:
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
    elif args.finetune_mode == "lora":
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        if not target_modules:
            raise ValueError("--lora-target-modules produced an empty target module list")
        raw_exclude = args.lora_exclude_modules.strip()
        if raw_exclude and any(ch in raw_exclude for ch in "|*()[]\\"):
            exclude_modules = raw_exclude
        else:
            exclude_modules = [x.strip() for x in raw_exclude.split(",") if x.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            exclude_modules=exclude_modules or None,
        )
        model = get_peft_model(model, lora_config)
    elif args.finetune_mode == "full":
        for param in model.parameters():
            param.requires_grad_(True)

    assert_no_vision_lora(model, enabled=args.forbid_vision_lora)

    action_head = None
    if args.use_action_head_kl:
        if args.action_head_max_id < 0:
            raise ValueError("--action-head-max-id must be >= 0 because candidate_-1 is modeled separately.")
        hidden_size = hidden_size_from_config(unwrap_model(model).config)
        action_head = torch.nn.Linear(hidden_size, int(args.action_head_max_id) + 2).to(device)
        action_head.train()
        if is_main:
            log_line(
                json.dumps(
                    {
                        "event": "action_head_init",
                        "hidden_size": hidden_size,
                        "num_classes": int(args.action_head_max_id) + 2,
                        "class_mapping": "candidate_id + 1; candidate_-1 maps to 0",
                    },
                    ensure_ascii=False,
                )
            )

    # ---- SGRC-Implicit module ON the decision-token computation path --------
    # (state gate -> ability activation -> gated reasoning -> residual fusion;
    #  decision logits are recomputed as lm_head(final_norm(h+P(r))). See
    #  docs/SGRC_implicit_design.md.) At inference the prompt stays minimal.
    sgrc_module = None
    sgrc_lm_head = sgrc_final_norm = None
    if args.use_sgrc:
        from embodied_nav_cot.sgrc_module import SGRCModule
        hs = hidden_size_from_config(unwrap_model(model).config)
        sgrc_module = SGRCModule(
            hidden=hs, num_states=args.sgrc_num_states, num_abilities=args.sgrc_num_abilities,
            expert_dim=args.sgrc_expert_dim,
            gate_state=(args.sgrc_gate_state == 1), gate_activation=(args.sgrc_gate_activation == 1),
            fuse=(args.sgrc_fuse == 1),
        ).to(device)
        sgrc_module.train()
        um = unwrap_model(model)
        sgrc_lm_head = um.lm_head
        sgrc_final_norm = um.model.language_model.norm
        if is_main:
            log_line(json.dumps({"event": "sgrc_module_init", "hidden_size": hs,
                                 "num_states": args.sgrc_num_states, "num_abilities": args.sgrc_num_abilities,
                                 "expert_dim": args.sgrc_expert_dim,
                                 "gate_state": args.sgrc_gate_state, "gate_activation": args.sgrc_gate_activation,
                                 "fuse": args.sgrc_fuse,
                                 "lambda": {"state": args.sgrc_state_lambda,
                                            "activation": args.sgrc_activation_lambda,
                                            "decision": args.sgrc_decision_lambda}}, ensure_ascii=False))

    if is_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    elif is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log_msg = {
            "event": "trainable_parameters",
            "trainable": trainable,
            "total": total,
            "trainable_pct": trainable / max(1, total) * 100.0,
            "finetune_mode": args.finetune_mode,
        }
        print(json.dumps(log_msg, ensure_ascii=False), flush=True)

    if is_dist() and args.finetune_mode == "full":
        if FSDP is None or MixedPrecision is None or ShardingStrategy is None or size_based_auto_wrap_policy is None:
            raise RuntimeError("PyTorch FSDP is unavailable; full fine-tuning requires FSDP in this script.")
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=args.fsdp_min_num_params,
        )
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
            use_orig_params=True,
        )
    elif is_dist():
        try:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                static_graph=True,
            )
        except TypeError:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            model._set_static_graph()
    if action_head is not None and is_dist():
        action_head = DDP(
            action_head,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    if args.use_sgrc and is_dist():
        sgrc_module = DDP(sgrc_module, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=True)

    teacher_cache = load_teacher_logprob_cache(args.teacher_logprob_jsonl)
    train_rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    val_rows = read_jsonl(Path(args.data_dir) / "val.jsonl")
    train_dataset = VisionPromptResponseDataset(train_rows, teacher_cache=teacher_cache)
    val_dataset = VisionPromptResponseDataset(val_rows, teacher_cache={})
    collator = VisionSFTCollator(
        processor,
        args.max_length,
        args.system_prompt,
        args.teacher_kl_max_topk,
        args.text_before_image,
        args.use_routing_field_loss_weight,
        args.use_decision_basis_field_loss_weight,
        args.arrival_certified_value_loss_weight,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_dist() else None
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if is_dist() and (args.finetune_mode == "full" or args.distributed_eval)
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Eval-only mode: skip the entire training loop, just score the val split
    # with the given (already-trained) adapter and write val_metrics.json.
    # Used by the ablation pipeline as a best-effort held-out text metric
    # before navigation eval; the caller ignores failures here.
    if getattr(args, "eval_only", False):
        if args.distributed_eval:
            val_loss = evaluate(unwrap_model(model), val_loader, device, args.bf16, distributed=True)
        elif is_main:
            val_loss = evaluate(unwrap_model(model), val_loader, device, args.bf16)
        else:
            val_loss = None
        if is_main:
            out_path = output_dir / "val_metrics.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump({"eval_only": True, "final_val_loss": val_loss,
                           "n_val": len(val_dataset)}, f, ensure_ascii=False, indent=2)
            log_line(json.dumps({"event": "eval_only_done", "final_val_loss": val_loss}, ensure_ascii=False))
            print(f"[eval-only] val_loss={val_loss:.6f} n_val={len(val_dataset)} -> {out_path}")
        if log_fh is not None:
            log_fh.close()
        if is_dist():
            dist.destroy_process_group()
        return

    optim_params = list(model.parameters())
    if action_head is not None:
        optim_params.extend(action_head.parameters())
    if sgrc_module is not None:
        optim_params.extend(sgrc_module.parameters())
    sgrc_heads_map = {"sgrc_module": sgrc_module} if args.use_sgrc else None
    sgrc_meta = {
        "num_states": args.sgrc_num_states, "num_abilities": args.sgrc_num_abilities,
        "expert_dim": args.sgrc_expert_dim, "image_token_id": 151655,
        "mechanism": "implicit_on_decision_path: h'=h+P(sum alpha_c E_c(h)); logits=lm_head(norm(h'))",
        "gate_state": args.sgrc_gate_state, "gate_activation": args.sgrc_gate_activation, "fuse": args.sgrc_fuse,
        "couple_stop_head": True, "sgrc_couple_lambda": args.sgrc_couple_lambda,
        "sgrc_couple_pos_weight": args.sgrc_couple_pos_weight,
        "sgrc_dec_use_stop_field_weight": bool(args.sgrc_dec_use_stop_field_weight),
    } if args.use_sgrc else None
    optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    if args.max_steps > 0:
        total_update_steps = min(total_update_steps, args.max_steps)
    warmup_steps = max(1, int(total_update_steps * args.warmup_ratio))
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

    metadata = {
        "model_path": args.model_path,
        "model_loader": model_loader_name,
        "data_dir": args.data_dir,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "distributed": {"enabled": is_dist(), "world_size": world_size},
        "args": vars(args),
        "teacher_cache_rows": len(teacher_cache),
    }
    if is_main:
        with (output_dir / "run_config.json").open("w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    if run is not None:
        run.config.update(metadata, allow_val_change=True)
        artifact = wandb.Artifact("exact_only_vision_sft_data", type="dataset")
        artifact.add_file(str(Path(args.data_dir) / "train.jsonl"), name="train.jsonl")
        artifact.add_file(str(Path(args.data_dir) / "val.jsonl"), name="val.jsonl")
        artifact.add_file(str(output_dir / "run_config.json"), name="run_config.json")
        run.log_artifact(artifact)

    global_step = max(0, args.start_global_step)
    best_val = float("inf")
    last_val_loss = None
    last_val_step = None
    optimizer.zero_grad(set_to_none=True)
    for _ in range(global_step):
        scheduler.step()
    model.train()
    if action_head is not None:
        action_head.train()

    loss_window: list = []
    mean_loss_interval = max(1, args.train_loss_every_steps)
    explicit_eval_steps = {
        int(x.strip())
        for x in str(args.eval_steps).split(",")
        if x.strip()
    }
    save_epoch_adapters = {
        int(x.strip())
        for x in str(args.save_epoch_adapters).split(",")
        if x.strip()
    }

    def do_optimizer_step(current_epoch: int, loss_value: float) -> None:
        nonlocal global_step, best_val, last_val_loss, last_val_step
        if is_fsdp_model(model):
            model.clip_grad_norm_(1.0)
            if action_head is not None:
                torch.nn.utils.clip_grad_norm_(action_head.parameters(), 1.0)
        else:
            clip_params = list(model.parameters())
            if action_head is not None:
                clip_params.extend(action_head.parameters())
            if sgrc_module is not None:
                clip_params.extend(sgrc_module.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.log_every_steps == 0:
            loss_window.append(loss_value)
            payload = {
                "event": "train_step",
                "epoch": current_epoch + 1,
                "global_step": global_step,
                "lr": scheduler.get_last_lr()[0],
                "loss": round(loss_value, 6),
                "time": time.time(),
            }
            log_line(json.dumps(payload, ensure_ascii=False))
            if run is not None:
                run.log(
                    {
                        "train/loss": payload["loss"],
                        "train/lr": payload["lr"],
                        "train/epoch": payload["epoch"],
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        if global_step % mean_loss_interval == 0 and len(loss_window) > 0:
            mean_loss = sum(loss_window) / len(loss_window)
            summary = {
                "event": "train_loss_eval",
                "global_step": global_step,
                "train_loss": round(mean_loss, 6),
                "window_steps": len(loss_window),
                "time": time.time(),
            }
            log_line(json.dumps(summary, ensure_ascii=False))
            if run is not None:
                run.log({"train/loss_window": mean_loss, "global_step": global_step}, step=global_step)
            loss_window.clear()

        if explicit_eval_steps:
            do_eval_step = (not args.disable_eval) and global_step in explicit_eval_steps
        else:
            do_eval_step = (not args.disable_eval) and args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0
        if do_eval_step:
            if args.distributed_eval:
                val_loss = evaluate(
                    unwrap_model(model), val_loader, device, args.bf16, distributed=True
                )
            elif args.finetune_mode == "full":
                val_loss = evaluate(model, val_loader, device, args.bf16, distributed=True)
            elif is_main:
                val_loss = evaluate(unwrap_model(model), val_loader, device, args.bf16)
            else:
                val_loss = None
            last_val_loss = val_loss
            last_val_step = global_step
            if is_main:
                log_line(json.dumps({"event": "eval", "global_step": global_step, "val_loss": val_loss}, ensure_ascii=False))
                if run is not None:
                    run.log({"eval/val_loss": val_loss, "global_step": global_step}, step=global_step)
            improved = val_loss is not None and val_loss < best_val
            should_save_best = improved and (args.finetune_mode == "lora" or args.save_best_on_eval)
            if improved:
                best_val = val_loss
            if should_save_best:
                best_dir = output_dir / ("best_model" if args.finetune_mode == "full" else "best_adapter")
                save_model_checkpoint(model, processor, best_dir, rank, full_state=args.finetune_mode == "full", sgrc_heads=sgrc_heads_map, sgrc_meta=sgrc_meta)
        if is_dist() and do_eval_step:
            dist.barrier()

        if global_step % args.save_every_steps == 0:
            ckpt = output_dir / f"checkpoint-step-{global_step}"
            save_model_checkpoint(model, processor, ckpt, rank, full_state=args.finetune_mode == "full", sgrc_heads=sgrc_heads_map, sgrc_meta=sgrc_meta)
        if is_dist() and global_step % args.save_every_steps == 0:
            dist.barrier()

    skipped_updates = 0
    stop_training = False
    for epoch in range(max(0, args.start_epoch), args.epochs):
        if stop_training:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break
            if skipped_updates < args.skip_train_steps:
                if step % args.grad_accum_steps == 0:
                    skipped_updates += 1
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_weights = batch.pop("loss_weights", None)
            task_family_ids = batch.pop("task_family_ids", None)
            loss_token_weights = batch.pop("loss_token_weights", None)
            unlikelihood_positions = batch.pop("unlikelihood_positions", None)
            unlikelihood_token_ids = batch.pop("unlikelihood_token_ids", None)
            unlikelihood_weights = batch.pop("unlikelihood_weights", None)
            teacher_positions = batch.pop("teacher_positions", None)
            teacher_token_ids = batch.pop("teacher_token_ids", None)
            teacher_probs = batch.pop("teacher_probs", None)
            teacher_mask = batch.pop("teacher_mask", None)
            action_head_positions = batch.pop("action_head_positions", None)
            action_head_labels = batch.pop("action_head_labels", None)
            action_head_base_labels = batch.pop("action_head_base_labels", None)
            action_head_weights = batch.pop("action_head_weights", None)
            action_head_kl_weights = batch.pop("action_head_kl_weights", None)
            sgrc_image_mask = batch.pop("sgrc_image_mask", None)
            sgrc_state_pos = batch.pop("sgrc_state_pos", None)
            sgrc_couple_pos = batch.pop("sgrc_couple_pos", None)
            sgrc_state_labels = batch.pop("sgrc_state_labels", None)
            sgrc_activation = batch.pop("sgrc_activation", None)
            sgrc_activation_valid = batch.pop("sgrc_activation_valid", None)
            sgrc_couple_stop = batch.pop("sgrc_couple_stop", None)
            sgrc_couple_move = batch.pop("sgrc_couple_move", None)
            sgrc_couple_active = batch.pop("sgrc_couple_active", None)
            sgrc_state_w = batch.pop("sgrc_state_w", None)
            sgrc_activation_w = batch.pop("sgrc_activation_w", None)
            sgrc_couple_w = batch.pop("sgrc_couple_w", None)
            # Static-graph DDP and gradient checkpointing require reducer hooks on
            # every microbatch in the supported Qwen multimodal training path.
            sync_context = contextlib.nullcontext()
            with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                use_custom_token_loss = (
                    normalization_power > 0
                    or args.arrival_certified_value_loss_weight != 1.0
                    or args.use_row_loss_weight
                    or args.use_decision_loss_weight
                    or args.use_action_field_loss_weight
                    or args.use_routing_field_loss_weight
                    or args.use_decision_basis_field_loss_weight
                    or args.use_teacher_kl
                    or args.use_action_head_kl
                )
                if use_custom_token_loss:
                    labels = batch.pop("labels")
                    if args.response_only_logits and (
                        args.use_action_unlikelihood
                        or args.use_stop_unlikelihood
                        or args.use_teacher_kl
                        or args.use_action_head_kl
                        or args.use_sgrc
                    ):
                        raise RuntimeError(
                            "response-only logits is incompatible with position-indexed auxiliary losses"
                        )
                    first_logit = 0
                    logits_to_keep = 0
                    if args.response_only_logits:
                        active_positions = labels.ne(-100).nonzero(as_tuple=False)
                        if active_positions.numel() == 0:
                            raise RuntimeError("response-only logits requires at least one supervised token")
                        first_target = int(active_positions[:, 1].min().item())
                        first_logit = max(0, first_target - 1)
                        logits_to_keep = int(labels.shape[1] - first_logit)
                    outputs = model(
                        **batch,
                        output_hidden_states=bool(action_head is not None or args.use_sgrc),
                        logits_to_keep=logits_to_keep,
                    )
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., first_logit + 1 :].contiguous()
                    token_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view_as(shift_labels)
                    active = shift_labels.ne(-100)
                    weights = torch.ones_like(token_loss, dtype=torch.float32)
                    if loss_token_weights is not None:
                        weights = loss_token_weights[..., first_logit + 1 :].contiguous().float()
                    if args.use_row_loss_weight and loss_weights is not None:
                        weights = weights * loss_weights.float().view(-1, 1)
                    weighted_loss = token_loss * weights
                    if normalization_power > 0:
                        active_float = active.float()
                        row_token_counts = active_float.sum(dim=1)
                        valid_rows = row_token_counts > 0
                        if valid_rows.any():
                            row_losses = (weighted_loss * active_float).sum(dim=1) / row_token_counts.clamp_min(1.0)
                            length_weights = row_token_counts[valid_rows].pow(1.0 - normalization_power)
                            if (
                                args.balance_task_family_loss
                                and task_family_ids is not None
                                and (task_family_ids[valid_rows] >= 0).any()
                            ):
                                valid_row_losses = row_losses[valid_rows]
                                valid_family_ids = task_family_ids[valid_rows]
                                valid_row_weights = (
                                    loss_weights.float()[valid_rows]
                                    if args.use_row_loss_weight and loss_weights is not None
                                    else torch.ones_like(length_weights)
                                )
                                family_losses = []
                                for family_id in (0, 1):
                                    family_mask = valid_family_ids.eq(family_id)
                                    if not family_mask.any():
                                        continue
                                    numerator = (
                                        valid_row_losses[family_mask]
                                        * length_weights[family_mask]
                                    ).sum()
                                    denominator = (
                                        valid_row_weights[family_mask]
                                        * length_weights[family_mask]
                                    ).sum().clamp_min(1e-8)
                                    family_losses.append(numerator / denominator)
                                if family_losses:
                                    loss = torch.stack(family_losses).mean()
                                else:
                                    loss = (
                                        valid_row_losses * length_weights
                                    ).sum() / length_weights.sum().clamp_min(1.0)
                            else:
                                loss = (
                                    row_losses[valid_rows] * length_weights
                                ).sum() / length_weights.sum().clamp_min(1.0)
                        else:
                            loss = weighted_loss.sum() * 0.0
                    else:
                        loss = weighted_loss[active].sum() / active.float().sum().clamp_min(1.0)
                    if (args.use_action_unlikelihood or args.use_stop_unlikelihood) and unlikelihood_positions is not None:
                        valid_ul = (
                            (unlikelihood_positions >= 0)
                            & (unlikelihood_token_ids >= 0)
                            & (unlikelihood_weights > 0)
                        )
                        if valid_ul.any():
                            probs = F.softmax(shift_logits.float(), dim=-1)
                            batch_idx = torch.arange(probs.shape[0], device=probs.device).view(-1, 1).expand_as(unlikelihood_positions)
                            safe_positions = unlikelihood_positions.clamp_min(0)
                            safe_tokens = unlikelihood_token_ids.clamp_min(0)
                            forbidden_probs = probs[batch_idx, safe_positions, safe_tokens].clamp(1e-6, 1.0 - 1e-6)
                            ul_loss = -torch.log1p(-forbidden_probs)
                            ul_loss = (ul_loss * unlikelihood_weights.float())[valid_ul].mean()
                            loss = loss + ul_loss
                    if (
                        args.use_teacher_kl
                        and args.teacher_kl_weight > 0
                        and teacher_positions is not None
                        and teacher_token_ids is not None
                        and teacher_probs is not None
                        and teacher_mask is not None
                    ):
                        valid_teacher = (teacher_positions >= 0) & teacher_mask.any(dim=-1)
                        if valid_teacher.any():
                            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                            batch_idx = torch.arange(log_probs.shape[0], device=log_probs.device).view(-1, 1, 1).expand_as(teacher_token_ids)
                            safe_positions = teacher_positions.clamp_min(0).unsqueeze(-1).expand_as(teacher_token_ids)
                            safe_tokens = teacher_token_ids.clamp_min(0)
                            gathered = log_probs[batch_idx, safe_positions, safe_tokens]
                            per_topk = -(teacher_probs.float() * gathered) * teacher_mask.float()
                            per_pos = per_topk.sum(dim=-1)
                            teacher_ce = per_pos[valid_teacher].mean()
                            loss = loss + float(args.teacher_kl_weight) * teacher_ce
                    if args.use_action_head_kl and action_head is not None:
                        hidden_states = getattr(outputs, "hidden_states", None)
                        if not hidden_states:
                            raise RuntimeError("--use-action-head-kl requires model outputs.hidden_states; pass output_hidden_states=True.")
                        hidden = hidden_states[-1]
                        if action_head_positions is None:
                            safe_positions = torch.zeros((hidden.shape[0],), device=hidden.device, dtype=torch.long)
                            valid_positions = torch.zeros((hidden.shape[0],), device=hidden.device, dtype=torch.bool)
                        else:
                            safe_positions = action_head_positions.clamp(min=0, max=max(0, hidden.shape[1] - 1))
                            valid_positions = action_head_positions >= 0
                        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                        action_hidden = hidden[batch_idx, safe_positions]
                        action_logits = action_head(action_hidden)
                        # Always touch the head so DDP sees the same graph even
                        # on rare batches without valid action-head labels.
                        loss = loss + action_logits.sum() * 0.0

                        num_action_classes = int(args.action_head_max_id) + 2
                        if action_head_labels is not None and action_head_weights is not None:
                            valid_action = (
                                valid_positions
                                & (action_head_weights > 0)
                                & (action_head_labels >= -1)
                                & (action_head_labels <= int(args.action_head_max_id))
                            )
                            if valid_action.any() and args.action_head_loss_weight > 0:
                                safe_labels = action_head_labels.clamp(min=-1, max=int(args.action_head_max_id)) + 1
                                per_row_ce = F.cross_entropy(action_logits.float(), safe_labels, reduction="none")
                                denom = action_head_weights[valid_action].float().sum().clamp_min(1e-6)
                                action_ce = (per_row_ce[valid_action] * action_head_weights[valid_action].float()).sum() / denom
                                loss = loss + float(args.action_head_loss_weight) * action_ce
                        if action_head_base_labels is not None and action_head_kl_weights is not None:
                            valid_base = (
                                valid_positions
                                & (action_head_kl_weights > 0)
                                & (action_head_base_labels >= -1)
                                & (action_head_base_labels <= int(args.action_head_max_id))
                            )
                            if valid_base.any() and args.action_head_base_kl_weight > 0:
                                safe_base_labels = action_head_base_labels.clamp(min=-1, max=int(args.action_head_max_id)) + 1
                                smoothing = min(max(float(args.action_head_base_label_smoothing), 0.0), 0.49)
                                log_probs = F.log_softmax(action_logits.float(), dim=-1)
                                if smoothing > 0 and num_action_classes > 1:
                                    target = torch.full_like(
                                        log_probs[valid_base],
                                        smoothing / float(num_action_classes - 1),
                                    )
                                    target.scatter_(1, safe_base_labels[valid_base].view(-1, 1), 1.0 - smoothing)
                                    per_row_base_ce = -(target * log_probs[valid_base]).sum(dim=-1)
                                else:
                                    per_row_base_ce = F.nll_loss(
                                        log_probs[valid_base],
                                        safe_base_labels[valid_base],
                                        reduction="none",
                                    )
                                denom = action_head_kl_weights[valid_base].float().sum().clamp_min(1e-6)
                                base_ce = (
                                    per_row_base_ce * action_head_kl_weights[valid_base].float()
                                ).sum() / denom
                                loss = loss + float(args.action_head_base_kl_weight) * base_ce
                    if args.use_sgrc and sgrc_module is not None:
                        hidden_states = getattr(outputs, "hidden_states", None)
                        if not hidden_states:
                            raise RuntimeError("--use-sgrc requires outputs.hidden_states.")
                        hidden = hidden_states[-1]              # (B,T,H) pre-final-norm
                        bsz, T, H = hidden.shape
                        # visual pooling for state/activation (navigation state lives in image tokens)
                        imask = sgrc_image_mask.to(hidden.device)
                        if imask.shape[1] != T:
                            m = torch.zeros((bsz, T), dtype=torch.bool, device=hidden.device)
                            k = min(imask.shape[1], T); m[:, :k] = imask[:, :k]; imask = m
                        imf = imask.unsqueeze(-1).to(hidden.dtype)
                        h_visual = (hidden * imf).sum(1) / imf.sum(1).clamp_min(1.0)   # (B,H)
                        # response-token positions on the DECISION path: labels(shifted) != -100.
                        # We recompute their logits through h' = h + P(gated_reasoning(h)),
                        # so decision generation literally flows through state->alpha->r.
                        resp_mask = shift_labels.ne(-100)              # (B, T-1)
                        # hidden aligned to shift: predicting token t uses hidden[t-1]
                        dec_hidden = hidden[:, :-1, :]                  # (B,T-1,H)
                        flat_mask = resp_mask.reshape(-1)
                        idx = flat_mask.nonzero(as_tuple=True)[0]
                        coupled_this_batch = False
                        if idx.numel() > 0 and args.sgrc_decision_lambda > 0:
                            flat_hidden = dec_hidden.reshape(-1, H)[idx]           # (N,H)
                            # map each response token to its sample index for h_visual
                            samp_idx = (torch.arange(bsz, device=hidden.device)
                                        .unsqueeze(1).expand(bsz, T-1).reshape(-1)[idx])
                            hv = h_visual[samp_idx]                                # (N,H)
                            out_sg = sgrc_module(flat_hidden, hv)
                            h_prime = out_sg["h_prime"]                            # (N,H)
                            recomputed = sgrc_lm_head(sgrc_final_norm(h_prime)).float()  # (N,V)
                            tgt_tok = shift_labels.reshape(-1)[idx]
                            if args.sgrc_dec_use_stop_field_weight and loss_token_weights is not None:
                                tok_w = loss_token_weights[..., 1:].reshape(-1)[idx].float()
                                per_tok_ce = F.cross_entropy(recomputed, tgt_tok, reduction="none")
                                l_dec = (per_tok_ce * tok_w).sum() / tok_w.sum().clamp_min(1e-6)
                            else:
                                l_dec = F.cross_entropy(recomputed, tgt_tok)
                            loss = loss + float(args.sgrc_decision_lambda) * l_dec
                            # aux losses reuse the module's state/alpha from these tokens' visual
                            state_logits = out_sg["state_logits"]                  # (N,S)
                            alpha_logits = out_sg["alpha_logits"]                  # (N,A)
                            # (1) state gate: supervise per-sample; broadcast label to its tokens
                            slab = sgrc_state_labels[samp_idx]                     # (N,)
                            vs = slab >= 0
                            if vs.any() and args.sgrc_state_lambda > 0:
                                loss = loss + float(args.sgrc_state_lambda) * F.cross_entropy(
                                    state_logits.float()[vs], slab[vs])
                            # (2) ability activation
                            av = sgrc_activation_valid[samp_idx]
                            if av.any() and args.sgrc_activation_lambda > 0:
                                tgt_a = sgrc_activation[samp_idx].float()
                                loss = loss + float(args.sgrc_activation_lambda) * F.binary_cross_entropy_with_logits(
                                    alpha_logits.float()[av], tgt_a[av])
                            # (3) reasoning->decision COUPLING on stop: a class-balanced BCE at
                            # the reasoning-terminal position (sgrc_couple_pos), predicted from
                            # r (the gated-reasoning output) rather than h' or the token-level
                            # CE. This is the causal bottleneck the design doc calls for: the
                            # stop decision must be DERIVABLE from s->alpha->r, and it gets a
                            # dedicated, non-token-diluted, class-balanced gradient (the
                            # token-level l_dec CE is dominated by JSON syntax tokens and
                            # ~6% positive should_stop=true, which starves this signal).
                            if (
                                args.sgrc_couple_lambda > 0
                                and sgrc_couple_pos is not None
                                and sgrc_couple_stop is not None
                                and sgrc_couple_active is not None
                            ):
                                cpos = sgrc_couple_pos
                                valid_c = (
                                    (cpos >= 0)
                                    & sgrc_couple_active
                                    & (sgrc_couple_stop >= 0)
                                )
                                if valid_c.any():
                                    b_idx = torch.arange(bsz, device=hidden.device)[valid_c]
                                    c_hidden = dec_hidden[b_idx, cpos[valid_c]]         # (Nc,H)
                                    c_visual = h_visual[b_idx]                          # (Nc,H)
                                    out_c = sgrc_module(c_hidden, c_visual)
                                    couple_logit = out_c["couple_stop_logit"].float()   # (Nc,)
                                    couple_tgt = sgrc_couple_stop[valid_c].float()
                                    if args.sgrc_couple_pos_weight > 0:
                                        pos_weight = torch.tensor(
                                            float(args.sgrc_couple_pos_weight), device=hidden.device,
                                        )
                                    else:
                                        n_pos = couple_tgt.sum().clamp_min(1.0)
                                        n_neg = (couple_tgt.numel() - couple_tgt.sum()).clamp_min(1.0)
                                        pos_weight = (n_neg / n_pos).detach()
                                    l_couple = F.binary_cross_entropy_with_logits(
                                        couple_logit, couple_tgt, pos_weight=pos_weight,
                                    )
                                    if sgrc_couple_w is not None:
                                        row_w = sgrc_couple_w[valid_c].float().mean().clamp_min(1e-6)
                                    else:
                                        row_w = 1.0
                                    loss = loss + float(args.sgrc_couple_lambda) * row_w * l_couple
                                    coupled_this_batch = True
                        if not coupled_this_batch:
                            # DDP graph safety: touch module params (incl. couple head) on
                            # every microbatch so DDP sees a consistent autograd graph even
                            # on batches without valid decision/coupling positions.
                            probe = sgrc_module(h_visual, h_visual)
                            loss = (
                                loss
                                + probe["h_prime"].sum() * 0.0
                                + probe["state_logits"].sum() * 0.0
                                + probe["couple_stop_logit"].sum() * 0.0
                            )
                else:
                    if args.response_only_logits:
                        labels = batch.pop("labels")
                        active_positions = labels.ne(-100).nonzero(as_tuple=False)
                        if active_positions.numel() == 0:
                            raise RuntimeError("response-only logits requires at least one supervised token")
                        first_target = int(active_positions[:, 1].min().item())
                        first_logit = max(0, first_target - 1)
                        logits_to_keep = int(labels.shape[1] - first_logit)
                        outputs = model(**batch, logits_to_keep=logits_to_keep)
                        shift_logits = outputs.logits[..., :-1, :].contiguous().float()
                        shift_labels = labels[..., first_logit + 1 :].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    else:
                        outputs = model(**batch)
                        loss = outputs.loss
                    if args.use_row_loss_weight and loss_weights is not None:
                        loss = loss * loss_weights.float().mean()
                loss = loss / args.grad_accum_steps
            loss.backward()
            epoch_loss += loss.detach().float().item()

            if step % args.grad_accum_steps == 0:
                do_optimizer_step(epoch, loss.detach().float().item() * args.grad_accum_steps)
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

        if not stop_training and len(train_loader) % args.grad_accum_steps != 0:
            remainder = len(train_loader) % args.grad_accum_steps
            scale = rescale_incomplete_accumulation(
                optim_params,
                expected_microbatches=args.grad_accum_steps,
                actual_microbatches=remainder,
            )
            do_optimizer_step(epoch, loss.detach().float().item() * args.grad_accum_steps * scale)
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True

        log_line(json.dumps({"event": "epoch_end", "epoch": epoch + 1, "epoch_loss": epoch_loss}, ensure_ascii=False))
        if run is not None:
            run.log({"train/epoch_loss": epoch_loss, "train/epoch": epoch + 1}, step=global_step)
        if (epoch + 1) in save_epoch_adapters:
            epoch_dir_name = f"epoch{epoch + 1}_{'model' if args.finetune_mode == 'full' else 'adapter'}"
            epoch_dir = output_dir / epoch_dir_name
            save_model_checkpoint(model, processor, epoch_dir, rank, full_state=args.finetune_mode == "full", sgrc_heads=sgrc_heads_map, sgrc_meta=sgrc_meta)
            log_line(
                json.dumps(
                    {
                        "event": "epoch_checkpoint",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "path": str(epoch_dir),
                    },
                    ensure_ascii=False,
                )
            )
        if is_dist() and (epoch + 1) in save_epoch_adapters:
            dist.barrier()

    if is_dist():
        dist.barrier()
    final_dir = output_dir / ("final_model" if args.finetune_mode == "full" else "final_adapter")
    save_model_checkpoint(model, processor, final_dir, rank, full_state=args.finetune_mode == "full", sgrc_heads=sgrc_heads_map, sgrc_meta=sgrc_meta)

    val_loss = last_val_loss if last_val_step == global_step else None
    if not args.disable_eval and last_val_step != global_step:
        if args.distributed_eval:
            val_loss = evaluate(
                unwrap_model(model), val_loader, device, args.bf16, distributed=True
            )
        elif args.finetune_mode == "full":
            val_loss = evaluate(model, val_loader, device, args.bf16, distributed=True)
        elif is_main:
            val_loss = evaluate(unwrap_model(model), val_loader, device, args.bf16)
    if is_main:
        with (output_dir / "final_metrics.json").open("w") as f:
            json.dump({"best_val_loss": best_val, "final_val_loss": val_loss, "global_step": global_step}, f, ensure_ascii=False, indent=2)
        log_line(json.dumps({"event": "done", "best_val_loss": best_val, "final_val_loss": val_loss, "global_step": global_step}, ensure_ascii=False))
    if is_dist():
        dist.barrier()
    if run is not None:
        final_log = {"final/best_val_loss": best_val, "global_step": global_step}
        if val_loss is not None:
            final_log["final/val_loss"] = val_loss
        run.log(final_log, step=global_step)
        final_artifact = wandb.Artifact("exact_only_vision_sft_outputs", type="model")
        final_artifact.add_dir(str(output_dir))
        run.log_artifact(final_artifact)
        run.finish()
    if log_fh is not None:
        log_fh.close()
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
