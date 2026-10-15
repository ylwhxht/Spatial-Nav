"""State-conditioned masked reasoning distillation primitives.

SC-MRD keeps the executable navigation surface on Qwen's native LM head.  A
training item may expose three views of the same current image:

* direct: image/goal/candidates -> compact decision;
* reason: image/goal/candidates -> action-blind spatial reasoning;
* reasoned: image/goal/candidates + gold reasoning -> compact decision.

Only the direct view exists at runtime.  This module intentionally contains no
decision classifier, latent-token loop, or simulator-facing feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable

import torch
import torch.nn.functional as F


PADDED_LEGAL_LOGIT = -1.0e9


ABILITIES = (
    "target_grounding",
    "relative_geometry",
    "spatial_relation",
    "local_affordance",
    "layout_topology",
)
ABILITY_SET = set(ABILITIES)
STATES = {"SEARCH", "TARGET_ENGAGED"}

ACTION_RE = re.compile(r"candidate_(0|[1-9][0-9]*)")
VALID_ACTION_LINE_RE = re.compile(
    r"Valid Move actions:\s*([^.\n]+)\.", re.IGNORECASE
)

# Spatial relations such as "left of the sofa" are legitimate.  The audit
# rejects executable policy language instead of banning geometric direction
# words wholesale.
ACTION_LEAK_PATTERNS = {
    "candidate_id": re.compile(r"\bcandidate(?:_-?\d+)?\b", re.IGNORECASE),
    "decision_field": re.compile(r"\b(?:action|decision|should_stop)\b", re.IGNORECASE),
    "turn_command": re.compile(
        r"\bturn(?:ing|ed)?\s+(?:left|right|around|back)\b", re.IGNORECASE
    ),
    "move_command": re.compile(
        r"\b(?:move|go|walk|head|proceed|advance|rotate)(?:s|d|ing)?\s+"
        r"(?:straight|forward|back(?:ward)?|left|right|toward|through|into|along)\b",
        re.IGNORECASE,
    ),
    "choice_command": re.compile(
        r"\b(?:choose|select|take|follow)\s+(?:the\s+)?(?:route|path|option|candidate)\b",
        re.IGNORECASE,
    ),
    "stop_command": re.compile(
        r"\b(?:stop|terminate)\s+(?:now|here|navigation|the episode)\b",
        re.IGNORECASE,
    ),
    "private_metric": re.compile(
        r"\b(?:distance_m|heading_deg|oracle|valid_stop|goal_reached)\b",
        re.IGNORECASE,
    ),
    "overlay_label": re.compile(r"\b(?:red[- ]?arrow|overlay id)\b", re.IGNORECASE),
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def legal_actions_from_prompt(prompt: str) -> list[str]:
    match = VALID_ACTION_LINE_RE.search(prompt)
    if match is None:
        raise ValueError("Move prompt has no public legal-action line")
    actions = list(dict.fromkeys(ACTION_RE.findall(match.group(1))))
    result = [f"candidate_{index}" for index in actions]
    if not result:
        raise ValueError("Move prompt has no legal candidate")
    if "candidate_-1" in prompt:
        raise ValueError("candidate_-1 is forbidden")
    return result


def move_action_from_response(response: str) -> str:
    action = decision_action_from_response(response)
    if re.fullmatch(r"candidate_(?:0|[1-9][0-9]*)", action) is None:
        raise ValueError(f"invalid direct Move response: {response!r}")
    return action


def decision_action_from_response(response: str) -> str:
    payload = parse_json_object(response)
    decision = payload.get("decision")
    action = str(decision.get("action") or "") if isinstance(decision, dict) else ""
    if not action:
        raise ValueError(f"decision has no action: {response!r}")
    return action


def direct_move_response(action: str) -> str:
    if re.fullmatch(r"candidate_(?:0|[1-9][0-9]*)", action) is None:
        raise ValueError(f"invalid Move action: {action!r}")
    return canonical_json({"decision": {"action": action}})


def direct_stop_response(action: str) -> str:
    if action not in {"Continue", "Stop"}:
        raise ValueError(f"invalid Stop-controller action: {action!r}")
    return canonical_json({"decision": {"action": action}})


def compact_reasoning_payload(response: Any) -> dict[str, Any]:
    """Project an audited long response to the five-ability action-blind view."""
    source = parse_json_object(response)
    state = str(source.get("navigation_state") or "")
    if state not in STATES:
        raise ValueError(f"invalid navigation state: {state!r}")
    raw_abilities = source.get("ability", source.get("abilities"))
    evidence = source.get("spatial_evidence", source.get("spatial_reasoning"))
    if not isinstance(raw_abilities, list) or not isinstance(evidence, dict):
        raise ValueError("reasoning is missing abilities or spatial evidence")

    abilities: list[str] = []
    compact_evidence: dict[str, dict[str, str]] = {}
    for raw in raw_abilities:
        ability = str(raw)
        if ability not in ABILITY_SET or ability in abilities:
            continue
        item = evidence.get(ability)
        if not isinstance(item, dict):
            continue
        visual = str(
            item.get("visual_evidence", item.get("visual_observation", ""))
        ).strip()
        limit = str(item.get("evidence_limit", "")).strip()
        if not visual or not limit:
            continue
        abilities.append(ability)
        compact_evidence[ability] = {
            "visual_evidence": visual,
            "evidence_limit": limit,
        }
        if len(abilities) == 3:
            break
    if not abilities:
        raise ValueError("no supported action-blind spatial ability remains")
    result = {
        "navigation_state": state,
        "ability": abilities,
        "spatial_evidence": compact_evidence,
    }
    leaks = reasoning_leak_hits(result)
    if leaks:
        raise ValueError("reasoning leaks executable policy: " + ",".join(leaks))
    return result


def reasoning_leak_hits(reasoning: Any) -> list[str]:
    text = canonical_json(reasoning)
    return sorted(
        name for name, pattern in ACTION_LEAK_PATTERNS.items() if pattern.search(text)
    )


def reasoning_prompt(goal: str, legal_actions: Iterable[str]) -> str:
    actions = ", ".join(legal_actions)
    abilities = ", ".join(ABILITIES)
    return (
        "Protocol: objectnav_sc_mrd_reason_v1\n\n"
        f"Goal object: nearest {goal}.\n"
        f"Valid Move actions: {actions}.\n"
        "Use only the current candidate-overlay RGB, the public goal, and visible "
        "scene content. Infer the navigation state and select one to three useful "
        "spatial abilities. For every selected ability, state one grounded visual "
        "fact and what this single view cannot establish. Do not choose, name, "
        "describe, or imply an action, candidate, turn, Stop decision, or route "
        "command.\n\n"
        "States: SEARCH, TARGET_ENGAGED.\n"
        f"Abilities: {abilities}.\n\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"navigation_state":"SEARCH|TARGET_ENGAGED","ability":["one_to_three"],'
        '"spatial_evidence":{"ability_name":{"visual_evidence":"...",'
        '"evidence_limit":"..."}}}'
    )


def reasoned_move_prompt(
    goal: str,
    legal_actions: Iterable[str],
    reasoning: dict[str, Any],
) -> str:
    actions = ", ".join(legal_actions)
    return (
        "Protocol: objectnav_sc_mrd_reasoned_move_v1\n\n"
        f"Goal object: nearest {goal}.\n"
        f"Valid Move actions: {actions}.\n"
        "Use the current candidate-overlay RGB, public goal, and the following "
        "training-only action-blind spatial deliberation to choose one legal "
        "Move action. A separate visual Stop controller owns termination.\n\n"
        "Spatial deliberation:\n"
        f"{canonical_json(reasoning)}\n\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"decision":{"action":"candidate_<number>"}}'
    )


def direct_stop_prompt(goal: str) -> str:
    return (
        "Protocol: objectnav_sc_mrd_direct_stop_v1\n\n"
        f"Goal object: nearest {goal}.\n"
        "Use only the current raw egocentric RGB image and the public goal. "
        "Decide whether this viewpoint is visually sufficient to terminate. "
        "A separate Move controller owns candidate selection.\n\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"decision":{"action":"Continue_or_Stop"}}'
    )


def stop_reasoning_prompt(goal: str) -> str:
    abilities = ", ".join(ABILITIES)
    return (
        "Protocol: objectnav_sc_mrd_stop_reason_v1\n\n"
        f"Goal object: nearest {goal}.\n"
        "Use only the current raw egocentric RGB image and the public goal. "
        "Infer the navigation state and select one to three useful spatial "
        "abilities. For every selected ability, state one grounded visual fact "
        "and what this single view cannot establish. Do not output or imply a "
        "terminal decision, Continue/Stop label, movement command, candidate, "
        "or answer synonym.\n\n"
        "States: SEARCH, TARGET_ENGAGED.\n"
        f"Abilities: {abilities}.\n\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"navigation_state":"SEARCH|TARGET_ENGAGED","ability":["one_to_three"],'
        '"spatial_evidence":{"ability_name":{"visual_evidence":"...",'
        '"evidence_limit":"..."}}}'
    )


def reasoned_stop_prompt(goal: str, reasoning: dict[str, Any]) -> str:
    return (
        "Protocol: objectnav_sc_mrd_reasoned_stop_v1\n\n"
        f"Goal object: nearest {goal}.\n"
        "Use the current raw egocentric RGB image, public goal, and the "
        "following training-only action-blind spatial deliberation to choose "
        "one legal terminal-controller action: Continue or Stop. A separate "
        "Move controller owns candidate selection.\n\n"
        "Spatial deliberation:\n"
        f"{canonical_json(reasoning)}\n\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"decision":{"action":"Continue_or_Stop"}}'
    )


@dataclass(frozen=True)
class ActionTokenSpec:
    """Native-LM token surface for one constrained legal decision."""

    response_token_offset: int
    legal_token_ids: tuple[int, ...]
    legal_scoring_sequences: tuple[tuple[int, ...], ...]
    gold_index: int


def action_token_spec(
    tokenizer: Any,
    response: str,
    legal_actions: Iterable[str],
) -> ActionTokenSpec:
    """Return an exact native-LM class surface for Move or binary Stop.

    Qwen3-VL tokenizes the common ``candidate_`` prefix identically. Common
    single-digit ids, ``Continue``, and ``Stop`` use one token. Multi-digit
    candidate ids are represented by a minimal prefix-free token sequence so
    the full legal set remains a well-defined native-LM distribution.
    """
    actions = list(legal_actions)
    payload = parse_json_object(response)
    if set(actions) == {"0", "1"} and len(actions) == 2:
        done = payload.get("done")
        if isinstance(done, bool) or not isinstance(done, int) or done not in (0, 1):
            raise ValueError("response has no integer binary done value")
        action = str(done)
    else:
        action = decision_action_from_response(response)
    if action not in actions:
        raise ValueError("gold action is not legal")
    candidate_actions = all(
        re.fullmatch(r"candidate_(?:0|[1-9][0-9]*)", candidate)
        for candidate in actions
    )
    if candidate_actions:
        matches = list(
            re.finditer(
                r'"action"\s*:\s*"candidate_(?P<value>[0-9]+)"',
                response,
            )
        )
        match = matches[-1] if matches else None
        if match is None:
            raise ValueError("response has no JSON decision.action candidate value")
        index_start, index_end = match.span("value")
        token_surfaces = [
            re.fullmatch(r"candidate_([0-9]+)", candidate)
            for candidate in actions
        ]
        legal_values = [
            value.group(1) for value in token_surfaces if value is not None
        ]
    elif set(actions) == {"Continue", "Stop"} and len(actions) == 2:
        match = re.search(
            r'"action"\s*:\s*"(?P<value>Continue|Stop)"', response
        )
        if match is None:
            raise ValueError("response has no binary Stop-controller value")
        index_start, index_end = match.span("value")
        legal_values = actions
    elif set(actions) == {"0", "1"} and len(actions) == 2:
        match = re.search(r'"done"\s*:\s*(?P<value>[01])', response)
        if match is None:
            raise ValueError("response has no binary done value")
        index_start, index_end = match.span("value")
        legal_values = actions
    else:
        raise ValueError(f"unsupported SC-MRD legal-action surface: {actions}")
    prefix_ids = tokenizer(
        response[:index_start], add_special_tokens=False
    ).get("input_ids", [])
    gold_ids = tokenizer(
        response[index_start:index_end], add_special_tokens=False
    ).get("input_ids", [])
    suffix_ids = tokenizer(
        response[index_end:], add_special_tokens=False
    ).get("input_ids", [])
    response_ids = tokenizer(response, add_special_tokens=False).get("input_ids", [])
    if not gold_ids:
        raise ValueError("SC-MRD paired action has an empty token sequence")
    if response_ids != [*prefix_ids, *gold_ids, *suffix_ids]:
        raise ValueError("candidate id is not an independently aligned token span")

    raw_sequences: list[tuple[int, ...]] = []
    for value in legal_values:
        ids = tokenizer(response[:index_start] + value, add_special_tokens=False).get(
            "input_ids", []
        )
        if ids[: len(prefix_ids)] != prefix_ids or len(ids) == len(prefix_ids):
            raise ValueError(f"legal action is not prefix aligned: {value}")
        raw_sequences.append(tuple(int(token) for token in ids[len(prefix_ids) :]))
    if len(set(raw_sequences)) != len(raw_sequences):
        raise ValueError("legal actions do not map to unique token sequences")

    scoring_sequences: list[tuple[int, ...]] = []
    for value, sequence in zip(legal_values, raw_sequences):
        is_prefix = any(
            len(other) > len(sequence) and other[: len(sequence)] == sequence
            for other in raw_sequences
        )
        if is_prefix:
            terminated = tokenizer(
                response[:index_start] + value + '"', add_special_tokens=False
            ).get("input_ids", [])
            if terminated[: len(prefix_ids)] != prefix_ids:
                raise ValueError("candidate terminator is not prefix aligned")
            terminated_sequence = tuple(
                int(token) for token in terminated[len(prefix_ids) :]
            )
            if (
                terminated_sequence[: len(sequence)] != sequence
                or len(terminated_sequence) != len(sequence) + 1
            ):
                raise ValueError("candidate terminator must add exactly one token")
            sequence = terminated_sequence
        scoring_sequences.append(sequence)
    if any(
        left != right and len(right) >= len(left) and right[: len(left)] == left
        for left in scoring_sequences
        for right in scoring_sequences
    ):
        raise ValueError("legal scoring sequences are not prefix-free")
    return ActionTokenSpec(
        response_token_offset=len(prefix_ids),
        legal_token_ids=tuple(sequence[0] for sequence in scoring_sequences),
        legal_scoring_sequences=tuple(scoring_sequences),
        gold_index=actions.index(action),
    )


def legal_action_logits(
    logits: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather padded legal-action logits from the original LM head."""
    if logits.ndim != 3:
        raise ValueError("expected (batch, sequence, vocabulary) logits")
    if positions.shape != (logits.shape[0],):
        raise ValueError("one native-LM action position is required per row")
    if token_ids.shape != token_mask.shape or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("legal token tensors do not align with the batch")
    safe_positions = positions.clamp(0, logits.shape[1] - 1)
    row = torch.arange(logits.shape[0], device=logits.device)
    position_logits = logits[row, safe_positions].float()
    gathered = position_logits.gather(1, token_ids.clamp_min(0))
    # Mixed-precision Qwen3.5 logits can transiently overflow on a fresh
    # delta adapter.  Keep every downstream margin/KD computation finite;
    # padded candidates remain effectively impossible after the clamp.
    gathered = torch.nan_to_num(
        gathered, nan=0.0, posinf=1.0e4, neginf=-1.0e4
    ).clamp(-1.0e4, 1.0e4)
    return gathered.masked_fill(~token_mask, PADDED_LEGAL_LOGIT)


def masked_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student legal-action surfaces must match")
    if teacher_logits.ndim != 2 or teacher_logits.shape[0] == 0:
        raise ValueError("distillation requires at least one teacher row")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Compute KL only over the actual legal-action support.  With Qwen3.5 the
    # distribution is sometimes sharp enough that a legal action underflows
    # to exactly zero probability.  ``F.kl_div`` may then evaluate the
    # mathematically-zero term as ``0 * log(0)`` and return NaN.  Explicitly
    # masking zero-probability terms preserves the exact KL definition and its
    # gradient without silently dropping an entire training example.
    teacher_input = teacher_logits.detach().float()
    student_input = student_logits.float()
    legal = teacher_input.gt(PADDED_LEGAL_LOGIT / 2.0) & student_input.gt(
        PADDED_LEGAL_LOGIT / 2.0
    )
    if not legal.any(dim=-1).all():
        raise ValueError("every distillation row requires a legal action")
    if not torch.isfinite(teacher_input[legal]).all() or not torch.isfinite(
        student_input[legal]
    ).all():
        raise FloatingPointError("legal distillation logits must be finite")
    negative_infinity = torch.tensor(
        float("-inf"), device=student_input.device, dtype=torch.float32
    )
    teacher_scaled = (teacher_input / temperature).masked_fill(
        ~legal, negative_infinity
    )
    student_scaled = (student_input / temperature).masked_fill(
        ~legal, negative_infinity
    )
    teacher_log_probability = F.log_softmax(teacher_scaled, dim=-1)
    student_log_probability = F.log_softmax(student_scaled, dim=-1)
    teacher_probability = teacher_log_probability.exp()
    active = legal & teacher_probability.gt(0)
    log_ratio = (teacher_log_probability - student_log_probability).masked_fill(
        ~active, 0.0
    )
    per_example = (teacher_probability * log_ratio).sum(dim=-1)
    per_example = per_example * float(temperature) ** 2
    if not torch.isfinite(per_example).all():
        raise FloatingPointError("distillation loss must be finite")
    if example_weights is None:
        return per_example.mean()
    if example_weights.shape != per_example.shape:
        raise ValueError("one distillation weight is required per example")
    weights = example_weights.detach().to(
        device=per_example.device, dtype=per_example.dtype
    )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("distillation weights must be finite and nonnegative")
    return (per_example * weights).mean()


def ema_sol_match_weights(
    teacher_logits: torch.Tensor,
    sol_indices: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    temperature: float = 0.5,
    margin_center: float = 0.0,
    weight_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return continuous full-KL weights from an online Teacher surface.

    ``teacher_logits`` is the privileged (Bridge) legal-action surface while
    ``sol_indices`` identifies the final action emitted by Sol.  The returned
    weight is a smooth function of the Teacher's Sol-action margin, so a
    Teacher that disagrees with Sol is retained for CE but contributes little
    or no KD pressure.  Rows with fewer than two legal candidates have no
    distributional information and receive zero weight.

    Returns ``(weights, match_score, sol_margin)``.  ``match_score`` is the
    sigmoid-calibrated agreement in ``[0, 1]`` and is diagnostic only.
    """
    if teacher_logits.ndim != 2 or sol_indices.ndim != 1:
        raise ValueError("invalid Teacher/Sol action surface dimensions")
    if teacher_logits.shape != (sol_indices.shape[0], legal_mask.shape[1]):
        raise ValueError("Teacher/Sol action surfaces do not align")
    if legal_mask.shape != teacher_logits.shape:
        raise ValueError("legal-action mask does not align with EMA logits")
    if (
        temperature <= 0
        or not math.isfinite(margin_center)
        or not 0.0 <= weight_floor < 1.0
    ):
        raise ValueError("invalid Teacher/Sol match calibration")
    counts = legal_mask.sum(dim=-1)
    valid = counts.ge(2)
    safe_indices = sol_indices.clamp_min(0).clamp_max(teacher_logits.shape[1] - 1)
    row = torch.arange(teacher_logits.shape[0], device=teacher_logits.device)
    sol_logit = teacher_logits[row, safe_indices]
    masked = teacher_logits.masked_fill(~legal_mask, float("-inf"))
    wrong = masked.clone()
    wrong[row, safe_indices] = float("-inf")
    best_wrong = wrong.max(dim=-1).values
    finite_wrong = torch.isfinite(best_wrong)
    margin = torch.where(finite_wrong, sol_logit - best_wrong, torch.zeros_like(sol_logit))
    match = torch.sigmoid(
        (margin - float(margin_center)) / float(temperature)
    )
    weights = float(weight_floor) + (1.0 - float(weight_floor)) * match
    # Do not hard-filter disagreement rows: the sigmoid is the adaptive
    # reliability mechanism.  A Teacher that disagrees with Sol contributes a
    # small but measurable soft signal, while a confident Sol match receives
    # nearly full weight.  Only forced single-candidate rows are zeroed.
    weights = torch.where(valid & finite_wrong, weights, torch.zeros_like(weights))
    return weights, match, margin


def action_marginal_residual_targets(
    teacher_logits: torch.Tensor,
    anchor_direct_logits: torch.Tensor,
    action_group_indices: torch.Tensor,
    legal_mask: torch.Tensor,
    example_weights: torch.Tensor,
    *,
    maximum_iterations: int = 256,
    tolerance: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove row and global action effects from privileged logit residuals.

    A Bridge Teacher can carry a global candidate-index preference in addition
    to sample-specific spatial information.  With a shared LM action token,
    row-level safety gates do not prevent that global preference from leaking
    into every example.  This routine fits the weighted additive model

    ``teacher - anchor = row_effect + action_effect + residual``

    over legal actions only.  The returned residual has zero mean within every
    row and zero weighted mean within every observed action group.  Therefore
    a squared-error residual objective has no initial global action-token
    gradient; it transfers only sample-specific deviations from the action
    marginal.  Zero-weight rows do not influence the fitted action effects,
    but receive residual targets under the fitted effects for auditing.
    """
    if teacher_logits.shape != anchor_direct_logits.shape:
        raise ValueError("teacher/anchor legal-action surfaces must match")
    if teacher_logits.ndim != 2:
        raise ValueError("action residual calibration requires rank-2 logits")
    if action_group_indices.shape != teacher_logits.shape:
        raise ValueError("action groups must align with legal-action logits")
    if legal_mask.shape != teacher_logits.shape or legal_mask.dtype != torch.bool:
        raise ValueError("legal mask must be boolean and align with logits")
    if example_weights.shape != (teacher_logits.shape[0],):
        raise ValueError("one residual calibration weight is required per row")
    if maximum_iterations <= 0 or tolerance <= 0:
        raise ValueError("residual calibration iteration controls must be positive")
    weights = example_weights.detach().float()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("residual calibration weights must be finite and nonnegative")
    if not weights.gt(0).any():
        raise ValueError("residual calibration requires at least one active row")
    groups = action_group_indices.detach().long()
    if bool((groups[legal_mask] < 0).any()):
        raise ValueError("legal actions require nonnegative action group ids")
    if not torch.isfinite(teacher_logits[legal_mask]).all() or not torch.isfinite(
        anchor_direct_logits[legal_mask]
    ).all():
        raise ValueError("legal residual calibration logits must be finite")

    # CPU float64 makes the offline fixed-effect fit deterministic and avoids
    # contaminating the training graph.  Candidate ids are dense nonnegative
    # integers in the formal Move course.
    mask = legal_mask.detach().cpu()
    group_cpu = groups.detach().cpu()
    delta = (
        teacher_logits.detach().double().cpu()
        - anchor_direct_logits.detach().double().cpu()
    ).masked_fill(~mask, 0.0)
    row_weights = weights.double().cpu()
    number_groups = int(group_cpu[mask].max()) + 1
    action_effects = torch.zeros(number_groups, dtype=torch.float64)
    counts = mask.sum(dim=-1).clamp_min(1).double()
    # The residual loss averages legal actions inside each row before applying
    # the example weight, so every legal cell carries weight w_i / |A_i|.
    cell_weight_rows = row_weights / counts
    row_effects = torch.zeros(delta.shape[0], dtype=torch.float64)
    for _ in range(maximum_iterations):
        gathered = action_effects[group_cpu.clamp_min(0)]
        row_effects = ((delta - gathered) * mask).sum(dim=-1) / counts
        updated = torch.zeros_like(action_effects)
        for group in range(number_groups):
            cells = mask & group_cpu.eq(group)
            cell_weights = cell_weight_rows[:, None].expand_as(delta)[cells]
            denominator = cell_weights.sum()
            if denominator > 0:
                centered_delta = delta - row_effects[:, None]
                updated[group] = (
                    centered_delta[cells] * cell_weights
                ).sum() / denominator
        # Additive row/action effects have one free gauge.  Pin the first
        # observed group without changing the fitted residual.
        observed = [
            group
            for group in range(number_groups)
            if bool((mask & group_cpu.eq(group) & row_weights[:, None].gt(0)).any())
        ]
        gauge = updated[observed[0]]
        updated = updated - gauge
        if float((updated - action_effects).abs().max()) <= tolerance:
            action_effects = updated
            break
        action_effects = updated
    gathered = action_effects[group_cpu.clamp_min(0)]
    row_effects = ((delta - gathered) * mask).sum(dim=-1) / counts
    residual = (delta - row_effects[:, None] - gathered).masked_fill(~mask, 0.0)
    return (
        residual.float().to(teacher_logits.device),
        action_effects.float().to(teacher_logits.device),
        row_effects.float().to(teacher_logits.device),
    )


def apply_action_marginal_residual_effects(
    teacher_logits: torch.Tensor,
    anchor_direct_logits: torch.Tensor,
    action_group_indices: torch.Tensor,
    legal_mask: torch.Tensor,
    action_effects: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply train-fitted action effects to another split and remove row means."""
    if not (
        teacher_logits.shape
        == anchor_direct_logits.shape
        == action_group_indices.shape
        == legal_mask.shape
    ):
        raise ValueError("residual target inputs must have matching shapes")
    if legal_mask.dtype != torch.bool:
        raise ValueError("legal mask must be boolean")
    groups = action_group_indices.long()
    if bool((groups[legal_mask] < 0).any()) or (
        legal_mask.any() and int(groups[legal_mask].max()) >= action_effects.numel()
    ):
        raise ValueError("action effect table does not cover legal action groups")
    delta = (teacher_logits.detach().float() - anchor_direct_logits.detach().float()).masked_fill(
        ~legal_mask, 0.0
    )
    gathered = action_effects.detach().float().to(delta.device)[groups.clamp_min(0)]
    counts = legal_mask.sum(dim=-1).clamp_min(1).to(delta.dtype)
    row_effects = ((delta - gathered) * legal_mask).sum(dim=-1) / counts
    residual = (delta - row_effects[:, None] - gathered).masked_fill(
        ~legal_mask, 0.0
    )
    return residual, row_effects


def masked_residual_logit_distillation_loss(
    target_residual: torch.Tensor,
    student_logits: torch.Tensor,
    anchor_direct_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match sample-specific privileged logit residuals in the Direct view.

    Both the current Student-minus-anchor delta and the fixed target are
    row-centered over legal actions.  This makes the objective invariant to a
    common logit shift and keeps it on the complete legal-action surface.
    """
    if not (
        target_residual.shape
        == student_logits.shape
        == anchor_direct_logits.shape
        == legal_mask.shape
    ):
        raise ValueError("residual KD surfaces and legal mask must match")
    if legal_mask.dtype != torch.bool:
        raise ValueError("residual KD legal mask must be boolean")
    mask = legal_mask.to(student_logits.device)
    counts = mask.sum(dim=-1).clamp_min(1).to(torch.float32)
    student_delta = student_logits.float() - anchor_direct_logits.detach().float()
    student_mean = (student_delta.masked_fill(~mask, 0.0).sum(dim=-1) / counts)
    student_residual = (student_delta - student_mean[:, None]).masked_fill(~mask, 0.0)
    target = target_residual.detach().float().to(student_logits.device)
    target_mean = target.masked_fill(~mask, 0.0).sum(dim=-1) / counts
    target = (target - target_mean[:, None]).masked_fill(~mask, 0.0)
    per_example = ((student_residual - target).square() * mask).sum(dim=-1) / counts
    if example_weights is None:
        return per_example.mean()
    if example_weights.shape != per_example.shape:
        raise ValueError("one residual KD weight is required per example")
    weights = example_weights.detach().to(per_example)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("residual KD weights must be finite and nonnegative")
    return (per_example * weights).mean()


def masked_representation_residual_distillation_loss(
    target_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    anchor_hidden: torch.Tensor,
    *,
    example_weights: torch.Tensor | None = None,
    minimum_scale: float = 1.0e-3,
) -> torch.Tensor:
    """Distill the Bridge-specific decision-state change into Direct.

    The frozen Direct anchor is subtracted from both views, so the objective
    transfers only what the privileged Bridge adds rather than forcing the
    Student to copy prompt-length- or token-identity-dependent absolute
    activations.  Per-row RMS normalization makes the loss measure direction
    and relative feature change instead of letting a few high-norm examples
    dominate.  This is ordinary supervised representation SFT/KD.
    """
    if not (
        target_hidden.shape == student_hidden.shape == anchor_hidden.shape
    ):
        raise ValueError("representation KD hidden surfaces must match")
    if target_hidden.ndim != 2:
        raise ValueError("representation KD expects (examples, hidden) tensors")
    target_delta = target_hidden.detach().float() - anchor_hidden.detach().float()
    student_delta = student_hidden.float() - anchor_hidden.detach().float()
    scale = target_delta.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
        float(minimum_scale)
    )
    per_example = ((student_delta - target_delta) / scale).square().mean(dim=-1)
    if example_weights is None:
        return per_example.mean()
    if example_weights.shape != per_example.shape:
        raise ValueError("one representation KD weight is required per example")
    weights = example_weights.detach().to(per_example)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("representation KD weights must be finite and nonnegative")
    return (per_example * weights).mean()


def masked_bridge_increment_representation_distillation_loss(
    bridge_hidden: torch.Tensor,
    no_bridge_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    direct_anchor_hidden: torch.Tensor,
    *,
    example_weights: torch.Tensor | None = None,
    minimum_scale: float = 1.0e-3,
) -> torch.Tensor:
    """Distill only the Bridge increment over the existing spatial prior."""
    if not (
        bridge_hidden.shape
        == no_bridge_hidden.shape
        == student_hidden.shape
        == direct_anchor_hidden.shape
    ):
        raise ValueError("Bridge-increment representation surfaces must match")
    if bridge_hidden.ndim != 2:
        raise ValueError("Bridge-increment representation KD expects 2D tensors")
    target_delta = bridge_hidden.detach().float() - no_bridge_hidden.detach().float()
    student_delta = student_hidden.float() - direct_anchor_hidden.detach().float()
    scale = target_delta.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
        float(minimum_scale)
    )
    per_example = ((student_delta - target_delta) / scale).square().mean(dim=-1)
    if example_weights is None:
        return per_example.mean()
    if example_weights.shape != per_example.shape:
        raise ValueError("one Bridge-increment KD weight is required per example")
    weights = example_weights.detach().to(per_example)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Bridge-increment KD weights must be finite and nonnegative")
    return (per_example * weights).mean()


def masked_action_projected_bridge_increment_loss(
    bridge_hidden: torch.Tensor,
    no_bridge_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    direct_anchor_hidden: torch.Tensor,
    legal_head_vectors: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    example_weights: torch.Tensor | None = None,
    minimum_scale: float = 1.0e-3,
) -> torch.Tensor:
    """Transfer the minimal action-visible Bridge increment.

    For each row, centered legal-action LM-head vectors span exactly the final
    hidden-state directions that can change relative legal-action scores.  The
    Bridge-minus-no-Bridge contrast is projected into that span.  Matching the
    full Student delta to this minimal projection transfers the causal action
    signal while explicitly retaining the Direct anchor in the orthogonal
    (decision-invisible) subspace.
    """
    if not (
        bridge_hidden.shape
        == no_bridge_hidden.shape
        == student_hidden.shape
        == direct_anchor_hidden.shape
    ):
        raise ValueError("action-projected Bridge hidden surfaces must match")
    if bridge_hidden.ndim != 2 or legal_head_vectors.ndim != 3:
        raise ValueError("action-projected Bridge KD expects 2D hidden and 3D heads")
    if legal_head_vectors.shape[0] != bridge_hidden.shape[0] or (
        legal_head_vectors.shape[2] != bridge_hidden.shape[1]
    ):
        raise ValueError("legal head vectors do not match hidden states")
    if legal_mask.shape != legal_head_vectors.shape[:2] or legal_mask.dtype != torch.bool:
        raise ValueError("legal head mask must be boolean and match action vectors")

    target_delta = bridge_hidden.detach().float() - no_bridge_hidden.detach().float()
    projected_rows: list[torch.Tensor] = []
    supported_rows: list[bool] = []
    for index in range(target_delta.shape[0]):
        heads = legal_head_vectors[index, legal_mask[index]].detach().float()
        if heads.shape[0] <= 1:
            projected_rows.append(torch.zeros_like(target_delta[index]))
            supported_rows.append(False)
            continue
        centered_heads = heads - heads.mean(dim=0, keepdim=True)
        _u, singular, vh = torch.linalg.svd(centered_heads, full_matrices=False)
        threshold = singular.max() * max(centered_heads.shape) * torch.finfo(
            singular.dtype
        ).eps
        rank = int((singular > threshold).sum())
        if rank == 0:
            projected_rows.append(torch.zeros_like(target_delta[index]))
            supported_rows.append(False)
            continue
        basis = vh[:rank]
        projected_rows.append(basis.T @ (basis @ target_delta[index]))
        supported_rows.append(True)
    target = torch.stack(projected_rows)
    student_delta = student_hidden.float() - direct_anchor_hidden.detach().float()
    scale = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
        float(minimum_scale)
    )
    per_example = ((student_delta - target) / scale).square().mean(dim=-1)
    if example_weights is None:
        weights = torch.tensor(supported_rows, device=per_example.device).to(per_example)
        return (per_example * weights).mean()
    if example_weights.shape != per_example.shape:
        raise ValueError("one action-projected Bridge KD weight is required per example")
    weights = example_weights.detach().to(per_example)
    weights = weights * torch.tensor(
        supported_rows, device=per_example.device
    ).to(per_example)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("action-projected Bridge KD weights must be finite and nonnegative")
    return (per_example * weights).mean()


def online_safe_full_distribution_weights(
    teacher_logits: torch.Tensor,
    anchor_direct_logits: torch.Tensor,
    student_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    *,
    temperature: float = 2.0,
    minimum_static_gold_logp_gain: float = 0.0,
    dynamic_gold_logp_gap_scale: float = 0.5,
    active_weight_floor: float = 0.25,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Gate full-distribution KD by causal validity and current need.

    The frozen privileged Teacher must first improve the externally supplied
    action's log probability over the *same frozen model* in Direct mode.
    This static comparison prevents KD from activating merely because the
    trainable Student later degrades.  Among valid rows, the Teacher-minus-
    current-Student gap is recomputed every forward pass, so KD turns off once
    the Direct Student catches up.  The returned weights are intended for
    :func:`masked_distillation_loss`; this function never replaces the full
    legal-action distribution with a pairwise gold-action objective.

    The caller is responsible for applying any independent Teacher correctness
    and margin requirements.  Returns detached ``(weights, static_valid,
    active, static_gain, dynamic_gap)``.
    """
    if not (
        teacher_logits.shape == anchor_direct_logits.shape == student_logits.shape
    ):
        raise ValueError("teacher/anchor/student legal-action surfaces must match")
    if teacher_logits.ndim != 2 or gold_indices.shape != (teacher_logits.shape[0],):
        raise ValueError("invalid safe full-distribution KD inputs")
    values = (
        temperature,
        minimum_static_gold_logp_gain,
        dynamic_gold_logp_gap_scale,
        active_weight_floor,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("safe full-distribution KD parameters must be finite")
    if temperature <= 0 or dynamic_gold_logp_gap_scale <= 0:
        raise ValueError("KD temperature and dynamic gap scale must be positive")
    if not 0.0 <= active_weight_floor <= 1.0:
        raise ValueError("active KD weight floor must be in [0, 1]")

    teacher = teacher_logits.detach().float()
    anchor = anchor_direct_logits.detach().float()
    student = student_logits.detach().float()
    row = torch.arange(teacher.shape[0], device=teacher.device)
    teacher_gold_logp = F.log_softmax(teacher / temperature, dim=-1)[
        row, gold_indices
    ]
    anchor_gold_logp = F.log_softmax(anchor / temperature, dim=-1)[
        row, gold_indices
    ]
    student_gold_logp = F.log_softmax(student / temperature, dim=-1)[
        row, gold_indices
    ]
    static_gain = teacher_gold_logp - anchor_gold_logp
    dynamic_gap = teacher_gold_logp - student_gold_logp
    static_valid = static_gain.gt(float(minimum_static_gold_logp_gain))
    active = static_valid & dynamic_gap.gt(0.0)
    strength = (
        dynamic_gap.clamp_min(0.0) / float(dynamic_gold_logp_gap_scale)
    ).clamp(max=1.0)
    weights = torch.where(
        active,
        strength.clamp_min(float(active_weight_floor)),
        torch.zeros_like(strength),
    )
    return (
        weights.detach(),
        static_valid.detach(),
        active.detach(),
        static_gain.detach(),
        dynamic_gap.detach(),
    )


def online_advantage_kd_weights(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    *,
    minimum_margin_advantage: float = 0.5,
    repair_weight: float = 1.0,
    confidence_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build detached repair-first KD weights from the current Student.

    A reliable Teacher should not perturb an already stronger Student merely
    because the row passed an offline gate.  Wrong Student decisions receive
    the full repair weight.  Correct Student decisions receive the lower
    confidence-shaping weight only when the Teacher has both a larger gold
    margin and a lower gold-action NLL.  All other rows are disabled.

    Returns ``(weights, repair, confidence, margin_gain, gold_logp_gain)``.
    The diagnostics are detached so the gate itself never becomes an
    optimization path.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student legal-action surfaces must match")
    if teacher_logits.ndim != 2 or gold_indices.shape != (teacher_logits.shape[0],):
        raise ValueError("invalid online KD advantage inputs")
    scalars = (
        minimum_margin_advantage,
        repair_weight,
        confidence_weight,
    )
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("online KD advantage parameters must be finite")
    if minimum_margin_advantage < 0:
        raise ValueError("minimum KD margin advantage must be nonnegative")
    if repair_weight < 0 or confidence_weight < 0:
        raise ValueError("online KD tier weights must be nonnegative")

    teacher_detached = teacher_logits.detach().float()
    student_detached = student_logits.detach().float()
    teacher_margin = gold_margin(teacher_detached, gold_indices)
    student_margin = gold_margin(student_detached, gold_indices)
    margin_gain = teacher_margin - student_margin

    row = torch.arange(teacher_logits.shape[0], device=teacher_logits.device)
    teacher_gold_logp = F.log_softmax(teacher_detached, dim=-1)[row, gold_indices]
    student_gold_logp = F.log_softmax(student_detached, dim=-1)[row, gold_indices]
    gold_logp_gain = teacher_gold_logp - student_gold_logp

    student_correct = student_detached.argmax(dim=-1).eq(gold_indices)
    repair = ~student_correct
    confidence = (
        student_correct
        & margin_gain.ge(float(minimum_margin_advantage))
        & gold_logp_gain.gt(0.0)
    )
    weights = torch.zeros_like(margin_gain)
    weights = torch.where(
        repair,
        torch.full_like(weights, float(repair_weight)),
        weights,
    )
    weights = torch.where(
        confidence,
        torch.full_like(weights, float(confidence_weight)),
        weights,
    )
    return (
        weights.detach(),
        repair.detach(),
        confidence.detach(),
        margin_gain.detach(),
        gold_logp_gain.detach(),
    )


def online_pairwise_adaptive_targets(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    *,
    minimum_margin_advantage: float = 0.25,
    repair_target_margin_cap: float = 0.5,
    confidence_increment_cap: float = 0.25,
    margin_weight_scale: float = 1.0,
    repair_weight: float = 1.0,
    confidence_weight: float = 0.25,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build on-policy pairwise KD targets from the current Student surface.

    The caller supplies rows that passed a detached reliable-Teacher gate; the
    Teacher may itself be an online EMA evaluated immediately before this
    function.  On every forward pass, the current Student chooses the
    competing wrong action.  A wrong Student is repaired toward a small
    positive margin; a correct Student is shaped only while the Teacher remains
    sufficiently stronger.  Both selection and target therefore evolve with
    training, while gradients never enter the Teacher branch.

    Returns detached ``(weights, repair, confidence, competitors, targets,
    margin_advantages)``.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student legal-action surfaces must match")
    if teacher_logits.ndim != 2 or gold_indices.shape != (teacher_logits.shape[0],):
        raise ValueError("invalid online pairwise KD inputs")
    scalars = (
        minimum_margin_advantage,
        repair_target_margin_cap,
        confidence_increment_cap,
        margin_weight_scale,
        repair_weight,
        confidence_weight,
    )
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("online pairwise KD parameters must be finite")
    if minimum_margin_advantage < 0:
        raise ValueError("minimum margin advantage must be nonnegative")
    if repair_target_margin_cap <= 0 or confidence_increment_cap <= 0:
        raise ValueError("online pairwise margin caps must be positive")
    if margin_weight_scale <= 0:
        raise ValueError("online pairwise margin weight scale must be positive")
    if repair_weight < 0 or confidence_weight < 0:
        raise ValueError("online pairwise tier weights must be nonnegative")

    teacher = teacher_logits.detach().float()
    student = student_logits.detach().float()
    batch = teacher.shape[0]
    row = torch.arange(batch, device=teacher.device)
    wrong_student = student.clone()
    wrong_student[row, gold_indices] = float("-inf")
    competitor_indices = wrong_student.argmax(dim=-1)
    competitor_scores = wrong_student[row, competitor_indices]
    has_competitor = torch.isfinite(competitor_scores) & (
        competitor_scores > PADDED_LEGAL_LOGIT / 2.0
    )
    student_pair_margin = (
        student[row, gold_indices] - student[row, competitor_indices]
    )
    teacher_pair_margin = (
        teacher[row, gold_indices] - teacher[row, competitor_indices]
    )
    margin_advantages = teacher_pair_margin - student_pair_margin
    student_correct = student.argmax(dim=-1).eq(gold_indices)
    repair = has_competitor & ~student_correct & margin_advantages.gt(0.0)
    confidence = (
        has_competitor
        & student_correct
        & margin_advantages.ge(float(minimum_margin_advantage))
    )

    continuous_strength = (
        margin_advantages.clamp_min(0.0) / float(margin_weight_scale)
    ).clamp(max=1.0)
    weights = torch.zeros_like(margin_advantages)
    weights = torch.where(
        repair,
        continuous_strength * float(repair_weight),
        weights,
    )
    weights = torch.where(
        confidence,
        continuous_strength * float(confidence_weight),
        weights,
    )

    repair_targets = teacher_pair_margin.clamp(
        min=0.0, max=float(repair_target_margin_cap)
    )
    confidence_targets = student_pair_margin + margin_advantages.clamp(
        min=0.0, max=float(confidence_increment_cap)
    )
    targets = torch.zeros_like(margin_advantages)
    targets = torch.where(repair, repair_targets, targets)
    targets = torch.where(confidence, confidence_targets, targets)
    active = repair | confidence
    competitor_indices = torch.where(active, competitor_indices, gold_indices)

    return (
        weights.detach(),
        repair.detach(),
        confidence.detach(),
        competitor_indices.detach(),
        targets.detach(),
        margin_advantages.detach(),
    )


def privileged_margin_rank_weights(
    teacher_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    minimum_margin: float = 0.1,
    minimum_weight: float = 0.5,
    maximum_weight: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weight correct privileged decisions by within-action margin rank.

    ``teacher_logits`` are produced by the frozen Student when it is given the
    training-only spatial reasoning/bridge.  A row is eligible only when that
    privileged view selects the externally annotated gold decision with at
    least ``minimum_margin``.  Eligible rows are ranked *within the same gold
    action group*, preventing naturally low-margin action positions from being
    systematically removed or downweighted.

    Ties receive their average rank.  With the default symmetric bounds, every
    non-singleton action group has mean weight exactly one, so this changes the
    allocation of KD rather than its nominal total mass.  Returns detached
    ``(weights, eligible, margins)``.
    """
    if teacher_logits.ndim != 2:
        raise ValueError("privileged logits must have shape (examples, actions)")
    batch = teacher_logits.shape[0]
    if gold_indices.shape != (batch,) or group_indices.shape != (batch,):
        raise ValueError("one gold and action-group index is required per example")
    scalars = (minimum_margin, minimum_weight, maximum_weight)
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("privileged-margin KD parameters must be finite")
    if minimum_weight < 0 or maximum_weight < minimum_weight:
        raise ValueError("privileged-margin KD weight bounds are invalid")

    logits = teacher_logits.detach().float()
    margins = gold_margin(logits, gold_indices).detach()
    correct = logits.argmax(dim=-1).eq(gold_indices)
    eligible = correct & margins.ge(float(minimum_margin))
    weights = torch.zeros_like(margins)
    midpoint = (float(minimum_weight) + float(maximum_weight)) / 2.0

    for group in torch.unique(group_indices.detach()):
        positions = (eligible & group_indices.eq(group)).nonzero(
            as_tuple=True
        )[0]
        count = int(positions.numel())
        if count == 0:
            continue
        if count == 1:
            weights[positions] = midpoint
            continue

        values = margins[positions]
        order = torch.argsort(values, stable=True)
        sorted_values = values[order]
        sorted_weights = torch.empty_like(sorted_values)
        start = 0
        while start < count:
            end = start + 1
            while end < count and bool(sorted_values[end].eq(sorted_values[start])):
                end += 1
            average_rank = (start + end - 1) / 2.0
            percentile = average_rank / float(count - 1)
            sorted_weights[start:end] = float(minimum_weight) + (
                float(maximum_weight) - float(minimum_weight)
            ) * percentile
            start = end
        weights[positions[order]] = sorted_weights

    return weights.detach(), eligible.detach(), margins.detach()


def privileged_correction_margin_targets(
    teacher_logits: torch.Tensor,
    anchor_direct_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    minimum_teacher_margin: float = 0.1,
    minimum_reasoning_gain: float = 0.25,
    target_margin_cap: float = 1.0,
    minimum_weight: float = 0.5,
    maximum_weight: float = 1.5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Calibrate repair-only privileged margins against the frozen Direct view.

    Full-distribution KL also copies how a privileged prompt allocates mass
    among irrelevant wrong actions.  This calibration instead retains only a
    causal-looking correction: the frozen Direct view is wrong, while the same
    frozen model with training-only reasoning selects ``gold_indices`` with a
    positive margin.  The competitor is fixed to the Direct view's mistaken
    top-1 action, and the target is only the Teacher's gold-vs-competitor
    margin, capped to avoid importing prompt-induced overconfidence.

    Eligible examples are ranked by privileged-minus-Direct pairwise margin
    gain within each gold-action group.  Returns detached
    ``(weights, eligible, competitor_indices, target_margins, margin_gains,
    teacher_gold_margins)``.
    """
    if teacher_logits.shape != anchor_direct_logits.shape:
        raise ValueError("privileged/Direct legal-action surfaces must match")
    if teacher_logits.ndim != 2:
        raise ValueError("correction logits must have shape (examples, actions)")
    batch = teacher_logits.shape[0]
    if gold_indices.shape != (batch,) or group_indices.shape != (batch,):
        raise ValueError("one gold and action-group index is required per example")
    scalars = (
        minimum_teacher_margin,
        minimum_reasoning_gain,
        target_margin_cap,
        minimum_weight,
        maximum_weight,
    )
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("correction-margin KD parameters must be finite")
    if minimum_reasoning_gain < 0:
        raise ValueError("minimum reasoning gain must be nonnegative")
    if target_margin_cap <= 0:
        raise ValueError("target margin cap must be positive")
    if minimum_weight < 0 or maximum_weight < minimum_weight:
        raise ValueError("correction-margin KD weight bounds are invalid")

    teacher = teacher_logits.detach().float()
    direct = anchor_direct_logits.detach().float()
    row = torch.arange(batch, device=teacher.device)
    wrong_direct = direct.clone()
    wrong_direct[row, gold_indices] = float("-inf")
    competitor_indices = wrong_direct.argmax(dim=-1)
    competitor_scores = wrong_direct[row, competitor_indices]
    has_competitor = torch.isfinite(competitor_scores) & (
        competitor_scores > PADDED_LEGAL_LOGIT / 2.0
    )

    direct_pair_margin = (
        direct[row, gold_indices] - direct[row, competitor_indices]
    )
    teacher_pair_margin = (
        teacher[row, gold_indices] - teacher[row, competitor_indices]
    )
    teacher_gold_margins = gold_margin(teacher, gold_indices)
    margin_gains = teacher_pair_margin - direct_pair_margin
    direct_correct = direct.argmax(dim=-1).eq(gold_indices)
    teacher_correct = teacher.argmax(dim=-1).eq(gold_indices)
    eligible = (
        has_competitor
        & ~direct_correct
        & teacher_correct
        & teacher_gold_margins.ge(float(minimum_teacher_margin))
        & margin_gains.ge(float(minimum_reasoning_gain))
    )
    # Inactive rows still travel through the batched loss with zero weight.
    # Point them at the always-valid gold column so variable-width batches
    # never expose a padding-only global calibration index.
    competitor_indices = torch.where(
        eligible, competitor_indices, gold_indices
    )
    target_margins = teacher_pair_margin.clamp(
        min=0.0, max=float(target_margin_cap)
    )
    target_margins = torch.where(
        eligible, target_margins, torch.zeros_like(target_margins)
    )

    weights = torch.zeros_like(margin_gains)
    midpoint = (float(minimum_weight) + float(maximum_weight)) / 2.0
    for group in torch.unique(group_indices.detach()):
        positions = (eligible & group_indices.eq(group)).nonzero(
            as_tuple=True
        )[0]
        count = int(positions.numel())
        if count == 0:
            continue
        if count == 1:
            weights[positions] = midpoint
            continue
        values = margin_gains[positions]
        order = torch.argsort(values, stable=True)
        sorted_values = values[order]
        sorted_weights = torch.empty_like(sorted_values)
        start = 0
        while start < count:
            end = start + 1
            while end < count and bool(sorted_values[end].eq(sorted_values[start])):
                end += 1
            average_rank = (start + end - 1) / 2.0
            percentile = average_rank / float(count - 1)
            sorted_weights[start:end] = float(minimum_weight) + (
                float(maximum_weight) - float(minimum_weight)
            ) * percentile
            start = end
        weights[positions[order]] = sorted_weights

    return (
        weights.detach(),
        eligible.detach(),
        competitor_indices.detach(),
        target_margins.detach(),
        margin_gains.detach(),
        teacher_gold_margins.detach(),
    )


def privileged_bridge_correction_margin_targets(
    teacher_logits: torch.Tensor,
    spatial_without_bridge_logits: torch.Tensor,
    anchor_direct_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    minimum_teacher_margin: float = 0.1,
    minimum_bridge_gain: float = 0.5,
    target_margin_cap: float = 0.5,
    minimum_weight: float = 0.5,
    maximum_weight: float = 1.5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Select corrections whose decision change is specific to the Bridge.

    A broad privileged-view gate cannot distinguish spatial evidence that was
    already sufficient from information added by the decision Bridge.  This
    stricter gate requires both the frozen Direct view and the action-blind
    spatial-only view to remain wrong, while the spatial-plus-Bridge Teacher is
    correct.  Bridge gain is measured on the gold-vs-frozen-Direct-error pair,
    so no unrelated wrong-action distribution is transferred.
    """
    if (
        teacher_logits.shape != spatial_without_bridge_logits.shape
        or teacher_logits.shape != anchor_direct_logits.shape
    ):
        raise ValueError("Bridge/ablation/Direct legal-action surfaces must match")
    if teacher_logits.ndim != 2:
        raise ValueError("Bridge correction logits must have shape (examples, actions)")
    batch = teacher_logits.shape[0]
    if gold_indices.shape != (batch,) or group_indices.shape != (batch,):
        raise ValueError("one gold and action-group index is required per example")
    scalars = (
        minimum_teacher_margin,
        minimum_bridge_gain,
        target_margin_cap,
        minimum_weight,
        maximum_weight,
    )
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("Bridge correction-margin parameters must be finite")
    if minimum_bridge_gain < 0:
        raise ValueError("minimum Bridge gain must be nonnegative")
    if target_margin_cap <= 0:
        raise ValueError("target margin cap must be positive")
    if minimum_weight < 0 or maximum_weight < minimum_weight:
        raise ValueError("Bridge correction-margin weight bounds are invalid")

    teacher = teacher_logits.detach().float()
    spatial = spatial_without_bridge_logits.detach().float()
    direct = anchor_direct_logits.detach().float()
    row = torch.arange(batch, device=teacher.device)
    wrong_direct = direct.clone()
    wrong_direct[row, gold_indices] = float("-inf")
    competitor_indices = wrong_direct.argmax(dim=-1)
    competitor_scores = wrong_direct[row, competitor_indices]
    has_competitor = torch.isfinite(competitor_scores) & (
        competitor_scores > PADDED_LEGAL_LOGIT / 2.0
    )

    teacher_pair_margin = (
        teacher[row, gold_indices] - teacher[row, competitor_indices]
    )
    spatial_pair_margin = (
        spatial[row, gold_indices] - spatial[row, competitor_indices]
    )
    bridge_gains = teacher_pair_margin - spatial_pair_margin
    teacher_gold_margins = gold_margin(teacher, gold_indices)
    direct_correct = direct.argmax(dim=-1).eq(gold_indices)
    spatial_correct = spatial.argmax(dim=-1).eq(gold_indices)
    teacher_correct = teacher.argmax(dim=-1).eq(gold_indices)
    eligible = (
        has_competitor
        & ~direct_correct
        & ~spatial_correct
        & teacher_correct
        & teacher_gold_margins.ge(float(minimum_teacher_margin))
        & bridge_gains.ge(float(minimum_bridge_gain))
    )
    competitor_indices = torch.where(eligible, competitor_indices, gold_indices)
    target_margins = teacher_pair_margin.clamp(
        min=0.0, max=float(target_margin_cap)
    )
    target_margins = torch.where(
        eligible, target_margins, torch.zeros_like(target_margins)
    )

    weights = torch.zeros_like(bridge_gains)
    midpoint = (float(minimum_weight) + float(maximum_weight)) / 2.0
    for group in torch.unique(group_indices.detach()):
        positions = (eligible & group_indices.eq(group)).nonzero(as_tuple=True)[0]
        count = int(positions.numel())
        if count == 0:
            continue
        if count == 1:
            weights[positions] = midpoint
            continue
        values = bridge_gains[positions]
        order = torch.argsort(values, stable=True)
        sorted_values = values[order]
        sorted_weights = torch.empty_like(sorted_values)
        start = 0
        while start < count:
            end = start + 1
            while end < count and bool(sorted_values[end].eq(sorted_values[start])):
                end += 1
            average_rank = (start + end - 1) / 2.0
            percentile = average_rank / float(count - 1)
            sorted_weights[start:end] = float(minimum_weight) + (
                float(maximum_weight) - float(minimum_weight)
            ) * percentile
            start = end
        weights[positions[order]] = sorted_weights

    return (
        weights.detach(),
        eligible.detach(),
        competitor_indices.detach(),
        target_margins.detach(),
        bridge_gains.detach(),
        teacher_gold_margins.detach(),
    )


def selective_pairwise_margin_loss(
    student_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    competitor_indices: torch.Tensor,
    target_margins: torch.Tensor,
    *,
    example_weights: torch.Tensor,
) -> torch.Tensor:
    """Hinge only the annotated action against a fixed Direct error action.

    The mean is intentionally over every supplied row, including zero-weight
    rows.  Consequently the nominal KD coefficient reflects both correction
    strength and selective coverage instead of silently renormalizing a small
    repair set into a full-dataset loss.
    """
    if student_logits.ndim != 2:
        raise ValueError("student logits must have shape (examples, actions)")
    batch = student_logits.shape[0]
    expected = (batch,)
    if any(
        value.shape != expected
        for value in (
            gold_indices,
            competitor_indices,
            target_margins,
            example_weights,
        )
    ):
        raise ValueError("one pairwise-margin target is required per example")
    if (competitor_indices < 0).any() or (
        competitor_indices >= student_logits.shape[1]
    ).any():
        raise ValueError("pairwise competitor index is out of bounds")
    weights = example_weights.detach().to(
        device=student_logits.device, dtype=student_logits.dtype
    )
    targets = target_margins.detach().to(
        device=student_logits.device, dtype=student_logits.dtype
    )
    if (
        not torch.isfinite(weights).all()
        or not torch.isfinite(targets).all()
        or (weights < 0).any()
        or (targets < 0).any()
    ):
        raise ValueError("pairwise margins and weights must be finite and nonnegative")
    row = torch.arange(batch, device=student_logits.device)
    student_margin = (
        student_logits[row, gold_indices]
        - student_logits[row, competitor_indices]
    )
    return (F.relu(targets - student_margin) * weights).mean()


def selective_pairwise_huber_margin_loss(
    student_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    competitor_indices: torch.Tensor,
    target_margins: torch.Tensor,
    *,
    example_weights: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """One-sided smooth margin correction with no force after the target.

    The loss is quadratic near the requested margin and linear only for large
    deficits.  Unlike a hard hinge, its gradient fades continuously as the
    Student reaches the Teacher-derived target.  The reduction remains over
    all rows so selective coverage is part of the effective KD strength.
    """
    if student_logits.ndim != 2:
        raise ValueError("student logits must have shape (examples, actions)")
    batch = student_logits.shape[0]
    expected = (batch,)
    if any(
        value.shape != expected
        for value in (
            gold_indices,
            competitor_indices,
            target_margins,
            example_weights,
        )
    ):
        raise ValueError("one pairwise-margin target is required per example")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("Huber beta must be finite and positive")
    if (competitor_indices < 0).any() or (
        competitor_indices >= student_logits.shape[1]
    ).any():
        raise ValueError("pairwise competitor index is out of bounds")
    weights = example_weights.detach().to(
        device=student_logits.device, dtype=student_logits.dtype
    )
    targets = target_margins.detach().to(
        device=student_logits.device, dtype=student_logits.dtype
    )
    if (
        not torch.isfinite(weights).all()
        or not torch.isfinite(targets).all()
        or (weights < 0).any()
        or (targets < 0).any()
    ):
        raise ValueError("pairwise margins and weights must be finite and nonnegative")
    row = torch.arange(batch, device=student_logits.device)
    student_margin = (
        student_logits[row, gold_indices]
        - student_logits[row, competitor_indices]
    )
    gap = F.relu(targets - student_margin)
    beta_value = float(beta)
    huber = torch.where(
        gap < beta_value,
        0.5 * gap.square() / beta_value,
        gap - 0.5 * beta_value,
    )
    return (huber * weights).mean()


def gold_margin(logits: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or gold_indices.shape != (logits.shape[0],):
        raise ValueError("invalid legal-action margin inputs")
    row = torch.arange(logits.shape[0], device=logits.device)
    gold = logits[row, gold_indices]
    wrong = logits.clone()
    wrong[row, gold_indices] = float("-inf")
    best_wrong = wrong.max(dim=-1).values
    # Forced one-action rows have no counter-action.  Their relative margin is
    # defined as zero instead of +inf so diagnostics and gates remain finite.
    has_wrong = torch.isfinite(best_wrong) & (
        best_wrong > PADDED_LEGAL_LOGIT / 2.0
    )
    return torch.where(has_wrong, gold - best_wrong, torch.zeros_like(gold))


def reliable_teacher_mask(
    logits: torch.Tensor,
    gold_indices: torch.Tensor,
    *,
    require_correct: bool,
    min_gold_margin: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select training-only teacher rows that are safe to distill.

    Returns ``(eligible, correct, gold_margin)`` so callers can report
    correctness and coverage independently instead of treating unpaired rows
    as incorrect examples.
    """
    margin = gold_margin(logits, gold_indices)
    correct = logits.argmax(dim=-1).eq(gold_indices)
    eligible = torch.ones_like(correct, dtype=torch.bool)
    if require_correct:
        eligible &= correct
    if min_gold_margin is not None:
        eligible &= margin >= float(min_gold_margin)
    return eligible, correct, margin


def counterfactual_margin_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    gold_indices: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    positive = gold_margin(positive_logits, gold_indices)
    negative = gold_margin(negative_logits.detach(), gold_indices)
    return F.relu(float(margin) - (positive - negative)).mean()
