"""Pure runtime helpers for public-observation navigation control."""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Iterable, Sequence


PRIVATE_INFERENCE_FIELDS = frozenset({
    "target_geometry",
    "distance_to_goal",
    "goal_distance",
    "geodesic_distance",
    "shortest_path",
    "nearest_target_position",
    "target_relative_position",
    "target_relative_distance",
    "target_relative_angle_deg",
    "object_positions",
    "view_positions",
})

PRIVATE_GOAL_FIELDS = frozenset({
    "view_points",
    "view_positions",
    "object_positions",
    "position",
    "image_position",
    "image_rotation",
    "nearest_target_position",
    "target_relative_position",
    "target_relative_distance",
    "target_relative_angle_deg",
    "distance_to_goal",
    "goal_distance",
    "geodesic_distance",
    "shortest_path",
})


def response_navigation_mode(response: Any) -> str:
    """Read the legacy mode or the authoritative state from a model response."""
    if not isinstance(response, dict):
        return ""
    value = response.get("navigation_mode")
    if value is None:
        value = response.get("navigation_state")
    return str(value).strip().lower() if value is not None else ""


def navigation_mode_allows_stop(mode: Any) -> bool:
    """Fail closed: an action-model Stop requires an explicit stop_at_goal state."""
    return str(mode or "").strip().lower() == "stop_at_goal"


def factorized_evidence_allows_stop(identity: Any, proximity: Any) -> bool:
    """The public model bottleneck permits Stop only under a strict conjunction."""
    return (
        str(identity or "").strip().lower() == "confirmed"
        and str(proximity or "").strip().lower() == "stop_ready"
    )


def terminal_certificate_allows_stop(response: Any) -> bool:
    """Fail closed on a visual terminal certificate.

    This is intentionally a discrete conjunction, not a temporal confidence
    accumulator: an independently trained visual certificate may terminate an
    episode only when it simultaneously predicts a terminal state, confirmed
    target visibility, sufficient proximity, and ``done=1`` from the same RGB
    frame.  No simulator geometry or repeated signal is consumed here.
    """
    if not isinstance(response, dict):
        return False

    def binary_true(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value) == 1
        return str(value or "").strip().lower() in {"1", "true", "yes"}

    state = str(response.get("terminal_state") or "").strip().lower()
    return bool(
        state == "terminal"
        and binary_true(response.get("target_visible"))
        and binary_true(response.get("near_enough"))
        and binary_true(response.get("done"))
    )


def visual_arrival_certificate_v2_allows_stop(response: Any) -> bool:
    """Fail closed on the v2 binary visual-arrival certificate.

    V2 deliberately makes no separately supervised claim about target
    visibility or metric proximity.  Its only deployable decision is whether
    the current *public RGB frame* is a sufficiently reliable terminal pose.
    The one-key schema prevents an old four-field certificate (or a generic
    ``done`` response) from silently being interpreted as V2 permission to
    stop.
    """
    if not isinstance(response, dict) or set(response) != {"arrival_certified"}:
        return False
    value = response.get("arrival_certified")
    # The SFT contract is JSON 0/1, not a free-form affirmative string or a
    # boolean.  ``bool`` is an ``int`` subclass, so exclude it explicitly.
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def state_conditioned_spatial_stop_allows_stop(response: Any) -> bool:
    """Validate the state-conditioned evidence bridge and fail closed.

    The schema itself is the activation mechanism.  SEARCH is forbidden from
    emitting proximity evidence; APPROACH and ARRIVE_VERIFY must emit it.  A
    Stop is accepted only after the complete ARRIVE_VERIFY identity and
    proximity chain.  Extra fields, booleans masquerading as integers, and
    inconsistent state/evidence combinations are rejected.
    """
    if not isinstance(response, dict):
        return False
    state = response.get("navigation_state")
    stop = response.get("stop")
    if not isinstance(stop, int) or isinstance(stop, bool) or stop not in (0, 1):
        return False
    if state == "SEARCH":
        return bool(
            set(response) == {"navigation_state", "identity_evidence", "stop"}
            and response.get("identity_evidence") == "NOT_CONFIRMED"
            and stop == 0
            and False
        )
    if state == "APPROACH":
        return bool(
            set(response)
            == {
                "navigation_state",
                "identity_evidence",
                "proximity_evidence",
                "stop",
            }
            and response.get("identity_evidence") == "CONFIRMED"
            and response.get("proximity_evidence") == "APPROACH_NEEDED"
            and stop == 0
            and False
        )
    if state == "ARRIVE_VERIFY":
        return bool(
            set(response)
            == {
                "navigation_state",
                "identity_evidence",
                "proximity_evidence",
                "stop",
            }
            and response.get("identity_evidence") == "CONFIRMED"
            and response.get("proximity_evidence") == "STOP_READY"
            and stop == 1
        )
    return False


def state_conditioned_evidence_stop_allows_stop(response: Any) -> bool:
    """Fail closed on the B2 state-routed visual-evidence contract.

    B2 makes the activated spatial abilities explicit and requires one
    non-empty, image-grounded judgment per activated ability.  SEARCH cannot
    emit proximity reasoning; APPROACH and ARRIVE_VERIFY must emit it.  Only
    the complete ARRIVE_VERIFY chain may terminate navigation.  This helper
    consumes no simulator geometry, history, score, or oracle metadata.
    """
    top_level = [
        "navigation_state",
        "ability",
        "spatial_reasoning",
        "decision_bridge",
        "stop",
    ]
    if not isinstance(response, dict) or list(response) != top_level:
        return False

    state = response.get("navigation_state")
    expected_abilities = {
        "SEARCH": ["target_visibility"],
        "APPROACH": ["target_visibility", "viewpoint_proximity"],
        "ARRIVE_VERIFY": ["target_visibility", "viewpoint_proximity"],
    }.get(state)
    abilities = response.get("ability")
    reasoning = response.get("spatial_reasoning")
    bridge = response.get("decision_bridge")
    stop = response.get("stop")
    if expected_abilities is None or abilities != expected_abilities:
        return False
    if not isinstance(reasoning, dict) or list(reasoning) != expected_abilities:
        return False
    if not all(
        isinstance(reasoning.get(name), str) and reasoning[name].strip()
        for name in expected_abilities
    ):
        return False
    if not isinstance(bridge, str) or not bridge.strip():
        return False
    if not isinstance(stop, bool) or stop is not (state == "ARRIVE_VERIFY"):
        return False
    return state == "ARRIVE_VERIFY"


def parse_state_conditioned_causal_stop_bottleneck_response(
    response: Any,
) -> dict[str, Any] | None:
    """Strictly parse B3 JSON without routing it through the action parser."""
    if isinstance(response, str):
        try:
            response = json.loads(response.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(response, dict) or list(response) != [
        "navigation_state", "ability", "spatial_reasoning"
    ]:
        return None
    state = response.get("navigation_state")
    expected = {
        "SEARCH": ["target_visibility"],
        "TARGET_ENGAGED": ["target_visibility", "viewpoint_proximity"],
    }.get(state)
    if expected is None or response.get("ability") != expected:
        return None
    reasoning = response.get("spatial_reasoning")
    if not isinstance(reasoning, dict) or list(reasoning) != expected:
        return None
    for ability in expected:
        item = reasoning.get(ability)
        if not isinstance(item, dict) or list(item) != [
            "visual_observation", "conclusion"
        ]:
            return None
        if not str(item.get("visual_observation") or "").strip():
            return None
    visibility = reasoning["target_visibility"].get("conclusion")
    if state == "SEARCH":
        if visibility != "NOT_CONFIRMED":
            return None
        return response
    proximity = reasoning["viewpoint_proximity"].get("conclusion")
    if visibility != "CONFIRMED":
        return None
    if proximity not in {
        "ADDITIONAL_APPROACH_REQUIRED", "NEAR_CURRENT_VIEWPOINT"
    }:
        return None
    return response


def state_conditioned_causal_stop_bottleneck_allows_stop(response: Any) -> bool:
    """Consume B3 spatial conclusions as the sole Stop decision bottleneck."""
    parsed = parse_state_conditioned_causal_stop_bottleneck_response(response)
    if parsed is None or parsed["navigation_state"] != "TARGET_ENGAGED":
        return False
    return (
        parsed["spatial_reasoning"]["viewpoint_proximity"]["conclusion"]
        == "NEAR_CURRENT_VIEWPOINT"
    )


def parse_visual_arrival_certificate_v2_runtime_response(
    response: Any,
) -> dict[str, int] | None:
    """Parse the raw V2 response exactly as the existing agent path does.

    The legacy generic response parser extracts the first balanced brace
    object, normalizes duplicated outer braces, then tries JSON followed by a
    Python literal.  Centralizing that V2-relevant subset here makes offline
    generation audits testable without claiming the runtime accepts only the
    compact training surface.  The final one-key integer check mirrors the V2
    gate for both the positive and the valid-negative response.
    """
    if not isinstance(response, str):
        return None
    text = response.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    candidate = None
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                break
    if not candidate:
        return None
    while candidate.startswith("{{"):
        candidate = candidate[1:]
    while candidate.count("{") < candidate.count("}"):
        candidate = candidate[:-1]

    parsed = None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(candidate)
        except Exception:
            continue
        if isinstance(value, dict) and "arrival_certified" in value:
            parsed = value
            break
    if not isinstance(parsed, dict) or set(parsed) != {"arrival_certified"}:
        return None
    value = parsed.get("arrival_certified")
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        return None
    return {"arrival_certified": int(value)}


def temporal_reference_contract_valid(relation: Any, reference_available: Any) -> bool:
    """Check that model output matches the number of public visual inputs."""
    relation_text = str(relation or "").strip().lower()
    if bool(reference_available):
        return relation_text in {
            "current_closer",
            "similar_range",
            "current_farther",
            "target_mismatch",
        }
    return relation_text == "no_reference"


def temporal_relation_allows_stop(relation: Any, reference_available: Any) -> bool:
    """Require a format-valid temporal judgment that does not contradict Stop.

    A reference-free decision may still stop from the current public RGB.  If a
    reference is present, ``current_farther`` and ``target_mismatch`` are
    explicit counter-evidence and therefore fail closed.
    """
    if not temporal_reference_contract_valid(relation, reference_available):
        return False
    relation_text = str(relation or "").strip().lower()
    if not bool(reference_available):
        return relation_text == "no_reference"
    return relation_text in {"current_closer", "similar_range"}


def temporal_reference_confirmation_eligible(
    *,
    navigation_state: Any,
    identity_evidence: Any,
    proximity_evidence: Any,
    temporal_relation: Any,
    reference_available: Any,
) -> bool:
    """Return whether a public frame may contribute to reference promotion.

    Promotion is deliberately narrower than identity confirmation: only a
    format-valid, non-terminal target-approach decision can build the stable
    reference.  Isolated confirmations and malformed temporal outputs cannot
    poison future observations.
    """
    return (
        str(navigation_state or "").strip().lower() == "approach_visible_target"
        and str(identity_evidence or "").strip().lower() == "confirmed"
        and str(proximity_evidence or "").strip().lower() == "approach_needed"
        and temporal_reference_contract_valid(
            temporal_relation,
            reference_available,
        )
    )


def public_inference_observation(obs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strip privileged target labels at the inference boundary."""
    clean = dict(obs)
    removed = []
    for key in PRIVATE_INFERENCE_FIELDS:
        if key in clean:
            clean.pop(key, None)
            removed.append(key)
    goal = clean.get("goal")
    if isinstance(goal, dict):
        public_goal = dict(goal)
        for key in PRIVATE_GOAL_FIELDS:
            if key in public_goal:
                public_goal.pop(key, None)
                removed.append(f"goal.{key}")
        clean["goal"] = public_goal
    return clean, sorted(removed)


def candidate_model_theta_to_yaw_command(model_theta_rad: float) -> float:
    """Convert the projected candidate bearing to Habitat's executed yaw."""
    return -float(model_theta_rad)


def candidate_model_displacement(
    radius: float,
    model_theta_rad: float,
) -> tuple[float, float, float]:
    """Return the candidate endpoint in the agent-local model frame."""
    radius = float(radius)
    theta = float(model_theta_rad)
    return (
        radius * math.sin(theta),
        0.0,
        -radius * math.cos(theta),
    )


def yaw_command_forward_displacement(
    radius: float,
    yaw_command_rad: float,
) -> tuple[float, float, float]:
    """Return local displacement after executing yaw and moving forward."""
    radius = float(radius)
    yaw = float(yaw_command_rad)
    return (
        -radius * math.sin(yaw),
        0.0,
        -radius * math.cos(yaw),
    )


def candidate_angle_metadata(model_theta_rad: float) -> dict[str, float]:
    """Record model bearing and simulator yaw without conflating their signs."""
    model_theta = float(model_theta_rad)
    yaw_command = candidate_model_theta_to_yaw_command(model_theta)
    return {
        "theta_deg_model_frame": math.degrees(model_theta),
        "theta_deg_execute_frame": math.degrees(yaw_command),
        "yaw_command_deg": math.degrees(yaw_command),
    }


def should_stop_after_same_pose_crosscheck(
    stop_signal: bool,
    action_number: Any,
) -> bool:
    """Require independent action-policy agreement before first-signal stop."""
    try:
        return bool(stop_signal) and int(action_number) == -1
    except (TypeError, ValueError):
        return False


def action_requests_stop_confirmation(
    action_number: Any,
    *,
    action_should_stop: bool = False,
    decision_action: Any = None,
    require_final_action_stop: bool = False,
) -> bool:
    """Return whether the action policy independently requested STOP.

    ``require_final_action_stop`` is for action-preserving verifier runs: only
    the final executable action may trigger the verifier.  The default retains
    legacy compatibility with structured action outputs whose auxiliary Stop
    fields were treated as a proposal.
    """
    try:
        final_action_is_stop = int(action_number) == -1
    except (TypeError, ValueError):
        final_action_is_stop = False
    if require_final_action_stop:
        return final_action_is_stop
    return bool(
        final_action_is_stop
        or
        action_should_stop
        or str(decision_action or "").strip().lower() in {"candidate_-1", "-1", "stop"}
    )


def blind_stop_crosscheck_prompt_suffix(goal: str) -> str:
    """Return a neutral stop-capable action contract without verifier leakage."""
    return (
        "\n\nIndependent current-pose decision:\n"
        "- This instruction provides no evidence that the target is nearby. "
        "Judge only from the current image.\n"
        f"- Choose action -1 only if the actual {goal} is unmistakably visible "
        "in the immediate foreground and no translation is needed.\n"
        "- If the target is uncertain, partly blocked, merely visible at approach "
        "distance, or has traversable space before it, choose the best numbered "
        "red-arrow action instead.\n"
        "- Return the normal action object. Use {'action': -1} only for an "
        "independently justified immediate stop."
    )


def use_consecutive_stop_fallback(
    consecutive_stop_signals: bool,
    blind_crosscheck: bool,
    allow_blind_fallback: bool = False,
    enabled: bool = True,
) -> bool:
    """Allow repeated stop signals to terminate when the protocol enables it."""
    return bool(enabled) and bool(consecutive_stop_signals) and (
        not bool(blind_crosscheck) or bool(allow_blind_fallback)
    )


def use_single_stop_direct_execute(
    current_stop_signal: bool,
    *,
    enabled: bool = False,
) -> bool:
    """Execute the current stopping model decision without temporal fallback.

    This is deliberately distinct from the historical two-consecutive-signal
    heuristic: one positive decision on the current public observation is a
    terminal Stop.  The flag defaults off so historical replay is unchanged.
    """
    return bool(enabled) and bool(current_stop_signal)


def compass_habitat_step_cost(
    radius_m: float,
    yaw_rad: float,
    *,
    is_stop: bool = False,
) -> int:
    """Approximate one PolarAction in CompassNav's Habitat-step units."""
    if is_stop:
        return 1
    radius = float(radius_m)
    yaw = float(yaw_rad)
    forward_steps = math.ceil(radius / 0.25) if radius > 0 else 0
    turn_degrees = abs(math.degrees(yaw))
    turn_steps = math.ceil(turn_degrees / 30.0) if turn_degrees > 0 else 0
    return max(1, int(forward_steps + turn_steps))


def strict_habitat_step_horizon_decision(
    consumed_steps: int,
    action_cost: int,
    budget: int,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Decide whether a macro action fits a strict Compass/Habitat horizon.

    CompassNav emits a *macro* ``PolarAction`` while the evaluation protocol
    counts its 0.25 m / 30 degree atomic-step equivalent.  Charging that cost
    only after the macro action has executed permits an overrun (for example,
    498 + a cost-10 action).  Strict evaluation must instead make the
    execution decision from the public action command before calling the
    simulator.

    The helper is deliberately pure so the hard-horizon property can be tested
    without importing Habitat.  When disabled it is observational only, which
    preserves legacy launcher behavior exactly.
    """
    try:
        consumed = max(0, int(consumed_steps))
    except (TypeError, ValueError):
        consumed = 0
    try:
        cost = max(0, int(action_cost))
    except (TypeError, ValueError):
        cost = 0
    try:
        limit = max(0, int(budget))
    except (TypeError, ValueError):
        limit = 0

    remaining = max(0, limit - consumed)
    if not bool(enabled) or limit <= 0:
        return {
            "strict": False,
            "allow_execution": True,
            "consumed_steps": consumed,
            "action_cost": cost,
            "budget": limit,
            "remaining_steps": remaining,
            "post_execution_steps": consumed + cost,
            "blocked_reason": "",
        }

    allowed = bool(consumed + cost <= limit)
    return {
        "strict": True,
        "allow_execution": allowed,
        "consumed_steps": consumed,
        "action_cost": cost,
        "budget": limit,
        "remaining_steps": remaining,
        # A blocked macro action is never executed or charged.  This is the
        # key invariant consumed by the runtime and its test.
        "post_execution_steps": consumed + cost if allowed else consumed,
        "blocked_reason": "" if allowed else "would_exceed_habitat_step_budget",
    }


def select_public_memory_candidate_indices(
    memories: Sequence[dict[str, Any]],
) -> list[int]:
    """Keep unblocked candidates near the least-visited safe destination."""
    unblocked = [
        (index, memory)
        for index, memory in enumerate(memories)
        if not bool(memory.get("online_blocked", False))
    ]
    if not unblocked:
        return []
    min_visits = min(int(memory.get("online_visit_count") or 0) for _, memory in unblocked)
    return [
        index
        for index, memory in unblocked
        if int(memory.get("online_visit_count") or 0) <= min_visits + 1
    ]


def cap_stop_confirmation_actions(
    current_actions: Iterable[Sequence[float]],
    *,
    cap: float,
    fov_deg: float,
) -> list[tuple[float, float]]:
    """Preserve visible headings while limiting first-stop translation."""
    cap = max(0.1, min(0.75, float(cap)))
    actions: list[tuple[float, float]] = []
    seen: set[float] = set()
    for action in current_actions:
        try:
            radius, theta = float(action[0]), float(action[1])
        except (TypeError, ValueError, IndexError):
            continue
        if radius <= 1e-6:
            continue
        key = round(theta, 4)
        if key in seen:
            continue
        seen.add(key)
        actions.append((min(radius, cap), theta))
    if not actions:
        angle = math.radians(float(fov_deg) / 2.0) * 0.7
        actions = [
            (cap, -angle),
            (cap, -angle / 4.0),
            (cap, angle / 4.0),
            (cap, angle),
        ]
    actions.sort(key=lambda item: item[1])
    return actions


def point_segment_distance_xz(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    """Return planar distance from a point to a finite XZ segment."""
    px, pz = float(point[0]), float(point[2])
    sx, sz = float(start[0]), float(start[2])
    ex, ez = float(end[0]), float(end[2])
    dx, dz = ex - sx, ez - sz
    denom = dx * dx + dz * dz
    if denom <= 1e-9:
        return math.hypot(px - sx, pz - sz)
    t = max(0.0, min(1.0, ((px - sx) * dx + (pz - sz) * dz) / denom))
    return math.hypot(px - (sx + t * dx), pz - (sz + t * dz))
