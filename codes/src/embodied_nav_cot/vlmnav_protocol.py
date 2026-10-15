from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "objectnav_cot_v1"
ABILITYMODE_PROTOCOL_ID = "objectnav_cot_v2_abilitymode"
ABILITYMODE_715_PROTOCOL_ID = "objectnav_cot_v2_abilitymode_715"
STATE_ROUTED_PROTOCOL_ID = "objectnav_cot_v3_state_routed"
DYNAMIC_STATE_ROUTED_PROTOCOL_ID = "objectnav_cot_v4_dynamic_state_routed"
EVIDENCE_BRIDGE_PROTOCOL_ID = "objectnav_cot_v5_scsra_evidence_bridge"
FACTORIZED_BRIDGE_PROTOCOL_ID = "objectnav_cot_v6_scsra_factorized_bridge"
TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID = "objectnav_cot_v7_scsra_temporal_reference_bridge"
STABLE_TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID = "objectnav_cot_v72_scsra_stable_temporal_reference_bridge"
PUBLIC_MEMORY_STATE_BRIDGE_PROTOCOL_ID = "objectnav_cot_v73_scsra_public_memory_state_bridge"
DIRECT_ACTION_PROTOCOL_ID = "objectnav_direct_action_v1"
AUX_SPATIAL_STATE_PROTOCOL_ID = "objectnav_aux_spatial_state_v3"
DIRECT_ACTION_WITH_STATE_PROTOCOL_ID = "objectnav_direct_action_with_state_v3"
V4_INTEGRATED_PROTOCOL_ID = "objectnav_spatial_cot_v4_integrated_action_v1"
STATE_CONDITIONED_COMPACT_ACTION_PROTOCOL_ID = (
    "objectnav_state_conditioned_compact_action_v1"
)
ADAPTIVE_SPATIAL_JOINT_B8_PROTOCOL_ID = "objectnav_adaptive_spatial_joint_b8_v1"

ABILITY_VOCAB = (
    "target_visibility",
    "metric_property_estimation",
    "route_planning",
    "scene_topological_reasoning",
    "cross_view_reasoning",
    "egocentric_allocentric_transformation",
)
STATE_VOCAB = (
    "near_goal_stop",
    "topology_correction",
    "target_guided_approach",
    "frontier_transition",
    "frontier_exploration",
)
MODE_VOCAB = (
    "explore_current_region",
    "enter_new_region",
    "approach_target",
    "replan",
    "stop",
)
SPATIAL_REASONING_KEYS = (
    "target_relation",
    "scene_relation",
    "spatial_inference",
    "navigation_use",
)
TOP_LEVEL_KEYS = (
    "ability",
    "evidence",
    "state",
    "spatial_reasoning",
    "reason",
    "decision",
)
ABILITYMODE_TOP_LEVEL_KEYS = (
    "ability",
    "navigation_mode",
    "spatial_reasoning",
    "candidate_comparison",
    "reason",
    "decision",
)

ABILITY_DESCRIPTIONS = {
    "target_visibility": "decide whether the goal is visible and whether visibility should dominate the next move",
    "metric_property_estimation": "judge near/far distance, stop boundary, and expected progress",
    "route_planning": "choose the locally executable action that best advances the navigation plan",
    "scene_topological_reasoning": "reason about rooms, openings, frontiers, and transitions between regions",
    "cross_view_reasoning": "compare current observation with short memory to detect revisits, loops, and recovery options",
    "egocentric_allocentric_transformation": "map red-arrow egocentric directions to persistent route or branch choices",
}

MODE_DESCRIPTIONS = {
    "explore_current_region": "search within the current visible area or room without committing to a new region",
    "enter_new_region": "move through an opening, doorway, hallway, or frontier into a new area",
    "approach_target": "move closer to a visible or strongly localized goal object",
    "replan": "recover from poor alignment, a blocked route, loop risk, or low progress",
    "stop": "terminate only when selecting candidate_-1 within success distance",
}

ABILITYMODE_ABILITY_VOCAB = (
    "target visibility",
    "distance measurement",
    "route planning",
    "topological reasoning",
    "spatial relations",
    "egocentric-allocentric transformation",
)
ABILITYMODE_NAVIGATION_MODE_VOCAB = (
    "search_current_area",
    "explore_new_area",
    "approach_visible_target",
    "recover_route",
    "stop_at_goal",
)
ABILITYMODE_ABILITY_DESCRIPTIONS = {
    "target visibility": "whether the goal object is visible, hidden, or visually anchored",
    "distance measurement": "whether the goal is close, far, within stop range, or still needs approach",
    "route planning": "which executable candidate best advances the next navigation step",
    "topological reasoning": "how rooms, openings, hallways, frontiers, and connected regions affect movement",
    "spatial relations": "relative geometry among the target, arrows, openings, obstacles, rooms, and viewpoint",
    "egocentric-allocentric transformation": "how the current egocentric arrow direction aligns with a persistent route or recovery heading",
}
ABILITYMODE_NAVIGATION_MODE_DESCRIPTIONS = {
    "search_current_area": "search within the current visible/local area because the target is not localized",
    "explore_new_area": "move through an opening, doorway, hallway, or frontier into a different area",
    "approach_visible_target": "move closer to a visible or strongly localized target",
    "recover_route": "recover from uncertainty, poor alignment, weak progress, or a bad heading",
    "stop_at_goal": "stop because the target is localized nearby and stopping is the selected action",
}


def candidate_label(action: int) -> str:
    return f"candidate_{int(action)}"


def normalize_decision_action(action: Any) -> str:
    text = str(action).strip()
    if text in {"stop", "STOP", "-1", "candidate_stop"}:
        return "candidate_-1"
    if text.startswith("candidate_"):
        return text
    try:
        return candidate_label(int(text))
    except Exception:
        return text


def raw_action_from_decision(action: Any) -> int | None:
    label = normalize_decision_action(action)
    if not label.startswith("candidate_"):
        return None
    try:
        return int(label.split("_", 1)[1])
    except Exception:
        return None


def final_action_line(action: Any) -> str:
    raw = raw_action_from_decision(action)
    if raw is None:
        raw = -1 if str(action).strip().lower() == "stop" else 0
    return "{'action': " + str(raw) + "}"


def final_done_line(done: bool) -> str:
    return "{'done': " + ("1" if done else "0") + "}"


def normalize_mode(mode: Any, *, should_stop: bool = False) -> str:
    text = str(mode or "").strip()
    if should_stop:
        return "stop"
    aliases = {
        "explore": "explore_current_region",
        "continue": "explore_current_region",
        "approach": "approach_target",
        "target_approach": "approach_target",
    }
    text = aliases.get(text, text)
    return text if text in MODE_VOCAB else "explore_current_region"


def normalize_cot(cot: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(cot)
    decision = dict(out.get("decision") if isinstance(out.get("decision"), Mapping) else {})
    action = normalize_decision_action(decision.get("action", out.get("action", "")))
    should_stop = bool(decision.get("should_stop")) or action == "candidate_-1"
    if should_stop:
        action = "candidate_-1"
    decision["action"] = action
    decision["mode"] = normalize_mode(decision.get("mode"), should_stop=should_stop)
    decision["should_stop"] = bool(should_stop)
    out["decision"] = decision
    out.setdefault("ability", [])
    out.setdefault("evidence", [])
    out.setdefault("state", "near_goal_stop" if should_stop else "frontier_exploration")
    out.setdefault("spatial_reasoning", {})
    out.setdefault("reason", "")
    return {key: out.get(key) for key in TOP_LEVEL_KEYS}


def render_response(cot: Mapping[str, Any]) -> str:
    normalized = normalize_cot(cot)
    return (
        json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        + "\n"
        + final_action_line(normalized["decision"]["action"])
    )


def ability_block() -> str:
    return "\n".join(f"- {name}: {ABILITY_DESCRIPTIONS[name]}" for name in ABILITY_VOCAB)


def mode_block() -> str:
    return "\n".join(f"- {name}: {MODE_DESCRIPTIONS[name]}" for name in MODE_VOCAB)


def action_mapping_block(action_vocab: Sequence[str], *, stop_allowed: bool) -> str:
    lines = [
        "Action mapping:",
        "- candidate_i refers to the red-arrow candidate with white-circle label i in the image.",
        "- Candidate numbers are identifiers, not preferences; lower numbers are not better by default.",
        "- The final executable line must use the same raw number: decision.action=candidate_i -> {'action': i}.",
        "- candidate_0 is a recovery-only turn-around/reorient fallback; it is not a red forward arrow and must not be treated as ordinary forward navigation.",
        "- Use candidate_0 only when forward red-arrow candidates are unreliable, blocked, badly aligned, or recovery/reorientation is needed.",
    ]
    if stop_allowed:
        lines.append("- candidate_-1 means stop; it maps only to {'action': -1}.")
    else:
        lines.append("- candidate_-1 is not available in this step; do not stop.")
    lines.append(f"- Valid actions for this step: {', '.join(action_vocab)}")
    return "\n".join(lines)


def visible_action_space_block(
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    *,
    context_name: str,
) -> str:
    actions = sorted(set(int(a) for a in valid_actions))
    forward_actions = [a for a in actions if a > 0]
    lines = [f"{context_name}:"]
    if -1 in actions:
        lines.append(
            "- candidate_-1: executable action -1; stop only when the goal is confirmed and close enough."
        )
    if 0 in actions:
        lines.append(
            "- candidate_0: executable action 0; recovery-only turn-around/reorient fallback."
        )
    if forward_actions:
        labels = ", ".join(str(a) for a in forward_actions)
        lines.append(
            f"- candidate_i for i in {{{labels}}}: executable action i; visible red-arrow label i."
        )
    elif candidate_lines:
        lines.extend(candidate_lines)
    else:
        lines.append("- none")
    return "\n".join(lines)


def stop_policy_block(goal: str, *, stop_allowed: bool) -> str:
    if not stop_allowed:
        availability = "candidate_-1 is not available in this step, so do not stop."
    else:
        availability = (
            "candidate_-1 is available, but choose it only for the confirmed-close state."
        )
    return (
        "Stop policy:\n"
        f"- visible {goal}: the goal is clearly visible, but this alone is not enough to stop; approach it unless it is close enough.\n"
        f"- suspected {goal}: the object may be a {goal}, is partly visible, ambiguous, far away, or only inferred; do not stop, improve viewpoint or continue search.\n"
        f"- confirmed-close {goal}: the object is clearly the goal, in the immediate foreground / within ObjectNav success distance, and no movement is needed; only then stop.\n"
        "- A distant, side-view, occluded, or ambiguous target is not a stop condition.\n"
        f"- {availability}"
    )


def strategy_hierarchy_block() -> str:
    return (
        "ObjectNav decision hierarchy, applied in order:\n"
        "1. stop only for a confirmed-close target;\n"
        "2. approach a visible but not-yet-close target;\n"
        "3. search for a hidden or uncertain target in the current area;\n"
        "4. enter the most promising topology such as a doorway, hallway, opening, frontier, or connected room;\n"
        "5. recover with candidate_0 or another recovery action only when progress is blocked, alignment is poor, or forward candidates are unreliable."
    )


def abilitymode_ability_block() -> str:
    return "\n".join(
        f"- {name}: {ABILITYMODE_ABILITY_DESCRIPTIONS[name]}"
        for name in ABILITYMODE_ABILITY_VOCAB
    )


def abilitymode_navigation_mode_block() -> str:
    return "\n".join(
        f"- {name}: {ABILITYMODE_NAVIGATION_MODE_DESCRIPTIONS[name]}"
        for name in ABILITYMODE_NAVIGATION_MODE_VOCAB
    )


def schema_block(*, should_stop_schema: str = "false") -> str:
    return (
        "Return exactly one JSON object first with this schema:\n"
        "{\n"
        '  "ability": ["string", "..."],\n'
        '  "evidence": ["string", "..."],\n'
        '  "state": "string",\n'
        '  "spatial_reasoning": {\n'
        '    "target_relation": "short spatial reasoning sentence",\n'
        '    "scene_relation": "short spatial reasoning sentence",\n'
        '    "spatial_inference": "short spatial reasoning sentence",\n'
        '    "navigation_use": "short spatial reasoning sentence"\n'
        "  },\n"
        '  "reason": "string",\n'
        '  "decision": {\n'
        '    "action": "candidate_1",\n'
        '    "mode": "string",\n'
        f'    "should_stop": {should_stop_schema}\n'
        "  }\n"
        "}\n"
    )


def vocabulary_block(action_vocab: Sequence[str], *, stop_allowed: bool) -> str:
    stop_rule = (
        "- candidate_-1 is available: decision.should_stop may be true only if decision.action is candidate_-1.\n"
        if stop_allowed
        else "- decision.should_stop must be false; do not choose candidate_-1 when it is not listed in the action vocabulary.\n"
    )
    return (
        "Vocabulary constraints:\n"
        f"- ability must be a JSON array using only: {', '.join(ABILITY_VOCAB)}\n"
        f"- state must be exactly one of: {', '.join(STATE_VOCAB)}\n"
        f"- decision.mode must be exactly one of: {', '.join(MODE_VOCAB)}\n"
        f"- decision.action must be exactly one of: {', '.join(action_vocab)}\n"
        f"{stop_rule}"
        "- evidence must use short grounded tokens such as target_visible, target_not_visible, "
        "target_distance_state:{within_stop_range|near|far|unknown}, stop_allowed, low_progress, loop_risk, "
        "candidate_1:preferred, candidate_1:unsafe, candidate_1:uncertain.\n"
        "- Do not output Observation/Reasoning bullet sections before the JSON.\n"
        "- After the JSON, output exactly one final line like {'action': <number>}.\n"
    )


def render_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    ability_targets: Sequence[str] | None = None,
    opd_hard_type: str | None = None,
    best_action: int | None = None,
    context_name: str = "Visible action space",
) -> str:
    action_vocab = [candidate_label(a) for a in valid_actions]
    stop_allowed = -1 in set(int(a) for a in valid_actions)
    target_block = ""
    memory_block = f"\n{memory}\n" if memory else ""
    stop_rule = (
        "Stop only with candidate_-1 when the target is immediate foreground/within success distance. "
        if stop_allowed
        else "Do not stop; candidate_-1 is not available. "
    )
    return (
        f"Protocol: {PROTOCOL_ID}\n"
        f"Goal: nearest {goal}. Choose the next executable ObjectNav action from the red-arrow candidates.\n"
        f"{memory_block}"
        f"{visible_action_space_block(candidate_lines, valid_actions, context_name=context_name)}\n\n"
        f"{action_mapping_block(action_vocab, stop_allowed=stop_allowed)}\n\n"
        f"{stop_policy_block(goal, stop_allowed=stop_allowed)}\n\n"
        f"{strategy_hierarchy_block()}\n\n"
        "Ability vocabulary for JSON field ability:\n"
        f"{ability_block()}\n\n"
        "Navigation mode vocabulary for JSON field decision.mode:\n"
        f"{mode_block()}\n\n"
        "Think spatially, but keep it compact. Choose only the useful ability names from the vocabulary above. "
        "Key checks: target visible/not visible, visible-but-not-close vs suspected vs confirmed-close, open route/frontier, loop/revisit risk, and which red-arrow label best advances the plan. "
        f"{stop_rule}"
        f"{target_block}"
        "Return compact JSON only, then the executable dict on the final line. "
        "The JSON shape below uses candidate_1/false only as concrete examples; candidate_1 is not a preference:\n"
        '{'
        '"ability":["<ability_name>"],'
        '"evidence":["target_visible or target_not_visible","target_distance_state:near","candidate_1:preferred"],'
        '"state":"<state_name>",'
        '"spatial_reasoning":{"target_relation":"short","scene_relation":"short","spatial_inference":"short","navigation_use":"short"},'
        '"reason":"one compact sentence linking evidence to the chosen action",'
        '"decision":{"action":"candidate_1","mode":"approach","should_stop":false}'
        '}\n'
        f"decision.action must be one of: {', '.join(action_vocab)}. "
        "Use concrete action strings, concrete mode strings, and JSON booleans only; never output generic template strings or angle-bracket placeholders. "
        f"state must be one of: {', '.join(STATE_VOCAB)}. "
        f"decision.mode must be one of: {', '.join(MODE_VOCAB)}. "
        "Use short evidence tokens, no extra prose. Final line exactly: {'action': <number>}."
    )


def render_adaptive_spatial_joint_b8_action_prompt(
    *,
    goal: str,
    **_: Any,
) -> str:
    """Expose the byte-aligned single-pass B8 prompt to VLMNav.

    Candidate availability is represented exclusively by the projected red
    arrows in the image.  Do not append legacy candidate tables, memory, or a
    separate Stop surface: the B8 SFT prompt is intentionally image + public
    goal only.
    """

    from .adaptive_spatial_joint_b8 import joint_prompt

    return joint_prompt(str(goal))


def render_direct_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Action candidates",
    **_: Any,
) -> str:
    """Aux-StopAction runtime prompt: direct decision only, no CoT."""
    actions = sorted(set(int(a) for a in valid_actions))
    action_vocab = [candidate_label(a) for a in actions]
    stop_allowed = -1 in actions
    memory_block = f"\n{memory}\n" if memory else ""
    lines = [f"{context_name}:"]
    if -1 in actions:
        lines.append(f"- candidate_-1: stop if the {goal} is confirmed and close enough to finish.")
    if 0 in actions:
        lines.append("- candidate_0: recovery/reorientation action, not a red forward arrow.")
    if candidate_lines:
        seen = set()
        for raw in candidate_lines:
            text = str(raw)
            match = re.search(r"candidate_(-?\d+)", text)
            if match:
                idx = int(match.group(1))
                if idx in {-1, 0}:
                    continue
                seen.add(idx)
                m_dist = re.search(r"distance_m=([-+]?\d+(?:\.\d+)?)", text)
                m_head = re.search(r"heading_deg=([-+]?\d+(?:\.\d+)?)", text)
                desc = f"- candidate_{idx}: red-arrow navigation candidate"
                if m_dist:
                    desc += f"; distance_m={float(m_dist.group(1)):.2f}"
                if m_head:
                    desc += f"; heading_deg={float(m_head.group(1)):.1f}"
                desc += "."
                lines.append(desc)
        for idx in [a for a in actions if a > 0 and a not in seen]:
            lines.append(f"- candidate_{idx}: red-arrow navigation candidate.")
    else:
        for idx in [a for a in actions if a > 0]:
            lines.append(f"- candidate_{idx}: red-arrow navigation candidate.")
    return f"""Protocol: {DIRECT_ACTION_PROTOCOL_ID}

You are an ObjectNav policy. Use the egocentric image and the red-arrow action candidates to navigate to the nearest {goal}.
{memory_block}
{chr(10).join(lines)}

Decision policy:
- Choose candidate_-1 only when the {goal} is visually confirmed and the current camera position already appears close enough to finish.
- If the {goal} is visible but still needs approach through clear free space, choose the candidate that best approaches it.
- If the {goal} is absent or uncertain, search or explore efficiently.
- Use candidate_0 only for recovery or reorientation when forward candidates are unreliable.
- Candidate numbers are identifiers, not preferences.

Valid actions: {', '.join(action_vocab)}

Return exactly one compact JSON object and nothing else:
{{"decision":{{"action":"<one valid candidate>","should_stop":true_or_false}}}}

Strict constraints:
- decision.action must be one valid candidate.
- decision.should_stop must be true iff decision.action is candidate_-1; otherwise false.
- Do not output reasoning, navigation_mode, ability, candidate tables, markdown, or extra prose.
"""


def render_v4_integrated_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Long-CoT runtime surface matching the V4 integrated SFT protocol."""
    del candidate_lines, memory, context_name
    actions = sorted(set(int(action) for action in valid_actions))
    action_vocab = [candidate_label(action) for action in actions]
    zero_rule = (
        "- candidate_0 is available in this image. It means turn around or reorient for recovery. "
        "Use it only when every visible movement candidate is blocked or unreliable; target absence or a missing doorway alone is not sufficient."
        if 0 in actions
        else "- candidate_0 is not available in this image and must not be selected."
    )
    return f"""Protocol: {V4_INTEGRATED_PROTOCOL_ID}

Goal object: nearest {goal}.

Valid actions: {', '.join(action_vocab)}
- candidate_-1 means stop. Use it only when the correct {goal} is clearly confirmed at the current camera position and no further approach is needed.
{zero_rule}
- candidate_1..candidate_N are the movement actions shown by the corresponding red-arrow labels in the image.

Generate one integrated annotation directly from this image. Do not rely on any previous annotation. Silently check the visual evidence, spatial interpretation, candidate alignment, and final action for mutual consistency before emitting JSON.

Choose 1 to 4 genuinely relevant abilities from this fixed vocabulary:
1. target_visibility: whether the {goal} is visible, partial, occluded, ambiguous, reflected, or absent.
2. spatial_relation: relative positions among the target, landmarks, furniture, walls, openings, obstacles, and free space.
3. local_affordance: immediate traversability, open floor, obstacles, blocked regions, clutter, and narrow passages.
4. self_orientation: the camera-centered facing direction and what lies ahead, left, right, or behind.
5. metric_distance: qualitative near/far/depth evidence for a visible target or key region; never invent an exact distance.
6. semantic_layout_topology: room type, boundaries, doorways, corridors, connected regions, and visible transitions.
7. object_location_prior: general knowledge about where a {goal} typically occurs, without claiming an unseen target is present.
8. scene_object_matching: whether visible scene cues raise or lower the plausibility of finding a {goal} here.
9. exploration_reasoning: which visible opening or region has greater information value for locating the target.

Return exactly these four top-level fields:
- ability: a unique list of 1 to 4 ability names from the vocabulary.
- spatial_reasoning: a JSON object, never an array. Each key is one selected ability name, and its value contains claim, visual_evidence, navigation_relevance, and evidence_limit.
- decision_bridge: from, candidate, and link.
- decision: action, mode, and should_stop.

Container types are strict:
- ability is a JSON array of strings.
- spatial_reasoning is a JSON object keyed by the exact strings in ability.
- decision_bridge.from is a JSON array of strings even when it contains only one ability.
- decision_bridge and decision are JSON objects.

Spatial-reasoning requirements:
- Base every scene-specific claim on visible image evidence.
- Keep the V3 ability semantics and full evidence-grounded reasoning quality.
- claim states the ability-specific spatial conclusion.
- visual_evidence names concrete visible objects, regions, boundaries, occlusions, or free space.
- navigation_relevance states what the conclusion implies for target search, approach, stopping, or recovery.
- evidence_limit states what the current image cannot establish, or "none" when directly supported.
- Do not mention candidate IDs, executable actions, or final decisions inside spatial_reasoning. Candidate mapping belongs only in decision_bridge.
- Keep navigation_relevance descriptive rather than prescriptive: do not tell the agent to move, turn, enter, search, approach, reorient, or stop.
- Select each ability only when it adds distinct, image-grounded evidence; do not include an ability merely because it is commonly useful for navigation.
- Do not force every selected ability to justify the action; only decision_bridge.from identifies the direct decision basis.

Decision-bridge requirements:
- from contains one or two selected abilities that provide the most direct decision evidence.
- candidate is exactly one string from Valid actions.
- link is one sentence of at most 24 English words explaining how the generated spatial conclusion aligns with that candidate.
- Use only a positive bridge. Do not discuss rejected candidates, alternatives, rankings, or scores.

Decision modes:
- explore_current_region: search within the current visible area or room.
- enter_new_region: pass through an opening, doorway, corridor, or frontier into another region.
- approach_target: move closer to a visible or strongly localized target.
- replan: recover from blockage, poor alignment, weak progress, or lack of a reliable forward route.
- stop: terminate at the confirmed nearby target.

Hard consistency rules:
- decision_bridge.from must be a subset of ability.
- decision_bridge.candidate must equal decision.action.
- candidate_-1, mode=stop, and should_stop=true must occur together and only together.
- candidate_0 requires mode=replan and should_stop=false.
- Even when candidate_0 is available, prefer an open movement candidate that can continue useful search. Do not use candidate_0 merely because the target is absent.
- Every movement candidate requires should_stop=false.
- approach_target requires image-grounded support that the target is localized and still needs approach.
- stop requires image-grounded support that the correct target is immediately close.
- Do not output confidence, target_state, route_state, candidate_observations, reject, reason, summary, markdown, or prose outside the JSON.
"""


def render_state_conditioned_compact_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Compact state-routed spatial evidence bridge for the Move controller."""
    del candidate_lines, memory, context_name
    actions = sorted({int(action) for action in valid_actions if int(action) >= 0})
    if not actions:
        raise ValueError("compact action prompt requires at least one Move candidate")
    action_vocab = ", ".join(candidate_label(action) for action in actions)
    zero_rule = (
        "candidate_0 is available only for recovery/reorientation."
        if 0 in actions
        else "candidate_0 is unavailable and must not be selected."
    )
    return f"""Protocol: {STATE_CONDITIONED_COMPACT_ACTION_PROTOCOL_ID}

Goal object: nearest {goal}.
Valid Move actions: {action_vocab}.
The numbered red arrows in the image are executable movement candidates; {zero_rule}
A separate visual Stop controller decides termination, so this action controller must never output candidate_-1 or a Stop decision.

Predict in this causal order: navigation_state -> activated ability -> compact spatial evidence -> decision bridge -> Move action.

State-conditioned activation:
- SEARCH: the target is not reliably localized. Activate 1 or 2 from local_affordance, semantic_layout_topology, scene_object_matching, exploration_reasoning.
- APPROACH: the target is visibly localized but still needs movement/alignment. Activate 1 or 2 from target_visibility, spatial_relation, metric_distance, local_affordance.
- RECOVER: the local route is blocked, unreliable, or poorly aligned. Activate 1 or 2 from local_affordance, self_orientation, spatial_relation, semantic_layout_topology.

Return exactly one compact JSON object with these top-level fields and no prose:
{{"navigation_state":"SEARCH_or_APPROACH_or_RECOVER","ability":["one_or_two_state_allowed_abilities"],"spatial_reasoning":{{"ability_name":{{"claim":"one short image-grounded conclusion","visual_evidence":"specific visible support"}}}},"decision_bridge":{{"from":["same activated ability names"],"candidate":"candidate_<number>","link":"one short positive link"}},"decision":{{"action":"candidate_<number>","mode":"explore_current_region_or_enter_new_region_or_approach_target_or_replan","should_stop":false}}}}

Hard constraints:
- ability contains 1 or 2 distinct abilities allowed by navigation_state; spatial_reasoning keys and decision_bridge.from equal ability exactly.
- SEARCH permits explore_current_region or enter_new_region; APPROACH requires approach_target; RECOVER requires replan.
- decision_bridge.candidate equals decision.action and must be one valid Move action.
- candidate_0 requires RECOVER and replan. Prefer a reliable numbered arrow whenever one is visibly traversable.
- Ground every claim in this image. Do not use pose, map, simulator distance, semantic masks, target coordinates, oracle geometry, or private metadata.
- Keep reasoning compact: only claim and visual_evidence for each activated ability; no rejected-candidate list, scores, long chain of thought, markdown, or extra prose.
"""


def render_aux_spatial_state_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Training/inference auxiliary prompt: spatial state only, no action."""
    actions = sorted(set(int(a) for a in valid_actions))
    memory_block = f"\n{memory}\n" if memory else ""
    lines = [f"{context_name}:"]
    if -1 in actions:
        lines.append(f"- candidate_-1: stop if the {goal} is confirmed and close enough to finish.")
    if 0 in actions:
        lines.append("- candidate_0: recovery/reorientation action, not a red forward arrow.")
    if candidate_lines:
        seen = set()
        for raw in candidate_lines:
            text = str(raw)
            match = re.search(r"candidate_(-?\d+)", text)
            if match:
                idx = int(match.group(1))
                if idx in {-1, 0}:
                    continue
                seen.add(idx)
                m_dist = re.search(r"distance_m=([-+]?\d+(?:\.\d+)?)", text)
                m_head = re.search(r"heading_deg=([-+]?\d+(?:\.\d+)?)", text)
                desc = f"- candidate_{idx}: red-arrow navigation candidate"
                if m_dist:
                    desc += f"; distance_m={float(m_dist.group(1)):.2f}"
                if m_head:
                    desc += f"; heading_deg={float(m_head.group(1)):.1f}"
                desc += "."
                lines.append(desc)
        for idx in [a for a in actions if a > 0 and a not in seen]:
            lines.append(f"- candidate_{idx}: red-arrow navigation candidate.")
    else:
        for idx in [a for a in actions if a > 0]:
            lines.append(f"- candidate_{idx}: red-arrow navigation candidate.")
    return f"""Protocol: {AUX_SPATIAL_STATE_PROTOCOL_ID}

You are an auxiliary spatial-state observer for ObjectNav. Use the egocentric image and the visible red-arrow candidates to describe the current navigation state for finding the nearest {goal}.
{memory_block}
{chr(10).join(lines)}

Task:
- Describe only the spatial state needed by a separate direct-action policy.
- Do not choose an action.
- Do not output candidate ids as a decision.
- Do not output should_stop, done, action, navigation_mode, ability, or executable dictionaries.
- Candidate numbers are identifiers only; lower numbers are not better.
- Use only visual/candidate evidence from this prompt. Do not use hidden simulator state, evaluation outcomes, dataset labels, or ground-truth distances.

Return exactly one compact JSON object and nothing else:
{{"spatial_state":{{"target_status":"short natural description of whether the {goal} is visually confirmed or uncertain","local_navigation_context":"short natural description of room/opening/obstacle/free-space context","remaining_path":"short natural description of whether useful navigation space remains before a reasonable goal viewpoint"}}}}
"""


def _compact_state_text(spatial_state: Any) -> str:
    if spatial_state is None:
        return ""
    if isinstance(spatial_state, Mapping):
        state = spatial_state.get("spatial_state", spatial_state)
        try:
            return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(state)
    text = str(spatial_state).strip()
    parsed = extract_first_json_object(text)
    if isinstance(parsed, dict):
        state = parsed.get("spatial_state", parsed)
        try:
            return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(state)
    return text[:1200]


def render_direct_action_with_state_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    spatial_state: Any = "",
    **_: Any,
) -> str:
    """Direct action prompt conditioned on a non-executable auxiliary state."""
    prompt = render_direct_action_prompt(
        goal=goal,
        candidate_lines=candidate_lines,
        valid_actions=valid_actions,
        memory=memory,
        context_name=context_name,
    )
    state_text = _compact_state_text(spatial_state)
    if not state_text:
        return prompt.replace(
            f"Protocol: {DIRECT_ACTION_PROTOCOL_ID}",
            f"Protocol: {DIRECT_ACTION_WITH_STATE_PROTOCOL_ID}",
            1,
        )
    state_block = (
        "\nAuxiliary spatial state from a separate observer. Treat it as non-executable evidence, "
        "not as an action command:\n"
        f"{state_text}\n"
    )
    prompt = prompt.replace(
        f"Protocol: {DIRECT_ACTION_PROTOCOL_ID}",
        f"Protocol: {DIRECT_ACTION_WITH_STATE_PROTOCOL_ID}",
        1,
    )
    marker = "\nDecision policy:\n"
    if marker in prompt:
        prompt = prompt.replace(marker, state_block + marker, 1)
    else:
        prompt += state_block
    return prompt


def render_abilitymode_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    return _render_selective_abilitymode_action_prompt(
        goal=goal,
        candidate_lines=candidate_lines,
        valid_actions=valid_actions,
        memory=memory,
        context_name=context_name,
    )


def _render_selective_abilitymode_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Baseline abilitymode prompt body used by the min3 ObjectNav runs."""
    action_vocab = [candidate_label(a) for a in valid_actions]
    stop_allowed = -1 in set(int(a) for a in valid_actions)
    memory_block = f"\n{memory}\n" if memory else ""
    forward_actions = sorted({int(a) for a in valid_actions if int(a) > 0})
    forward_set = ",".join(str(a) for a in forward_actions)
    action_space_lines = []
    if stop_allowed:
        action_space_lines.append(
            f"- candidate_-1: stop. Use when the {goal} is clearly confirmed and close enough for ObjectNav success."
        )
    if 0 in {int(a) for a in valid_actions}:
        action_space_lines.append(
            "- candidate_0: recovery-only reorientation action. It is not a red forward arrow."
        )
    if forward_actions:
        action_space_lines.append(
            f"- candidate_i for i in {{{forward_set}}}: red-arrow navigation candidate with white-circle label i."
        )
    if not action_space_lines:
        action_space_lines.append("- none")
    candidate_geometry_block = ""
    if candidate_lines:
        candidate_geometry_block = (
            "\nCandidate geometry hints:\n"
            + "\n".join(str(line) for line in candidate_lines)
            + "\n"
            "Use these as local motion hints: distance_m is how far that arrow can execute, "
            "heading_deg is its egocentric turn direction. If a visible target still has an "
            "aligned forward candidate with meaningful distance_m through free space, prefer "
            "approach_visible_target over stop_at_goal.\n"
        )
    example_action = next(
        (label for label in action_vocab if label not in {"candidate_-1", "candidate_0"}),
        action_vocab[0] if action_vocab else "candidate_0",
    )
    example_should_stop = "true" if example_action == "candidate_-1" else "false"
    return f"""Protocol: {ABILITYMODE_PROTOCOL_ID}

You are an ObjectNav policy. Use the egocentric image and the red-arrow action candidates to navigate to the nearest {goal}.
{memory_block}
Action space:
{chr(10).join(action_space_lines)}
{candidate_geometry_block}

Action mapping:
- candidate_i refers exactly to the red-arrow label i in the image.
- Candidate numbers are identifiers, not preferences.
- decision.action=candidate_i maps to executable action i; candidate_-1 maps to executable action -1.
- Valid actions: {', '.join(action_vocab)}.

Decision priority:
1. stop_at_goal: If the {goal} is clearly the requested object and the current camera position is already in the near-goal stopping region, choose candidate_-1. The object does not need perfect centering. Do not keep moving just to improve viewpoint, alignment, or confidence once the target is clearly localized at close range.
2. approach_visible_target: If the {goal} is clearly visible but still far, small, across the room, or blocked by obstacles, choose the candidate that best approaches it through free space.
3. search_current_area: If the {goal} is hidden or only suspected in the current area, search efficiently without stopping.
4. explore_new_area: Prefer meaningful transitions into promising rooms, doorways, hallways, openings, frontiers, or connected regions over moving within already checked or irrelevant space.
5. recover_route: Use candidate_0 only when forward candidates are blocked, unreliable, badly aligned, or recovery/reorientation is needed.

Stop calibration:
- Confirmed-close {goal}: clearly matches the target class and is in the immediate interaction zone: directly beside the agent, facing the agent at close range, occupying the near foreground with reachable floor contact, or separated only by a very short final gap. Stop immediately.
- Visible-not-close {goal}: clearly visible but not yet close enough. Approach instead of stopping.
- Suspected {goal}: ambiguous, partial, far, occluded, reflected, picture-like, or merely likely from room context. Do not stop.
- Do not stop just because the {goal} is visible, centered, or large in the image if it is still across a room, across furniture, beyond a doorway, behind a table/bed/counter, or separated by a clear navigable walking gap.
- When confirmed-close holds, stop beats every navigation candidate because extra movement can overshoot or leave the success region.

Stop/action consistency:
- If distance measurement says far, distant, moderate distance, across the room, not close enough, or still needs approach, decision.action must not be candidate_-1.
- If distance measurement says close enough, immediate foreground, within success distance, no movement is needed, or further approach is unnecessary, decision.action must be candidate_-1 and should_stop must be true.
- If decision.action is candidate_-1, navigation_mode must be stop_at_goal and distance measurement must state that the target is close from the current agent position, not merely that a future candidate could approach it.

Allowed ability values:
- target visibility
- distance measurement
- route planning
- topological reasoning
- spatial relations
- egocentric-allocentric transformation

Allowed navigation_mode values:
- search_current_area
- explore_new_area
- approach_visible_target
- recover_route
- stop_at_goal

Ability usage:
- Include target visibility when judging whether the {goal} is visible, hidden, or only suspected.
- Include distance measurement when deciding whether to stop or approach.
- Include route planning for normal movement decisions.
- Include topological reasoning when choosing rooms, openings, hallways, frontiers, or connected regions.
- Include spatial relations when the decision depends on relative positions of the {goal}, arrows, obstacles, openings, or viewpoint.
- Include egocentric-allocentric transformation only for heading recovery or route alignment.
- Choose only abilities that matter for this image and action choice. Do not add topological, spatial, route, or egocentric abilities just to fill the 2 to 4 slots.
- Each spatial_reasoning value must be grounded in this scene: mention concrete target evidence, candidate IDs, red-arrow geometry, openings, obstacles, room layout, or viewpoint relations where relevant.
- Do not use generic template text such as "the decision compares the relative layout", "local topology is used only insofar", or "egocentric arrow direction is interpreted as a route".

Return exactly one compact JSON object:
{{
  "ability": ["two to four allowed ability values"],
  "navigation_mode": "one allowed navigation_mode value",
  "spatial_reasoning": {{
    "<each ability value listed in ability>": "concise free-form spatial judgment for this image and goal",
    "navigation_rationale": "integrated spatial reasoning that combines the listed abilities and explains why the selected decision.action follows"
  }},
  "candidate_comparison": "concrete comparison between the chosen candidate and the main alternatives using candidate IDs and scene geometry",
  "reason": "one short natural-language sentence",
  "decision": {{"action": "{example_action}", "should_stop": {example_should_stop}}}
}}

Strict output constraints:
- decision.action must be one valid candidate.
- decision.should_stop must be the JSON boolean true only when decision.action is candidate_-1; otherwise false.
- decision.should_stop must be true iff decision.action is candidate_-1; otherwise false.
- The schema example uses {example_action}; choose the actual valid candidate that best satisfies the policy for this image.
- ability must contain 2 to 4 values. Do not always use only target visibility and distance measurement; include route planning, spatial relations, topological reasoning, or egocentric-allocentric transformation when they matter.
- navigation_mode and ability must use only the allowed values above.
- spatial_reasoning must contain exactly one key for every listed ability plus navigation_rationale. Do not omit an ability key and do not add keys that are not listed in ability.
- spatial_reasoning.navigation_rationale must synthesize the listed ability judgments and explicitly connect them to decision.action.
- candidate_comparison is required and must compare the selected candidate with plausible alternatives using concrete candidate IDs, geometry, target evidence, or stop/recovery constraints.
- If navigation_mode is stop_at_goal, decision.action must be candidate_-1. If decision.action is candidate_-1, navigation_mode must be stop_at_goal. If decision.action is candidate_0, navigation_mode must be recover_route.
- Do not output evidence, state, decision.mode, scores, candidate tables, markdown, a separate action dict, or extra prose.
- The executor parses decision.action directly."""


def render_abilitymode715_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Locked prompt contract from the verified 71.5% SR full-1000 run."""
    prompt = _render_selective_abilitymode_action_prompt(
        goal=goal,
        candidate_lines=candidate_lines,
        valid_actions=valid_actions,
        memory=memory,
        context_name=context_name,
    )
    prompt = prompt.replace("explore_new_area", "enter_new_area")
    for line in (
        "- Choose only abilities that matter for this image and action choice. Do not add topological, spatial, route, or egocentric abilities just to fill the 2 to 4 slots.\n",
        "- Each spatial_reasoning value must be grounded in this scene: mention concrete target evidence, candidate IDs, red-arrow geometry, openings, obstacles, room layout, or viewpoint relations where relevant.\n",
        '- Do not use generic template text such as "the decision compares the relative layout", "local topology is used only insofar", or "egocentric arrow direction is interpreted as a route".\n',
        "- The schema example uses {example_action}; choose the actual valid candidate that best satisfies the policy for this image.\n",
        "- ability must contain 2 to 4 values. Do not always use only target visibility and distance measurement; include route planning, spatial relations, topological reasoning, or egocentric-allocentric transformation when they matter.\n",
        "- spatial_reasoning.navigation_rationale must synthesize the listed ability judgments and explicitly connect them to decision.action.\n",
        "- candidate_comparison is required and must compare the selected candidate with plausible alternatives using concrete candidate IDs, geometry, target evidence, or stop/recovery constraints.\n",
        "- If navigation_mode is stop_at_goal, decision.action must be candidate_-1. If decision.action is candidate_-1, navigation_mode must be stop_at_goal. If decision.action is candidate_0, navigation_mode must be recover_route.\n",
    ):
        prompt = prompt.replace(line, "")
    prompt = prompt.replace(
        '"ability": ["two to four allowed ability values"]',
        '"ability": ["one or more allowed ability values"]',
    )
    prompt = re.sub(
        r'"decision": \{"action": "candidate_-?\d+", "should_stop": (?:true|false)\}',
        '"decision": {"action": "candidate_<number>", "should_stop": <boolean>}',
        prompt,
    )
    prompt = re.sub(
        r"- The schema example uses candidate_-?\d+; choose the actual valid candidate that best satisfies the policy for this image\.\n",
        "",
        prompt,
    )
    prompt = prompt.replace(
        '"<each ability value listed in ability>": "concise free-form spatial judgment for this image and goal"',
        '"<ability value from ability>": "concise spatial judgment for this image and goal"',
    )
    prompt = prompt.replace(
        '"navigation_rationale": "integrated spatial reasoning that combines the listed abilities and explains why the selected decision.action follows"',
        '"navigation_rationale": "how the selected action follows from the spatial judgment"',
    )
    prompt = prompt.replace(
        '"candidate_comparison": "concrete comparison between the chosen candidate and the main alternatives using candidate IDs and scene geometry"',
        '"candidate_comparison": "why the chosen candidate is better than the main alternatives"',
    )
    prompt = prompt.replace(
        "- spatial_reasoning must contain exactly one key for every listed ability plus navigation_rationale. Do not omit an ability key and do not add keys that are not listed in ability.\n",
        "- spatial_reasoning must contain navigation_rationale plus exactly one key for each listed ability.\n",
    )
    prompt = prompt.replace(
        "- decision.should_stop must be the JSON boolean true only when decision.action is candidate_-1; otherwise false.\n",
        "- <boolean> means true only when decision.action is candidate_-1, otherwise false; never output the literal token <boolean>.\n",
    )
    return prompt


def render_state_routed_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Single-pass causal state -> ability -> reasoning -> action protocol."""
    prompt = render_abilitymode715_action_prompt(
        goal=goal,
        candidate_lines=candidate_lines,
        valid_actions=valid_actions,
        memory=memory,
        context_name=context_name,
    ).replace(
        f"Protocol: {ABILITYMODE_PROTOCOL_ID}",
        f"Protocol: {STATE_ROUTED_PROTOCOL_ID}",
        1,
    )
    old_abilities = """Allowed ability values:
- target visibility
- distance measurement
- route planning
- topological reasoning
- spatial relations
- egocentric-allocentric transformation
"""
    new_abilities = """Allowed ability values:
- target_visibility
- spatial_relation
- local_affordance
- self_orientation
- metric_distance
- semantic_layout_topology
- object_location_prior
- scene_object_matching
- exploration_reasoning
"""
    prompt = prompt.replace(old_abilities, new_abilities, 1)
    usage_start = prompt.index("Ability usage:\n")
    usage_end = prompt.index("\nReturn exactly one compact JSON object:\n", usage_start)
    prompt = (
        prompt[:usage_start]
        + """Ability usage:
- Select exactly three abilities from the state-specific pool below.
- Each selected ability must contribute distinct image-grounded evidence.
- Do not generate or use reasoning for an ability outside the selected route.
"""
        + prompt[usage_end:]
    )
    routing = f"""
State-conditioned spatial reasoning activation:
- First emit navigation_state before any ability or reasoning tokens.
- navigation_state must equal one allowed navigation_mode value.
- Activate exactly three distinct abilities from the pool assigned to that state:
  - search_current_area -> target_visibility, semantic_layout_topology, object_location_prior, scene_object_matching, exploration_reasoning, local_affordance
  - enter_new_area -> semantic_layout_topology, exploration_reasoning, local_affordance, self_orientation, spatial_relation
  - approach_visible_target -> target_visibility, metric_distance, spatial_relation, local_affordance, self_orientation
  - recover_route -> self_orientation, local_affordance, spatial_relation, exploration_reasoning, semantic_layout_topology
  - stop_at_goal -> target_visibility, metric_distance, spatial_relation
- spatial_reasoning must contain only the three activated abilities. Do not reason with inactive abilities.
- The final Move-Stop decision must follow from the activated reasoning and the image. navigation_mode must equal navigation_state.
"""
    marker = "\nReturn exactly one compact JSON object:\n"
    prompt = prompt.replace(marker, routing + marker, 1)
    prompt = prompt.replace(
        '  "ability": ["one or more allowed ability values"],\n'
        '  "navigation_mode": "one allowed navigation_mode value",',
        '  "navigation_state": "one allowed navigation_mode value",\n'
        '  "ability": ["exactly three state-assigned ability values"],\n'
        '  "navigation_mode": "same value as navigation_state",',
    )
    prompt = prompt.replace(
        "- navigation_mode and ability must use only the allowed values above.\n",
        "- navigation_state and navigation_mode must use only the allowed values above and must be equal.\n"
        "- ability must exactly match the three-ability route assigned to navigation_state, with no extra or missing ability.\n",
    )
    return prompt


def render_dynamic_state_routed_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """Single-pass state -> dynamic Top-3 abilities -> reasoning -> action."""
    prompt = render_state_routed_action_prompt(
        goal=goal,
        candidate_lines=candidate_lines,
        valid_actions=valid_actions,
        memory=memory,
        context_name=context_name,
    ).replace(
        f"Protocol: {STATE_ROUTED_PROTOCOL_ID}",
        f"Protocol: {DYNAMIC_STATE_ROUTED_PROTOCOL_ID}",
        1,
    )
    fixed_routing = """State-conditioned spatial reasoning activation:
- First emit navigation_state before any ability or reasoning tokens.
- navigation_state must equal one allowed navigation_mode value.
- Activate exactly three distinct abilities from the pool assigned to that state:
  - search_current_area -> target_visibility, semantic_layout_topology, object_location_prior, scene_object_matching, exploration_reasoning, local_affordance
  - enter_new_area -> semantic_layout_topology, exploration_reasoning, local_affordance, self_orientation, spatial_relation
  - approach_visible_target -> target_visibility, metric_distance, spatial_relation, local_affordance, self_orientation
  - recover_route -> self_orientation, local_affordance, spatial_relation, exploration_reasoning, semantic_layout_topology
  - stop_at_goal -> target_visibility, metric_distance, spatial_relation
- spatial_reasoning must contain only the three activated abilities. Do not reason with inactive abilities.
- The final Move-Stop decision must follow from the activated reasoning and the image. navigation_mode must equal navigation_state.
"""
    dynamic_routing = """State-conditioned dynamic spatial reasoning activation:
- First emit navigation_state before any ability or reasoning tokens.
- navigation_state must equal one allowed navigation_mode value.
- Then dynamically activate exactly three distinct abilities from the full nine-ability vocabulary.
- Use the state as a prior, not a rigid ability whitelist:
  - search_current_area: prioritize target confirmation, scene matching, topology, object-location prior, exploration value, or local affordance.
  - enter_new_area: prioritize topology, exploration value, local affordance, orientation, or spatial relation.
  - approach_visible_target: prioritize target confirmation and the most uncertain two of distance, spatial relation, affordance, orientation, or scene matching.
  - recover_route: prioritize orientation, affordance, spatial relation, exploration value, or topology according to the observed failure.
  - stop_at_goal: verify both target identity and closeness; use scene_object_matching when identity is ambiguous and spatial_relation when intervening geometry is ambiguous.
- An ability outside the usual state priorities is allowed only when it resolves concrete image-grounded uncertainty.
- spatial_reasoning must contain only the three activated abilities. Do not reason with inactive abilities.
- The final Move-Stop decision must follow from the activated reasoning and the image. navigation_mode must equal navigation_state.
"""
    if fixed_routing not in prompt:
        raise RuntimeError("state-routed prompt template changed; dynamic replacement failed")
    prompt = prompt.replace(fixed_routing, dynamic_routing, 1)
    prompt = prompt.replace(
        "- Select exactly three abilities from the state-specific pool below.",
        "- Select exactly three abilities dynamically from the full vocabulary using the state prior and visible uncertainty.",
        1,
    )
    prompt = prompt.replace(
        '"ability": ["exactly three state-assigned ability values"]',
        '"ability": ["exactly three dynamically activated ability values"]',
        1,
    )
    prompt = prompt.replace(
        "- ability must exactly match the three-ability route assigned to navigation_state, with no extra or missing ability.",
        "- ability must contain exactly three distinct allowed values selected for the current state and image, with no extra or missing ability.",
        1,
    )
    return prompt


def upgrade_dynamic_prompt_to_evidence_bridge(prompt: str) -> str:
    """Replace the verbose V4 response contract with the compact V5 bridge."""
    protocol_line = f"Protocol: {DYNAMIC_STATE_ROUTED_PROTOCOL_ID}"
    if protocol_line not in prompt:
        raise ValueError("evidence bridge requires a dynamic-state-routed prompt")
    prompt = prompt.replace(
        protocol_line,
        f"Protocol: {EVIDENCE_BRIDGE_PROTOCOL_ID}",
        1,
    )
    marker = "Decision priority:\n"
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("decision block missing from dynamic prompt")
    bridge_contract = """State-conditioned spatial evidence bridge:
- Predict in this causal order: navigation_state -> ability -> spatial_evidence -> stop_evidence -> decision.
- navigation_state: search_current_area, enter_new_area, approach_visible_target, recover_route, or stop_at_goal.
- Activate exactly three abilities from: target_visibility, spatial_relation, local_affordance, self_orientation, metric_distance, semantic_layout_topology, object_location_prior, scene_object_matching, exploration_reasoning.
- spatial_evidence has exactly those three ability keys; each value is one short claim grounded in the image or public memory. No long chain of thought.
- stop_evidence is confirmed_close when the target is visually confirmed at the current reachable stop pose; visible_not_close when visible/localized but an approach gap remains; otherwise route_not_stop.
- confirmed_close requires stop_at_goal and candidate_-1. visible_not_close requires approach_visible_target and Move. route_not_stop requires search, enter, or recover and Move.
- For Move, compare the red-arrow candidates and choose the locally executable one supporting the state. candidate_0 is recovery only.
- Never use simulator distance, oracle geometry, target coordinates, or private metadata.

Return JSON only:
{"navigation_state":"...","ability":["...","...","..."],"spatial_evidence":{"ability_1":"one visible claim","ability_2":"one visible claim","ability_3":"one visible claim"},"stop_evidence":"confirmed_close|visible_not_close|route_not_stop","action_evidence":"one short candidate comparison","decision":{"action":"candidate_<number>","should_stop":false}}

Constraints: action must be valid; should_stop is true iff action is candidate_-1; evidence keys exactly match ability; output no markdown or extra prose. The executor parses decision.action directly.
"""
    return prompt[:start] + bridge_contract


def render_evidence_bridge_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """V5 SCSRA: state -> abilities -> atomic evidence -> gate -> action."""
    return upgrade_dynamic_prompt_to_evidence_bridge(
        render_dynamic_state_routed_action_prompt(
            goal=goal,
            candidate_lines=candidate_lines,
            valid_actions=valid_actions,
            memory=memory,
            context_name=context_name,
        )
    )


def upgrade_dynamic_prompt_to_factorized_bridge(prompt: str) -> str:
    """Compact SCSRA contract with a factorized reasoning-to-action bottleneck."""
    protocol_line = f"Protocol: {DYNAMIC_STATE_ROUTED_PROTOCOL_ID}"
    if protocol_line not in prompt:
        raise ValueError("factorized bridge requires a dynamic-state-routed prompt")
    prompt = prompt.replace(protocol_line, f"Protocol: {FACTORIZED_BRIDGE_PROTOCOL_ID}", 1)
    start = prompt.find("Decision priority:\n")
    if start < 0:
        raise ValueError("decision block missing from dynamic prompt")
    contract = """State-conditioned factorized spatial reasoning bridge:
- Predict in this causal order: navigation_state -> ability -> spatial_evidence -> identity_evidence + proximity_evidence -> candidate_evidence -> decision.
- navigation_state routes reasoning and is one of search_current_area, enter_new_area, approach_visible_target, recover_route, stop_at_goal.
- Dynamically activate exactly two abilities from: target_visibility, spatial_relation, local_affordance, self_orientation, metric_distance, semantic_layout_topology, object_location_prior, scene_object_matching, exploration_reasoning.
- spatial_evidence has exactly those two keys; each value is one short image-grounded claim. Do not write a long chain of thought.
- identity_evidence is confirmed only when the requested target is visibly identified; otherwise unconfirmed.
- proximity_evidence is stop_ready only when the confirmed target is at the current reachable stopping pose; approach_needed when meaningful translation remains; otherwise unknown.
- candidate_evidence names one valid candidate and gives one short visual reason it best realizes the activated reasoning.
- Stop iff identity_evidence=confirmed AND proximity_evidence=stop_ready. Otherwise choose a Move candidate. candidate_0 is recovery only.
- Never use simulator distance, oracle geometry, target coordinates, or private metadata.

Return JSON only:
{"navigation_state":"...","ability":["...","..."],"spatial_evidence":{"ability_1":"one visible claim","ability_2":"one visible claim"},"identity_evidence":"confirmed|unconfirmed","proximity_evidence":"stop_ready|approach_needed|unknown","candidate_evidence":{"candidate":"candidate_<number>","support":"one short visible reason"},"decision":{"action":"candidate_<number>","should_stop":false}}

Constraints: candidate_evidence.candidate equals decision.action; action is valid; should_stop is true iff action is candidate_-1; evidence keys exactly match ability; output no markdown or extra prose. The executor parses decision.action directly.
"""
    return prompt[:start] + contract


def render_factorized_bridge_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    return upgrade_dynamic_prompt_to_factorized_bridge(
        render_dynamic_state_routed_action_prompt(
            goal=goal,
            candidate_lines=candidate_lines,
            valid_actions=valid_actions,
            memory=memory,
            context_name=context_name,
        )
    )


def upgrade_dynamic_prompt_to_temporal_reference_bridge(prompt: str) -> str:
    """SCSRA bridge whose range evidence is grounded by a public visual reference."""
    protocol_line = f"Protocol: {DYNAMIC_STATE_ROUTED_PROTOCOL_ID}"
    if protocol_line not in prompt:
        raise ValueError("temporal-reference bridge requires a dynamic-state-routed prompt")
    prompt = prompt.replace(
        protocol_line,
        f"Protocol: {TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID}",
        1,
    )
    start = prompt.find("Decision priority:\n")
    if start < 0:
        raise ValueError("decision block missing from dynamic prompt")
    contract = """State-conditioned temporal-reference spatial reasoning bridge:
- Image convention: with two images, Image 1 is the most recent earlier frame in which this policy visually confirmed the requested target, and Image 2 is the current action-candidate image. With one image, it is the current image and no target reference exists.
- Predict in this causal order: navigation_state -> ability -> spatial_evidence -> temporal_relation -> identity_evidence + proximity_evidence -> candidate_evidence -> decision.
- navigation_state routes reasoning and is one of search_current_area, enter_new_area, approach_visible_target, recover_route, stop_at_goal.
- Dynamically activate exactly two abilities from: target_visibility, spatial_relation, local_affordance, self_orientation, metric_distance, semantic_layout_topology, object_location_prior, scene_object_matching, exploration_reasoning.
- spatial_evidence has exactly those two keys; each value is one short claim grounded only in the supplied image(s).
- temporal_relation is no_reference for one image; otherwise compare the same target across the reference and current images and output current_closer, similar_range, current_farther, or target_mismatch.
- identity_evidence is confirmed only when the requested target is visibly identified in the current image; otherwise unconfirmed.
- proximity_evidence is stop_ready only when the current confirmed target is at the reachable stopping pose; approach_needed when translation remains; otherwise unknown. Treat current_farther and target_mismatch as evidence against Stop; current_farther can still be stop_ready only when current visual scale independently shows a reachable stopping pose.
- candidate_evidence names one valid candidate and gives one short visible reason it realizes the activated reasoning.
- Stop iff identity_evidence=confirmed AND proximity_evidence=stop_ready. Otherwise choose a Move candidate. candidate_0 is recovery only.
- The reference image is public policy memory, not a stop vote. Do not require repeated Stop predictions. Never use simulator distance, poses, oracle geometry, target coordinates, or private metadata.

Return JSON only:
{"navigation_state":"...","ability":["...","..."],"spatial_evidence":{"ability_1":"one visible claim","ability_2":"one visible claim"},"temporal_relation":"no_reference|current_closer|similar_range|current_farther|target_mismatch","identity_evidence":"confirmed|unconfirmed","proximity_evidence":"stop_ready|approach_needed|unknown","candidate_evidence":{"candidate":"candidate_<number>","support":"one short visible reason"},"decision":{"action":"candidate_<number>","should_stop":false}}

Constraints: temporal_relation must match the number and content of supplied images; candidate_evidence.candidate equals decision.action; action is valid; should_stop is true iff action is candidate_-1; evidence keys exactly match ability; output no markdown or extra prose. The executor parses decision.action directly.
"""
    return prompt[:start] + contract


def render_temporal_reference_bridge_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    return upgrade_dynamic_prompt_to_temporal_reference_bridge(
        render_dynamic_state_routed_action_prompt(
            goal=goal,
            candidate_lines=candidate_lines,
            valid_actions=valid_actions,
            memory=memory,
            context_name=context_name,
        )
    )


def upgrade_dynamic_prompt_to_stable_temporal_reference_bridge(prompt: str) -> str:
    """V7.2 bridge aligned with the runtime's stable public-reference cache."""
    protocol_line = f"Protocol: {DYNAMIC_STATE_ROUTED_PROTOCOL_ID}"
    if protocol_line not in prompt:
        raise ValueError("stable temporal-reference bridge requires a dynamic-state-routed prompt")
    prompt = prompt.replace(
        protocol_line,
        f"Protocol: {STABLE_TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID}",
        1,
    )
    start = prompt.find("Decision priority:\n")
    if start < 0:
        raise ValueError("decision block missing from dynamic prompt")
    contract = """State-conditioned stable temporal-reference spatial reasoning bridge:
- Image convention is strict. With one image, only the current action-candidate image exists and temporal_relation MUST be no_reference. With two images, Image 1 is a stable earlier public RGB promoted only after this policy made at least two consecutive protocol-valid approach confirmations; Image 2 is current.
- Predict in this causal order: navigation_state -> ability -> spatial_evidence -> temporal_relation -> identity_evidence + proximity_evidence -> candidate_evidence -> decision.
- navigation_state routes reasoning and is one of search_current_area, enter_new_area, approach_visible_target, recover_route, stop_at_goal.
- Dynamically activate exactly two abilities from: target_visibility, spatial_relation, local_affordance, self_orientation, metric_distance, semantic_layout_topology, object_location_prior, scene_object_matching, exploration_reasoning.
- spatial_evidence is an object keyed by the exact two selected ability names. Never use placeholder keys such as ability_1 or ability_2. Each value is one short image-grounded claim.
- temporal_relation is no_reference for one image. For two images, compare the requested target and output exactly one of current_closer, similar_range, current_farther, target_mismatch.
- identity_evidence is confirmed only when the requested target is visibly identified in the CURRENT image; the reference alone cannot confirm identity.
- proximity_evidence is stop_ready only when the current confirmed target is at a reachable stopping pose; approach_needed when translation remains; otherwise unknown.
- With a reference, current_farther and target_mismatch prohibit Stop. Without a reference, a single current image may still justify Stop when identity and proximity are visually decisive.
- candidate_evidence names one valid candidate and gives one short visible reason connecting the two activated abilities to that candidate.
- Stop iff navigation_state=stop_at_goal, identity_evidence=confirmed, proximity_evidence=stop_ready, and temporal_relation does not prohibit Stop. Otherwise choose a Move candidate. candidate_0 is recovery only.
- The stable reference is public policy memory, not a Stop vote. Never use simulator distance, poses, oracle geometry, target coordinates, or private metadata.

Return JSON only, following this container pattern (ability names shown are illustrative but keys must match the selected names):
{"navigation_state":"...","ability":["target_visibility","metric_distance"],"spatial_evidence":{"target_visibility":"one visible claim","metric_distance":"one visible claim"},"temporal_relation":"no_reference|current_closer|similar_range|current_farther|target_mismatch","identity_evidence":"confirmed|unconfirmed","proximity_evidence":"stop_ready|approach_needed|unknown","candidate_evidence":{"candidate":"candidate_<number>","support":"one short visible reason"},"decision":{"action":"candidate_<number>","should_stop":false}}

Constraints: temporal_relation must match the supplied image count; candidate_evidence.candidate equals decision.action; action is valid; should_stop is true iff action is candidate_-1; spatial_evidence keys exactly equal ability; output no markdown or extra prose. The executor parses decision.action directly.
"""
    return prompt[:start] + contract


def render_stable_temporal_reference_bridge_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    return upgrade_dynamic_prompt_to_stable_temporal_reference_bridge(
        render_dynamic_state_routed_action_prompt(
            goal=goal,
            candidate_lines=candidate_lines,
            valid_actions=valid_actions,
            memory=memory,
            context_name=context_name,
        )
    )


def render_public_memory_state_bridge_action_prompt(
    *,
    goal: str,
    candidate_lines: Sequence[str],
    valid_actions: Sequence[int],
    temporal_reference_available: bool,
    memory: str = "",
    context_name: str = "Visible action space",
    **_: Any,
) -> str:
    """V7.3 prompt with an explicit, public policy-memory input state.

    Reference availability is known to the executor because the reference is
    policy-owned RGB memory.  It is neither simulator geometry nor an oracle
    label.  Exposing this input removes the brittle requirement that the VLM
    infer message arity from visual-token boundaries, while the two-image
    spatial comparison remains a learned visual judgment.
    """
    prompt = render_stable_temporal_reference_bridge_action_prompt(
        goal=goal,
        candidate_lines=candidate_lines,
        valid_actions=valid_actions,
        memory=memory,
        context_name=context_name,
    )
    prompt = prompt.replace(
        f"Protocol: {STABLE_TEMPORAL_REFERENCE_BRIDGE_PROTOCOL_ID}",
        f"Protocol: {PUBLIC_MEMORY_STATE_BRIDGE_PROTOCOL_ID}",
        1,
    )
    state = (
        "stable_reference_present" if temporal_reference_available
        else "stable_reference_absent"
    )
    image_contract = (
        "Exactly two images follow: Image 1 is the stable public reference and Image 2 is current. "
        "Visually compare the requested target and do not output no_reference."
        if temporal_reference_available
        else "Exactly one current image follows. No reference comparison is active, so temporal_relation MUST be no_reference."
    )
    marker = "State-conditioned stable temporal-reference spatial reasoning bridge:\n"
    public_state = (
        "Public policy-memory input (executor supplied; non-oracle):\n"
        f"- policy_memory_state: {state}\n"
        f"- {image_contract}\n"
    )
    if marker not in prompt:
        raise ValueError("stable temporal-reference contract marker missing")
    return prompt.replace(marker, marker + public_state, 1)


def render_stopping_prompt(*, goal: str) -> str:
    return f"""Protocol: {PROTOCOL_ID}_stopping_cot_visibility_distance_v1

You are the ObjectNav stopping verifier. The agent is navigating to the nearest {goal}.
Use only the current single egocentric image to judge whether the current camera position is
visually consistent with being sufficiently close to a valid viewpoint for the {goal}.

Do not estimate metric distance from the camera to the object. The stopping condition concerns
proximity to a valid goal viewpoint, not proximity to the object's surface or center.

Use coarse visual evidence such as target visibility, target scale or cropping, remaining
traversable space, depth ordering, and intervening structures. No single cue, including apparent
object size, is sufficient by itself.

Treat stop=true as requiring both reliable target confirmation and a viewpoint that already
appears close enough to finish. If the target is merely visible but there is still clear
traversable space, a doorway threshold, a counter, large furniture, glass, or another intervening
structure between the camera and a reasonable viewing position, prefer stop=false.

A stop target does not need perfect centering, a frontal view, or a nicer alignment step. However,
continue when the image clearly indicates that substantial navigation toward a valid viewing
position remains.

Return exactly one compact JSON object:
{{
  "spatial_reasoning": {{
    "target_visibility": "short natural judgment about whether the {goal} is visually confirmed",
    "distance_measurement": "short natural judgment about whether the current viewpoint is close enough to stop"
  }},
  "reason": "one concise sentence linking the visual evidence to the stop decision",
  "stop": <true or false>
}}

Strict constraints:
- spatial_reasoning must contain exactly the two keys shown above and no navigation rationale.
- stop must be true only when the target is visually confirmed and the current viewpoint already appears close enough for stopping.
- stop must be false when the target is absent, ambiguous, likely wrong, or visible but still separated from a reasonable stopping viewpoint.
- Do not output candidate labels, movement choices, routes, navigation modes, old stopping fields, markdown, or extra prose.
- Output JSON only, with no markdown or extra prose."""


def render_stopping715_prompt(*, goal: str) -> str:
    """Locked stopping contract from the verified 71.5% SR run."""
    return f"""Protocol: {PROTOCOL_ID}_stopping_cot_visibility_distance_v1

You are the ObjectNav stopping verifier. The agent is navigating to the nearest {goal}.
Use only the current egocentric image to decide whether the agent should stop at this exact camera position.
Focus the reasoning only on target visibility and target distance. Do not plan navigation actions.

Stop decision policy:
- Default to stop=false. Use stop=true only with strong visual evidence that the agent is already at the reachable stop pose.
- stop=true requires all of these: the object identity is clear, the object is in the immediate reachable foreground, there is visible nearby floor/contact geometry or clear direct adjacency, and no meaningful walking step or alignment step remains.
- stop=false when the {goal} is absent, uncertain, a different object, reflected/picture-like, merely inferred from room context, or visible but still separated by open floor, furniture, a doorway, a side view, or any meaningful walking/alignment gap.
- Large, centered, or salient in the image is not enough. A wall-mounted TV, object across a room, object beyond furniture, object at the end of a corridor, or object with uncertain floor/contact relation must be stop=false.
- For beds, sofas, chairs, toilets, plants, and TVs, stop only when the agent appears directly beside or facing the reachable object at near interaction distance. If you would need one more movement to touch/reach it, output stop=false.
- Do not require perfect centering or full-object visibility for a confirmed-close target, but when uncertain, output stop=false.

Return exactly one compact JSON object:
{{
  "spatial_reasoning": {{
    "target_visibility": "one concise free-form judgment about whether the {goal} is visually confirmed in the current image",
    "distance_measurement": "one concise free-form judgment about whether the confirmed {goal} is close enough to stop or still separated"
  }},
  "reason": "one short stop-related spatial reason",
  "stop": <true or false>
}}

Strict constraints:
- spatial_reasoning must contain exactly the two keys shown above and no navigation rationale.
- stop must be true only when the target is visually confirmed and the current camera pose is close enough for stopping.
- stop must be false when the target is absent, ambiguous, the wrong object, or visible but still too far, blocked, side-viewed, or separated by a movement gap.
- Do not output candidate labels, movement choices, routes, navigation modes, old stopping fields, markdown, or extra prose.
- Output JSON only, with no markdown or extra prose."""


def render_direct_stopping_prompt(*, goal: str) -> str:
    return f"""Protocol: objectnav_direct_stop_v1

You are the ObjectNav stopping verifier. The agent is navigating to the nearest {goal}.
Use only the current single egocentric image to judge whether this exact camera position already appears close enough to finish.

Stop boundary:
- Default to done=0.
- Use done=1 only when the {goal} is visually confirmed and the viewpoint already appears close enough for a reliable stop.
- Do not estimate metric distance to the object surface or center; judge whether the current viewpoint is close enough to a valid goal viewpoint.
- Use coarse visual evidence: target visibility, target scale/cropping, remaining traversable space, depth ordering, and intervening structures.
- A target does not need perfect centering, a frontal view, or a nicer alignment step.
- However, if the target is merely visible but there is still clear traversable space, a doorway threshold, counter, large furniture, glass, or another intervening structure before a reasonable viewing position, output done=0.
- If one more movement or alignment step still seems necessary, output done=0.

Return exactly one compact JSON object and nothing else:
{{"done":0_or_1}}
"""


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def parse_final_action_line(text: str) -> int | None:
    for raw in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        if "action" not in raw:
            continue
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            continue
        if isinstance(obj, Mapping) and "action" in obj:
            try:
                return int(obj["action"])
            except Exception:
                return None
    return None


def validate_response(
    text: str,
    valid_actions: Sequence[int] | None = None,
    *,
    require_final_action: bool = False,
    protocol_id: str = ABILITYMODE_PROTOCOL_ID,
) -> list[str]:
    issues: list[str] = []
    obj = extract_first_json_object(text)
    if obj is None:
        return ["missing_json"]
    top_level_keys = ABILITYMODE_TOP_LEVEL_KEYS if protocol_id == ABILITYMODE_PROTOCOL_ID else TOP_LEVEL_KEYS
    missing = [key for key in top_level_keys if key not in obj]
    issues.extend(f"missing_key:{key}" for key in missing)
    decision = obj.get("decision") if isinstance(obj.get("decision"), Mapping) else {}
    action_label = normalize_decision_action(decision.get("action", ""))
    raw = raw_action_from_decision(action_label)
    if raw is None:
        issues.append("bad_decision_action")
    if require_final_action:
        final_raw = parse_final_action_line(text)
        if final_raw is None:
            issues.append("missing_final_action")
        elif raw is not None and final_raw != raw:
            issues.append("final_action_mismatch")
    if valid_actions is not None and raw is not None and raw not in set(int(a) for a in valid_actions):
        issues.append("action_not_in_valid_actions")
    should_stop = bool(decision.get("should_stop"))
    if (raw == -1) != should_stop:
        issues.append("stop_action_should_stop_mismatch")
    reasoning = obj.get("spatial_reasoning")
    if not isinstance(reasoning, Mapping):
        issues.append("bad_spatial_reasoning")
    elif protocol_id == ABILITYMODE_PROTOCOL_ID:
        abilities = obj.get("ability")
        if not isinstance(abilities, list) or not abilities:
            issues.append("bad_ability")
            ability_values: list[str] = []
        else:
            ability_values = [str(item) for item in abilities]
            if len(ability_values) < 2 or len(ability_values) > 4:
                issues.append("bad_ability_count")
            if len(set(ability_values)) != len(ability_values):
                issues.append("duplicate_ability")
            for ability in ability_values:
                if ability not in ABILITYMODE_ABILITY_VOCAB:
                    issues.append(f"unknown_ability:{ability}")
        mode = str(obj.get("navigation_mode", ""))
        if mode not in ABILITYMODE_NAVIGATION_MODE_VOCAB:
            issues.append(f"unknown_navigation_mode:{mode}")
        expected_reasoning_keys = set(ability_values) | {"navigation_rationale"}
        for key in expected_reasoning_keys:
            if not str(reasoning.get(key, "")).strip():
                issues.append(f"missing_spatial_reasoning:{key}")
        extra_reasoning_keys = set(reasoning) - expected_reasoning_keys
        for key in sorted(extra_reasoning_keys):
            issues.append(f"extra_spatial_reasoning:{key}")
        if raw == -1 and mode != "stop_at_goal":
            issues.append("stop_action_without_stop_mode")
        if raw != -1 and mode == "stop_at_goal":
            issues.append("stop_mode_without_stop_action")
        if raw == 0 and mode != "recover_route":
            issues.append("candidate0_without_recover_route")
        for forbidden in ("evidence", "state"):
            if forbidden in obj:
                issues.append(f"forbidden_key:{forbidden}")
        if isinstance(decision, Mapping) and "mode" in decision:
            issues.append("forbidden_key:decision.mode")
    else:
        for key in SPATIAL_REASONING_KEYS:
            if not str(reasoning.get(key, "")).strip():
                issues.append(f"missing_spatial_reasoning:{key}")
    return issues


def prompt_has_protocol(prompt: str) -> bool:
    return bool(re.search(rf"Protocol:\s*{re.escape(PROTOCOL_ID)}\b", prompt))
