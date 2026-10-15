"""Public-input protocol for the fixed-B0 SC-RLE Move branch."""

from __future__ import annotations

import re
from typing import Any, Sequence


# Exact public ObjectNav labels exposed by HM3D and MP3D navigation runtimes.
# MP3D's ``tv_monitor`` is rendered as ``tv screen`` before prompting.
PUBLIC_GOALS = {
    "bed", "bathtub", "cabinet", "chair", "chest_of_drawers", "clothes",
    "counter", "cushion", "fireplace", "gym_equipment", "picture", "plant",
    "seating", "shower", "sink", "sofa", "stool", "table", "toilet",
    "towel", "tv screen",
}
GOAL_PATTERNS = (
    re.compile(r"Goal object:\s*nearest\s+([^.\n]+)", re.IGNORECASE),
    re.compile(r"Goal:\s*nearest\s+([^.\n]+)", re.IGNORECASE),
    re.compile(
        r"navigate\s+to\s+the\s+nearest\s+([^.\n]+)",
        re.IGNORECASE,
    ),
    re.compile(r"navigate\s+to\s+nearest\s+([^.\n]+)", re.IGNORECASE),
)
CANDIDATE_RE = re.compile(r"\bcandidate_(-?\d+)\b")


def normalize_goal(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def build_move_sc_rle_prompt(
    *,
    goal: str,
    actions: Sequence[str],
) -> str:
    goal = normalize_goal(goal)
    if goal not in PUBLIC_GOALS:
        raise ValueError(f"unsupported public ObjectNav goal: {goal!r}")
    action_numbers: list[int] = []
    for action in actions:
        match = re.fullmatch(r"candidate_(\d+)", str(action))
        if match is None:
            raise ValueError(f"invalid public Move action: {action!r}")
        action_numbers.append(int(match.group(1)))
    if (
        not action_numbers
        or action_numbers != sorted(set(action_numbers))
        or action_numbers[0] not in (0, 1)
        or action_numbers
        != list(range(action_numbers[0], action_numbers[-1] + 1))
    ):
        raise ValueError(
            "Move actions must be unique and contiguous, starting at "
            "candidate_0 or candidate_1"
        )
    action_text = ", ".join(f"candidate_{value}" for value in action_numbers)
    return (
        "Protocol: objectnav_state_conditioned_spatial_move_sc_rle_v1\n\n"
        f"Goal object: nearest {goal}.\n"
        f"Valid Move actions: {action_text}.\n"
        "First route the current visual state:\n"
        "Use only this current candidate-overlay egocentric RGB image, the "
        "public goal, and the listed candidates. Select one to three useful "
        "spatial abilities without a fixed state-to-ability mapping. State one "
        "visible fact and its current-view limit for each selected ability, "
        "then choose one legal Move action supported by that evidence. A "
        "separate Stop model owns termination.\n\n"
        "Return exactly one compact JSON object and no prose; decision must be "
        "last:\n"
        '{"navigation_state":"SEARCH_or_TARGET_ENGAGED","ability":["..."],'
        '"spatial_evidence":{"ability_name":{"visual_evidence":"...",'
        '"evidence_limit":"current RGB only"}},"decision":{"action":'
        '"candidate_<number>"}}'
    )


def _latest_user_text(messages: Sequence[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")) != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
            if text:
                return text
    raise ValueError("Move request has no current user prompt")


def public_move_inputs_from_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Extract only the public goal and candidate identifiers.

    The legacy action prompt can mention candidate_-1 because older Move
    models owned Stop.  This branch has an independent Stop model, so -1 is
    deliberately discarded and can never reach the Move generation surface.
    """

    text = _latest_user_text(messages)
    goal = ""
    for pattern in GOAL_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            goal = normalize_goal(match.group(1))
            break
    if goal not in PUBLIC_GOALS:
        raise ValueError(f"could not recover a supported public goal: {goal!r}")

    numbers = sorted(
        {
            int(value)
            for value in CANDIDATE_RE.findall(text)
            if int(value) >= 0
        }
    )
    if not numbers:
        raise ValueError("Move request exposes no non-Stop candidates")
    # Public Move surfaces are contiguous candidate_0..N when the recovery
    # action is available, and candidate_1..N when it is not. Preserve the
    # exposed identifiers exactly: never invent candidate_0 or renumber the
    # visible arrows.
    if (
        numbers[0] not in (0, 1)
        or numbers != list(range(numbers[0], numbers[-1] + 1))
    ):
        raise ValueError(f"non-contiguous public Move candidates: {numbers}")
    return goal, [f"candidate_{value}" for value in numbers]


def canonicalize_move_messages(
    messages: Sequence[dict[str, Any]],
    *,
    image_count: int,
) -> list[dict[str, Any]]:
    if image_count != 1:
        raise ValueError(
            f"Move closed-loop protocol requires one current RGB, got {image_count}"
        )
    goal, actions = public_move_inputs_from_messages(messages)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_move_sc_rle_prompt(
                        goal=goal,
                        actions=actions,
                    ),
                },
                {"type": "image"},
            ],
        }
    ]
