"""The historical action-only Move request, independent of serving backend."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from embodied_nav_cot.move_sc_rle_protocol import public_move_inputs_from_messages


SYSTEM_PROMPT = "You are a navigation spatial reasoning assistant."
PROTOCOL_ID = "objectnav_direct_move_only_v2"


def build_sc_esr_direct_move_prompt(*, goal: str, actions: Sequence[str]) -> str:
    action_text = ", ".join(str(action) for action in actions)
    return (
        "Protocol: objectnav_direct_move_only_v2\n\n"
        f"Goal object: nearest {goal}.\n"
        f"Valid Move actions: {action_text}.\n"
        "A separate visual Stop controller decides termination. Inspect the "
        "current candidate-overlay RGB image and return exactly one action-only "
        "JSON object using a valid Move action.\n\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"decision":{"action":"candidate_<number>"}}'
    )


def canonicalize_sc_esr_move_messages(
    messages: Sequence[dict[str, Any]], *, image_count: int
) -> list[dict[str, Any]]:
    """Match the historical SC-ESR proxy's messages exactly, before image binding."""
    if image_count != 1:
        raise ValueError(f"SC-ESR Move requires one current RGB, got {image_count}")
    goal, actions = public_move_inputs_from_messages(messages)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_sc_esr_direct_move_prompt(goal=goal, actions=actions)},
            ],
        },
    ]


def canonicalize_sc_esr_openai_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace the old navigation prompt without re-encoding its single RGB.

    Only the public goal, visible candidate IDs and image survive the conversion.
    In particular, candidate_-1, prior instructions and should_stop are omitted.
    """
    image_urls = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                value = item.get("image_url", {})
                url = value.get("url") if isinstance(value, dict) else None
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("SC-ESR Move requires an inline base64 RGB image")
                image_urls.append(url)
    canonical = canonicalize_sc_esr_move_messages(messages, image_count=len(image_urls))
    for message in canonical:
        message["content"] = [
            {"type": "image_url", "image_url": {"url": image_urls[0]}}
            if item["type"] == "image" else item
            for item in message["content"]
        ]
    return canonical


def validate_move_output(text: str, *, legal: Sequence[str]) -> str:
    """Validate the historical action-only schema; never manufacture an action."""
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError) as exc:
        raise ValueError("completion is not exactly one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"decision"} or not isinstance(value["decision"], dict):
        raise ValueError("Move completion has the wrong top-level schema")
    decision = value["decision"]
    if set(decision) != {"action"}:
        raise ValueError("Move completion has the wrong decision schema")
    action = decision["action"]
    if not isinstance(action, str) or re.fullmatch(r"candidate_\d+", action) is None:
        raise ValueError("Move completion has an invalid action")
    if action not in legal:
        raise ValueError(f"Move completion is outside legal actions: {action}")
    return action


def isolate_valid_move_json(text: str, *, legal: Sequence[str]) -> str:
    """Match the historical proxy: strip prose only around a valid Move object.

    Invalid schemas, missing actions and out-of-range candidates raise instead
    of being coerced into a navigation decision. The client can then retry.
    """
    try:
        validate_move_output(text, legal=legal)
        return text
    except ValueError as original_error:
        start = 0
        while True:
            start = text.find("{", start)
            if start < 0:
                raise original_error
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, len(text)):
                char = text[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            validate_move_output(candidate, legal=legal)
                            return candidate
                        except ValueError:
                            break
            start += 1
