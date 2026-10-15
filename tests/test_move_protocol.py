"""CPU-only protocol checks: python -m unittest discover -s tests -v."""

import __future__
import ast
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes" / "src"))

from embodied_nav_cot.sc_esr_move_protocol import (
    SYSTEM_PROMPT,
    canonicalize_sc_esr_openai_messages,
    isolate_valid_move_json,
    validate_move_output,
)


def request(goal="chair", candidates="candidate_-1, candidate_0, candidate_1, candidate_2"):
    return [{"role": "user", "content": [
        {"type": "text", "text": (
            f"Protocol: objectnav_direct_action_v1\nGoal object: nearest {goal}.\n"
            f"Valid actions: {candidates}.\nReturn action and should_stop."
        )},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,dGVzdA=="}},
    ]}]


def runtime_client(runtime):
    """Load only the actual OpenAI client methods, without GPU dependencies."""
    source = ROOT / "codes" / "navigation" / runtime / "vlm.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenAIVLM")
    names = {"_completion_kwargs", "_completion_content", "call", "call_chat", "call_chat_with_max_tokens", "call_stop_with_logprobs"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"os": os, "json": __import__("json"), "logging": Mock(), "time": Mock()}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    methods_by_name = {node.name: namespace[node.name] for node in methods}
    client = type("Client", (), methods_by_name)()
    client.model = "spatial_nav_move"
    client.send_sampling_args = client.send_max_tokens = True
    client._chat_temperature, client._chat_max_tokens = 0.0, 24
    client.history = []
    client._record_completion_tokens = Mock()
    client._user_text_for_prompt = lambda text: text
    client._system_prefix_messages = lambda: []
    client._image_contents_from_images = lambda images: [request()[0]["content"][1]]
    client.client = Mock()
    return client


class MoveProtocolTest(unittest.TestCase):
    def test_exact_historical_message(self):
        original = request()
        untouched = copy.deepcopy(original)
        actual = canonicalize_sc_esr_openai_messages(original)
        self.assertEqual(original, untouched)
        self.assertEqual(actual, [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                original[0]["content"][1],
                {"type": "text", "text": (
                    "Protocol: objectnav_direct_move_only_v2\n\n"
                    "Goal object: nearest chair.\n"
                    "Valid Move actions: candidate_0, candidate_1, candidate_2.\n"
                    "A separate visual Stop controller decides termination. Inspect the "
                    "current candidate-overlay RGB image and return exactly one action-only "
                    "JSON object using a valid Move action.\n\n"
                    "Return exactly one compact JSON object and no prose:\n"
                    '{"decision":{"action":"candidate_<number>"}}'
                )},
            ]},
        ])

    def test_all_goals_and_candidate_zero_presence(self):
        for goal in (
            "bed", "bathtub", "cabinet", "chair", "chest_of_drawers", "clothes",
            "counter", "cushion", "fireplace", "gym_equipment", "picture",
            "plant", "seating", "shower", "sink", "sofa", "stool", "table",
            "toilet", "towel", "tv screen",
        ):
            for first in (0, 1):
                with self.subTest(goal=goal, first=first):
                    labels = ", ".join(f"candidate_{i}" for i in [-1, *range(first, 5)])
                    actual = canonicalize_sc_esr_openai_messages(request(goal, labels))
                    text = actual[1]["content"][1]["text"]
                    self.assertIn(f"nearest {goal}.", text)
                    self.assertNotIn("candidate_-1", text)
                    self.assertNotIn("should_stop", text)
                    self.assertEqual("candidate_0" in text, first == 0)

    def test_invalid_goal_or_candidates_fail(self):
        for args in [("unknown", "candidate_0"), ("chair", "candidate_-1"),
                     ("chair", "candidate_0, candidate_2"), ("chair", "candidate_2")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                canonicalize_sc_esr_openai_messages(request(*args))

    def test_requires_exactly_one_inline_image(self):
        for count in (0, 2):
            messages = request()
            image = messages[0]["content"].pop()
            messages[0]["content"].extend([image] * count)
            with self.subTest(count=count), self.assertRaises(ValueError):
                canonicalize_sc_esr_openai_messages(messages)
        messages = request()
        messages[0]["content"][1]["image_url"]["url"] = "https://example.invalid/image.jpg"
        with self.assertRaises(ValueError):
            canonicalize_sc_esr_openai_messages(messages)

    def test_replaces_prior_system_and_is_idempotent(self):
        messages = [{"role": "system", "content": "old system"}, *request()]
        actual = canonicalize_sc_esr_openai_messages(messages)
        self.assertEqual(actual, canonicalize_sc_esr_openai_messages(actual))

    def test_output_accepts_only_legal_action_schema(self):
        legal = ("candidate_0", "candidate_1", "candidate_2")
        valid = '{"decision":{"action":"candidate_1"}}'
        self.assertEqual(validate_move_output(valid, legal=legal), "candidate_1")
        for text in (" " + valid + "\n", "Answer: " + valid, "```json\n" + valid + "\n```",
                     '{"irrelevant":"a } brace"} ' + valid):
            with self.subTest(text=text):
                actual = isolate_valid_move_json(text, legal=legal)
                self.assertEqual(validate_move_output(actual, legal=legal), "candidate_1")
        for text in ("", valid[:-1], "1", "null", '{"decision":{"action":1}}',
                     '{"decision":{"action":"candidate_9"}}',
                     '{"decision":{"action":"candidate_-1"}}',
                     '{"decision":{"action":"candidate_1","should_stop":false}}',
                     '{"action":"candidate_1"}', '{"decision":null}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                isolate_valid_move_json(text, legal=legal)

    def test_real_client_paths_validate_retry_and_leave_noncanonical_stop_unchanged(self):
        valid = '{"decision":{"action":"candidate_1"}}'
        prompt = request()[0]["content"][0]["text"]
        for runtime in ("hm3dv1", "hm3dv2", "mp3d"):
            client = runtime_client(runtime)
            calls = [lambda: client.call([], prompt), lambda: client.call_chat(0, [], prompt)]
            if hasattr(client, "call_chat_with_max_tokens"):
                calls.append(lambda: client.call_chat_with_max_tokens(0, [], prompt, 24))
            with patch.dict(os.environ, {
                "VLMNAV_ACTION_CANONICAL_MOVE": "1", "VLMNAV_ACTION_MODEL": "spatial_nav_move",
            }, clear=True):
                for call in calls:
                    create = client.client.chat.completions.create
                    create.return_value = SimpleNamespace(choices=[
                        SimpleNamespace(message=SimpleNamespace(content="Answer: " + valid))])
                    self.assertEqual(call(), valid)
                    create.return_value.choices[0].message.content = '{"decision":{"action":"candidate_9"}}'
                    create.reset_mock()
                    self.assertEqual(call(), "OPENAI API ERROR")
                    self.assertEqual(create.call_count, 5)
                    self.assertEqual(client.history, [])
                response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="raw"))])
                client.model = "spatial_nav_stop"
                self.assertEqual(client._completion_content(response, []), "raw")
                client.model = "spatial_nav_move"
                os.environ["VLMNAV_ACTION_CANONICAL_MOVE"] = "0"
                self.assertEqual(client._completion_content(response, []), "raw")

    def test_every_runtime_only_canonicalizes_enabled_move(self):
        # Compile the actual client method without importing GPU/vision packages.
        # This also detects a missing hook in any of the three runtimes.
        for runtime in ("hm3dv1", "hm3dv2", "mp3d"):
            source = ROOT / "codes" / "navigation" / runtime / "vlm.py"
            tree = ast.parse(source.read_text(encoding="utf-8"))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenAIVLM")
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_completion_kwargs")
            namespace = {"os": os, "json": __import__("json")}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
            client = type("Client", (), {
                "model": "spatial_nav_move", "send_sampling_args": True,
                "send_max_tokens": True, "_chat_temperature": 0.0, "_chat_max_tokens": 24,
            })()
            call = namespace["_completion_kwargs"]
            with self.subTest(runtime=runtime), patch.dict(os.environ, {
                "VLMNAV_ACTION_CANONICAL_MOVE": "1", "VLMNAV_ACTION_MODEL": "spatial_nav_move",
                "VLMNAV_OPENAI_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":false}',
            }, clear=True):
                payload = call(client, request())
                self.assertEqual(payload["messages"], canonicalize_sc_esr_openai_messages(request()))
                self.assertEqual((payload["temperature"], payload["top_p"], payload["n"], payload["max_tokens"]), (0, 1, 1, 24))
                self.assertEqual(payload["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
                client.model = "spatial_nav_stop"
                stop = [{"role": "user", "content": "Return done=0_or_1"}]
                self.assertEqual(call(client, stop)["messages"], stop)
                client.model = "spatial_nav_move"
                os.environ["VLMNAV_ACTION_CANONICAL_MOVE"] = "0"
                self.assertEqual(call(client, request())["messages"], request())


if __name__ == "__main__":
    unittest.main()
