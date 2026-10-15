"""Stop auditing captures SDK inputs, not the projected Move visualization."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes/src"))
from embodied_nav_cot.stop_request_audit import record_stop_exchange
from test_move_protocol import runtime_client


class StopRequestAuditTest(unittest.TestCase):
    def test_disabled_is_a_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            record_stop_exchange({"not_serializable": object()}, "", {})

    def test_body_roundtrip_hash_deduplication_and_no_mutation(self):
        request = {"model": "stop", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Stop boundary prompt"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,dGVzdA=="}},
        ]}], "temperature": 0, "max_tokens": 16, "logprobs": True}
        original = copy.deepcopy(request)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "VLMNAV_STOP_REQUEST_AUDIT_DIR": directory,
            "VLMNAV_STOPPING_OPENAI_API_KEY": "do-not-capture-me",
        }, clear=True):
            for margin in (1.75, 2.25):
                record_stop_exchange(request, '{"done":1}', {"margin_stop_minus_continue": margin})
            root = Path(directory)
            bodies = list((root / "requests").glob("*.json"))
            self.assertEqual(len(bodies), 1)
            raw = bodies[0].read_bytes()
            self.assertEqual(bodies[0].stem, hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads(raw), request)
            responses = [json.loads(path.read_text()) for path in (root / "responses").glob("*.json")]
            self.assertEqual(len(responses), 2)
            self.assertEqual({item["request_sha256"] for item in responses}, {bodies[0].stem})
            self.assertNotIn("do-not-capture-me", raw.decode())
        self.assertEqual(request, original)

    def test_real_clients_audit_after_success_without_changing_request_or_retrying_audit_errors(self):
        for runtime in ("hm3dv1", "hm3dv2", "mp3d"):
            with self.subTest(runtime=runtime), patch.dict(os.environ, {}, clear=True):
                client = runtime_client(runtime)
                client.model = "spatial_nav_stop"
                binary = {"valid": True, "margin_stop_minus_continue": 1.75}
                client._stop_binary_logprobs_from_response = Mock(return_value=binary)
                client.client.chat.completions.create.return_value = SimpleNamespace(choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"done":1}'))])
                with patch("embodied_nav_cot.stop_request_audit.record_stop_exchange") as record:
                    result = client.call_stop_with_logprobs([], "actual stop prompt")
                    sent = client.client.chat.completions.create.call_args.kwargs
                    self.assertEqual(result, ('{"done":1}', binary))
                    record.assert_called_once_with(sent, '{"done":1}', binary)
                    record.side_effect = OSError("audit disk full")
                    client.client.chat.completions.create.reset_mock()
                    with self.assertRaisesRegex(OSError, "audit disk full"):
                        client.call_stop_with_logprobs([], "actual stop prompt")
                    client.client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
