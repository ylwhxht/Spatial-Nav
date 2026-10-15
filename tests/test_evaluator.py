"""CPU-only checks for reproducible evaluation configuration and subsets."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "codes" / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class EvaluatorTest(unittest.TestCase):
    def test_base_protocols_thresholds_and_no_adapters(self):
        for architecture, protocol, canonical in (
            ("qwen3-vl-8b", "objectnav_direct_move_only_v2", "1"),
            ("internvl3-8b", "objectnav_direct_action_v1", "0"),
        ):
            with self.subTest(architecture=architecture):
                command = [sys.executable, str(SCRIPT), "--architecture", architecture,
                           "--variant", "base", "--base-model", "missing-base",
                           "--dataset", "hm3d_v2", "--scene-root", "missing-scenes",
                           "--objectnav-root", "missing-episodes", "--run-dir", "not-created", "--dry-run"]
                config = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
                self.assertEqual(config["move_wire_protocol"], protocol)
                self.assertEqual(config["stop_threshold"], 0.0)
                self.assertIsNone(config["move_checkpoint"])
                self.assertIsNone(config["stop_checkpoint"])
                env = config["runtime_environment"]
                self.assertEqual(env["VLMNAV_PROTOCOL_ID"], "objectnav_direct_action_v1")
                self.assertEqual(env["VLMNAV_ACTION_CANONICAL_MOVE"], canonical)
                self.assertEqual(env["VLMNAV_ACTION_OPENAI_MAX_TOKENS"], "24")
                self.assertEqual(env["VLMNAV_STOPPING_OPENAI_MAX_TOKENS"], "16")
                self.assertEqual(env["VLMNAV_STOP_VETO_RUNTIME_FAR"], "0")
                self.assertEqual(env["VLMNAV_LEGACY_FOUR_CELL_ACTION_PROPOSER_SCORE"], "0")
                for commands in config["server_commands"].values():
                    for cmd in commands:
                        self.assertNotIn("--enable-lora", cmd)
                        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "3072")
                overridden = json.loads(subprocess.run(
                    command + ["--base-move-protocol", "objectnav_direct_action_v1", "--stop-threshold", "1.5"],
                    capture_output=True, text=True, check=True).stdout)
                self.assertEqual(overridden["move_wire_protocol"], "objectnav_direct_action_v1")
                self.assertEqual(overridden["runtime_environment"]["VLMNAV_ACTION_CANONICAL_MOVE"], "0")
                self.assertEqual(overridden["stop_threshold"], 1.5)
                command[command.index("base")] = "checkpoint"
                rejected = subprocess.run(command + ["--base-move-protocol", protocol],
                                          capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("only valid with --variant base", rejected.stderr)

    def test_dataset_specific_server_sequence_limits(self):
        for dataset, expected in (("hm3d_v1", 32), ("hm3d_v2", 64), ("mp3d", 48)):
            with self.subTest(dataset=dataset):
                command = [sys.executable, str(SCRIPT), "--architecture", "qwen3-vl-8b",
                           "--variant", "checkpoint", "--base-model", "missing-base",
                           "--dataset", dataset, "--scene-root", "missing-scenes",
                           "--objectnav-root", "missing-episodes", "--run-dir", "not-created",
                           "--dry-run"]
                config = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
                self.assertEqual(config["max_num_seqs"], expected)

    def test_dry_run_records_four_card_layout_and_historical_flags(self):
        command = [
            sys.executable, str(SCRIPT), "--architecture", "qwen3-vl-8b", "--variant", "checkpoint",
            "--base-model", "missing-base", "--dataset", "hm3d_v2",
            "--scene-root", "missing-scenes", "--objectnav-root", "missing-episodes",
            "--run-dir", "not-created", "--move-gpus", "0,1", "--stop-gpus", "2,3",
            "--sim-gpus", "0,1,2,3", "--workers-per-gpu", "10",
            "--allow-shared-server-gpus", "--gpu-memory-utilization", "0.7", "--dry-run",
        ]
        with patch.dict(os.environ, {"VLMNAV_STOP_VETO_RUNTIME_FAR": "1",
                                     "VLMNAV_UNKNOWN_LEFTOVER": "1",
                                     "VLMNAV_ACTION_OPENAI_API_KEY": "test-secret"}):
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        config = json.loads(result.stdout)
        self.assertEqual(config["workers"], 40)
        self.assertEqual(config["workers_per_gpu"], {str(i): 10 for i in range(4)})
        self.assertEqual(sum(len(items) for items in config["model_servers"].values()), 4)
        self.assertEqual(config["episode_indices"], list(range(1000)))
        self.assertTrue(config["stop_checkpoint"].endswith("stop_merged"))
        self.assertEqual(config["move_wire_protocol"], "objectnav_direct_move_only_v2")
        self.assertEqual(config["request_routing"], "least-inflight")
        self.assertEqual(set(config["request_routers"]), {"move", "stop"})
        for role in ("move", "stop"):
            router = config["request_routers"][role]
            self.assertEqual(config[f"worker_{role}_urls"], [router["url"]] * 40)
            self.assertEqual(router["backends"], [server["url"] for server in config["model_servers"][role]])
        static = json.loads(subprocess.run(command + ["--request-routing", "static"],
                                           capture_output=True, text=True, check=True).stdout)
        self.assertEqual(static["request_routers"], {})
        self.assertEqual(len(set(static["worker_move_urls"])), 2)
        env = config["runtime_environment"]
        self.assertEqual(env["VLMNAV_STOP_VETO_RUNTIME_FAR"], "0")
        self.assertEqual(env["VLMNAV_LEGACY_FOUR_CELL_ACTION_PROPOSER_SCORE"], "0")
        self.assertNotIn("VLMNAV_UNKNOWN_LEFTOVER", env)
        self.assertNotIn("test-secret", result.stdout)
        self.assertNotIn("VLMNAV_STOP_REQUEST_AUDIT_DIR", env)
        audited = json.loads(subprocess.run(command + ["--audit-stop-inputs"],
                                           capture_output=True, text=True, check=True).stdout)
        self.assertTrue(audited["runtime_environment"]["VLMNAV_STOP_REQUEST_AUDIT_DIR"].endswith("stop_request_audit"))
        for role, limit in (("move", "3072"), ("stop", "4096")):
            for command in config["server_commands"][role]:
                self.assertEqual(command[command.index("--max-model-len") + 1], limit)
                self.assertEqual(command[command.index("--max-num-seqs") + 1], "64")

    def test_dataset_specific_routing_and_single_backend_bypass(self):
        for architecture in ("qwen3-vl-8b", "internvl3-8b"):
            for dataset in ("hm3d_v1", "hm3d_v2", "mp3d"):
                command = [sys.executable, str(SCRIPT), "--architecture", architecture,
                           "--variant", "checkpoint", "--base-model", "missing-base", "--dataset", dataset,
                           "--scene-root", "missing-scenes", "--objectnav-root", "missing-episodes",
                           "--run-dir", "not-created", "--dry-run"]
                config = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
                expected = "least-inflight" if architecture == "internvl3-8b" or dataset == "hm3d_v2" else "static"
                self.assertEqual(config["request_routing"], expected)
                self.assertEqual(config["request_routers"], {})

    def test_fixed_indices_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.json"
            args = argparse.Namespace(dataset="hm3d_v2", episodes=None, episode_indices=path)
            for values in ([3, 900, 0], [0, 0], [], [False], [-1], [1000], {"a": 0}):
                path.write_text(json.dumps(values), encoding="utf-8")
                if values == [3, 900, 0]:
                    self.assertEqual(evaluate.episode_indices(args), values)
                else:
                    with self.subTest(values=values), self.assertRaises(ValueError):
                        evaluate.episode_indices(args)
            args.episodes = 3
            with self.assertRaises(ValueError):
                evaluate.episode_indices(args)
        args.episode_indices = None
        self.assertEqual(evaluate.episode_indices(args), [0, 1, 2])

    def test_hardware_override_is_preserved_but_policy_override_is_not(self):
        with patch.dict(os.environ, {"VLMNAV_NVIDIA_EGL_VENDOR_JSON": "/driver.json",
                                     "VLMNAV_PROTOCOL_ID": "stale"}, clear=True):
            self.assertEqual(evaluate.clean_environment(), {"VLMNAV_NVIDIA_EGL_VENDOR_JSON": "/driver.json"})

    def test_preflight_rejects_missing_scene_and_navmesh(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(evaluate.EPISODE_COUNTS, {"hm3d_v1": 1}):
            root = Path(directory)
            scene_root, episode_root = root / "scenes", root / "episodes"
            folder = scene_root / "00800-fixture"
            folder.mkdir(parents=True)
            content = episode_root / "val/content"
            content.mkdir(parents=True)
            (scene_root / "hm3d.scene_dataset_config.json").write_text("{}")
            payload = {"episodes": [{"scene_id": "fixture", "episode_id": "0"}]}
            (content / "fixture.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "scene GLB"):
                evaluate.validate_eval_assets("hm3d_v1", scene_root, episode_root)
            (folder / "fixture.basis.glb").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "navmesh"):
                evaluate.validate_eval_assets("hm3d_v1", scene_root, episode_root)
            (folder / "fixture.basis.navmesh").write_bytes(b"fixture")
            inventory = evaluate.validate_eval_assets("hm3d_v1", scene_root, episode_root)
            self.assertEqual(inventory["episodes"], 1)
            self.assertEqual(len(inventory["navmesh_sha256"]), 1)
            payload["episodes"] *= 2
            (content / "fixture.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unique"):
                evaluate.validate_eval_assets("hm3d_v1", scene_root, episode_root)


if __name__ == "__main__":
    unittest.main()
