"""Check the selected mainline's execution settings without loading a model."""
import argparse
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "codes" / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class TrainingConfigTest(unittest.TestCase):
    def test_run_identity_rejects_smoke_reuse_and_changed_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            data = root / "data"
            model.mkdir()
            data.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
            (data / "manifest.json").write_text('{"schema":"fixture"}')
            args = argparse.Namespace(
                architecture="qwen3-vl-8b", base_model=model, data=data,
                output=root / "output", gpus=8, smoke=True,
                foundation_seed=1, move_seed=2, move_opd_seed=3, stop_seed=4,
            )
            train.verify_run_identity(args)
            record = args.output / "training_run.json"
            original = record.read_bytes()
            train.verify_run_identity(args)
            self.assertEqual(record.read_bytes(), original)
            for field, value in (("smoke", False), ("gpus", 4), ("move_seed", 42)):
                previous = getattr(args, field)
                setattr(args, field, value)
                with self.assertRaisesRegex(RuntimeError, "identity differs"):
                    train.verify_run_identity(args)
                self.assertEqual(record.read_bytes(), original)
                setattr(args, field, previous)
            identity = json.loads(original)
            identity["recipe_sha256"]["train.py"] = "0" * 64
            record.write_text(json.dumps(identity))
            with self.assertRaisesRegex(RuntimeError, "recipe_sha256"):
                train.verify_run_identity(args)

    def test_run_identity_rejects_untracked_outputs_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}")
            (root / "manifest.json").write_text("{}")
            output = root / "output"
            output.mkdir()
            old = output / "existing-adapter"
            old.write_text("preserve")
            args = argparse.Namespace(
                architecture="qwen3-vl-8b", base_model=root, data=root,
                output=output, gpus=8, smoke=False,
                foundation_seed=1, move_seed=2, move_opd_seed=3, stop_seed=4,
            )
            with self.assertRaisesRegex(RuntimeError, "lacks training_run.json"):
                train.verify_run_identity(args)
            self.assertEqual(old.read_text(), "preserve")
            self.assertFalse((output / "training_run.json").exists())

    def test_smoke_rejects_zero_and_unchanged_adapters(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("Install training requirements for tensor validation")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            base = {"layer.lora_A.weight": torch.ones(2, 2),
                    "layer.lora_B.weight": torch.zeros(2, 2)}
            def save(stage, weights):
                path = train.adapter_file(output, stage)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(weights, str(path))
            save("foundation", base)
            with self.assertRaisesRegex(RuntimeError, "remains zero"):
                train.verify_smoke_update(output, "foundation")
            base["layer.lora_B.weight"] = torch.ones(2, 2)
            save("foundation", base)
            self.assertGreater(train.verify_smoke_update(output, "foundation")["changed_tensors"], 0)
            save("move_no_kd", base)
            with self.assertRaisesRegex(RuntimeError, "equal parent"):
                train.verify_smoke_update(output, "move_no_kd")
            changed = {key: value.clone() for key, value in base.items()}
            changed["layer.lora_B.weight"][0, 0] += 0.1
            save("move_no_kd", changed)
            self.assertEqual(train.verify_smoke_update(output, "move_no_kd")["changed_tensors"], 1)
            changed["layer.lora_B.weight"][0, 0] = float("nan")
            save("stop", changed)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                train.verify_smoke_update(output, "stop")

    def test_smoke_runs_past_zero_learning_rate_warmup(self):
        for architecture in train.DEFAULT_SEEDS:
            args = argparse.Namespace(
                architecture=architecture, data=ROOT / "datasets", output=ROOT / "missing-output",
                base_model=ROOT / "missing-model", gpus=4, smoke=True, dry_run=True,
                foundation_seed=1, move_seed=2, move_opd_seed=3, stop_seed=4,
            )
            for stage in train.DATA_DIR:
                command = train.training_command(args, stage)
                with self.subTest(architecture=architecture, stage=stage):
                    self.assertEqual(command[command.index("--max-steps") + 1], "2")
                    for option in ("--diagnostic-samples", "--max-train-diagnostic-samples", "--max-dev-samples"):
                        if option in command:
                            self.assertLessEqual(int(command[command.index(option) + 1]), 16)

    def test_mainline_microbatch_accumulation_and_workers(self):
        expected = {
            "qwen3-vl-8b": {"foundation": (1, 12, 0), "move_no_kd": (4, 1, 2),
                           "move_opd": (1, 4, 0), "stop": (1, 4, 0)},
            "internvl3-8b": {"foundation": (2, 6, 4), "move_no_kd": (1, 4, 4),
                            "move_opd": (1, 4, 4), "stop": (1, 4, 4)},
        }
        for architecture, stages in expected.items():
            args = argparse.Namespace(
                architecture=architecture, data=ROOT / "datasets", output=ROOT / "missing-output",
                base_model=ROOT / "missing-model", gpus=8, smoke=False, dry_run=True,
                foundation_seed=1, move_seed=2, move_opd_seed=3, stop_seed=4,
            )
            for stage, wanted in stages.items():
                command = train.training_command(args, stage)
                actual = tuple(int(command[command.index(name) + 1]) for name in
                               ("--per-device-batch-size", "--grad-accum-steps", "--num-workers"))
                with self.subTest(architecture=architecture, stage=stage):
                    self.assertEqual(actual, wanted)
                    if stage == "move_no_kd":
                        self.assertTrue(any(value.endswith("train_sc_esr_move.py") for value in command))


if __name__ == "__main__":
    unittest.main()
