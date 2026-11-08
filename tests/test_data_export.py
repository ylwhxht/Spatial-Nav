"""Native image export must preserve bytes, order and labels, not just row counts."""
from collections import Counter
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


exporter = module("export_data", "tools/export_data.py")
trainer = module("train_for_data_test", "codes/scripts/train.py")


class DataExportTest(unittest.TestCase):
    def fixture(self, root):
        source, cache, output = (root / name for name in ("source", "cache", "export"))
        source.mkdir()
        cache.mkdir()
        inventory = []
        for name, size in (("native.png", (1920, 1080)), ("small.jpg", (512, 288))):
            image = source / name
            Image.new("RGB", size, (12, 34, 56)).save(image)
            inventory.append({"path": str(image), "sha256": exporter.sha256(image),
                              "width": size[0], "height": size[1], "mode": "RGB",
                              "bytes": image.stat().st_size})
        inv_path = source / "original_images.sha256.jsonl"
        inv_path.write_text("".join(json.dumps(x) + "\n" for x in inventory))
        stages, contract_stages, original_rows = {}, {}, {}
        for index, (name, (target, splits)) in enumerate(exporter.STAGES.items()):
            stages[name] = (target, {split: (0 if count == 0 else 2) for split, count in splits.items()})
            path = source / "stages" / name
            path.mkdir(parents=True)
            records = {}
            for split, count in stages[name][1].items():
                rows = [{"sample_id": f"{name}:{split}:{i}", "image_path": item["path"],
                         "prompt": "unchanged prompt", "response": "unchanged label",
                         "mix_role": "direct_move_only_policy" if i == 0 else "auxiliary"}
                        for i, item in enumerate(inventory[:count])]
                data_path = path / f"{split}.jsonl"
                data_path.write_text("".join(json.dumps(x) + "\n" for x in rows))
                records[data_path.name] = {"rows": count, "sha256": exporter.sha256(data_path),
                                          "resolution_counts": dict(Counter(
                                              f"{x['width']}x{x['height']}" for x in inventory[:count]))}
                original_rows[(target, split)] = rows
            contract_stages[name] = {"splits": records}
        exporter.write_json(source / "manifest.json", {
            "schema": "raw_training_contract_restored_v1", "passed": True,
            "uses_resized_512_dataset": False, "labels_and_row_order_unchanged": True,
            "unique_images": len(inventory), "image_manifest_sha256": exporter.sha256(inv_path),
            "stages": contract_stages,
        })
        exporter.write_json(cache / "sample_ids.json", ["move_no_kd:train:0"])
        (cache / "evidence.safetensors").write_bytes(b"fixture: verifier checks bytes without loading tensors")
        exporter.write_json(cache / "manifest.json", {
            "passed": True, "training_only": True, "runtime_loads_cache": False,
            "sample_ids_sha256": exporter.sha256(cache / "sample_ids.json"),
            "shards": {"evidence.safetensors": 1}, "evidence_items": 1,
            "shard_sha256": {"evidence.safetensors": exporter.sha256(cache / "evidence.safetensors")},
        })
        return source, cache, output, stages, original_rows, inventory

    def test_preserves_native_png_jpeg_and_verifies_all_images(self):
        with tempfile.TemporaryDirectory() as directory:
            source, cache, output, stages, originals, inventory = self.fixture(Path(directory))
            expected = {target: splits for target, splits in stages.values()}
            with patch.object(exporter, "STAGES", stages), patch.object(exporter, "DIRECT_MOVE_ROWS", 1), contextlib.redirect_stdout(io.StringIO()):
                exporter.export(source, cache, output, 2)
            for (stage, split), rows in originals.items():
                exported = list(map(json.loads, (output / stage / f"{split}.jsonl").read_text().splitlines()))
                self.assertEqual([r["sample_id"] for r in exported], [r["sample_id"] for r in rows])
                for row, old in zip(exported, rows):
                    for key in ("prompt", "response", "mix_role"):
                        self.assertEqual(row[key], old[key])
                    self.assertEqual((output / stage / row["image_path"]).read_bytes(), Path(old["image_path"]).read_bytes())
            with patch.object(trainer, "EXPECTED_SPLITS", expected):
                self.assertEqual(trainer.verify_data(output)["image_contract"], "original_bytes")
                image = next((output / "images").iterdir())
                image.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "image hash mismatch"):
                    trainer.verify_data(output)

    def test_rejects_resized_source_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, cache, output, *_ = self.fixture(Path(directory))
            contract_path = source / "manifest.json"
            contract = json.loads(contract_path.read_text())
            contract["uses_resized_512_dataset"] = True
            exporter.write_json(contract_path, contract)
            with self.assertRaisesRegex(ValueError, "original image bytes"):
                exporter.export(source, cache, output, 1)
            self.assertFalse(output.exists())

    def test_training_rejects_legacy_512_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter.write_json(root / "manifest.json", {"schema": "spatial-nav-data-v1", "passed": True,
                                                         "all_image_hashes_verified": True})
            with self.assertRaisesRegex(RuntimeError, "resized 512x288 bundle"):
                trainer.verify_data(root)

    def test_exact_sample_id_resolver_recovers_ephemeral_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source, cache, output, stages, _, inventory = self.fixture(Path(directory))
            data_path = source / "stages/move_no_kd/train.jsonl"
            rows = list(map(json.loads, data_path.read_text().splitlines()))
            resolver = source / "resolver.jsonl"
            resolver.write_text(json.dumps({"sample_id": rows[0]["sample_id"], "image_path": inventory[0]["path"]}) + "\n")
            rows[0]["image_path"] = "/dev/shm/no-longer-existing/image.png"
            data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            contract_path = source / "manifest.json"
            contract = json.loads(contract_path.read_text())
            contract["resolver_files"] = {str(resolver): exporter.sha256(resolver)}
            contract["stages"]["move_no_kd"]["splits"]["train.jsonl"]["sha256"] = exporter.sha256(data_path)
            exporter.write_json(contract_path, contract)
            with self.assertRaisesRegex(ValueError, "exact --resolver"):
                exporter.export(source, cache, output, 1)
            with patch.object(exporter, "STAGES", stages), patch.object(exporter, "DIRECT_MOVE_ROWS", 1), contextlib.redirect_stdout(io.StringIO()):
                exporter.export(source, cache, output, 1, [resolver])
            row = json.loads((output / "stage2_move_nokd/train.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["image_sha256"], inventory[0]["sha256"])


if __name__ == "__main__":
    unittest.main()
