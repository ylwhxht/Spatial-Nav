"""Response supervision must not depend on another row's padding length."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


class FoundationMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from PIL import Image
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"Requires the training environment: {exc}")
        cls.torch, cls.Image = torch, Image
        path = Path(__file__).resolve().parents[1] / "codes/scripts/train_cold_start_vision_sft.py"
        spec = importlib.util.spec_from_file_location("foundation_mask_test_trainer", path)
        cls.trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.trainer
        spec.loader.exec_module(cls.trainer)

    def test_mixed_length_rows_only_supervise_response_for_both_padding_sides(self):
        torch = self.torch

        class Processor:
            tokenizer = None

            def __init__(self, side):
                self.side = side

            def apply_chat_template(self, messages, **kwargs):
                text = next(item["text"] for item in messages[0]["content"] if item["type"] == "text")
                ids = [11] * len(text)
                return ids + ([91, 92] if messages[-1]["role"] == "assistant" else [])

            def __call__(self, text, **kwargs):
                width = max(map(len, text))
                ids, masks = [], []
                for row in text:
                    padding = [0] * (width - len(row))
                    ids.append(padding + row if self.side == "left" else row + padding)
                    active = [1] * len(row)
                    masks.append(padding + active if self.side == "left" else active + padding)
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            self.Image.new("RGB", (8, 8)).save(image_path)
            samples = [self.trainer.VisionSample(
                sample_id=str(index), image_path=str(image_path), image_paths=[str(image_path)],
                prompt=prompt, response="OK",
            ) for index, prompt in enumerate(("short", "a much longer prompt"))]
            for side in ("left", "right"):
                with self.subTest(padding_side=side):
                    collator = self.trainer.VisionSFTCollator(Processor(side), 128, system_prompt="")
                    batch = collator(samples)
                    for index, sample in enumerate(samples):
                        labels = batch["labels"][index]
                        self.assertEqual(labels[labels != -100].tolist(), [91, 92])
                        single = collator([sample])["labels"][0]
                        active = batch["attention_mask"][index].bool()
                        self.assertTrue(torch.equal(labels[active], single))
                        self.assertTrue(torch.all(labels[~active] == -100))


if __name__ == "__main__":
    unittest.main()
