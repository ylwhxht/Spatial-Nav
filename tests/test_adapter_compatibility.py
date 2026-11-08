"""Exercise adapter loading with distributed state, not just base-model import."""
import tempfile
from pathlib import Path
import unittest


class DistributedAdapterCompatibilityTest(unittest.TestCase):
    def test_adapter_reload_inside_process_group(self):
        try:
            import torch
            import torch.distributed as dist
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            self.skipTest(f"Requires the training environment: {exc}")
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("CPU gloo process groups are unavailable")
        self.assertFalse(dist.is_initialized(), "Run this test outside a training process")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                                    rank=0, world_size=1)
            try:
                model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)),
                                       LoraConfig(r=2, lora_alpha=4, target_modules=["0"]))
                model.save_pretrained(root / "adapter")
                loaded = PeftModel.from_pretrained(
                    torch.nn.Sequential(torch.nn.Linear(4, 4)), root / "adapter", is_trainable=True)
                loss = loaded(torch.ones(2, 4)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None for p in loaded.parameters() if p.requires_grad))
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
