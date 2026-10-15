"""The environment preflight must exercise rendering rather than just loading."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "codes/scripts/check_environment.py"
spec = importlib.util.spec_from_file_location("check_environment", SCRIPT)
environment_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment_check)


def fake_habitat(frame, navmesh=True):
    sim = MagicMock()
    sim.pathfinder.is_loaded = navmesh
    sim.get_sensor_observations.return_value = {"environment_check_rgb": frame}
    context = MagicMock()
    context.__enter__.return_value = sim
    module = SimpleNamespace(
        SimulatorConfiguration=SimpleNamespace,
        CameraSensorSpec=SimpleNamespace,
        SensorType=SimpleNamespace(COLOR="color"),
        agent=SimpleNamespace(AgentConfiguration=SimpleNamespace),
        Configuration=lambda config, agents: SimpleNamespace(config=config, agents=agents),
        Simulator=MagicMock(return_value=context),
    )
    return module, sim, context


class EnvironmentCheckTest(unittest.TestCase):
    def test_local_clients_do_not_eagerly_import_gemini(self):
        root = SCRIPT.parents[2]
        for dataset in ("hm3dv1", "hm3dv2", "mp3d"):
            with self.subTest(dataset=dataset):
                tree = ast.parse((root / "codes/navigation" / dataset / "vlm.py").read_text())
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        self.assertFalse(any(item.name.startswith("google.") for item in node.names))
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith("google."))

    def test_scene_check_requests_rgb_render_and_physics(self):
        module, sim, context = fake_habitat(np.zeros((64, 64, 4), dtype=np.uint8))
        with patch.dict(sys.modules, {"habitat_sim": module}):
            environment_check.check_sim_scene(Path("fixture.glb"), 3)
        config = module.Simulator.call_args.args[0]
        self.assertTrue(config.config.enable_physics)
        self.assertEqual(config.config.gpu_device_id, 3)
        self.assertEqual(config.agents[0].sensor_specifications[0].sensor_type, "color")
        self.assertEqual(config.agents[0].sensor_specifications[0].resolution, [64, 64])
        sim.get_sensor_observations.assert_called_once_with()
        context.__exit__.assert_called_once()

    def test_scene_check_rejects_missing_navmesh(self):
        module, sim, context = fake_habitat(None, navmesh=False)
        with patch.dict(sys.modules, {"habitat_sim": module}), self.assertRaisesRegex(RuntimeError, "navmesh"):
            environment_check.check_sim_scene(Path("fixture.glb"), 0)
        sim.get_sensor_observations.assert_not_called()
        context.__exit__.assert_called_once()

    def test_scene_check_rejects_missing_or_invalid_frame(self):
        for frame in (None, np.zeros((32, 32, 4), dtype=np.uint8), np.zeros((64, 64, 4))):
            with self.subTest(frame_type=type(frame), shape=getattr(frame, "shape", None)):
                module, _, context = fake_habitat(frame)
                with patch.dict(sys.modules, {"habitat_sim": module}), self.assertRaisesRegex(RuntimeError, "RGB sensor"):
                    environment_check.check_sim_scene(Path("fixture.glb"), 0)
                context.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
