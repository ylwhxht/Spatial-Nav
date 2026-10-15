#!/usr/bin/env python3
"""Check one of the isolated environments used by Spatial-Nav."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path


MODULES = {
    "train": ("torch", "PIL", "cv2", "peft", "safetensors", "transformers", "accelerate", "huggingface_hub"),
    "worker": (
        "torch", "PIL", "yaml", "pandas", "requests", "cv2", "matplotlib",
        "quaternion", "openai", "habitat_sim", "magnum",
    ),
    "serve": ("torch", "vllm", "transformers", "peft", "huggingface_hub"),
}


def check_sim_scene(scene: Path, gpu_device_id: int) -> None:
    """Exercise native context creation and a real RGB render, not just imports."""
    import habitat_sim
    import numpy as np

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(scene.resolve())
    config.gpu_device_id = gpu_device_id
    config.enable_physics = True  # Same request as the navigation runtimes.
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "environment_check_rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [64, 64]
    sensor.position = [0.0, 1.25, 0.0]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    with habitat_sim.Simulator(habitat_sim.Configuration(config, [agent])) as sim:
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("scene navmesh was not loaded")
        rgb = sim.get_sensor_observations().get(sensor.uuid)
        if not isinstance(rgb, np.ndarray) or rgb.shape != (64, 64, 4) or rgb.dtype != np.uint8:
            raise RuntimeError("RGB sensor did not return the expected 64x64 uint8 RGBA frame")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(MODULES), required=True)
    parser.add_argument("--min-gpus", type=int, default=0,
                        help="Fail unless this many CUDA GPUs are visible")
    parser.add_argument("--sim-scene", type=Path, default=None,
                        help="For --role worker, load this GLB/navmesh and render a real RGB frame through EGL")
    parser.add_argument("--sim-gpu-device-id", type=int, default=0,
                        help="Habitat-Sim GPU index for --sim-scene (default: 0)")
    args = parser.parse_args()
    if args.min_gpus < 0:
        parser.error("--min-gpus cannot be negative")
    if args.sim_scene is not None and args.role != "worker":
        parser.error("--sim-scene is only valid with --role worker")
    if args.sim_gpu_device_id < 0:
        parser.error("--sim-gpu-device-id cannot be negative")

    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    failures: list[str] = []
    for module_name in MODULES[args.role]:
        try:
            importlib.import_module(module_name)
            print(f"OK import {module_name}")
        except Exception as exc:  # Native extension import failures are useful diagnostics too.
            failures.append(module_name)
            print(f"FAIL import {module_name}: {type(exc).__name__}: {exc}")

    if args.role == "worker":
        for path in (Path(sys.prefix) / "conda-meta").glob("habitat-sim-*.json"):
            metadata = json.loads(path.read_text())
            if metadata.get("name") == "habitat-sim":
                print(f"habitat-sim={metadata['version']} build={metadata.get('build')}")
                print(f"habitat-sim-package-sha256={metadata.get('sha256')}")

    try:
        import torch

        print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"cuda_available={torch.cuda.is_available()} visible_gpus={gpu_count}")
        for index in range(gpu_count):
            print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
        if args.min_gpus and (not torch.cuda.is_available() or gpu_count < args.min_gpus):
            failures.append(f"need at least {args.min_gpus} visible CUDA GPUs")
    except Exception as exc:
        failures.append("torch runtime")
        print(f"FAIL torch runtime: {type(exc).__name__}: {exc}")

    if args.sim_scene is not None:
        if not args.sim_scene.is_file():
            failures.append(f"scene GLB missing: {args.sim_scene}")
        else:
            try:
                check_sim_scene(args.sim_scene, args.sim_gpu_device_id)
                print(f"OK Habitat-Sim EGL context, navmesh and RGB render: {args.sim_scene}")
            except Exception as exc:
                failures.append("Habitat-Sim EGL/scene")
                print(f"FAIL Habitat-Sim EGL/scene: {type(exc).__name__}: {exc}")

    if failures:
        print("Environment check failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    print(f"Environment check passed: {args.role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
