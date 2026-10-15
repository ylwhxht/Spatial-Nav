#!/usr/bin/env python3
"""Generate the 11 validation navmeshes without a renderer or pretrained model.

Run in the pinned Habitat-Sim 0.2.5 environment. The output directory must not
already exist. Existing scenes and navmeshes are read-only inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import habitat_sim


EXPECTED_SHA256 = {
    "2azQ1b91cZZ": "25380afaf281e0833ff82f2c1a0e57be3bcf78d8363bdf67643b0525471d97ad",
    "8194nk5LbLH": "d96e93521a4a8fc829c77167eee2eeee91bce276dcc9a3ec5685b43b18589414",
    "EU6Fwq7SyZv": "8510c4241576edd6e0b243e02466d60247f260a406e7498c85b1bdc1cd218542",
    "QUCTc6BB5sX": "c2030eba4e1af8ba2c731c6981cfc5e4746a3320873a346965ae19ec43286462",
    "TbHJrupSAjP": "1a7e849633d28b18798a8cba2c2ee71710814e37a6074e61271274c1d23e8505",
    "X7HyMhZNoso": "78ec0bb3c74c31926d3c6c1eee3bd0ab01b40bcd9e68d63a88b05de270a10ab7",
    "Z6MFQCViBuw": "d07334628bf8a6f4d94afd8832fe34c6ee09f34eea51a83f179521057bbf34a0",
    "oLBMNvg9in8": "1ad368353fd2a98ab837c272795d552a9e4f9d7ff001f0b1dca3bf08bcfa4642",
    "pLe4wQe7qrG": "8a034324d9cf8e4c84a10f618fc9716cb06d8a9d0c7835a425a7ede2dd05aabf",
    "x8F5xyUWy9e": "de8d183c089c2ed3e4e7417e2332411cc3dbf78829c214480d1d86da7679d66f",
    "zsNo4HB9uLZ": "437eaed1b143111baa93e007d45defc67c20aeb554f094110349d334e6fb1820",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--objectnav-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if habitat_sim.__version__ != "0.2.5":
        parser.error("Use the pinned Habitat-Sim 0.2.5 environment from ENVIRONMENT.md")
    scene_root = args.scene_root.resolve()
    content = args.objectnav_root.resolve() / "val" / "content"
    output = args.output_root.resolve()
    scene_config = scene_root / "mp3d.scene_dataset_config.json"
    if not scene_config.is_file() or not content.is_dir():
        parser.error("Scene config and ObjectNav val/content directory are required")
    if output.exists():
        parser.error("--output-root must be a new directory; existing files are not overwritten")
    shards = {}
    for path in sorted(content.iterdir()):
        if not path.name.endswith((".json", ".json.gz")):
            continue
        scene = path.name.removesuffix(".gz").removesuffix(".json")
        if scene not in shards or path.name.endswith(".gz"):
            shards[scene] = path
    if set(shards) != set(EXPECTED_SHA256):
        parser.error("Episode shards must cover the 11 reported MP3D validation scenes")
    episode_count = 0
    for scene, path in shards.items():
        if not (scene_root / scene / f"{scene}.glb").is_file():
            parser.error(f"Missing scene GLB: {scene}")
        raw = path.read_bytes()
        payload = json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)
        episode_count += len(payload["episodes"])
    if episode_count != 2195:
        parser.error(f"Expected 2195 validation episodes; found {episode_count}")
    output.mkdir(parents=True)
    for scene in sorted(shards):
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene_root / scene / f"{scene}.glb")
        backend.scene_dataset_config_file = str(scene_config)
        backend.enable_physics = False
        backend.create_renderer = False
        agent = habitat_sim.agent.AgentConfiguration()
        agent.radius = 0.18
        agent.height = 0.88
        agent.sensor_specifications = []
        sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
        try:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = 0.18
            settings.agent_height = 0.88
            settings.agent_max_climb = 0.20
            settings.agent_max_slope = 45.0
            settings.cell_size = 0.05
            settings.cell_height = 0.20
            if not sim.recompute_navmesh(sim.pathfinder, settings) or not sim.pathfinder.is_loaded:
                raise RuntimeError(f"Navmesh generation failed: {scene}")
            target = output / scene / f"{scene}.navmesh"
            target.parent.mkdir()
            if not sim.pathfinder.save_nav_mesh(str(target)):
                raise RuntimeError(f"Navmesh save failed: {scene}")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != EXPECTED_SHA256[scene]:
                raise RuntimeError(f"Generated navmesh differs from the reported asset: {scene}. "
                                   "Check the scene download and pinned Habitat-Sim build; existing assets are unchanged.")
            print(f"Generated and hash-verified {scene}: {target.stat().st_size} bytes", flush=True)
        finally:
            sim.close()
    print("Generated all 11 validation navmeshes", flush=True)


if __name__ == "__main__":
    main()
