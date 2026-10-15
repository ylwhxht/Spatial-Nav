#!/usr/bin/env python3
"""Evaluate a raw base model or the bundled Move/Stop checkpoints on ObjectNav."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
NAVIGATION = ROOT / "codes" / "navigation"
DATASET_ENV = {
    "hm3d_v1": ("VLMNAV_HM3D_V1_SCENE_ROOT", "VLMNAV_OBJECTNAV_V1_ROOT"),
    "hm3d_v2": ("VLMNAV_HM3D_V2_SCENE_ROOT", "VLMNAV_OBJECTNAV_V2_ROOT"),
    "mp3d": ("VLMNAV_MP3D_SCENE_ROOT", "VLMNAV_OBJECTNAV_MP3D_ROOT"),
}
RUNTIME = {"hm3d_v1": "hm3dv1", "hm3d_v2": "hm3dv2", "mp3d": "mp3d"}
MODEL_NAMES = {"move": "spatial_nav_move", "stop": "spatial_nav_stop"}
EPISODE_COUNTS = {"hm3d_v1": 2000, "hm3d_v2": 1000, "mp3d": 2195}
DEFAULT_STOP_THRESHOLDS = {"qwen3-vl-8b": 2.0, "internvl3-8b": 0.0}
BASE_MOVE_PROTOCOLS = {
    "qwen3-vl-8b": "objectnav_direct_move_only_v2",
    "internvl3-8b": "objectnav_direct_action_v1",
}
DEFAULT_MAX_NUM_SEQS = {"hm3d_v1": 32, "hm3d_v2": 64, "mp3d": 48}
HISTORICAL_REQUEST_ROUTING = {
    "qwen3-vl-8b": {"hm3d_v1": "static", "hm3d_v2": "least-inflight", "mp3d": "static"},
    "internvl3-8b": {dataset: "least-inflight" for dataset in RUNTIME},
}


def episode_indices(args: argparse.Namespace) -> list[int]:
    """Select a fixed subset without changing canonical dataset indices."""
    total = EPISODE_COUNTS[args.dataset]
    if args.episode_indices is not None:
        if args.episodes is not None:
            raise ValueError("Use --episode-indices or --episodes, not both")
        values = json.loads(args.episode_indices.read_text(encoding="utf-8"))
        if (not isinstance(values, list) or not values
                or any(type(index) is not int or not 0 <= index < total for index in values)):
            raise ValueError(f"--episode-indices must contain a nonempty JSON list of integers in [0, {total})")
        if len(values) != len(set(values)):
            raise ValueError("--episode-indices must not contain duplicates")
        return values
    count = total if args.episodes is None else args.episodes
    if not 1 <= count <= total:
        raise ValueError(f"--episodes must be in [1, {total}]")
    return list(range(count))


def clean_environment() -> dict[str, str]:
    """Prevent shell leftovers from silently changing the public policy."""
    hardware = {"VLMNAV_NVIDIA_DRIVER_LIB_DIR", "VLMNAV_NVIDIA_EGL_VENDOR_JSON"}
    runtime_controls = {
        "VLMNAV_DISABLE_IMAGE_LOGGING",
        "VLMNAV_SAVE_EVERY_STEP_IMAGE",
        "VLMNAV_SAVE_STOP_SIGNAL_IMAGES",
        "VLMNAV_MODEL_IO_LOG",
        "VLMNAV_MODEL_IO_STDOUT",
    }
    return {key: value for key, value in os.environ.items()
            if not key.startswith("VLMNAV_") or key in hardware or key in runtime_controls}


def environment_versions(python: Path) -> dict:
    probe = (
        "import importlib.metadata as m, json, pathlib, platform, sys\n"
        "packages = {}\n"
        "for name in ('torch', 'torchvision', 'transformers', 'tokenizers', 'peft', 'accelerate', 'vllm', 'numpy', 'Pillow', 'openai', 'httpx', 'opencv-python', 'opencv-python-headless', 'numpy-quaternion', 'numba', 'scipy', 'flashinfer-python', 'quack-kernels'):\n"
        "    try: packages[name] = m.version(name)\n"
        "    except m.PackageNotFoundError: pass\n"
        "for path in (pathlib.Path(sys.prefix) / 'conda-meta').glob('habitat-sim-*.json'):\n"
        "    meta = json.loads(path.read_text())\n"
        "    if meta.get('name') == 'habitat-sim':\n"
        "        packages['habitat-sim'] = meta['version']\n"
        "        packages['habitat-sim-build'] = meta.get('build')\n"
        "        packages['habitat-sim-package-sha256'] = meta.get('sha256')\n"
        "print(json.dumps({'python': platform.python_version(), 'executable': sys.executable, 'packages': packages}))"
    )
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def worker_gpu_layout(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    """Resolve one simulator GPU per worker, preserving the single-GPU CLI."""
    if args.sim_gpus is not None and args.sim_gpu is not None:
        parser.error("Use --sim-gpus or --sim-gpu, not both")
    gpu_ids = (
        [item.strip() for item in args.sim_gpus.split(",")]
        if args.sim_gpus is not None else [args.sim_gpu or "2"]
    )
    if not gpu_ids or any(not item.isdecimal() for item in gpu_ids):
        parser.error("--sim-gpus must be a comma-separated list of GPU indices")
    if len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--sim-gpus must not repeat a GPU index")
    if args.workers is not None and args.workers_per_gpu is not None:
        parser.error("Use --workers or --workers-per-gpu, not both")
    if args.workers_per_gpu is not None:
        if args.workers_per_gpu < 1:
            parser.error("--workers-per-gpu must be positive")
        layout = [gpu for gpu in gpu_ids for _ in range(args.workers_per_gpu)]
    else:
        total = args.workers if args.workers is not None else len(gpu_ids)
        if total < 1:
            parser.error("--workers must be positive")
        layout = [gpu_ids[index % len(gpu_ids)] for index in range(total)]
    if len(layout) > args.episodes:
        parser.error("Total workers must not exceed --episodes")
    return layout


def model_gpu_layout(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    """Assign one Move or Stop server to each selected model GPU."""
    if bool(args.move_url) != bool(args.stop_url):
        parser.error("Provide both --move-url and --stop-url, or neither")
    if args.move_url and (args.move_gpus is not None or args.stop_gpus is not None):
        parser.error("--move-gpus/--stop-gpus require locally managed model servers")
    role_gpus = {}
    for role in MODEL_NAMES:
        many = getattr(args, f"{role}_gpus")
        single = getattr(args, f"{role}_gpu")
        values = [item.strip() for item in many.split(",")] if many is not None else [single]
        if not values or any(not item.isdecimal() for item in values):
            parser.error(f"--{role}-gpus must be a comma-separated list of GPU indices")
        if len(set(values)) != len(values):
            parser.error(f"--{role}-gpus must not repeat a GPU index")
        role_gpus[role] = values
    if not args.move_url and set(role_gpus["move"]) & set(role_gpus["stop"]):
        parser.error("Move and Stop model GPU lists must be disjoint: one model per GPU")
    return role_gpus


def validate_eval_assets(dataset: str, scene_root: Path, objectnav_root: Path) -> dict:
    """Fail before model startup if any validation scene or episode shard is absent."""
    content = objectnav_root / "val" / "content"
    if not content.is_dir():
        raise ValueError(f"Missing ObjectNav validation episode directory: {content}")
    episode_files = sorted(path for path in content.iterdir()
                           if path.name.endswith((".json", ".json.gz")))
    gz_stems = {path.name[:-8] for path in episode_files if path.name.endswith(".json.gz")}
    duplicates = [path for path in episode_files if path.name.endswith(".json") and path.name[:-5] in gz_stems]
    if dataset == "mp3d":
        # Match the existing MP3D runtime, which prefers compressed shards.
        episode_files = [path for path in episode_files if path not in duplicates]
    elif duplicates or len(episode_files) != len(list(content.iterdir())):
        raise ValueError(f"Ambiguous HM3D content directory (duplicate/extra files): {content}")
    shard_records, episode_ids = {}, []
    for path in episode_files:
        raw = path.read_bytes()
        payload = json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)
        episodes = payload["episodes"]
        # Official shards reuse episode_id (often "0"). Identity is the full
        # episode payload, not that non-unique display field.
        episode_ids.extend(hashlib.sha256(json.dumps(e, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                           for e in episodes)
        shard_records[path.name] = {
            "episodes": len(episodes), "sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
    if len(episode_ids) != EPISODE_COUNTS[dataset] or len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"Expected {EPISODE_COUNTS[dataset]} unique {dataset} episodes; got {len(episode_ids)} rows / {len(set(episode_ids))} unique")
    scenes = sorted({path.name.removesuffix(".json.gz").removesuffix(".json")
                     for path in episode_files})
    if not scenes:
        raise ValueError(f"No ObjectNav validation episode shards found: {content}")
    if dataset == "mp3d":
        config_paths = [scene_root / "mp3d.scene_dataset_config.json"]
        if len(scenes) != 11:
            raise ValueError(f"Expected 11 MP3D validation scenes, found {len(scenes)} in {content}")
    else:
        config_paths = list(scene_root.glob("*.scene_dataset_config.json"))
        config_paths += list((scene_root / "val").glob("*.scene_dataset_config.json"))
    if not any(path.is_file() for path in config_paths):
        raise ValueError(f"Missing scene dataset configuration under {scene_root}")
    missing, navmeshes = [], {}
    for scene in scenes:
        if dataset == "mp3d":
            folders = [scene_root / scene]
        else:
            folders = list(scene_root.glob(f"*-{scene}"))
            folders += list((scene_root / "val").glob(f"*-{scene}"))
        folders = [folder for folder in folders if folder.is_dir()]
        pattern = f"{scene}.glb" if dataset == "mp3d" else f"{scene}.basis.glb"
        meshes = [path for folder in folders for path in folder.glob(pattern)]
        if not meshes:
            missing.append(f"{scene}: scene GLB")
        elif not any(path.with_suffix(".navmesh").is_file() for path in meshes):
            missing.append(f"{scene}: navmesh")
        else:
            for path in meshes:
                navmesh = path.with_suffix(".navmesh")
                if navmesh.is_file():
                    navmeshes[str(navmesh.relative_to(scene_root))] = hashlib.sha256(navmesh.read_bytes()).hexdigest()
    if missing:
        preview = ", ".join(missing[:8])
        hint = " HM3D-v0.1 mainline uses v0.1 episodes with v0.2 basis scene assets/navmeshes." if dataset == "hm3d_v1" else ""
        raise ValueError(f"Missing evaluation assets ({len(missing)} scenes): {preview}.{hint}")
    return {"episodes": len(episode_ids), "scenes": len(scenes), "episode_shards": shard_records,
            "navmesh_sha256": navmeshes,
            "scene_config_sha256": {str(path.relative_to(scene_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in config_paths if path.is_file()}}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, process: subprocess.Popen | None, timeout: int = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Model server exited early ({process.returncode}): {url}")
        try:
            with urlopen(url.rstrip("/").removesuffix("/v1") + "/health", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Model server not ready: {url}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()


def server_command(args: argparse.Namespace, role: str, port: int, adapter: Path | None) -> list[str]:
    # Both released Stop checkpoints are merged HF models; Move uses LoRA.
    merged_model = adapter is not None and (adapter / "model.safetensors.index.json").is_file()
    model_path = adapter if merged_model else args.base_model
    command = [
        str(args.serve_python), "-m", "vllm.entrypoints.cli.main", "serve",
        str(model_path.resolve()), "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", MODEL_NAMES[role],
        "--dtype", "bfloat16", "--tokenizer-mode", "slow",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len or getattr(args, f"{role}_max_model_len")),
        "--max-num-batched-tokens", "4096", "--max-num-seqs", str(args.max_num_seqs),
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--mm-processor-cache-gb", "2", "--mm-processor-kwargs",
        '{"use_fast":true}' if args.architecture == "internvl3-8b" else '{"use_fast":false}',
        "--enforce-eager",
    ]
    if adapter is not None and not merged_model:
        command += [
            "--enable-lora", "--lora-modules", f"{MODEL_NAMES[role]}={adapter.resolve()}",
            "--max-lora-rank", "128",
        ]
    return command


def summarize(run_root: Path, run_name: str, requested: set[int], metadata: dict) -> dict:
    log_root = run_root / "logs" / f"ObjectNav_{run_name}"
    records: dict[int, dict] = {}
    for path in log_root.glob("*_of_*/*/episode_summary.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        index = int(str(item["run_name"]).split("_", 1)[0])
        if index in requested:
            if index in records:
                raise RuntimeError(f"Duplicate episode summary for index {index}")
            records[index] = item
    missing = sorted(requested - records.keys())
    if missing:
        raise RuntimeError(f"Missing {len(missing)} episode summaries; first: {missing[:10]}")
    values = list(records.values())
    success = sum(bool(row["goal_reached"]) for row in values) / len(values)
    spl = sum(float(row["spl"]) for row in values) / len(values)
    result = {**metadata, "episodes": len(values), "sr": success, "spl": spl,
              "sr_percent": round(success * 100, 2), "spl_percent": round(spl * 100, 2)}
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("qwen3-vl-8b", "internvl3-8b"), required=True)
    parser.add_argument("--variant", choices=("base", "checkpoint"), required=True)
    parser.add_argument("--base-move-protocol", choices=tuple(sorted(set(BASE_MOVE_PROTOCOLS.values()))),
                        help="Base-only prompt override; defaults: Qwen move-only v2, InternVL direct-action v1")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_ENV), required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--objectnav-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes to evaluate (default: full validation split)")
    parser.add_argument("--episode-indices", type=Path, default=None,
                        help="JSON list of canonical episode indices for fixed-subset validation")
    parser.add_argument("--workers", type=int, default=None,
                        help="Total simulator workers (round-robin across --sim-gpus)")
    parser.add_argument("--workers-per-gpu", type=int, default=None,
                        help="Simulator workers on each GPU in --sim-gpus")
    parser.add_argument("--worker-gpu-device-id", type=int, default=0,
                        help="Habitat-Sim device id inside each worker (use -1 for CPU/Mesa)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--move-checkpoint", type=Path, default=None)
    parser.add_argument("--stop-checkpoint", type=Path, default=None)
    parser.add_argument("--move-url", default="", help="Existing OpenAI-compatible /v1 endpoint")
    parser.add_argument("--stop-url", default="", help="Existing OpenAI-compatible /v1 endpoint")
    parser.add_argument("--serve-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--worker-python", type=Path, default=Path(sys.executable),
        help="Python executable for Habitat-Sim workers (can differ from the controller and vLLM environments)",
    )
    parser.add_argument("--move-gpu", default="0")
    parser.add_argument("--stop-gpu", default="1")
    parser.add_argument("--move-gpus", default=None,
                        help="Comma-separated GPUs for Move model replicas; overrides --move-gpu")
    parser.add_argument("--stop-gpus", default=None,
                        help="Comma-separated GPUs for Stop model replicas; overrides --stop-gpu")
    parser.add_argument("--sim-gpu", default=None,
                        help="Legacy single Habitat-Sim GPU (default: 2)")
    parser.add_argument("--sim-gpus", default=None,
                        help="Comma-separated Habitat-Sim GPUs, e.g. 2,3,4,5")
    parser.add_argument("--allow-shared-server-gpus", action="store_true",
                        help="Permit Habitat workers on the Move/Stop model-server GPUs")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--image-res", type=int, default=1920)
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override the context limit for both model roles")
    parser.add_argument("--move-max-model-len", type=int, default=3072)
    parser.add_argument("--stop-max-model-len", type=int,
                        help="Default: 4096 for checkpoints, 3072 for unmodified bases")
    parser.add_argument("--max-num-seqs", type=int,
                        help="Historical defaults: HM3D-v0.1 32, HM3D-v0.2 64, MP3D 48")
    parser.add_argument("--request-routing", choices=("historical", "static", "least-inflight"),
                        default="historical", help="Replica dispatch policy; historical uses the selected checkpoint's dataset-specific launcher")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-stop-inputs", action="store_true",
                        help="Save exact Stop SDK request bodies (including inline images) and parsed responses; diagnostic runs only")
    args = parser.parse_args()
    if args.base_move_protocol is not None and args.variant != "base":
        parser.error("--base-move-protocol is only valid with --variant base")
    move_protocol = (
        args.base_move_protocol or BASE_MOVE_PROTOCOLS[args.architecture]
        if args.variant == "base" else "objectnav_direct_move_only_v2"
    )
    canonical_move = move_protocol == "objectnav_direct_move_only_v2"
    if args.stop_max_model_len is None:
        args.stop_max_model_len = 3072 if args.variant == "base" else 4096
    if args.max_num_seqs is None:
        args.max_num_seqs = DEFAULT_MAX_NUM_SEQS[args.dataset]
    try:
        selected_indices = episode_indices(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.episodes = len(selected_indices)
    if any(value is not None and value < 1 for value in
           (args.max_model_len, args.move_max_model_len, args.stop_max_model_len, args.max_num_seqs)):
        parser.error("Context limits and --max-num-seqs must be positive")
    worker_gpus = worker_gpu_layout(args, parser)
    args.workers = len(worker_gpus)
    role_gpus = model_gpu_layout(args, parser)
    model_gpu_set = set(role_gpus["move"] + role_gpus["stop"])
    shared_server_gpus = not args.move_url and bool(model_gpu_set & set(worker_gpus))
    if shared_server_gpus and not args.allow_shared_server_gpus:
        parser.error("Habitat-Sim GPUs overlap model-server GPUs; pass "
                     "--allow-shared-server-gpus to opt in")
    if shared_server_gpus and args.gpu_memory_utilization > 0.75:
        parser.error("When sharing model-server GPUs with Habitat workers, "
                     "set --gpu-memory-utilization to 0.75 or less")
    asset_inventory = None
    if not args.dry_run:
        for path in (args.base_model / "config.json", args.scene_root, args.objectnav_root):
            if not path.exists():
                parser.error(f"Missing input: {path}")
        try:
            asset_inventory = validate_eval_assets(args.dataset, args.scene_root, args.objectnav_root)
        except ValueError as exc:
            parser.error(str(exc))
        if args.run_dir.exists():
            parser.error(f"Run directory already exists: {args.run_dir}")
        if not args.worker_python.is_file():
            parser.error(f"Habitat worker Python not found: {args.worker_python}")
        if not args.move_url and not args.serve_python.is_file():
            parser.error(f"vLLM serving Python not found: {args.serve_python}")
    checkpoint_root = args.checkpoint_root or ROOT / "checkpoints" / args.architecture
    adapters = {"move": None, "stop": None}
    if args.variant == "checkpoint":
        default_stop = checkpoint_root / "stop_merged"
        adapters = {
            "move": args.move_checkpoint or checkpoint_root / "move_composed",
            "stop": args.stop_checkpoint or default_stop,
        }
        if not args.dry_run and not args.move_url:
            for adapter in adapters.values():
                is_adapter = (adapter / "adapter_model.safetensors").is_file()
                is_merged = (adapter / "config.json").is_file() and (adapter / "model.safetensors.index.json").is_file()
                if not (is_adapter or is_merged):
                    parser.error(f"Checkpoint must be a PEFT adapter or merged HF model: {adapter}")
    run_dir = args.run_dir.resolve()
    run_name = run_dir.name.replace(" ", "_")
    servers: dict[str, list[dict]] = {}
    reserved_ports: set[int] = set()
    for role in MODEL_NAMES:
        if args.move_url:
            servers[role] = [{"gpu": None, "port": None, "url": getattr(args, f"{role}_url").rstrip("/")}]
        else:
            servers[role] = []
            for gpu in role_gpus[role]:
                port = free_port()
                while port in reserved_ports:
                    port = free_port()
                reserved_ports.add(port)
                servers[role].append({"gpu": gpu, "port": port,
                                      "url": f"http://127.0.0.1:{port}/v1"})
    sim_gpu_order = list(dict.fromkeys(worker_gpus))
    worker_urls = {
        role: [servers[role][sim_gpu_order.index(gpu) % len(servers[role])]["url"]
               for gpu in worker_gpus]
        for role in MODEL_NAMES
    }
    routing = (HISTORICAL_REQUEST_ROUTING[args.architecture][args.dataset]
               if args.request_routing == "historical" else args.request_routing)
    routers = {}
    if routing == "least-inflight" and not args.move_url:
        for role, entries in servers.items():
            # One backend needs no dispatcher; the small-subset topology has
            # no cross-replica routing and cannot validate the multi-GPU path.
            if len(entries) < 2:
                continue
            port = free_port()
            while port in reserved_ports:
                port = free_port()
            reserved_ports.add(port)
            url = f"http://127.0.0.1:{port}/v1"
            command = [str(args.worker_python), str(SCRIPTS / "openai_inflight_router.py"),
                       "--port", str(port), "--backends", ",".join(item["url"] for item in entries),
                       "--timeout", "600"]
            routers[role] = {"url": url, "port": port, "backends": [item["url"] for item in entries],
                             "command": command}
            worker_urls[role] = [url] * args.workers
    metadata = {
        "architecture": args.architecture, "variant": args.variant, "dataset": args.dataset,
        "evaluation_assets": asset_inventory,
        "base_model": str(args.base_model.resolve()),
        "move_checkpoint": str(adapters["move"].resolve()) if adapters["move"] and not args.move_url else None,
        "stop_checkpoint": str(adapters["stop"].resolve()) if adapters["stop"] and not args.stop_url else None,
        "move_url": servers["move"][0]["url"], "stop_url": servers["stop"][0]["url"],
        "model_servers": servers,
        "worker_move_urls": worker_urls["move"],
        "worker_stop_urls": worker_urls["stop"],
        "request_routing": "external" if args.move_url else routing,
        "request_routing_requested": args.request_routing,
        "request_routers": routers,
        "stop_threshold": (
            args.stop_threshold
            if args.stop_threshold is not None
            else (0.0 if args.variant == "base" else DEFAULT_STOP_THRESHOLDS[args.architecture])
        ),
        "episodes_requested": args.episodes, "max_steps": args.max_steps,
        "episode_indices": selected_indices,
        "scene_root": str(args.scene_root.resolve()),
        "objectnav_root": str(args.objectnav_root.resolve()),
        "image_res": args.image_res, "workers": args.workers,
        "worker_gpu_device_id": args.worker_gpu_device_id,
        "sim_gpus": sim_gpu_order,
        "workers_per_gpu": {
            gpu: worker_gpus.count(gpu) for gpu in dict.fromkeys(worker_gpus)
        },
        "worker_gpu_map": worker_gpus,
        "move_gpu": None if args.move_url else role_gpus["move"][0],
        "stop_gpu": None if args.stop_url else role_gpus["stop"][0],
        "move_gpus": None if args.move_url else role_gpus["move"],
        "stop_gpus": None if args.stop_url else role_gpus["stop"],
        "shared_server_gpus": shared_server_gpus,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "move_max_model_len": args.max_model_len or args.move_max_model_len,
        "stop_max_model_len": args.max_model_len or args.stop_max_model_len,
        "move_wire_protocol": move_protocol,
    }
    env = clean_environment()
    # Historical base controls also disable the policy-changing fallbacks.
    env.update(json.loads((NAVIGATION / "checkpoint_profile.json").read_text(encoding="utf-8")))
    if args.architecture == "qwen3-vl-8b" or args.variant == "base":
        # vLLM 0.12 accepts chat_template_kwargs per request, but does not
        # expose --default-chat-template-kwargs as a server CLI option.
        env["VLMNAV_OPENAI_CHAT_TEMPLATE_KWARGS"] = '{"enable_thinking":false}'
    env["PYTHONPATH"] = str(ROOT / "codes" / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    driver_lib = env.get("VLMNAV_NVIDIA_DRIVER_LIB_DIR")
    if driver_lib:
        env["LD_LIBRARY_PATH"] = driver_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    vendor_json = env.get("VLMNAV_NVIDIA_EGL_VENDOR_JSON")
    if vendor_json:
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor_json
    env["EMBODIED_NAV_COT_SRC"] = str(ROOT / "codes" / "src")
    env["VLMNAV_LOGS_ROOT"] = str(run_dir / "logs")
    env["VLMNAV_ACTION_MODEL"] = MODEL_NAMES["move"]
    env["VLMNAV_ACTION_CANONICAL_MOVE"] = "1" if canonical_move else "0"
    env["VLMNAV_STOPPING_MODEL"] = MODEL_NAMES["stop"]
    if args.audit_stop_inputs:
        env["VLMNAV_STOP_REQUEST_AUDIT_DIR"] = str(run_dir / "stop_request_audit")
    env["VLMNAV_ACTION_OPENAI_BASE_URL"] = servers["move"][0]["url"]
    env["VLMNAV_STOPPING_OPENAI_BASE_URL"] = servers["stop"][0]["url"]
    env["VLMNAV_ACTION_OPENAI_API_KEY"] = "EMPTY"
    env["VLMNAV_STOPPING_OPENAI_API_KEY"] = "EMPTY"
    env["VLMNAV_ACTION_OPENAI_SEND_SAMPLING_ARGS"] = "1"
    env["VLMNAV_ACTION_OPENAI_SEND_MAX_TOKENS"] = "1"
    env["VLMNAV_ACTION_OPENAI_TEMPERATURE"] = "0"
    env["VLMNAV_ACTION_OPENAI_MAX_TOKENS"] = "24"
    env["VLMNAV_STOPPING_OPENAI_SEND_SAMPLING_ARGS"] = "1"
    env["VLMNAV_STOPPING_OPENAI_SEND_MAX_TOKENS"] = "1"
    env["VLMNAV_STOPPING_OPENAI_TEMPERATURE"] = "0"
    env["VLMNAV_STOPPING_OPENAI_MAX_TOKENS"] = "16"
    env["VLMNAV_ACTION_OPENAI_TIMEOUT"] = "600"
    env["VLMNAV_STOPPING_OPENAI_TIMEOUT"] = "600"
    env["VLMNAV_STOPPING_MAX_IMAGE_RES"] = str(args.image_res)
    env["VLMNAV_STOPPING_IMAGE_ENCODE_FORMAT"] = "jpeg"
    env["VLMNAV_STOPPING_JPEG_QUALITY"] = "85"
    # The runtime renders v1; the Qwen/base and checkpoint client adapters
    # then convert it to the exact v2 wire message. InternVL base stays v1.
    env["VLMNAV_PROTOCOL_ID"] = "objectnav_direct_action_v1"
    env["VLMNAV_MOVE_RUNTIME_PROMPT_PROFILE"] = "finetuned_direct_v2" if canonical_move else "direct_action_v1"
    env["VLMNAV_SINGLE_STOP_DIRECT_EXECUTE"] = "1"
    env["VLMNAV_SINGLEVIEW_STOPBOUNDARY_DONE_PROMPT"] = "1"
    env["VLMNAV_COT_STOPPING_PROMPT"] = "0"
    env["VLMNAV_STOP_LOGIT_THRESHOLD_ENABLE"] = "1"
    env["VLMNAV_STOP_LOGIT_THRESHOLD"] = str(metadata["stop_threshold"])
    env["VLMNAV_HABITAT_STEP_BUDGET"] = str(args.max_steps)
    env["VLMNAV_REQUIRE_HABITAT_STEP_BUDGET"] = "1"
    env["SUCCESS_THRESHOLD"] = "1.0"
    env["VLMNAV_REQUIRE_EXPLICIT_STOP_FOR_SUCCESS"] = "0"
    env["VLMNAV_FORCE_STOP_AT_EPISODE_BUDGET"] = "0"
    env["VLMNAV_FORCE_STOP_ON_STAGE_TIMEOUT"] = "0"
    env["VLMNAV_ENFORCE_PUBLIC_INFERENCE"] = "1"
    env["VLMNAV_ORIGINAL_OBJECTNAV_LOGIC"] = "1"
    env["VLMNAV_DISABLE_IMAGE_LOGGING"] = "1"
    env["VLMNAV_EPISODE_LIST_EXIT_STAGGER_S"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    scene_env, objectnav_env = DATASET_ENV[args.dataset]
    env[scene_env] = str(args.scene_root.resolve())
    env[objectnav_env] = str(args.objectnav_root.resolve())
    if args.dataset == "mp3d":
        env.update(json.loads((NAVIGATION / "mp3d_profile.json").read_text(encoding="utf-8")))
    metadata["runtime_environment"] = {
        key: value for key, value in sorted(env.items())
        if (key.startswith("VLMNAV_") and "API_KEY" not in key)
        or key in {"SUCCESS_THRESHOLD", "PYOPENGL_PLATFORM", "__EGL_VENDOR_LIBRARY_FILENAMES",
                   "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    }
    metadata["server_commands"] = {
        role: [server_command(args, role, item["port"], adapters[role])
               for item in entries if item["port"] is not None]
        for role, entries in servers.items()
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    metadata["environments"] = {
        "worker": environment_versions(args.worker_python),
        "serving": environment_versions(args.serve_python) if not args.move_url else {"external": True},
    }
    sources = [*SCRIPTS.glob("*.py"), *NAVIGATION.glob("*.json"), *NAVIGATION.glob("*.yaml"),
               *(NAVIGATION / RUNTIME[args.dataset]).glob("*.py"),
               *(ROOT / "codes" / "src" / "embodied_nav_cot").glob("*.py")]
    metadata["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    owned: list[subprocess.Popen] = []
    try:
        health_checks: list[tuple[str, subprocess.Popen | None]] = []
        for role, entries in servers.items():
            for entry in entries:
                if entry["port"] is None:
                    health_checks.append((entry["url"], None))
                    continue
                server_env = env.copy()
                server_env["CUDA_VISIBLE_DEVICES"] = entry["gpu"]
                server_env["PYTHONNOUSERSITE"] = "1"
                log = (run_dir / f"{role}_gpu{entry['gpu']}_server.log").open("w", encoding="utf-8")
                process = subprocess.Popen(server_command(args, role, entry["port"], adapters[role]),
                                           env=server_env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                log.close()
                owned.append(process)
                health_checks.append((entry["url"], process))
        # Launch replicas concurrently, then wait for every endpoint before
        # starting workers; this avoids serial model-loading time on eight GPUs.
        for url, process in health_checks:
            wait_for_health(url, process, timeout=30 if process is None else 600)
        for role, entry in routers.items():
            log = (run_dir / f"{role}_router.log").open("w", encoding="utf-8")
            process = subprocess.Popen(entry["command"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            log.close()
            owned.append(process)
            wait_for_health(entry["url"], process, timeout=30)
        workers: list[subprocess.Popen] = []
        requested = set(selected_indices)
        for worker in range(args.workers):
            indices = selected_indices[worker::args.workers]
            index_path = run_dir / f"indices_{worker}.json"
            index_path.write_text(json.dumps(indices) + "\n", encoding="utf-8")
            command = [
                str(args.worker_python), str(SCRIPTS / "eval_worker.py"),
                "--runtime", str(NAVIGATION / RUNTIME[args.dataset]),
                "--config", str(NAVIGATION / "ObjectNav.qwen3vl.local.yaml"),
                "--dataset", args.dataset, "--name", run_name,
                "--worker", str(worker), "--workers", str(args.workers),
                "--index-list", str(index_path), "--max-steps", str(args.max_steps),
                "--image-res", str(args.image_res),
                "--gpu-device-id", str(args.worker_gpu_device_id),
            ]
            worker_env = env.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = worker_gpus[worker]
            worker_env["VLMNAV_ACTION_OPENAI_BASE_URL"] = worker_urls["move"][worker]
            worker_env["VLMNAV_STOPPING_OPENAI_BASE_URL"] = worker_urls["stop"][worker]
            log = (run_dir / f"worker_{worker}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            log.close()
            owned.append(process)
            workers.append(process)
        bad = []
        for index, process in enumerate(workers):
            return_code = process.wait()
            if return_code != 0:
                bad.append((index, return_code))
        if bad:
            raise RuntimeError(f"Navigation workers failed: {bad}; inspect worker_*.log")
        if routers:
            stats = {}
            for role, entry in routers.items():
                with urlopen(entry["url"] + "/stats", timeout=5) as response:
                    stats[role] = json.load(response)
            (run_dir / "router_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summarize(run_dir, run_name, requested, metadata), indent=2))
    finally:
        for process in reversed(owned):
            stop_process(process)


if __name__ == "__main__":
    # nohup/background shells can inherit ignored SIGINT. Convert both normal
    # shutdown signals to an exception so owned workers/servers are cleaned.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    main()
