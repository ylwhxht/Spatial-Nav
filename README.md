# Spatial-Nav

<div align="center">
  <img src="figures/teaser.png" width="900" alt="Spatial-Nav overview" />
</div>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2609.29934) [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/hx24/spatial-nav-100k-dataset) [![Models](https://img.shields.io/badge/Models-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/hx24/spatial-nav-models)

</div>

Spatial-Nav studies how spatial supervision transfers from vision-language benchmarks to embodied navigation. The release contains the Spatial-Nav-100K data interface, the Qwen3-VL-8B end-to-end training chain, and the checkpoint evaluator for Qwen3-VL-8B and InternVL3-8B.

The complete reproduction protocol, pinned environments, commands, recorded logs, and metrics are in [REPRODUCTION.md](REPRODUCTION.md). The original project figure is shown above; paper, dataset, and model files are linked in the badges.

## Quick Start

Run the commands below from a Linux x86-64 host with eight visible GPUs. Large base-model weights, simulator scenes, navigation meshes, and training images are downloaded separately and are not stored in GitHub.

### 1. Prepare the dataset and checkpoints

Create a writable cache and download the released dataset/checkpoint bundle:

```bash
export NAV_CACHE=/path/to/spatial-nav-cache
export NAV_RESULTS=/path/to/spatial-nav-results
export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"
export SPATIAL_NAV_DATASET_ID=hx24/spatial-nav-100k-dataset
export SPATIAL_NAV_CHECKPOINT_REPO=hx24/spatial-nav-models

bash tools/download_hf_assets.sh
```

For evaluation, also prepare the licensed HM3D-v1, HM3D-v2, or MP3D scene/objectnav assets. MP3D evaluation additionally needs the validation navmeshes; see [REPRODUCTION.md](REPRODUCTION.md#3-checkpoint-evaluation).

### 2. Create the pinned environments

The helper creates separate training, simulator-worker, and model-serving environments from the files under `environment/`:

```bash
tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"

"$TRAIN_PY" codes/scripts/train.py \
  --data "$SPATIAL_NAV_DATA_ROOT" --verify-only
```

The tested stack and exact package pins are recorded in [REPRODUCTION.md](REPRODUCTION.md) and `requirements-*.txt`.

### 3. Train Qwen3-VL-8B from scratch

Run the full six-stage chain on GPUs 0–7:

```bash
"$TRAIN_PY" codes/scripts/train.py \
  --architecture qwen3-vl-8b \
  --base-model "$NAV_CACHE/models/Qwen3-VL-8B-Instruct" \
  --data "$SPATIAL_NAV_DATA_ROOT" \
  --output "$NAV_RESULTS/runs/qwen3-vl-8b" \
  --stage all --gpus 8 --gpu-ids 0,1,2,3,4,5,6,7
```

`--stage all` runs Foundation, Move no-KD, Move OPD, adapter composition, Stop, and merged Stop. The resulting Move and Stop checkpoints can be passed directly to the evaluator. Phase settings and recorded training logs are listed in [REPRODUCTION.md](REPRODUCTION.md#2-qwen3-vl-8b-training).

### 4. Evaluate a checkpoint

Use one command per dataset (`hm3d_v1`, `hm3d_v2`, or `mp3d`). The verified layout starts four Move servers on GPUs 0–3, four Stop servers on GPUs 4–7, and ten simulator workers per GPU:

```bash
export DATASET=hm3d_v2
export SCENE_ROOT=/path/to/scene-root
export OBJECTNAV_ROOT=/path/to/objectnav-root
export RUN_DIR="$NAV_RESULTS/eval/qwen3-vl-8b/$DATASET"

"$TRAIN_PY" codes/scripts/evaluate.py \
  --architecture qwen3-vl-8b --variant checkpoint \
  --base-model "$NAV_CACHE/models/Qwen3-VL-8B-Instruct" \
  --dataset "$DATASET" \
  --scene-root "$SCENE_ROOT" --objectnav-root "$OBJECTNAV_ROOT" \
  --checkpoint-root "$SPATIAL_NAV_CHECKPOINT_ROOT/qwen3-vl-8b" \
  --run-dir "$RUN_DIR" \
  --worker-python "$SIM_PY" --serve-python "$SERVE_PY" \
  --move-gpus 0,1,2,3 --stop-gpus 4,5,6,7 \
  --sim-gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 10 \
  --allow-shared-server-gpus --gpu-memory-utilization 0.70 \
  --request-routing historical --max-steps 500 --image-res 1920 \
  --max-num-seqs 64 --move-max-model-len 3072 --stop-max-model-len 4096 \
  --worker-gpu-device-id 0 --stop-threshold 2.0
```

For a freshly trained Qwen3 run, replace `--checkpoint-root` with `--move-checkpoint` and `--stop-checkpoint` pointing to the two output directories. For InternVL3, change `--architecture`, `--base-model`, checkpoint root, and use `--stop-threshold 0.0`. The evaluator writes `metrics.json` and `run_config.json` under `RUN_DIR`.

## Recorded results

The released full-split results are percentages (`SR / SPL`):

| checkpoint | HM3D-v1 | HM3D-v2 | MP3D |
|---|---:|---:|---:|
| Qwen3-VL-8B, fresh 8-GPU training | 60.25 / 30.47 | 76.80 / 34.86 | 48.43 / 20.47 |
| InternVL3-8B, published checkpoint | 57.80 / 31.22 | 72.40 / 34.66 | 42.23 / 19.53 |

The corresponding logs and JSON metrics are under `reproduction/`. Checkpoint provenance and SHA-256 records are in [CHECKPOINTS.md](CHECKPOINTS.md).

## Repository layout

- `codes/`: training, serving, evaluation, and simulator preparation scripts.
- `environment/`: pinned environment definitions for training, serving, and workers.
- `tools/`: Hugging Face download/verification and MP3D preparation helpers.
- `reproduction/`: compact run metadata, logs, and recorded metrics.
- `checkpoint_configs/`, `checkpoints/`: public checkpoint metadata and tokenizer/config files; large weights remain on Hugging Face.
- `tests/`: lightweight integrity and command checks.

## Citation

```bibtex
@article{huang2026spatialnav,
  title   = {Beyond Spatial Benchmarks: From Spatial Reasoning to Navigation},
  author  = {Huang, Xun and Zhao, Shijia and Qu, Rongsheng and Li, Jiayuan and Lu, Xin and Li, Weixin and Wen, Chenglu and Wang, Cheng},
  journal = {arXiv preprint arXiv:2609.29934},
  year    = {2026}
}
```

See [README_CN.md](README_CN.md) for a concise Chinese entry point.
