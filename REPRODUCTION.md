# Reproduction guide

This guide records the exact Qwen3-VL-8B training run and the Qwen3/InternVL3 checkpoint evaluations used for the released numbers. Commands are run from the repository root on Linux x86-64.

## 1. Workload and hardware

- Training: eight GPUs (`0-7`) with one distributed training process per GPU.
- Evaluation: four Move model servers on GPUs `0-3`, four Stop model servers on GPUs `4-7`, and ten simulator workers per GPU.
- Training data: Spatial-Nav-100K, with 40,000 Foundation rows, 40,000 Move no-KD rows, 13,317 Move OPD rows, and 14,125 Stop rows.
- Evaluation splits: HM3D-v1 (2,000 episodes), HM3D-v2 (1,000 episodes), and MP3D (2,195 episodes).
- The scene/objectnav assets and base-model weights are downloaded separately. MP3D also requires the validation navigation meshes.

The public assets are [hx24/spatial-nav-100k-dataset](https://huggingface.co/datasets/hx24/spatial-nav-100k-dataset) and [hx24/spatial-nav-models](https://huggingface.co/hx24/spatial-nav-models).

## 2. Environment setup

Use separate environments for training, simulator workers, and model serving:

```bash
export NAV_CACHE=/path/on/a/large-disk/spatial-nav-cache
export NAV_RESULTS=/path/on/a/large-disk/spatial-nav-results

tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"
```

The observed run used the following stack:

| environment | Python | PyTorch | CUDA | main packages |
|---|---:|---:|---:|---|
| training | 3.10.22 | 2.2.2 | 12.1 | Transformers 4.57.6, PEFT 0.15.2 |
| serving | 3.10.22 | 2.10.0 | 12.8 | vLLM 0.18.1 |
| Qwen3 workers | 3.9.23 | 2.7.0+cu126 | CUDA 12.6 | Habitat-Sim 0.2.5 |
| InternVL3 workers | 3.9.19 | 2.0.1+cu118 | CUDA 11.8 | Habitat-Sim 0.2.5 |

The corresponding package files are `requirements-train.txt`, `requirements-serve.txt`, `requirements-eval.txt`, and `environment/*.yml`. Download and verify the data bundle before training:

```bash
export SPATIAL_NAV_DATASET_ID=hx24/spatial-nav-100k-dataset
export SPATIAL_NAV_CHECKPOINT_REPO=hx24/spatial-nav-models
export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"

bash tools/download_hf_assets.sh
"$TRAIN_PY" codes/scripts/train.py --data "$SPATIAL_NAV_DATA_ROOT" --verify-only
```

The base-model revisions used for the recorded run were Qwen/Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and OpenGVLab/InternVL3-8B-hf revision `259a3b64a14623c0ec91a045cb43f7c5af5fa6af`.

## 3. Qwen3-VL-8B training

Run all stages with:

```bash
"$TRAIN_PY" codes/scripts/train.py \
  --architecture qwen3-vl-8b \
  --base-model "$NAV_CACHE/models/Qwen3-VL-8B-Instruct" \
  --data "$SPATIAL_NAV_DATA_ROOT" \
  --output "$NAV_RESULTS/runs/qwen3-vl-8b" \
  --stage all --gpus 8 --gpu-ids 0,1,2,3,4,5,6,7
```

`--stage all` runs Foundation, Move no-KD, Move OPD, adapter composition, Stop, and merged Stop. The resulting directories used by evaluation are:

```text
$NAV_RESULTS/runs/qwen3-vl-8b/compose/final_adapter
$NAV_RESULTS/runs/qwen3-vl-8b/stop_merged
```

The observed phase settings and optimizer steps were:

| phase | rows | epochs | batch / accumulation | max length | learning rate | optimizer steps | observed time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Foundation | 40,000 | 2 | 1 / 12 | 4096 | 1e-5 | 834 | 2 h 22 min |
| Move no-KD | 40,000 | 1 | 4 / 1 | 1536 | 1e-5 backbone | 1,250 | 1 h 47 min |
| Move OPD | 13,317 | 1 | 1 / 4 | 2048 | 5e-7 | 417 | 32 min |
| Stop | 14,125 | 1 | 1 / 4 | 2048 | 1e-6 | 442 | 25 min |
| composition and merge | — | — | — | — | — | — | about 1 min |

The full observed training window was approximately **5 h 10 min** (03:00–08:10, UTC+08:00). The phase summaries are in `reproduction/logs/qwen3-training/`. The diagnostic records report the expected optimizer steps and successful pipeline completion; for example, Move no-KD reports 75.20% action accuracy and 52.86% ability micro-F1 on its diagnostic split.

## 4. Checkpoint evaluation

Use one evaluation run per dataset. Set `ARCH`, `BASE_MODEL`, and `CHECKPOINT_ROOT` for the selected model. The command below uses the released checkpoint bundle:

```bash
export ARCH=qwen3-vl-8b                 # or internvl3-8b
export BASE_MODEL="$NAV_CACHE/models/Qwen3-VL-8B-Instruct"  # use the InternVL3 base for InternVL3
export CHECKPOINT_ROOT="$SPATIAL_NAV_CHECKPOINT_ROOT/$ARCH"
export DATASET=hm3d_v2                  # hm3d_v1, hm3d_v2, or mp3d
export SCENE_ROOT=/path/to/scene-root
export OBJECTNAV_ROOT=/path/to/objectnav-root
export RUN_DIR="$NAV_RESULTS/eval/$ARCH/$DATASET"

CHECKPOINT_ARGS=(--checkpoint-root "$CHECKPOINT_ROOT")
# For a fresh Qwen3 run instead use:
# CHECKPOINT_ARGS=(--move-checkpoint "$NAV_RESULTS/runs/qwen3-vl-8b/compose/final_adapter" \
#   --stop-checkpoint "$NAV_RESULTS/runs/qwen3-vl-8b/stop_merged")

"$TRAIN_PY" codes/scripts/evaluate.py \
  --architecture "$ARCH" --variant checkpoint \
  --base-model "$BASE_MODEL" --dataset "$DATASET" \
  --scene-root "$SCENE_ROOT" --objectnav-root "$OBJECTNAV_ROOT" \
  "${CHECKPOINT_ARGS[@]}" --run-dir "$RUN_DIR" \
  --worker-python "$SIM_PY" --serve-python "$SERVE_PY" \
  --move-gpus 0,1,2,3 --stop-gpus 4,5,6,7 \
  --sim-gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 10 \
  --allow-shared-server-gpus --gpu-memory-utilization 0.70 \
  --request-routing historical --max-steps 500 --image-res 1920 \
  --max-num-seqs 64 --move-max-model-len 3072 --stop-max-model-len 4096 \
  --worker-gpu-device-id 0 --stop-threshold 2.0
```

For InternVL3, use `--stop-threshold 0.0`. Every run writes `metrics.json` and `run_config.json` under a new `RUN_DIR`. MP3D requires the validation navmeshes; prepare them with `tools/generate_mp3d_navmeshes.py` and `codes/scripts/prepare_mp3d_navmesh.py`.

## 5. Evaluation time and results

The recorded Qwen3 evaluation launcher ran the three datasets sequentially:

| dataset | episodes | observed elapsed time | SR | SPL |
|---|---:|---:|---:|---:|
| HM3D-v1 | 2,000 | 1 h 57 min 35 s | 60.25 | 30.47 |
| HM3D-v2 | 1,000 | 53 min 09 s | 76.80 | 34.86 |
| MP3D | 2,195 | 2 h 09 min 10 s | 48.43 | 20.47 |
| **total** | **5,195** | **4 h 59 min 54 s** | — | — |

The corresponding InternVL3 checkpoint results are:

| dataset | episodes | SR | SPL |
|---|---:|---:|---:|
| HM3D-v1 | 2,000 | 57.80 | 31.22 |
| HM3D-v2 | 1,000 | 72.40 | 34.66 |
| MP3D | 2,195 | 42.23 | 19.53 |

All values are percentages. The machine used for the recorded evaluation did not provide NVIDIA EGL userspace libraries, so simulator workers ran with CPU/Mesa while the model servers used GPUs 0–7. On a host with working EGL, keep `--worker-gpu-device-id 0`; to reproduce the recorded fallback, use `--worker-gpu-device-id -1`.

The compact commands, launch metadata, metrics, and compatibility note are kept under `reproduction/`:

```text
reproduction/logs/qwen3-training/
reproduction/logs/evaluation/
reproduction/results/qwen3-vl-8b/
reproduction/results/internvl3-8b/
```

## 6. Expected output layout

```text
$NAV_RESULTS/
├── runs/qwen3-vl-8b/
│   ├── foundation/
│   ├── move_no_kd/
│   ├── move_opd/
│   ├── compose/final_adapter/
│   ├── stop/
│   └── stop_merged/
└── eval/
    ├── qwen3-vl-8b/<dataset>/{metrics.json,run_config.json}
    └── internvl3-8b/<dataset>/{metrics.json,run_config.json}
```
