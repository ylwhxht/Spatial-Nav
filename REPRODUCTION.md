# Reproduction

All commands below are run from the repository root on Linux x86-64. The
reported run used cards `0-7` for training and evaluation. Evaluation starts
four Move model servers on cards `0-3`, four Stop model servers on cards `4-7`,
and ten simulator workers per card.

## 0. Released assets

[![Model: Hugging Face](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/hx24/spatial-nav-models) [![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/hx24/spatial-nav-100k-dataset)

- Model checkpoints: [hx24/spatial-nav-models](https://huggingface.co/hx24/spatial-nav-models). Use the architecture-specific files for published evaluation under checkpoints/<architecture>, or pass explicit checkpoint paths.
- Training dataset: [hx24/spatial-nav-100k-dataset](https://huggingface.co/datasets/hx24/spatial-nav-100k-dataset). Use it to populate the dataset bundle expected by datasets/ before verify-only or training.

## 1. Environment

Choose writable locations on a disk with enough space, then create the three
pinned environments from the declarative files. NAV_CACHE is only for
software environments; NAV_RESULTS is for checkpoints and evaluation logs.

```bash
export NAV_CACHE=/path/on/a/large-disk/spatial-nav-cache
export NAV_RESULTS=/path/on/a/large-disk/spatial-nav-results

tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"
```
The tested stack is: training Python 3.10.22 / PyTorch 2.2.2 / CUDA 12.1 /
Transformers 4.57.6 / PEFT 0.15.2; serving Python 3.10.22 / vLLM 0.18.1 /
PyTorch 2.10.0 / CUDA 12.8. Qwen3 evaluation workers used Python 3.9.23 /
PyTorch 2.7.0+cu126 / Habitat-Sim 0.2.5; InternVL3 evaluation workers used
Python 3.9.19 / PyTorch 2.0.1+cu118 / Habitat-Sim 0.2.5.
requirements-eval.txt contains shared worker packages; install the CUDA-matched
worker torch wheels separately. Exact pins are in environment/*.yml and the package
requirements files; recorded per-run snapshots in reproduction/logs/evaluation/.

Download the released dataset, checkpoints, and pinned base models before
training or evaluation. The helper uses the two Hugging Face releases linked
above and verifies the training bundle:

```bash
export SPATIAL_NAV_DATASET_ID=hx24/spatial-nav-100k-dataset
export SPATIAL_NAV_CHECKPOINT_REPO=hx24/spatial-nav-models
export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"
tools/download_hf_assets.sh
"$TRAIN_PY" codes/scripts/train.py --data "$SPATIAL_NAV_DATA_ROOT" --verify-only
```

The pinned bases are Qwen/Qwen3-VL-8B-Instruct revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and
OpenGVLab/InternVL3-8B-hf revision
`259a3b64a14623c0ec91a045cb43f7c5af5fa6af`.

## 2. Qwen3-VL-8B training

The verified end-to-end command is:

```bash
"$TRAIN_PY" codes/scripts/train.py \
  --architecture qwen3-vl-8b \
  --base-model "$NAV_CACHE/models/Qwen3-VL-8B-Instruct" \
  --data "$SPATIAL_NAV_DATA_ROOT" \
  --output "$NAV_RESULTS/runs/qwen3-vl-8b" \
  --stage all --gpus 8 --gpu-ids 0,1,2,3,4,5,6,7
```

`--stage all` runs Foundation, Move no-KD, Move OPD, adapter composition,
Stop, and merged Stop. The command expands to the scripts in
`codes/scripts/`; it records each `run_config.json` and refuses mismatched
data, base adapters, or row counts.

To evaluate the weights produced by this run, pass these two paths to the
evaluation command in the next section:

```bash
--move-checkpoint "$NAV_RESULTS/runs/qwen3-vl-8b/compose/final_adapter" \
--stop-checkpoint "$NAV_RESULTS/runs/qwen3-vl-8b/stop_merged"
```

For a fresh training output, uncomment the second CHECKPOINT_ARGS line in the evaluation block. The published checkpoint bundle uses `--checkpoint-root` directly.

Verified Qwen3 phase settings:

| phase | rows | epochs | per-GPU batch | accumulation | length | learning rate | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Foundation | 40,000 | 2 | 1 | 12 | 4096 | 1e-5 | 834 |
| Move no-KD | 40,000 | 1 | 4 | 1 | 1536 | 1e-5 backbone | 1250 |
| Move OPD | 13,317 | 1 | 1 | 4 | 2048 | 5e-7 | 417 |
| Stop | 14,125 | 1 | 1 | 4 | 2048 | 1e-6 | 442 |

All phases use LoRA `r=64`, `alpha=128`, BF16, vision LoRA disabled,
`--no-gradient-checkpointing`, fixed seeds from the generated command, and
the historical native-LM loss semantics. The reproduced phase summaries are
in `reproduction/logs/qwen3-training/`.

## 3. Checkpoint evaluation

Use `codes/scripts/evaluate.py` once per dataset. Replace the model, base,
checkpoint and asset variables for each architecture; keep the serving and
worker options unchanged.

```bash
export ARCH=qwen3-vl-8b                 # or internvl3-8b
export BASE_MODEL="$NAV_CACHE/models/Qwen3-VL-8B-Instruct"  # use InternVL3-8B for InternVL3
export CHECKPOINT_ROOT="$SPATIAL_NAV_CHECKPOINT_ROOT/$ARCH"
export DATASET=hm3d_v2                  # hm3d_v1, hm3d_v2, or mp3d
export SCENE_ROOT=/path/to/scene-root
export OBJECTNAV_ROOT=/path/to/objectnav-root
export RUN_DIR="$NAV_RESULTS/eval/$ARCH/$DATASET"

# Use the published checkpoints by default. For a fresh Qwen3 run, use the
# two explicit output paths shown in the training section instead.
CHECKPOINT_ARGS=(--checkpoint-root "$CHECKPOINT_ROOT")
# CHECKPOINT_ARGS=(--move-checkpoint "$NAV_RESULTS/runs/qwen3-vl-8b/compose/final_adapter" --stop-checkpoint "$NAV_RESULTS/runs/qwen3-vl-8b/stop_merged")

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

For InternVL3 use `--stop-threshold 0.0`. The evaluator writes
`metrics.json` and `run_config.json`; each run must use a new empty directory.
The full validation splits contain HM3D-v1 2,000, HM3D-v2 1,000, and MP3D
2,195 episodes. MP3D requires the bundled 11 validation navmeshes; generate
and install them with `tools/generate_mp3d_navmeshes.py` and
`codes/scripts/prepare_mp3d_navmesh.py` before evaluation.

## 4. Recorded full-split results

Values are percentages (`SR / SPL`) from the JSON files under
`reproduction/results/`.

| checkpoint | HM3D-v1 (2,000) | HM3D-v2 (1,000) | MP3D (2,195) |
|---|---:|---:|---:|
| Qwen3-VL-8B (fresh 8-GPU training output) | 60.25 / 30.47 | 76.80 / 34.86 | 48.43 / 20.47 |
| InternVL3-8B (published checkpoint) | 57.80 / 31.22 | 72.40 / 34.66 | 42.23 / 19.53 |

The recorded host required simulator workers to use CPU/Mesa mode because its
NVIDIA EGL userspace library was unavailable; model servers still used the
eight requested GPUs. To reproduce that host run, set
`--worker-gpu-device-id -1`; on a host with working EGL keep the default `0`.
The run metadata records the requested `historical` routing and the effective
least-inflight router used for the eight replica endpoints.
