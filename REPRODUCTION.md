# Anonymous reproduction instructions

All commands are run from the repository root on Linux x86-64. The release
does not contain scenes, base-model weights, training images, or large
checkpoints. The anonymous artifact package supplies the corresponding
dataset and model identifiers.

## 1. Environments

Choose writable paths on a large disk:

    export NAV_CACHE=/path/to/spatial-nav-cache
    export NAV_RESULTS=/path/to/spatial-nav-results
    tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
    export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
    export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
    export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"

The tested stack is pinned in environment/*.yml and the requirements files.
Simulator workers may require CPU/Mesa mode on hosts without NVIDIA EGL.

## 2. External assets

Set the anonymous identifiers supplied with the review package:

    export SPATIAL_NAV_DATASET_ID=ANONYMOUS_ACCOUNT/ANONYMOUS_DATASET
    export SPATIAL_NAV_CHECKPOINT_REPO=ANONYMOUS_ACCOUNT/ANONYMOUS_MODELS
    export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
    export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
    export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"
    tools/download_hf_assets.sh

The helper also downloads the pinned public base-model revisions and runs the
data verification entry point. It never stores credentials in the repository.

## 3. Qwen3-VL training

    "$TRAIN_PY" codes/scripts/train.py \
      --architecture qwen3-vl-8b \
      --base-model "$NAV_CACHE/models/Qwen3-VL-8B-Instruct" \
      --data "$SPATIAL_NAV_DATA_ROOT" \
      --output "$NAV_RESULTS/runs/qwen3-vl-8b" \
      --stage all --gpus 8 --gpu-ids 0,1,2,3,4,5,6,7

The command records run_config.json and checks data row counts, base adapters,
and Move/Stop decision kinds. The phase contract is:

| phase | rows | per-GPU batch | accumulation | length | learning rate | steps |
|---|---:|---:|---:|---:|---:|---:|
| Foundation | 40,000 | 1 | 12 | 4096 | 1e-5 | 834 |
| Move no-KD | 40,000 | 4 | 1 | 1536 | 1e-5 | 1250 |
| Move OPD | 13,317 | 1 | 4 | 2048 | 5e-7 | 417 |
| Stop | 14,125 | 1 | 4 | 2048 | 1e-6 | 442 |

## 4. Evaluation

Use codes/scripts/evaluate.py once per dataset. Supply explicit local paths
for the scenes, ObjectNav metadata, base model, and checkpoint root. Keep the
serving and worker options fixed across architectures and datasets. The
evaluation contract records the requested routing mode, simulator worker
configuration, stop threshold, episode count, and effective GPU mapping.

The validation splits contain HM3D-v1 2,000 episodes, HM3D-v2 1,000 episodes,
and MP3D 2,195 episodes. MP3D requires the bundled validation navigation
meshes to be generated with tools/generate_mp3d_navmeshes.py.

Every evaluation directory must be new and must retain metrics.json,
run_config.json, and the per-episode output. Do not overwrite prior runs.

## 5. Scope and provenance

This is an anonymous code-only release. Large artifacts and full run logs are
distributed separately so that the Git history stays reviewable and does not
expose local paths or author metadata. CHECKPOINTS.md and the manifest provide
the architecture revisions, relative paths, and checksums needed to audit the
external artifact bundle.
