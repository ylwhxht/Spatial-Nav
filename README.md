# Spatial-Nav: Anonymous Reproducibility Release

This repository is the anonymous code and reproducibility package for the
Spatial-Nav navigation study. It contains the training and evaluation code,
environment pins, protocol checks, configuration schemas, and checkpoint
verification metadata.

Large binary assets are deliberately kept outside this Git repository:

- training images and JSONL bundles;
- simulator scenes and licensed navigation meshes;
- base-model weights;
- trained adapters and merged checkpoints;
- full per-episode logs.

The anonymous dataset and model artifact identifiers are supplied separately
with the review package. They are intentionally not hard-coded here because
the repository must not disclose an author account or a non-anonymous mirror.

## Quick start

Run on Linux with Python and CUDA environments matching the files in
environment/:

    export NAV_CACHE=/path/to/a/writable/cache
    export NAV_RESULTS=/path/to/results
    tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
    export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
    export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
    export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"

Obtain the anonymous dataset and checkpoint artifact identifiers from the
review package, then set:

    export SPATIAL_NAV_DATASET_ID=ANONYMOUS_ACCOUNT/ANONYMOUS_DATASET
    export SPATIAL_NAV_CHECKPOINT_REPO=ANONYMOUS_ACCOUNT/ANONYMOUS_MODELS
    export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
    export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
    export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"
    tools/download_hf_assets.sh

The download helper requires explicit asset identifiers and never embeds an
author account. It verifies the data contract before training.

## Reproduction entry points

- codes/scripts/train.py runs the staged Qwen3-VL training chain.
- codes/scripts/evaluate.py runs checkpoint evaluation.
- tools/download_hf_assets.sh obtains the external assets.
- REPRODUCTION.md describes the tested command structure.
- CHECKPOINTS.md lists architecture, revision, path, and checksum metadata.
- tests/ contains protocol and configuration checks.

The verified Qwen3-VL phases are Foundation, Move no-KD, Move OPD, adapter
composition, Stop, and merged Stop. Move and Stop use separate data
contracts and decision interfaces.

## Anonymous release scope

The repository is initialized with a fresh history and generic commit
metadata. It contains no author names, private server paths, access tokens,
or links to a non-anonymous source account. Asset values that must be supplied
at evaluation time are explicit environment variables.

Licensed simulator scenes and base models remain subject to their original
licenses. This repository does not redistribute them.
