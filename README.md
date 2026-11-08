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

The external assets are available through anonymous read-only mirrors:

- training dataset: [https://anonymous-hf.com/a/qw045c5e6efg/](https://anonymous-hf.com/a/qw045c5e6efg/)
- base-model bundle: [https://anonymous-hf.com/a/yd05u6f8p76h/](https://anonymous-hf.com/a/yd05u6f8p76h/)

The public paths expose only anonymous identifiers. The model bundle contains
the pinned Qwen3-VL-8B and InternVL3-8B base weights; the trained Move/Stop
checkpoint components listed in CHECKPOINTS.md are produced by the staged
training run or supplied as a separate checkpoint bundle.

## Quick start

Run on Linux with Python and CUDA environments matching the files in
environment/:

    export NAV_CACHE=/path/to/a/writable/cache
    export NAV_RESULTS=/path/to/results
    tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
    export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
    export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
    export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"

Download the anonymous dataset and base-model bundle, then set the local
paths used by training and evaluation:

    export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
    export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
    export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"
    tools/download_anonymous_assets.sh

The anonymous helper uses the two public proxy URLs in
ANONYMOUS_ARTIFACTS.md and requires no account token. It verifies the expected
manifest and base-model configuration files. Checkpoint directories are
created by training or supplied separately.

## Reproduction entry points

- codes/scripts/train.py runs the staged Qwen3-VL training chain.
- codes/scripts/evaluate.py runs checkpoint evaluation.
- tools/download_anonymous_assets.sh obtains the anonymous data and base models.
- tools/download_hf_assets.sh supports a separately supplied complete checkpoint bundle.
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
