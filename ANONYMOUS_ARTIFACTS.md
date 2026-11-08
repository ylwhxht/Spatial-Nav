# Anonymous artifact access

The release uses two read-only Anonymous HF proxy URLs. Their public paths
contain only anonymous identifiers and do not expose the source account or
repository name.

| Asset | Anonymous URL | Contents |
|---|---|---|
| Training dataset | [https://anonymous-hf.com/a/qw045c5e6efg/](https://anonymous-hf.com/a/qw045c5e6efg/) | Training JSONL bundles, image shards, evidence cache, and manifests |
| Base-model bundle | [https://anonymous-hf.com/a/yd05u6f8p76h/](https://anonymous-hf.com/a/yd05u6f8p76h/) | Qwen3-VL-8B and InternVL3-8B base-model weights and tokenizer/configuration files |

The links are active and provide ZIP downloads through the proxy. The dataset
archive is the input to the four training stages. The model archive currently
contains `qwen3-vl-8b-instruct/` and `internvl3-8b-hf/`; it is a base-weight
bundle, not a trained Move/Stop checkpoint bundle. The checkpoint inventory and
expected checksums are recorded in CHECKPOINTS.md.

## Download

From the repository root, use:

    export SPATIAL_NAV_DATA_ROOT=/path/to/cache/datasets
    export SPATIAL_NAV_MODEL_ROOT=/path/to/cache/models
    tools/download_anonymous_assets.sh

The helper resolves each anonymous page to its read-only ZIP endpoint, extracts
the files, checks the dataset manifest, and creates the model-directory names
expected by the training commands. It does not require a Hugging Face login or
store credentials.

If a separate complete checkpoint bundle is supplied, place it under
`SPATIAL_NAV_CHECKPOINT_ROOT` according to CHECKPOINTS.md and run the existing
checkpoint verification steps before evaluation.
