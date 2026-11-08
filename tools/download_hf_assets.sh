#!/usr/bin/env bash
set -Eeuo pipefail

# Download the assets that are intentionally kept outside the code release.
# The two repository IDs are explicit inputs so a public mirror can be used
# without editing this script or putting credentials in the repository.

usage() {
  cat <<'EOF'
Usage:
  SPATIAL_NAV_DATASET_ID=ORG/DATASET \
  SPATIAL_NAV_CHECKPOINT_REPO=ORG/MODELS \
  tools/download_hf_assets.sh

Optional variables:
  SPATIAL_NAV_DATASET_REVISION       Dataset revision (default: main)
  SPATIAL_NAV_CHECKPOINT_REVISION    Model revision (default: main)
  SPATIAL_NAV_QWEN_MODEL_REVISION    Qwen base revision (default: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b)
  SPATIAL_NAV_INTERNVL_MODEL_REVISION InternVL base revision (default: 259a3b64a14623c0ec91a045cb43f7c5af5fa6af)
  SPATIAL_NAV_DATA_ROOT              Training bundle directory (default: ./datasets)
  SPATIAL_NAV_MODEL_ROOT             Base-model directory (default: ./models)
  SPATIAL_NAV_CHECKPOINT_ROOT        Checkpoint directory (default: ./checkpoints)
  HF_BIN                             Hugging Face CLI executable (default: hf)
  TRAIN_PY                           Python executable for verification (default: python)

The script never removes files. Existing directories are populated by
`hf download`; use a fresh checkout for a clean provenance record.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${SPATIAL_NAV_DATASET_ID:?Set SPATIAL_NAV_DATASET_ID to the training Dataset Hub ID}"
: "${SPATIAL_NAV_CHECKPOINT_REPO:?Set SPATIAL_NAV_CHECKPOINT_REPO to the checkpoint Model Hub ID}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_BIN="${HF_BIN:-hf}"
TRAIN_PY="${TRAIN_PY:-python}"
DATASET_REVISION="${SPATIAL_NAV_DATASET_REVISION:-main}"
CHECKPOINT_REVISION="${SPATIAL_NAV_CHECKPOINT_REVISION:-main}"
QWEN_REVISION="${SPATIAL_NAV_QWEN_MODEL_REVISION:-0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
INTERNVL_REVISION="${SPATIAL_NAV_INTERNVL_MODEL_REVISION:-259a3b64a14623c0ec91a045cb43f7c5af5fa6af}"
DATA_ROOT="${SPATIAL_NAV_DATA_ROOT:-${ROOT_DIR}/datasets}"
MODEL_ROOT="${SPATIAL_NAV_MODEL_ROOT:-${ROOT_DIR}/models}"
CHECKPOINT_ROOT="${SPATIAL_NAV_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "Hugging Face CLI not found: $HF_BIN (install huggingface_hub in spatial-nav-train)" >&2
  exit 1
fi
if ! command -v "$TRAIN_PY" >/dev/null 2>&1; then
  echo "Training Python not found: $TRAIN_PY (activate spatial-nav-train or set TRAIN_PY)" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$CHECKPOINT_ROOT"

echo "Downloading training data: $SPATIAL_NAV_DATASET_ID@$DATASET_REVISION -> $DATA_ROOT"
"$HF_BIN" download "$SPATIAL_NAV_DATASET_ID" \
  --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$DATA_ROOT"

echo "Downloading checkpoints: $SPATIAL_NAV_CHECKPOINT_REPO@$CHECKPOINT_REVISION -> $CHECKPOINT_ROOT"
"$HF_BIN" download "$SPATIAL_NAV_CHECKPOINT_REPO" \
  --repo-type model --revision "$CHECKPOINT_REVISION" --local-dir "$CHECKPOINT_ROOT"

echo "Downloading Qwen base model@$QWEN_REVISION -> $MODEL_ROOT/Qwen3-VL-8B-Instruct"
"$HF_BIN" download Qwen/Qwen3-VL-8B-Instruct \
  --revision "$QWEN_REVISION" --local-dir "$MODEL_ROOT/Qwen3-VL-8B-Instruct"

echo "Downloading InternVL base model@$INTERNVL_REVISION -> $MODEL_ROOT/InternVL3-8B"
"$HF_BIN" download OpenGVLab/InternVL3-8B-hf \
  --revision "$INTERNVL_REVISION" --local-dir "$MODEL_ROOT/InternVL3-8B"

echo "Verifying the downloaded training bundle"
"$TRAIN_PY" "$ROOT_DIR/codes/scripts/train.py" --data "$DATA_ROOT" --verify-only

echo "Checking the required checkpoint directories"
for architecture in qwen3-vl-8b internvl3-8b; do
  test -f "$CHECKPOINT_ROOT/$architecture/move_composed/adapter_model.safetensors" \
    || { echo "Missing Move adapter for $architecture" >&2; exit 1; }
  test -f "$CHECKPOINT_ROOT/$architecture/stop_merged/config.json" \
    || { echo "Missing merged Stop model for $architecture" >&2; exit 1; }
done

test -f "$MODEL_ROOT/Qwen3-VL-8B-Instruct/config.json" \
  || { echo "Missing Qwen base model: $MODEL_ROOT/Qwen3-VL-8B-Instruct/config.json" >&2; exit 1; }
test -f "$MODEL_ROOT/InternVL3-8B/config.json" \
  || { echo "Missing InternVL base model: $MODEL_ROOT/InternVL3-8B/config.json" >&2; exit 1; }

echo "Hugging Face asset download and verification completed."
