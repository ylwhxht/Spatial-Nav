#!/usr/bin/env bash
set -Eeuo pipefail

# Download the public anonymous data and base-model bundles used by the
# reproducibility release. No account token is required.

DATASET_URL="${SPATIAL_NAV_DATASET_URL:-https://anonymous-hf.com/a/qw045c5e6efg/}"
MODEL_URL="${SPATIAL_NAV_MODEL_URL:-https://anonymous-hf.com/a/yd05u6f8p76h/}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SPATIAL_NAV_DATA_ROOT:-${ROOT_DIR}/datasets}"
MODEL_ROOT="${SPATIAL_NAV_MODEL_ROOT:-${ROOT_DIR}/models}"

usage() {
  cat <<'EOF'
Usage:
  tools/download_anonymous_assets.sh

Optional variables:
  SPATIAL_NAV_DATASET_URL  Anonymous dataset page (default: the release URL)
  SPATIAL_NAV_MODEL_URL    Anonymous base-model page (default: the release URL)
  SPATIAL_NAV_DATA_ROOT    Dataset extraction directory (default: ./datasets)
  SPATIAL_NAV_MODEL_ROOT   Model extraction directory (default: ./models)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required" >&2
  exit 1
fi

fetch_zip() {
  local source_url="$1"
  local destination="$2"
  local label="$3"
  local endpoint="$source_url"
  local anonymous_id
  if [[ "$source_url" =~ ^https?://anonymous-hf\.com/a/([^/]+)/? ]]; then
    anonymous_id="${BASH_REMATCH[1]}"
    endpoint="https://anonymous-hf.com/api/a/${anonymous_id}/download/"
  fi
  mkdir -p "$destination"
  local archive="${destination%/}.zip.part"
  rm -f "$archive"
  echo "Downloading $label from anonymous proxy"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --retry-delay 2 --output "$archive" "$endpoint"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet --show-progress --output-document="$archive" "$endpoint"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
  unzip -q -o "$archive" -d "$destination"
  rm -f "$archive"
}

fetch_zip "$DATASET_URL" "$DATA_ROOT" "training dataset"
fetch_zip "$MODEL_URL" "$MODEL_ROOT" "base-model bundle"

# The anonymous model archive preserves its neutral source directory names.
# Add the local names expected by the training/evaluation commands.
if [[ -d "$MODEL_ROOT/qwen3-vl-8b-instruct" && ! -e "$MODEL_ROOT/Qwen3-VL-8B-Instruct" ]]; then
  ln -s "qwen3-vl-8b-instruct" "$MODEL_ROOT/Qwen3-VL-8B-Instruct"
fi
if [[ -d "$MODEL_ROOT/internvl3-8b-hf" && ! -e "$MODEL_ROOT/InternVL3-8B" ]]; then
  ln -s "internvl3-8b-hf" "$MODEL_ROOT/InternVL3-8B"
fi

test -f "$DATA_ROOT/manifest.json" \
  || { echo "Missing dataset manifest: $DATA_ROOT/manifest.json" >&2; exit 1; }
test -f "$MODEL_ROOT/Qwen3-VL-8B-Instruct/config.json" \
  || { echo "Missing Qwen base model under $MODEL_ROOT" >&2; exit 1; }
test -f "$MODEL_ROOT/InternVL3-8B/config.json" \
  || { echo "Missing InternVL base model under $MODEL_ROOT" >&2; exit 1; }

echo "Anonymous dataset and base-model bundles are ready."
echo "Trained Move/Stop checkpoints must be generated or supplied separately; see CHECKPOINTS.md."
