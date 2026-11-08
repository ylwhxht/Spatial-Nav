#!/usr/bin/env bash
set -Eeuo pipefail

# Build the three isolated environments used by the public reproduction
# recipe. No private paths or dated environment names are required.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX_ROOT="${ROOT_DIR}/.conda"
CONDA_BIN="${CONDA_EXE:-conda}"

usage() {
  cat <<'EOF'
Usage: tools/bootstrap_envs.sh [--prefix PATH] [--conda PATH]

Creates PREFIX/train, PREFIX/sim, and PREFIX/serve for the three named
environments. Existing environments are not removed; use a fresh prefix for a
clean reproduction.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX_ROOT="$2"; shift 2 ;;
    --conda) CONDA_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "conda/mamba executable not found: $CONDA_BIN" >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then SOLVER="mamba"; else SOLVER="$CONDA_BIN"; fi
mkdir -p "$PREFIX_ROOT"

create_env() {
  local name="$1"; local spec="$2"; local prefix="$3"
  if [[ -x "$prefix/bin/python" ]]; then
    echo "Environment already exists: $prefix"
  else
    "$SOLVER" env create --prefix "$prefix" --file "$ROOT_DIR/environment/$spec" --yes
  fi
}

TRAIN_PREFIX="$PREFIX_ROOT/train"
SIM_PREFIX="$PREFIX_ROOT/sim"
SERVE_PREFIX="$PREFIX_ROOT/serve"
create_env spatial-nav-train train.yml "$TRAIN_PREFIX"
create_env spatial-nav-sim sim.yml "$SIM_PREFIX"
create_env spatial-nav-serve serve.yml "$SERVE_PREFIX"

# Install CUDA-specific wheels before the repository requirements so the
# dependency resolver cannot replace them with a CPU/default-index wheel.
"$CONDA_BIN" run -p "$TRAIN_PREFIX" python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
"$CONDA_BIN" run -p "$TRAIN_PREFIX" python -m pip install --upgrade -r "$ROOT_DIR/requirements-train.txt"

"$CONDA_BIN" run -p "$SIM_PREFIX" python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1 torchvision==0.15.2
"$CONDA_BIN" run -p "$SIM_PREFIX" python -m pip install --upgrade -r "$ROOT_DIR/requirements-eval.txt"

# vLLM's validated wheel is from the standard Linux index; requirements-serve
# pins its matching Torch/CUDA build and the required native dependencies.
"$CONDA_BIN" run -p "$SERVE_PREFIX" python -m pip install --upgrade -r "$ROOT_DIR/requirements-serve.txt"

echo "Environments created. Run the verification commands in REPRODUCTION.md before training or evaluation."
