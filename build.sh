#!/usr/bin/env bash
# build.sh — compile le binaire release `malaga` dans build/.
#
#   ./build.sh            CPU
#   ./build.sh --cuda     GPU NVIDIA (nvcc requis ; CUDA_COMPUTE_CAP=89 pour une RTX 40xx, 80 A100, 90 H100…)
#   ./build.sh --metal    GPU Apple
#
# Non interactif (utilisable en CI) ; n'écrit que dans target/ et build/.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
FEATURES=(); SUFFIX=cpu
case "${1:-}" in
  --cuda)
    [ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
    command -v nvcc >/dev/null || { echo "nvcc introuvable : installer le CUDA Toolkit" >&2; exit 1; }
    if [ -z "${CUDA_COMPUTE_CAP:-}" ] && command -v nvidia-smi >/dev/null; then
      CUDA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
      export CUDA_COMPUTE_CAP
    fi
    FEATURES=(--features cuda); SUFFIX=cuda ;;
  --metal) FEATURES=(--features metal); SUFFIX=metal ;;
  "") ;;
  *) echo "option inconnue : $1 (--cuda, --metal)" >&2; exit 1 ;;
esac
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
cargo build --release --locked -p malaga "${FEATURES[@]}"
mkdir -p build
OUT="build/malaga-$(uname -s | tr '[:upper:]' '[:lower:]')-$SUFFIX-$VERSION"
cp target/release/malaga "$OUT"
"$OUT" --version
echo "artefact : $OUT"
