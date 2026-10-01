#!/usr/bin/env bash
# setup.sh — prépare un poste de développement : prérequis, dépendances Rust,
# vérification de compilation. Idempotent et non interactif.
#
#   ./setup.sh            CPU
#   ./setup.sh --cuda     avec le support GPU NVIDIA
set -euo pipefail
cd "$(dirname "$0")"
FEATURES=()
[ "${1:-}" = "--cuda" ] && FEATURES=(--features cuda)

./prereq.sh
export PATH="$HOME/.cargo/bin:$PATH"
# Vérification des outils indispensables (prereq.sh les installe s'ils manquent).
for tool in cargo rustc git curl; do
  command -v "$tool" >/dev/null || { echo "outil manquant : $tool (relancer ./prereq.sh)" >&2; exit 1; }
done
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
if [ "${1:-}" = "--cuda" ] && ! command -v nvcc >/dev/null; then
  echo "nvcc introuvable : installer le CUDA Toolkit ou lancer ./setup.sh sans --cuda" >&2; exit 1
fi
mkdir -p models/hf models/gguf models/data
cargo fetch --locked || { echo "cargo fetch a échoué (réseau ?)" >&2; exit 1; }
cargo check --workspace --locked "${FEATURES[@]}"
echo "setup OK — étape suivante : ./build.sh ${1:-} puis scripts/download.sh"
