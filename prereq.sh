#!/usr/bin/env bash
# prereq.sh — vérifie (et installe si besoin) la chaîne d'outils de malaga.
#
#   ./prereq.sh          vérifie et installe ce qui manque (Rust via rustup, paquets de build)
#   CHECK_ONLY=1 ./prereq.sh   vérifie seulement, sans rien installer
#
# Le support GPU NVIDIA demande en plus le CUDA Toolkit (nvcc) : il n'est pas
# installé automatiquement (taille, version liée au driver), le script indique
# seulement s'il est présent. Idempotent : peut être relancé sans effet de bord.
set -euo pipefail

ok()   { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✘\033[0m %s\n' "$*" >&2; exit 1; }
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo -n"

OS=$(uname -s)
echo "Système : $OS"
case "$OS" in
  Linux)
    if [ "${CHECK_ONLY:-0}" != 1 ] && ! command -v cc >/dev/null; then
      if command -v apt-get >/dev/null; then
        $SUDO apt-get update && $SUDO apt-get install -y build-essential pkg-config curl git \
          || die "installation des paquets de build impossible (relancer avec sudo)"
      elif command -v dnf >/dev/null; then
        $SUDO dnf install -y gcc gcc-c++ make pkgconf-pkg-config curl git || die "dnf a échoué"
      elif command -v pacman >/dev/null; then
        $SUDO pacman -S --needed --noconfirm base-devel curl git || die "pacman a échoué"
      else
        die "gestionnaire de paquets non reconnu : installer un compilateur C/C++, pkg-config, curl et git"
      fi
    fi ;;
  Darwin)
    xcode-select -p >/dev/null 2>&1 || die "installer les Xcode Command Line Tools : xcode-select --install" ;;
  MINGW*|MSYS*|CYGWIN*)
    warn "Windows : installer Visual Studio Build Tools (C++) ; CUDA via le CUDA Toolkit NVIDIA" ;;
  *) warn "système non testé : $OS" ;;
esac
command -v cc >/dev/null || command -v cl >/dev/null && ok "compilateur C/C++" || die "compilateur C/C++ absent"
command -v git >/dev/null && ok "git" || die "git absent"
command -v curl >/dev/null && ok "curl" || die "curl absent"

if ! command -v cargo >/dev/null && [ -x "$HOME/.cargo/bin/cargo" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null; then
  [ "${CHECK_ONLY:-0}" = 1 ] && die "Rust absent (https://rustup.rs)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    || die "installation de rustup impossible"
  export PATH="$HOME/.cargo/bin:$PATH"
fi
# rust-toolchain.toml épingle la version : rustup l'installe au premier appel.
ok "Rust $(cargo --version | cut -d' ' -f2) (épinglé par rust-toolchain.toml)"

NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)
if [ -n "$NVCC" ]; then
  ok "CUDA : $("$NVCC" --version | grep -o 'release [0-9.]*')  → ./build.sh --cuda"
else
  warn "nvcc introuvable : build CPU uniquement (installer le CUDA Toolkit ≥ 12 pour le GPU)"
fi
command -v python3 >/dev/null && ok "python3 (optionnel : scripts d'évaluation)" || warn "python3 absent (optionnel)"

echo "Vérification finale :"
cargo --version >/dev/null && rustc --version >/dev/null && ok "prérequis OK"
