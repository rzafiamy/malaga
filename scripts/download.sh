#!/usr/bin/env bash
# Downloads an NLLB-200 checkpoint from Hugging Face and converts it to GGUF.
#
#   scripts/download.sh [600M|1.3B|3.3B] [preset ...]
#
# Presets: f32 f16 q8_0 q6_k q5_k_m q4_k_m (default: q8_0 q4_k_m).
# Set SHORTLIST=1 to also embed a Malagasy vocabulary shortlist (opt-in
# `--fast-vocab` mode, built from Malagasy Wikipedia).
set -euo pipefail
SIZE=${1:-600M}; shift || true
PRESETS=${*:-q8_0 q4_k_m}
case $SIZE in
  600M) REPO=facebook/nllb-200-distilled-600M ;;
  1.3B) REPO=facebook/nllb-200-distilled-1.3B ;;
  3.3B) REPO=facebook/nllb-200-3.3B ;;
  *) echo "unknown size $SIZE (600M, 1.3B, 3.3B)"; exit 1 ;;
esac
NAME=$(basename "$REPO")
HF=models/hf/$NAME
BIN=${MALAGA_BIN:-./target/release/malaga}
mkdir -p "$HF" models/gguf models/data

files=$(curl -fsSL "https://huggingface.co/api/models/$REPO" | python3 -c \
  "import json,sys; print(' '.join(s['rfilename'] for s in json.load(sys.stdin)['siblings']))")
for f in $files; do
  case $f in
    config.json|generation_config.json|tokenizer.json|*.safetensors|pytorch_model*.bin|*.index.json)
      [ -s "$HF/$f" ] || { echo "downloading $f"; curl -fL --progress-bar -o "$HF/$f" "https://huggingface.co/$REPO/resolve/main/$f"; } ;;
  esac
done

SL=()
if [ "${SHORTLIST:-0}" = 1 ]; then
  if [ ! -s models/data/mg-wiki.txt ]; then
    curl -fL -o models/data/mg-wiki.parquet \
      "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.mg/train-00000-of-00001.parquet"
    python3 -c "import pyarrow.parquet as pq; t=pq.read_table('models/data/mg-wiki.parquet',columns=['text']).column('text').to_pylist(); open('models/data/mg-wiki.txt','w').write('\n'.join(x.replace('\n',' ') for x in t))"
  fi
  SL=(--shortlist mg=models/data/mg-wiki.txt)
fi

for p in $PRESETS; do
  "$BIN" convert --hf "$HF" --name "$REPO" --preset "$p" "${SL[@]}" --out "models/gguf/$NAME-$p.gguf"
done

# FLORES-200 (evaluation data, used by scripts/eval_flores.sh)
if [ ! -d models/data/flores200_dataset ]; then
  curl -fsSL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | tar xz -C models/data
fi
