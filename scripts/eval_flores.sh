#!/usr/bin/env bash
# Evaluates GGUF models on FLORES-200 devtest (BLEU / chrF++ with sacrebleu).
# usage: scripts/eval_flores.sh SRC TGT model1.gguf [model2.gguf ...]
#   SRC/TGT are NLLB codes, e.g. fra_Latn plt_Latn.
# Needs: ./target/release/malaga, sacrebleu (pip install sacrebleu) and the
# FLORES-200 data in models/data (see scripts/download.sh).
set -euo pipefail
SRC=$1 TGT=$2; shift 2
[ $# -ge 3 ] || { echo "usage : $0 SRC TGT model.gguf [...]  (ex. fra_Latn plt_Latn)" >&2; exit 1; }
DATA=${FLORES_DIR:-models/data/flores200_dataset/devtest}
BIN=${MALAGA_BIN:-./target/release/malaga}
[ -x "$BIN" ] || { echo "binaire introuvable : $BIN (./build.sh ou MALAGA_BIN=...)" >&2; exit 1; }
command -v sacrebleu >/dev/null || { echo "sacrebleu absent : pip install sacrebleu" >&2; exit 1; }
for f in "$DATA/$SRC.devtest" "$DATA/$TGT.devtest"; do
  [ -f "$f" ] || { echo "données FLORES absentes : $f (scripts/download.sh)" >&2; exit 1; }
done
mkdir -p models/eval
printf "%-45s %8s %8s %8s\n" model BLEU chrF++ seconds
for m in "$@"; do
  out=models/eval/$(basename "$m" .gguf).$SRC-$TGT.txt
  start=$(date +%s.%N)
  "$BIN" translate --lines -m "$m" -f "$SRC" -t "$TGT" < "$DATA/$SRC.devtest" > "$out" 2>"$out.log" \
    || { echo "échec de la traduction avec $m (voir $out.log)" >&2; exit 1; }
  secs=$(echo "$(date +%s.%N) - $start" | bc)
  scores=$(sacrebleu "$DATA/$TGT.devtest" -i "$out" -m bleu chrf --chrf-word-order 2 -b -w 2 | tr -d '[] \n')
  printf "%-45s %8s %8s %8.1f\n" "$(basename "$m")" "${scores%,*}" "${scores#*,}" "$secs"
done
