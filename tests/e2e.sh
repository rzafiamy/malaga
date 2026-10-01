#!/usr/bin/env bash
# tests/e2e.sh — bout en bout sur le vrai modèle NLLB-200 600M.
# Prérequis : ./build.sh [--cuda] et scripts/download.sh 600M q4_k_m.
#   tests/e2e.sh [binaire] [modèle.gguf]
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${1:-$(ls -t build/malaga-* 2>/dev/null | head -1 || echo target/release/malaga)}
MODEL=${2:-models/gguf/nllb-200-distilled-600M-q4_k_m.gguf}
[ -x "$BIN" ] || { echo "binaire introuvable : $BIN (lancer ./build.sh)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "modèle introuvable : $MODEL (lancer scripts/download.sh 600M q4_k_m)" >&2; exit 1; }
fail() { echo "ÉCHEC : $*" >&2; exit 1; }

out=$("$BIN" translate -m "$MODEL" "Merci beaucoup pour votre aide." 2>/dev/null)
echo "fr→mg : $out"
[[ "$out" == *Misaotra* ]] || fail "traduction inattendue : $out"

out=$(printf 'Good morning.\n\nWhere is the market?' | "$BIN" translate -m "$MODEL" -f en 2>/dev/null)
[ "$(echo "$out" | wc -l)" -eq 3 ] || fail "mise en page non conservée"

PORT=${PORT:-18977}
"$BIN" serve -m "$MODEL" --port "$PORT" --model-id malaga-e2e >/dev/null 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/health" >/dev/null || fail "le serveur ne répond pas"
r=$(curl -sf "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"malaga:fr-mg","messages":[{"role":"user","content":"Je t aime."}]}')
[[ "$r" == *Tiako* ]] || fail "réponse chat inattendue : $r"
echo "OK"
