# Tests manuels

Résultats du 2026-10-01 (v0.1.0), RTX 4090, Ubuntu 24.04, CUDA 12.8.

## MT-01 — Parité avec Hugging Face transformers (REQ-QUA-001, REQ-INF-001)

1. `scripts/download.sh 600M f32`
2. Référence Python (`pip install torch transformers sentencepiece`) :
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   d = "models/hf/nllb-200-distilled-600M"
   tok = AutoTokenizer.from_pretrained(d, src_lang="fra_Latn"); m = AutoModelForSeq2SeqLM.from_pretrained(d)
   for s in ["Bonjour, comment allez-vous aujourd'hui ?", "Merci beaucoup pour votre aide."]:
       out = m.generate(**tok(s, return_tensors="pt"), forced_bos_token_id=tok.convert_tokens_to_ids("plt_Latn"))
       print(tok.batch_decode(out, skip_special_tokens=True)[0])
   ```
3. `malaga translate --device cpu -m models/gguf/nllb-200-distilled-600M-f32.gguf "<phrase>"`

Attendu : mêmes ids de tokens et même texte. Résultat : ✅ identiques sur les 3 phrases testées
(« Salama, manao ahoana ny fiainanao androany? », « Misaotra betsaka anao nanampy ahy. »,
« Nosy lehibe any amin'ny Ranomasimbe Indianina i Madagasikara, … »), CPU et CUDA.

## MT-02 — Qualité FLORES-200 (REQ-QUA-002, REQ-GPU-001, REQ-PRF-002)

```bash
scripts/eval_flores.sh fra_Latn plt_Latn models/gguf/nllb-200-distilled-600M-*.gguf
sacrebleu flores/plt_Latn.devtest -i f32.txt q8_0.txt q4_k_m.txt -m chrf --chrf-word-order 2 --paired-bs
malaga translate --reference --lines -m f32.gguf < flores/fra_Latn.devtest > ref.txt   # comparer à la sortie optimisée
```

Attendu : p > 0,05 pour chaque preset ; sortie optimisée f32 ≈ référence. Résultat : ✅ (tableau
dans [docs/performance.md](../docs/performance.md)) ; 1008/1012 phrases identiques optimisé vs
référence en f32 ; q4_0 rejeté (p = 0,001) et retiré ; shortlist non exacte (29 phrases perdent un
mot rare) → désactivée par défaut.

## MT-03 — GPU partagé, mémoire insuffisante (REQ-GPU-002)

Avec ~850 Mo de VRAM libres (un llama-server 27B chargé à côté) :
`MALAGA_LAZY_EMBEDDINGS=1 malaga translate --fast-vocab --lines -m q4_k_m.gguf < flores/fra_Latn.devtest`

Attendu : avertissements « out of GPU memory … » puis traduction complète. Résultat : ✅ 1012
phrases traduites (graphes abandonnés, batchs réduits), chrF++ 44,68.

## MT-04 — Latence (REQ-PRF-001)

`malaga bench -m models/gguf/nllb-200-distilled-600M-q4_k_m.gguf` sur GPU libre.
Attendu : < 20 ms par phrase, < 1 ms/token. Résultat : ✅ 16,4 ms, 0,78 ms/token.

## MT-05 — Contrat zallama (REQ-ZAL-001)

```bash
malaga serve --model q4_k_m.gguf --model-id malaga-fr-mg --host 127.0.0.1 --port 18931 --device cuda &
until curl -sf localhost:18931/health; do sleep 1; done
curl -s localhost:18931/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"malaga-fr-mg","messages":[{"role":"user","content":"Je t aime."}]}'
```

Attendu : `/health` ne répond qu'après « warm-up done » dans les logs ; réponse « Tiako ianao. ».
Résultat : ✅
