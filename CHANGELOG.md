# Changelog

Évolutions notables de malaga. Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
versions : [SemVer](https://semver.org/lang/fr/).

## [0.1.0] - 2026-10-01

### Added
- `malaga convert` : checkpoints Hugging Face NLLB-200 / M2M100 (`.bin` PyTorch ou `.safetensors`)
  vers un GGUF unique (poids, tokenizer, configuration) ; presets `f32`, `f16`, `q8_0`, `q6_k`,
  `q5_k_m`, `q4_k_m` ; fusion QKV et pré-mise à l'échelle des requêtes à la conversion.
- Inférence NLLB sur candle (CPU, CUDA, Metal), greedy et beam search, traduction de documents
  avec découpage en phrases et mise en page conservée.
- Décodage GPU greedy : noyaux CUDA fusionnés, CUDA Graphs, KV-cache f16, argmax parallèle,
  repli automatique en cas de manque de VRAM.
- Mode optionnel `--fast-vocab` (shortlist de vocabulaire, `convert --shortlist mg=corpus.txt`).
- `malaga serve` : API native `/v1/translate`, OpenAI `/v1/chat/completions` (JSON et SSE),
  Ollama `/api/generate`, `/api/chat`, `/api/tags` ; batching dynamique ; contrat de lancement
  des backends zallama (`--model-id`, `--device`, `/health` après préchauffage).
- `malaga bench`, `scripts/eval_flores.sh`, `scripts/download.sh`, `tests/e2e.sh`.
- Tests d'intégration sur mini-modèle généré, dont la parité GPU optimisé / référence.

### Removed
- Preset `q4_0` : dégradation significative sur FLORES-200 (chrF++ 43,89 contre 44,53, p = 0,001).
