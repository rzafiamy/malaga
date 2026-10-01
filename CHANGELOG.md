# Changelog

Évolutions notables de malaga. Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
versions : [SemVer](https://semver.org/lang/fr/).

## [0.2.0] - 2026-10-01

### Added
- **Synthèse vocale malgache** (Meta MMS-TTS / VITS, `facebook/mms-tts-mlg`, licence CC-BY-NC-4.0) :
  `malaga convert` reconnaît les checkpoints VITS et écrit un GGUF `vits` autonome (configuration et
  alphabet inclus). Les tenseurs d'entraînement sont retirés (36,3 M → 28,3 M paramètres) ; weight
  norm, QKV, facteurs d'échelle et moyenne des resblocks HiFi-GAN sont repliés dans les poids ;
  f16 par défaut (145 → 54 Mo).
- `malaga speak` (WAV, `--print-text`, `--iters` pour mesurer) ; `malaga serve` sur un GGUF `vits`
  expose `POST /v1/audio/speech` (forme OpenAI, WAV ou PCM) et `POST /v1/audio/normalize`.
- Normalisation du texte malgache : nombres (unités d'abord, `amby` / `sy`), décimales, ordinaux,
  heures, dates, téléphones, unités et devises, symboles, URL et e-mails, sigles, mots étrangers
  (lexique intégré, `--lexicon` pour l'enrichir, règles de réécriture français / anglais) ;
  parenthèses, deux-points et virgules donnent une pause courte.
- Plancher de durée des voyelles (45 ms, 65 ms en fin de mot) : les voyelles avalées par MMS
  (≤ 32 ms) passent de 32 % à 0 %.
- HiFi-GAN décodé par fenêtres avec contexte : sortie identique, VRAM bornée quelle que soit la
  longueur (phrase de 41 s : 3,1 → 1,0 Go en f32, 760 Mo en f16).

### Performances
- RTX 4090, paragraphe de 21 s : 103 ms en f16 (≈ 200× le temps réel), 155 ms en f32 ;
  sortie f32 identique à `transformers` échantillon par échantillon (CPU et CUDA). Détails dans
  `docs/tts.md`.

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
