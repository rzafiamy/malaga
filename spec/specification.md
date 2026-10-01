# Spécification

## Objectif

Traduire du français et de l'anglais vers le malgache avec les modèles NLLB-200 de Meta, depuis un
fichier GGUF unique, avec la latence et l'empreinte VRAM les plus faibles possibles **sans
dégrader la qualité** par rapport au modèle d'origine, et servir ces traductions à zallama.

## Contexte et utilisateurs

Le malgache est peu servi par les outils de traduction auto-hébergeables. NLLB-200 le couvre
(`plt_Latn`), mais son exécution de référence (PyTorch) est lourde et llama.cpp / Ollama ne
supportent pas son architecture encodeur-décodeur. malaga comble ce manque pour zallama.

Utilisateurs :
- **Intégrateur zallama** : déclare malaga comme backend, attend un `/health` fiable, une API
  OpenAI et une empreinte VRAM connue pour cohabiter avec l'ASR et la vision.
- **Développeur d'application** : appelle `/v1/translate` (une phrase ou un lot) et attend une
  réponse en quelques dizaines de millisecondes.
- **Utilisateur en ligne de commande** : traduit un texte ou un fichier en conservant sa mise en
  page.
- **Mainteneur** : convertit de nouveaux checkpoints (1.3B, 3.3B) et vérifie leur qualité.

Parcours principal : `scripts/download.sh` → `malaga convert` → `malaga serve` (zallama) →
requêtes de traduction ; le texte est découpé en phrases, traduit en batch sur GPU puis réassemblé.

## Exigences fonctionnelles

| ID | Exigence |
|---|---|
| REQ-CNV-001 | Convertir un checkpoint Hugging Face NLLB (`.bin` ou `.safetensors`, éventuellement découpé) en GGUF, pour chaque preset `f32`, `f16`, `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m`. |
| REQ-CNV-002 | Le GGUF est autosuffisant : configuration et tokenizer embarqués ; aucun autre fichier n'est requis pour traduire. |
| REQ-CNV-003 | `--shortlist LANG=corpus` stocke une shortlist contenant les tokens vus au moins `--shortlist-min-count` fois, les tokens spéciaux et le code de langue. |
| REQ-INF-001 | Traduire sur CPU, CUDA ou Metal depuis le GGUF. |
| REQ-INF-002 | La traduction d'une phrase ne dépend pas des autres phrases de son batch (même résultat seule ou en batch, pour un même chemin de calcul en f32). |
| REQ-INF-003 | Beam search (`--beam N`) déterministe ; `N = 1` équivaut au greedy. |
| REQ-INF-004 | Un document est découpé en phrases ; sauts de ligne et indentation sont conservés. |
| REQ-LNG-001 | Langues désignées par alias (`fr`, `en`, `mg`, `français`, `malgache`, `fr-FR`…) ou code NLLB ; une langue inconnue est une erreur explicite. |
| REQ-SRV-001 | `GET /health` répond `{"status":"ok"}` quand le serveur est prêt. |
| REQ-SRV-002 | `POST /v1/translate` traduit un texte ou une liste de textes, langues par défaut configurables. |
| REQ-SRV-003 | `POST /v1/chat/completions` compatible OpenAI, JSON et SSE ; langues tirées du nom de modèle. |
| REQ-SRV-004 | `POST /api/generate`, `/api/chat` et `GET /api/tags` compatibles Ollama, corps JSON accepté sans `Content-Type`. |
| REQ-SRV-005 | `GET /v1/models` expose l'identifiant `--model-id`. |
| REQ-SRV-006 | Une requête invalide (langue inconnue, JSON invalide) renvoie 400 avec un message. |
| REQ-SRV-007 | Les requêtes concurrentes sont regroupées en batch et chacune reçoit sa propre traduction. |
| REQ-ZAL-001 | Contrat de backend zallama : `serve --model --model-id --host --port --device --device-id --threads`, `/health` prêt après préchauffage. |

## Exigences de qualité et de performance

| ID | Exigence |
|---|---|
| REQ-QUA-001 | En f32, sorties identiques à `transformers` (`M2M100ForConditionalGeneration.generate`, greedy). |
| REQ-QUA-002 | Chaque preset proposé a une qualité FLORES-200 (chrF++) non significativement différente de f32 (paired bootstrap, p > 0,05) ; sinon il est retiré. |
| REQ-GPU-001 | Le décodage GPU optimisé (noyaux fusionnés, CUDA Graphs, KV-cache f16) produit les mêmes tokens que l'implémentation générique. |
| REQ-GPU-002 | En cas de manque de VRAM, malaga continue (libération des graphes, batchs réduits) au lieu d'échouer, tant que le modèle tient en mémoire. |
| REQ-PRF-001 | NLLB-200 600M q4_k_m sur RTX 4090 : une phrase de ~20 tokens en moins de 20 ms, décodeur sous 1 ms/token. |
| REQ-PRF-002 | Optimisations non exactes uniquement en option explicite (`--fast-vocab`), désactivées par défaut. |

## Spécification technique

- Architecture et format GGUF : [docs/architecture.md](../docs/architecture.md).
- Mesures : [docs/performance.md](../docs/performance.md).
- Intégration : [docs/zallama.md](../docs/zallama.md).
- Contraintes : Rust 1.97.1, candle 0.11, CUDA ≥ 12 pour le GPU ; noyaux maison compilés en PTX
  (`compute_$CUDA_COMPUTE_CAP`, 70 par défaut) ; dimensions supportées par les noyaux fusionnés :
  `d_model ≤ 2048`, `head_dim` multiple de 8 (NLLB : 1024/2048 et 64).
