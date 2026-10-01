# malaga

**Traduction français / anglais → malgache**, en Rust, avec les modèles
[NLLB-200](https://ai.meta.com/research/no-language-left-behind/) de Meta convertis en **GGUF**,
optimisée pour la **latence** et la **VRAM**, sans sacrifier la qualité.

- Un seul fichier `.gguf` par modèle (poids quantifiés + tokenizer intégrés).
- Inférence GPU NVIDIA : noyaux CUDA fusionnés, décodage rejoué en **CUDA Graphs**, KV-cache f16,
  sélection du token sur le GPU — **16 ms** pour traduire une phrase (RTX 4090, q4_k_m).
- **Qualité identique à la référence** : sorties identiques à Hugging Face `transformers` en f32 ;
  chaque preset de quantification a été validé sur FLORES-200 (aucune différence significative).
- Serveur HTTP : API native, compatible **OpenAI** (`/v1/chat/completions`, SSE) et **Ollama**,
  prêt à être piloté par [zallama](docs/zallama.md).

```console
$ malaga translate -m models/gguf/nllb-200-distilled-600M-q4_k_m.gguf "Madagascar est une grande île située dans l'océan Indien."
Nosy lehibe any amin'ny Ranomasimbe Indianina i Madagasikara.
```

Sommaire : [Features](#features) · [Installation](#installation) · [Utilisation](#utilisation) ·
[Configuration](#configuration) · [API HTTP](#api-http) · [Performances et qualité](#performances-et-qualité) ·
[Tests](#tests) · [Limitations connues](#limitations-connues) · [Licence](#licence)

## Features

Identifiants `REQ-…` : [spec/specification.md](spec/specification.md), traçabilité vers le code et
les tests dans [spec/matrix.md](spec/matrix.md).

Cœur (bloquant pour l'usage) :

- **Conversion Hugging Face → GGUF** (`.bin` PyTorch ou `.safetensors`, presets f32 → q4_k_m),
  tokenizer embarqué, fusion QKV et mise à l'échelle des requêtes pré-calculées — REQ-CNV-001/002 —
  [`convert.rs`](crates/malaga-core/src/convert.rs)
- **Inférence NLLB / M2M100** encodeur-décodeur sur poids quantifiés (CPU, CUDA, Metal) — REQ-INF-001 —
  [`model.rs`](crates/malaga-core/src/model.rs)
- **Traduction de documents** : découpage en phrases, mise en page conservée, phrases traduites
  ensemble en batchs triés par longueur, résultat indépendant du batch — REQ-INF-002/004 —
  [`translator.rs`](crates/malaga-core/src/translator.rs), [`segment.rs`](crates/malaga-core/src/segment.rs)
- **Décodage GPU optimisé** et vérifié contre l'implémentation de référence — REQ-GPU-001 —
  [`graph.rs`](crates/malaga-core/src/graph.rs), [`kernels/malaga.cu`](crates/malaga-core/kernels/malaga.cu)
- **Serveur HTTP** (natif, OpenAI, Ollama) avec batching dynamique des requêtes concurrentes — REQ-SRV-001…007 —
  [`server.rs`](crates/malaga/src/server.rs)

Secondaires :

- **Beam search** (`--beam N`) — REQ-INF-003
- **Langues** : alias `fr`, `en`, `mg`, `français`, `malgache`… et tout code NLLB-200 (`deu_Latn`…) — REQ-LNG-001 —
  [`langs.rs`](crates/malaga-core/src/langs.rs)
- **Mode `--fast-vocab`** (opt-in, non exact) : shortlist de vocabulaire malgache — REQ-CNV-003
- **Repli automatique** sous pression mémoire GPU (batchs réduits, décodage sans graphes) — REQ-GPU-002
- **Benchmark intégré** (`malaga bench`) et évaluation FLORES-200 ([`scripts/eval_flores.sh`](scripts/eval_flores.sh))

Architecture détaillée : [docs/architecture.md](docs/architecture.md).

## Installation

### Prérequis

| | Version | Remarque |
|---|---|---|
| Rust | 1.97.1 (épinglé par [`rust-toolchain.toml`](rust-toolchain.toml)) | installé par `./prereq.sh` via rustup |
| Compilateur C/C++, `pkg-config`, `git`, `curl` | — | `build-essential` (Debian/Ubuntu) |
| CUDA Toolkit (`nvcc`) | ≥ 12.0 | GPU NVIDIA uniquement ; driver compatible |
| Python 3 + `sacrebleu` | — | optionnel, évaluation qualité |
| Tauri / webview | non requis | pas d'interface graphique : ni Tauri CLI ni webkit2gtk |

Plateformes : **Linux** (CPU et CUDA, testé : Ubuntu 24.04, RTX 4090, CUDA 12.8), **macOS** (CPU,
Metal), **Windows** (CPU ; CUDA via le CUDA Toolkit, non testé).

### Commandes

```bash
git clone https://github.com/rzafiamy/malaga && cd malaga
./prereq.sh               # vérifie/installe Rust et les paquets de build (CHECK_ONLY=1 pour vérifier seulement)
./setup.sh --cuda         # prérequis + dépendances + cargo check (sans --cuda : CPU)
./build.sh --cuda         # binaire release dans build/ (détecte la compute capability du GPU)
scripts/download.sh 600M  # télécharge NLLB-200 600M et produit models/gguf/*-q8_0.gguf et *-q4_k_m.gguf
```

`scripts/download.sh 1.3B q4_k_m` ou `3.3B` pour les modèles plus grands ; `SHORTLIST=1` ajoute la
shortlist malgache du mode `--fast-vocab` (télécharge Wikipédia en malgache, ~22 Mo).

### Vérifier l'installation

```bash
./build/malaga-linux-cuda-0.1.0 --version
./build/malaga-linux-cuda-0.1.0 translate -m models/gguf/nllb-200-distilled-600M-q4_k_m.gguf "Bonjour."
# → Miarahaba anao.
cargo test --workspace
```

## Utilisation

```bash
# Traduire (texte en argument ou sur stdin ; mise en page conservée)
malaga translate -m MODEL.gguf -f fr -t mg "Bonjour tout le monde."
malaga translate -m MODEL.gguf -f en < article.txt > lahatsoratra.txt
malaga translate -m MODEL.gguf --lines < phrases.txt      # une ligne = un segment, en batch

# Convertir un checkpoint Hugging Face
malaga convert --hf models/hf/nllb-200-distilled-600M --preset q4_k_m -o model.gguf
#   presets : f32 f16 q8_0 q6_k q5_k_m q4_k_m

# Mesurer
malaga bench -m MODEL.gguf

# Servir
malaga serve -m MODEL.gguf --host 0.0.0.0 --port 8080
```

Choix du modèle : **`q4_k_m`** est le meilleur compromis (le plus rapide, ~1,2 Go de VRAM au total,
qualité égale à f32). `q8_0` si la marge de VRAM est confortable. Les modèles 1.3B / 3.3B
améliorent la qualité au prix de la latence.

## Configuration

**Emplacement** : il n'y a pas de fichier de configuration. Tout se règle par options de ligne de
commande (`malaga <commande> --help`) et, pour le serveur, par variables d'environnement
(modèle : [`malaga.example.env`](malaga.example.env), à copier en `.env`). Pour modifier un
réglage, changer l'option ou la variable puis relancer le processus ; l'option de ligne de
commande a priorité sur la variable. Rien n'est écrit en dehors des fichiers demandés.

| Option / variable | Rôle | Défaut |
|---|---|---|
| `--model`, `MALAGA_MODEL` | Fichier GGUF | — |
| `--device`, `MALAGA_DEVICE` | `auto` (CUDA, Metal, sinon CPU), `cpu`, `cuda`, `metal` | `auto` |
| `--device-id` | GPU à utiliser | `0` |
| `--threads` | Threads CPU (fixe `RAYON_NUM_THREADS`) | tous les cœurs |
| `--host`, `MALAGA_HOST` / `--port`, `MALAGA_PORT` | Adresse d'écoute du serveur | `127.0.0.1:8080` |
| `--model-id` | Nom du modèle exposé par l'API | `general.name` du GGUF |
| `--default-source` / `--default-target` | Langues quand la requête n'en précise pas | `fr` / `mg` |
| `--beam` | Taille du beam (1 = greedy, le plus rapide) | `1` |
| `--max-batch` | Phrases décodées ensemble | `32` |
| `--max-new-tokens` | Longueur maximale de sortie | `2 × source + 16`, ≤ 200 |
| `--fast-vocab` | Shortlist de vocabulaire (plus rapide, **non exact**) | désactivé |
| `--reference` | Implémentation générique non fusionnée (vérification) | désactivé |
| `MALAGA_GRAPH_CACHE_MB` | Budget VRAM des graphes CUDA et de leurs buffers | `512` |
| `MALAGA_NO_CUDA_GRAPH=1` | Décodage fusionné sans capture de graphes | graphes activés |
| `MALAGA_LAZY_EMBEDDINGS=1` | Ne charge la table d'embeddings complète sur GPU qu'au besoin (avec `--fast-vocab`) | chargée |
| `RUST_LOG` | Niveau de log (`info`, `debug`…) | `info` |
| `CUDA_COMPUTE_CAP`, `NVCC` | Build CUDA : architecture cible, chemin de `nvcc` | `70`, `nvcc` |

Exemple de lancement serveur pour un GPU partagé :

```bash
MALAGA_GRAPH_CACHE_MB=256 malaga serve -m models/gguf/nllb-200-distilled-600M-q4_k_m.gguf \
  --device cuda --model-id malaga --port 8104
curl -s localhost:8104/health   # {"status":"ok"} : modèle chargé et préchauffé
```

**Vérifier la configuration** : `malaga serve` journalise au démarrage le modèle, le
périphérique retenu (`loaded … on Cuda(…)`) et la durée du préchauffage, et refuse de démarrer
si le modèle ou le périphérique est invalide ; `curl localhost:PORT/health` puis
`curl localhost:PORT/v1/models` confirment le modèle servi. `malaga bench -m MODEL.gguf` vérifie
GPU, latence et VRAM.

⚠ Sécurité : le serveur n'a **pas d'authentification** et écoute sur `127.0.0.1` par défaut.
Pour l'exposer, le placer derrière un proxy authentifiant (zallama, nginx…) ; ne mettez ni jeton
ni secret dans la ligne de commande. Les textes traduits ne sont ni journalisés ni stockés.

## API HTTP

| Méthode | Chemin | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok"}` une fois le modèle chargé et préchauffé |
| `POST` | `/v1/translate` | API native : `{"text": "…" \| "texts": [...], "source": "fr", "target": "mg", "beam_size": 1}` |
| `POST` | `/v1/chat/completions` | OpenAI (JSON ou SSE `stream: true`) : traduit le dernier message `user` |
| `GET` | `/v1/models` | Modèle servi |
| `POST` | `/api/generate`, `/api/chat` | Ollama (NDJSON) |
| `GET` | `/api/tags` | Ollama |

Pour les API OpenAI / Ollama, les langues se donnent par le nom de modèle (`"model": "malaga:fr-mg"`,
`"nllb-en-mg"`) ou par les champs `source` / `target` du corps.

```bash
curl -s localhost:8080/v1/translate -H 'content-type: application/json' \
  -d '{"texts": ["Good morning.", "Where is the market?"], "source": "en"}'
# {"translation":["Tsara ny maraina.","Aiza ny tsena?"],"source":"en","target":"mg",...}
```

Les requêtes concurrentes sont fusionnées en un seul batch GPU (batching dynamique) sans ajouter de
latence à une requête isolée. Intégration zallama : [docs/zallama.md](docs/zallama.md).

## Performances et qualité

NLLB-200 distilled 600M, RTX 4090, une phrase de 21 tokens → 18 tokens (`malaga bench`).
Détails, méthode et historique des optimisations : [docs/performance.md](docs/performance.md).

| Configuration | Latence / phrase | Décodeur | Batch de 16 | VRAM totale* |
|---|---|---|---|---|
| Portage candle direct (point de départ, q8_0) | 31,8 ms | 1,59 ms/token | 198 phrases/s | — |
| **malaga q4_k_m (défaut recommandé)** | **16,4 ms** | **0,78 ms/token** | **363 phrases/s** | **1,2 Go** |
| malaga q8_0 | 19,1 ms | 0,92 ms/token | 318 phrases/s | 1,4 Go |
| malaga q4_k_m `--fast-vocab` (non exact) | 12,5 ms | 0,58 ms/token | 433 phrases/s | 0,97 Go** |
| CPU 32 cœurs, q4_k_m | 835 ms | 39 ms/token | 2 phrases/s | — |

\* contexte CUDA inclus (~430 Mo, incompressible). \*\* avec `MALAGA_LAZY_EMBEDDINGS=1`.

Qualité FLORES-200 devtest (1012 phrases, chrF++ / BLEU, sacrebleu) — aucune différence
statistiquement significative avec f32 (paired bootstrap, p > 0,05) :

| Modèle | fr → mg chrF++ | fr → mg BLEU | en → mg chrF++ | en → mg BLEU |
|---|---|---|---|---|
| f32 (= Hugging Face `transformers`) | 44,53 | 13,15 | 46,74 | 15,34 |
| q8_0 | 44,45 | 13,09 | 46,68 | 15,24 |
| q4_k_m | 44,72 | 13,53 | 46,77 | 15,46 |

`q4_0` a été retiré : il dégrade significativement (chrF++ 43,89, p = 0,001).

## Tests

```bash
cargo test --workspace                         # CPU : unitaires + intégration (mini-modèle généré, sans téléchargement)
cargo test --workspace --features malaga/cuda  # + parité GPU optimisé == référence (GPU NVIDIA requis)
cargo clippy --workspace --all-targets -- -D warnings
tests/e2e.sh                                   # bout en bout sur le vrai modèle (après scripts/download.sh)
scripts/eval_flores.sh fra_Latn plt_Latn models/gguf/*.gguf   # qualité FLORES-200
```

Les tests d'intégration génèrent un NLLB miniature aléatoire (même architecture, quelques centaines
de Ko) et vérifient conversion → chargement → traduction, l'indépendance au batch, le beam search,
la shortlist, l'API HTTP et — sur GPU — que le chemin optimisé choisit exactement les mêmes tokens
que l'implémentation de référence. Tests manuels (parité Hugging Face, FLORES, mémoire) :
[spec/manual-tests.md](spec/manual-tests.md).

## Limitations connues

- **Licence des poids** : NLLB-200 est sous **CC-BY-NC 4.0** (usage non commercial). Le code de
  malaga est libre (MIT / Apache-2.0), les modèles convertis héritent de la licence de Meta.
- `llama.cpp` / Ollama ne savent pas exécuter l'architecture NLLB (encodeur-décodeur M2M100) : les
  `.gguf` produits ici se servent avec `malaga serve`.
- Le chemin CPU n'utilise pas les noyaux fusionnés (≈ 40 ms/token sur 32 cœurs).
- `--fast-vocab` n'est **pas exact** : ~3 % des phrases FLORES fr→mg changent (mots rares absents
  de la shortlist : `vaksin`, `galaksi`, `GDP`…), d'où sa désactivation par défaut.
- En q4/q5/q8, les noyaux quantifiés rendent la sortie sensible à la taille du batch (quasi
  ex-aequo) ; la qualité mesurée est identique. En f32 le résultat est stable.
- NLLB traduit phrase par phrase, sans contexte entre phrases ; le malgache produit reste celui de
  NLLB-200 (BLEU ~13-15 sur FLORES).

## Licence

Code sous double licence [MIT](LICENSE-MIT) ou [Apache-2.0](LICENSE-APACHE), au choix.
Modèles NLLB-200 : CC-BY-NC 4.0 (Meta). Dépendances et sources tierces : [CREDITS.md](CREDITS.md).
Historique : [CHANGELOG.md](CHANGELOG.md). Présentation : [portfolio/](portfolio/README.md).
