# Credits

malaga est distribué sous licence MIT ou Apache-2.0. Il repose sur les travaux ci-dessous, chacun
utilisé sous sa propre licence. Versions = `Cargo.lock`.

## Auteurs

- Rija Z. ([@rzafiamy](https://github.com/rzafiamy)) — conception, responsable du projet.
- Développé avec l'assistance de Claude (Anthropic) via Claude Code.

## Dépendances Rust

| Dépendance | Version | Rôle | Licence | Source |
|---|---|---|---|---|
| `candle-core` | 0.11.0 | Tenseurs, matmul quantifiés (CPU/CUDA/Metal), lecture/écriture GGUF, CUDA Graphs via cudarc | MIT OR Apache-2.0 | https://github.com/huggingface/candle |
| `candle-nn` | 0.11.0 | LayerNorm, softmax, KV-cache | MIT OR Apache-2.0 | https://github.com/huggingface/candle |
| `tokenizers` | 0.22.2 | Tokenizer SentencePiece/BPE de NLLB | Apache-2.0 | https://github.com/huggingface/tokenizers |
| `half` | 2.7.1 | Type f16 du KV-cache | MIT OR Apache-2.0 | https://github.com/starkat99/half-rs |
| `serde` | 1.0.229 | Sérialisation | MIT OR Apache-2.0 | https://serde.rs |
| `serde_json` | 1.0.151 | `config.json`, API JSON | MIT OR Apache-2.0 | https://github.com/serde-rs/json |
| `anyhow` | 1.0.104 | Erreurs | MIT OR Apache-2.0 | https://github.com/dtolnay/anyhow |
| `tracing` | 0.1.44 | Journalisation | MIT | https://github.com/tokio-rs/tracing |
| `tracing-subscriber` | 0.3.23 | Sortie des logs, `RUST_LOG` | MIT | https://github.com/tokio-rs/tracing |
| `clap` | 4.6.7 | Ligne de commande | MIT OR Apache-2.0 | https://github.com/clap-rs/clap |
| `axum` | 0.8.9 | Serveur HTTP | MIT | https://github.com/tokio-rs/axum |
| `tokio` | 1.53.1 | Runtime asynchrone du serveur | MIT | https://github.com/tokio-rs/tokio |
| `tempfile` | 3.27.0 | Répertoires temporaires des tests (dev) | MIT OR Apache-2.0 | https://github.com/Stebalien/tempfile |
| `tower` | 0.5.3 | Appel du routeur dans les tests (dev) | MIT | https://github.com/tower-rs/tower |

Aucune bibliothèque Tauri : malaga est une CLI / un serveur HTTP.

## Modèles et données

| Élément | Usage | Licence | Source |
|---|---|---|---|
| NLLB-200 (600M, 1.3B, 3.3B), Meta AI | Modèles de traduction convertis | CC-BY-NC 4.0 | https://huggingface.co/facebook/nllb-200-distilled-600M |
| FLORES-200 | Évaluation (devtest) | CC-BY-SA 4.0 | https://github.com/facebookresearch/flores |
| Wikipédia en malgache (dump 2023-11-01) | Corpus de la shortlist `--fast-vocab` | CC-BY-SA 4.0 | https://huggingface.co/datasets/wikimedia/wikipedia |

## Sources tierces

- Les noyaux de matmul quantifiés (MMVQ/MMQ, formats GGUF/k-quants) utilisés via candle viennent
  de [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).
- L'architecture M2M100 suit l'implémentation de référence de
  [transformers](https://github.com/huggingface/transformers) (Apache-2.0), utilisée pour valider
  les sorties ; aucun code n'en est copié.
- Les noyaux de `crates/malaga-core/kernels/malaga.cu` sont écrits pour malaga.
- Outils d'évaluation (non distribués) : [sacrebleu](https://github.com/mjpost/sacrebleu) (Apache-2.0).
