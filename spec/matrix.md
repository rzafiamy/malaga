# Matrice de traçabilité

Exigence ([specification.md](specification.md)) → feature → module → test. Les tests automatisés
portent l'identifiant dans un commentaire `/// covers: REQ-…` ; les tests manuels sont décrits dans
[manual-tests.md](manual-tests.md).

| ID | Requirement | Feature | Module | Test | Statut | Notes |
|---|---|---|---|---|---|---|
| REQ-CNV-001 | Conversion HF → GGUF, 6 presets | Conversion | `crates/malaga-core/src/convert.rs` | `convert_load_and_translate_every_preset` | ✅ | |
| REQ-CNV-002 | GGUF autosuffisant | Conversion | `crates/malaga-core/src/convert.rs`, `translator.rs` | `convert_load_and_translate_every_preset` | ✅ | chargé sans autre fichier |
| REQ-CNV-003 | Shortlist | `--fast-vocab` | `crates/malaga-core/src/convert.rs` | `shortlist_keeps_corpus_tokens_and_the_language_token` | ✅ | |
| REQ-INF-001 | Traduction CPU/CUDA/Metal | Inférence | `crates/malaga-core/src/model.rs` | `convert_load_and_translate_every_preset`, manuel MT-01 | ✅ | Metal non testé |
| REQ-INF-002 | Indépendance au batch | Traduction | `crates/malaga-core/src/translator.rs` | `batching_does_not_change_translations` | ✅ | |
| REQ-INF-003 | Beam search | Traduction | `crates/malaga-core/src/translator.rs` | `beam_search_is_deterministic_and_beam1_is_greedy` | ✅ | |
| REQ-INF-004 | Mise en page conservée | Segmentation | `crates/malaga-core/src/segment.rs` | `document_layout_is_preserved`, `layout_is_preserved`, `sentences` | ✅ | |
| REQ-LNG-001 | Alias de langues | Langues | `crates/malaga-core/src/langs.rs` | `aliases`, `unknown_language_is_an_error` | ✅ | |
| REQ-SRV-001 | `/health` | Serveur | `crates/malaga/src/server.rs` | `every_api_answers` | ✅ | |
| REQ-SRV-002 | `/v1/translate` | Serveur | `crates/malaga/src/server.rs` | `every_api_answers` | ✅ | |
| REQ-SRV-003 | OpenAI chat, SSE | Serveur | `crates/malaga/src/server.rs` | `every_api_answers`, `model_name_langs` | ✅ | |
| REQ-SRV-004 | API Ollama | Serveur | `crates/malaga/src/server.rs` | `every_api_answers` | ✅ | |
| REQ-SRV-005 | `/v1/models` | Serveur | `crates/malaga/src/server.rs` | `every_api_answers` | ✅ | |
| REQ-SRV-006 | Erreurs 400 | Serveur | `crates/malaga/src/server.rs` | `bad_requests_are_400_with_a_message` | ✅ | |
| REQ-SRV-007 | Batching dynamique | Serveur | `crates/malaga/src/server.rs` | `concurrent_requests_get_their_own_translation` | ✅ | |
| REQ-ZAL-001 | Contrat backend zallama | Serveur | `crates/malaga/src/main.rs` | manuel MT-05 | ✅ | |
| REQ-QUA-001 | Parité `transformers` | Inférence | `crates/malaga-core/src/model.rs` | manuel MT-01 | ✅ | 3 phrases identiques, token par token |
| REQ-QUA-002 | Qualité des presets | Conversion | `crates/malaga-core/src/convert.rs` | manuel MT-02 | ✅ | q4_0 retiré |
| REQ-GPU-001 | GPU optimisé == référence | Décodage GPU | `crates/malaga-core/src/graph.rs`, `kernels.rs` | `fused_gpu_decoding_matches_the_reference_path`, manuel MT-02 | ✅ | test GPU (`--features malaga/cuda`) |
| REQ-GPU-002 | Repli mémoire | Décodage GPU | `crates/malaga-core/src/graph.rs`, `translator.rs` | manuel MT-03 | ✅ | |
| REQ-PRF-001 | Latence | Décodage GPU | `crates/malaga-core/src/graph.rs` | manuel MT-04 | ✅ | 16,4 ms, 0,78 ms/token |
| REQ-PRF-002 | Pas d'approximation par défaut | Configuration | `crates/malaga-core/src/translator.rs` | manuel MT-02 | ✅ | `GenOptions::default().shortlist == false` |

Mise à jour : 2026-10-01 (v0.1.0).
