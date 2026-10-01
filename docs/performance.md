# Performances et qualité

Mesures du 2026-10-01, RTX 4090 (24 Go), CUDA 12.8, driver 570, Rust 1.97.1,
NLLB-200 distilled 600M. Latence : `malaga bench` (phrase de 93 caractères, 21 tokens source →
18 tokens produits, médiane de 20 essais après préchauffage). Qualité : FLORES-200 devtest
(1012 phrases), sacrebleu 2.6, chrF++ (`--chrf-word-order 2`) et BLEU.

## Règle : la qualité d'abord

Une optimisation n'est retenue que si elle ne change pas la qualité :

- en f32, la sortie est **identique** à Hugging Face `transformers` (test manuel MT-01) ;
- le chemin GPU optimisé est comparé à l'implémentation générique : 1008/1012 phrases FLORES
  strictement identiques en f32 ; les 4 autres sont des quasi-ex-aequo de qualité équivalente
  (ex. « North Carolina » / « Carolina Avaratra ») ; test automatique sur mini-modèle ;
- chaque preset est comparé à f32 par paired bootstrap (1000 tirages) : retenu si p > 0,05.

## Historique des optimisations (q8_0, une phrase)

| Étape | Latence | Décodeur | Effet |
|---|---|---|---|
| Portage candle direct | 31,8 ms | 1,59 ms/token | point de départ : ~500 lancements de noyaux par token |
| CUDA Graphs sur pas statique | 26,6 ms | 1,32 ms/token | le CPU ne lance plus chaque noyau |
| Noyaux fusionnés + argmax parallèle | 20,6 ms | 1,00 ms/token | l'argmax candle sur 256k logits coûtait 140 µs (1 seul bloc) |
| LayerNorm en registres, attention 1 thread/clé, KV f16 | 19,1 ms | 0,92 ms/token | |
| Preset q4_k_m | 16,4 ms | 0,78 ms/token | moitié moins d'octets lus par token |

Optimisations évaluées puis **écartées ou rendues optionnelles** pour la qualité :

- **Shortlist de vocabulaire** (37k tokens malgaches au lieu de 256k) : −25 % de latence, mais
  29 phrases FLORES sur 1012 changent parce que le modèle voulait un token absent de la liste
  (`▁vaksin`, `▁galaksi`, `▁GDP`, `▁30.000`…). Score global inchangé, mais ce sont de vrais mots
  perdus → option `--fast-vocab`, désactivée par défaut.
- **q4_0** : chrF++ 43,89 contre 44,53 en f32, différence significative (p = 0,001) → retiré.

## Qualité par preset (fr → mg)

| Preset | Taille | BLEU | chrF++ | p vs f32 |
|---|---|---|---|---|
| f32 | 2,4 Go | 13,15 | 44,53 | — |
| f16 | 1,3 Go | 13,16 | 44,53 | 0,34 |
| q8_0 | 0,67 Go | 13,09 | 44,45 | 0,10 |
| q6_k | 0,52 Go | 13,29 | 44,55 | 0,32 |
| q5_k_m | 0,49 Go | 13,11 | 44,41 | 0,14 |
| q4_k_m | 0,46 Go | 13,53 | 44,72 | 0,09 |
| ~~q4_0~~ | 0,43 Go | 12,86 | 43,89 | **0,001** |

en → mg : f32 15,34 / 46,74 ; q8_0 15,24 / 46,68 ; q4_k_m 15,46 / 46,77.
Tailles hors shortlist optionnelle.

Les presets quantifiés passent les activations en q8_1 dans les matmul (MMVQ pour ≤ 8 lignes, MMQ
au-delà) : la sortie dépend légèrement de la taille du batch (≈ 1/3 des phrases changent entre
batch 1 et batch 32, y compris avec l'implémentation de référence), sans effet mesurable sur la
qualité. Pour des sorties bit à bit reproductibles, utiliser f32 ou f16.

## Débit et mémoire

| Configuration | Batch 16 | VRAM (contexte CUDA ~430 Mo inclus) |
|---|---|---|
| q4_k_m | 363 phrases/s | 1,2 Go |
| q8_0 | 318 phrases/s | 1,4 Go |
| q4_k_m `--fast-vocab`, `MALAGA_LAZY_EMBEDDINGS=1` | 433 phrases/s | 0,97 Go |

FLORES complet (1012 phrases) : ~4,9 s en q4_k_m. Chaque graphe CUDA capturé réserve ~32 Mo ;
le cache est borné par `MALAGA_GRAPH_CACHE_MB` (512 par défaut).

## Reproduire

```bash
malaga bench -m models/gguf/nllb-200-distilled-600M-q4_k_m.gguf
scripts/eval_flores.sh fra_Latn plt_Latn models/gguf/*.gguf
malaga translate --reference --lines -m MODEL.gguf < flores/fra_Latn.devtest > ref.txt   # chemin générique
```
