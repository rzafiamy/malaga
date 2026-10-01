# Synthèse vocale (MMS-TTS / VITS)

malaga sert aussi la voix malgache de Meta, [`facebook/mms-tts-mlg`](https://huggingface.co/facebook/mms-tts-mlg)
(VITS, 36 M paramètres, 16 kHz, une seule voix). **Licence CC-BY-NC-4.0 : usage non commercial.**

```bash
# HF -> GGUF (f16 par défaut ; --preset f32 pour la référence exacte)
malaga convert --hf models/hf/mms-tts-mlg -o models/gguf/mms-tts-mlg-f16.gguf
malaga speak  -m models/gguf/mms-tts-mlg-f16.gguf -o salama.wav "Manao ahoana ianao?"
malaga serve  -m models/gguf/mms-tts-mlg-f16.gguf --port 8080   # POST /v1/audio/speech
```

`convert` reconnaît VITS au `model_type` de `config.json` ; `serve` choisit le serveur TTS d'après
`general.architecture = "vits"` du GGUF.

## API

`POST /v1/audio/speech` (forme OpenAI) : `{"input": "...", "speed": 1.0, "response_format": "wav"|"pcm"}`
→ `audio/wav` 16 bits mono. `voice` est accepté et ignoré. Réglages en plus : `noise_scale`,
`noise_scale_duration`, `seed` (audio reproductible, identique CPU/GPU).

## Normalisation du texte malgache (`mg_norm.rs`)

MMS ne lit que 30 caractères (lettres malgaches et espace) : tout le reste était supprimé en
silence. Avant la synthèse, le texte est donc réécrit tel qu'on le prononce, en orthographe
malgache. `malaga speak --print-text` affiche le résultat ; `POST /v1/audio/normalize` aussi.

| Écrit | Lu |
|---|---|
| `2024` | efatra amby roapolo sy roa arivo (unités d'abord, `amby` puis `sy`, `iraika` devant `amby`) |
| `12,5%` | roa ambin'ny folo faingo dimy isan-jato |
| `3 500 000 Ar` | dimy hetsy amby telo tapitrisa ariary |
| `28°C`, `45 km/h` | valo amby roapolo degre selsiosy, dimy amby efapolo kilometatra isan'ora |
| `14h30`, `26/06/1960` | efatra ambin'ny folo ora sy telopolo minitra ; enina amby roapolo jona … |
| `faha-3`, `21e` | fahatelo, fahiraika amby roapolo |
| `034 12 345 67` | groupe par groupe, avec une pause entre chaque |
| `SMS`, `4G`, `S24` | esy ema esy ; efatra je ; esy efatra amby roapolo |
| `info@malaga.mg` | infô arobasy malaga teboka ema je |
| `iPhone`, `Android`, `WiFi` | aifaona, andrôida, oaifay (lexique) |
| `Paris`, `Macron`, `santé` | parisy, makrôna, sante (règles) |
| `( ) : ; —` | pause courte (120 ms) |

Mots étrangers : d'abord le lexique (une centaine d'entrées, `--lexicon fichier.tsv` pour en ajouter
ou corriger, une ligne `mot<TAB>prononciation`), puis des règles de réécriture pour tout mot qui ne
peut pas être malgache (lettres c q u w x, consonne finale hors élision `amin'ny` / `isan-jato`,
groupe de consonnes absent du malgache). Les règles visent le français et l'anglais (`ou` → o,
`o` → ô, `ch` → s, `c` → s/k, finale consonantique + y/a…) : approximatives par nature, le lexique
prime. Les noms étrangers à l'orthographe d'apparence malgache (Tokyo, Toronto) ne sont détectables
que par le lexique. Les sigles de 2–3 lettres ou sans voyelle sont épelés ; JIRAMA est lu comme un mot.

## Fins de mots « hachées »

Le signal ne s'arrête pas brutalement (énergie des 20 dernières ms ≈ 1 % du corps) : ce sont les
voyelles qui sont avalées. Même sans bruit, MMS donne ≤ 32 ms à 40 % des voyelles (lecture rapide
de la Bible), et surtout aux voyelles atones finales ; le bruit de durée n'y change presque rien.
Correction : plancher de durée par voyelle, 45 ms (`--vowel-floor-ms`) et 65 ms en fin de mot
(`--final-vowel-floor-ms`). Voyelles ≤ 32 ms : 32 % → 0 %, audio +15 % plus long, vitesse inchangée.
`0` rend les durées du modèle telles quelles.

## Ce que fait la conversion (`vits_convert.rs`)

| Optimisation | Effet |
|---|---|
| Encodeur postérieur, `post_flows` et flow 1 du prédicteur de durée supprimés (entraînement seulement) | 36,3 M → 28,3 M paramètres |
| Weight norm (`weight_g`/`weight_v`) fusionnée | 64 tenseurs en moins, aucun calcul à l'exécution |
| Q, K, V fusionnés, Q pré-multiplié par `1/√head_dim` ; embedding × `√hidden` | 1 matmul au lieu de 3 |
| Logits largeur/hauteur des splines pré-divisés par `√192` | |
| Moyenne `/3` des resblocks HiFi-GAN repliée dans le conv suivant (leaky ReLU est homogène) | 4 passes mémoire en moins |
| Stockage f16 | 145 Mo → 54 Mo |

## Inférence (`vits.rs`)

- Une passe VITS par phrase (`segment.rs`) + 250 ms de silence : le modèle a été entraîné sans
  ponctuation, la normalisation MMS la supprime et les pauses disparaissaient.
- Normalisation : NFC (un `ô` décomposé n'est plus perdu), minuscules, lettre de base pour les
  accents hors vocabulaire (`é` → `e`), le reste (chiffres, ponctuation, `c`, `q`, `u`, `w`, `x`) est
  supprimé.
- Blancs intercalés comme le MMS d'origine (`intersperse(ids, 0)`, 2n+1 jetons). `transformers` en
  perd un quand le texte commence/finit par `a` ou contient `aa` (le blanc *est* le jeton `a`, déclaré
  spécial, donc le texte est découpé dessus) : malaga suit l'original.
- Spline rationnelle-quadratique et flow affine sur l'hôte (quelques centaines de scalaires).
- Convolution depthwise du prédicteur de durée en `k` multiplications décalées : candle exécute
  `groups = c` comme `c` convolutions séparées.
- `conv_transpose1d` lancé avec `padding = 0` puis rogné : seul ce cas prend le chemin col2im rapide
  de candle.
- HiFi-GAN par fenêtres de 384 trames avec 24 trames de contexte de chaque côté : sortie identique
  (champ réceptif mesuré : 12 trames suffisent, écart ≤ 1 LSB en 16 bits), VRAM bornée quelle que soit
  la longueur de la phrase.

## Mesures (RTX 4090, paragraphe de 5 phrases, 21 s d'audio)

| | meilleur temps | × temps réel | pic VRAM (phrase de 41 s) |
|---|---|---|---|
| PyTorch CPU f32 (16 threads, référence) | 2 834 ms | 6× | — |
| malaga CPU f32 | 13 600 ms | 1,6× | — |
| malaga CUDA f32 | 155 ms | 138× | 3,1 Go → **1,0 Go** (fenêtres) |
| malaga CUDA f16 (`--dtype auto`) | **103 ms** | **203×** | 1,4 Go → **760 Mo** |

HiFi-GAN représente ~85 % du temps GPU. Requête HTTP isolée (2 phrases, 4,2 s d'audio) : 44 ms.

Exactitude (bruits à 0, comparaison échantillon par échantillon avec `transformers`) : CPU et CUDA
f32 identiques (corrélation 1,00000, écart max 0,0000) ; GGUF f16 0,99915 ; calcul f16 0,99972.
Débogage : `MALAGA_VITS_DUMP=<dir>` écrit les tenseurs intermédiaires, `MALAGA_VITS_PROFILE=1` le
temps par étape.

## Pistes

- **Lexique** : à enrichir à l'usage (noms propres, marques) ; les règles ne remplacent pas une
  vraie phonétisation G2P du français et de l'anglais.
- **CPU** : 4× plus lent que PyTorch, car le conv1d CPU de candle (im2col + gemm) est faible face à
  oneDNN. Il faudrait un conv direct ou la feature `mkl` pour un hébergement sans GPU.
- **GPU** : cuDNN (feature `cudnn`, non installé ici) ou un noyau fusionné leaky-ReLU + conv dilaté
  pour les étages 32/64 canaux de HiFi-GAN.
- **Streaming** : le découpage en fenêtres de HiFi-GAN permet déjà d'émettre l'audio par morceaux.
