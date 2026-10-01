# Héberger malaga dans zallama

`malaga serve` suit le contrat de lancement des backends zallama (comme `parakeet-rs-server`) :

```
malaga serve --model <fichier.gguf> --model-id <nom> --host 127.0.0.1 --port <port> \
             [--device auto|cpu|cuda] [--device-id N] [--threads N] [--beam N] [--max-batch N]
```

- `GET /health` répond `200` seulement une fois le modèle chargé **et préchauffé** (compilation
  JIT des noyaux CUDA, première capture de graphe) : le délai de démarrage est absorbé par le
  health-check de zallama, pas par la première requête.
- `POST /v1/chat/completions` (JSON ou SSE) : zallama le relaie déjà pour la modalité `text`.
- Le nom du modèle choisit la paire de langues : `malaga-fr-mg`, `malaga-en-mg`, `malaga:mg-fr`…
  (ou champs `source` / `target` dans le corps).

## 1. Installer le binaire

```bash
./build.sh --cuda
cp build/malaga-linux-cuda-0.1.0 /opt/zallama/bin/malaga
scripts/download.sh 600M q4_k_m
cp models/gguf/nllb-200-distilled-600M-q4_k_m.gguf /bank2/zallama/models/
```

## 2. Déclarer le backend (`server/backends.py`)

```python
class MalagaServerBackend:
    """malaga serve — traduction FR/EN -> malgache (NLLB-200 en GGUF).

    Contrat : --model --model-id --host --port, GET /health (prêt après préchauffage),
    POST /v1/chat/completions (traduit le dernier message user ; JSON ou SSE),
    POST /v1/translate (API native, batch). La paire de langues vient du nom de modèle
    (`malaga-fr-mg`) ou des champs `source`/`target` du corps.
    """
    name = "malaga-server"
    binary_name = "malaga"
    modalities = {TEXT}

    _PARAM_MAP = {
        "device": "--device",
        "device_id": "--device-id",
        "threads": "--threads",
        "beam": "--beam",
        "max_batch": "--max-batch",
        "default_source": "--default-source",
        "default_target": "--default-target",
    }

    def build_args(self, binary, port, model_path, entry, merged_params, artifacts):
        args = [binary, "serve",
                "--model", str(model_path),
                "--model-id", entry["name"],
                "--host", "127.0.0.1",
                "--port", str(port)]
        for key, flag in self._PARAM_MAP.items():
            if merged_params.get(key) not in (None, ""):
                args += [flag, str(merged_params[key])]
        if merged_params.get("fast_vocab"):
            args.append("--fast-vocab")
        return args

    def health_path(self) -> str:
        return "/health"
```

puis l'ajouter au dictionnaire `_BACKENDS` (ou `register_backend(MalagaServerBackend())`).

## 3. Entrée de registre (`registry.yaml`)

```yaml
  - name: malaga-fr-mg
    file: nllb-200-distilled-600M-q4_k_m.gguf
    modality: text
    backend: malaga-server
    mem_gb: 1.3
    evict_group: translation
    aliases: [malaga]
    description: "Traduction français -> malgache (NLLB-200 600M, q4_k_m)"
    params:
      device: cuda
      default_source: fr
      default_target: mg
```

Pour l'anglais, une seconde entrée `malaga-en-mg` (`default_source: en`) — ou une seule entrée en
passant `source` dans le corps de la requête, ce qui évite un second processus.

`mem_gb` : 1,2 Go mesuré en q4_k_m (contexte CUDA ~0,43 Go inclus, batchs jusqu'à 32 phrases) ;
1,4 Go en q8_0. Avec `fast_vocab: true` et la variable `MALAGA_LAZY_EMBEDDINGS=1` : ~1,0 Go, au
prix d'une traduction non exacte (voir [performance.md](performance.md)). Pour une entrée CPU,
`device: cpu` et pas de `mem_gb`.

## 4. Appeler

```bash
curl -s localhost:<port-zallama>/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "malaga-fr-mg",
  "messages": [{"role": "user", "content": "Le marché ouvre très tôt le matin."}]
}'
```

## Cohabitation sur le GPU

Prévu pour tourner avec d'autres modèles (ASR, vision) : si la VRAM vient à manquer, malaga
libère ses graphes CUDA et continue sans eux, puis réduit la taille des batchs, au lieu d'échouer.
`MALAGA_GRAPH_CACHE_MB` (512 par défaut) borne la mémoire de ces graphes.
