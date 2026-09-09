# MUSCAL-ColabAPI-ProviderLLM

LoRA / QLoRA-Finetuning der **Liquid AI LFM2.5**-Modelle auf Colab-GPUs —
Text, MoE (`LFM2.5-8B-A1B`), Vision (`LFM2.5-VL`) und Audio (`LFM2.5-Audio`).

> **Kein Tunnel, kein VPN, keine Erreichbarkeit.** Das Notebook läuft *in* Colab:
> Daten rein über Google Drive, Adapter raus über Google Drive. Der lokale Rechner
> wird nur für Datenvorbereitung und zum Zusammenfügen des Adapters gebraucht.

## Schnellstart

1. **Daten vorbereiten** (lokal, kein GPU nötig):

   ```bash
   python scripts/prepare_dataset.py --input meine_daten.csv --format qa \
       --system "Du bist ein hilfreicher Assistent." --out data/train.jsonl
   ```

   Das schreibt JSONL im `messages`-Format und meckert, wenn Zeilen kaputt sind.

2. **`data/train.jsonl` nach Google Drive** (z. B. `MyDrive/muscal-lfm/data/`) und
   Notebook öffnen: [`notebooks/01_lfm_text_qlora.ipynb`](notebooks/01_lfm_text_qlora.ipynb).
   Im Notebook: Runtime → **GPU** (T4 reicht für 1.2B/2.6B).

3. **Adapter liegt danach** in `MyDrive/muscal-lfm/outputs/<run>/adapter/`.
   Zusammenführen (Merge) und GGUF-Export gehen im Notebook oder lokal:

   ```bash
   python scripts/merge_adapter.py \
       --base LiquidAI/LFM2.5-1.2B-Instruct \
       --adapter outputs/lfm25-1.2b-text/adapter \
       --out outputs/lfm25-1.2b-text/merged
   ```

Ohne Notebook, direkt in Colab per Kommandozeile:

```bash
git clone https://github.com/INDIEaner84/MUSCAL-ColabAPI-ProviderLLM.git
cd MUSCAL-ColabAPI-ProviderLLM && pip install -q -r requirements-colab.txt
python -m muscal_lfm --config configs/text_qlora.yaml \
    --set data.train_file=/content/drive/MyDrive/muscal-lfm/data/train.jsonl
```

Alle Config-Werte sind per `--set section.key=value` überschreibbar,
`--dry-run` zeigt die aufgelöste Config, ohne ein Modell zu laden.

## Modellmatrix

| Track | Modell | Colab-Tier | Anmerkung |
|---|---|---|---|
| Text (klein) | `LFM2.5-350M`, `LFM2.5-1.2B-Instruct` | **T4 free** | QLoRA, 1–3 Epochen in Minuten |
| Text (groß) | `LFM2.5-2.6B` | **T4 free** (knapp) / L4 | bs=1–2, `max_length` 1024 |
| MoE | `LFM2.5-8B-A1B` (8.3B / 1.5B aktiv) | **A100/L4 empfohlen** | 4-bit ≈ 5.5 GB Gewichte; auf der T4 nur mit bs=1 + 512 Tokens |
| Vision | `LFM2.5-VL-450M`, `-1.6B`, `-3B` | T4 (450M/1.6B) / L4 (3B) | Bild-Token runterdrehen bei OOM |
| Audio | `LFM2.5-Audio-1.5B` | **A100/L4** | `liquid-audio` macht *volles* Finetuning, kein LoRA |

Details, VRAM-Abschätzungen und Settings pro GPU: [docs/model-matrix.md](docs/model-matrix.md).
LFM2-Audio-1.5B (die alte Generation) ist laut Liquid **nicht** trainierbar —
für Audio immer die `LFM2.5-Audio`-Checkpoints nehmen.

## Datenformate

Alle HF-Tracks erwarten Konversationen:

```json
{"messages": [
  {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
  {"role": "user", "content": "Was ist die Hauptstadt von Frankreich?"},
  {"role": "assistant", "content": "Paris."}
]}
```

Vision braucht flache Zeilen (`image`, `question`, `answer`), Audio läuft über
`liquid-audio` eigene Preprocessing-Kette. Alles inkl. Beispiele:
[docs/datasets.md](docs/datasets.md).

**Die zwei Regeln, die am meisten Ärger sparen:**

* Mit dem *eigenen* Chat-Template des Modells trainieren — `apply_chat_template`,
  Format zeichengleich zur Produktion.
* Lieber 500–5.000 saubere Beispiele, die die echte Eingabeverteilung treffen,
  als 100k Rauschen.

## Repo-Struktur

```
configs/          vier Beispiel-Runs (text / moe / vl / audio)
muscal_lfm/       Python-Paket: config, data, model, train, export, cli
scripts/          lokale Helfer (Datenaufbereitung, Merge)
notebooks/        Colab-Notebooks, eines pro Track
docs/             Modellmatrix, Datenformate, GGUF-Export
```

## Nach dem Training

Der Adapter allein ist klein (wenige MB) und nur mit dem Basis-Modell zusammen
nutzbar. Für den Einsatz auf dem Gerät: mergen → GGUF → llama.cpp/Ollama/LM Studio.
Siehe [docs/gguf-export.md](docs/gguf-export.md).

## Bekannte Fallstricke

* **T4 kann kein bf16.** `bnb_4bit_compute_dtype: auto` und `bf16: auto` wechseln
  automatisch auf fp16 — wer bf16 hart setzt, bekommt kryptische bnb-Fehler.
* **MoE braucht alle Gewichte im Speicher**, auch wenn nur 1.5B aktiv sind.
* **Audio = volles Finetune**, kein LoRA-Schalter in `liquid-audio` → deutlich mehr VRAM.
* **LFM2 ist hybrid** (gated Convs + GQA). `flash_attention_2` ist nicht überall
  supportet; Default ist `sdpa`, das läuft überall.
* Adapter-Ziele (`target_modules`) matchen die MoE-Experten per Suffix, deshalb
  funktioniert dieselbe Liste für dichte und MoE-Checkpoints.

## Quellen

* Liquid AI Docs – Modellbibliothek & Finetuning: <https://docs.liquid.ai>
* TRL-Rezepte (SFT/DPO/VLM) für LFM2.5: <https://docs.liquid.ai/lfm/fine-tuning/trl>
* Unsloth LFM2.5-Support: <https://docs.liquid.ai/lfm/fine-tuning/unsloth>
* `liquid-audio` (Audio-Finetuning ab v1.2.0): <https://github.com/Liquid4All/liquid-audio>
* Liquid AI Cookbook (Notebooks & Beispiele): <https://github.com/Liquid4All/cookbook>

Modelle stehen unter der **LFM Open License v1.0**.
