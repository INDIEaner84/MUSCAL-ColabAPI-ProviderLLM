# Datenformate

## Text & MoE — Konversationen

Eine Zeile = ein Trainingsbeispiel, Datei `data/train.jsonl`:

```json
{"messages": [
  {"role": "system", "content": "Du bist ein Support-Assistent für MUSCAL."},
  {"role": "user", "content": "Wie starte ich den Server?"},
  {"role": "assistant", "content": "Mit `python -m muscal_lfm --config configs/text_qlora.yaml`."}
]}
```

Regeln, die der Loader prüft ([`muscal_lfm/data.py`](../muscal_lfm/data.py)):

* Spalte `messages` muss existieren.
* Rollen nur `system` / `user` / `assistant` / `tool`.
* Die **letzte** Nachricht muss von `assistant` kommen — sonst gibt es nichts zu lernen.
* Mindestens eine `assistant`-Nachricht pro Zeile.

Mehrere Runden (Multi-Turn) sind erlaubt: `user` und `assistant` abwechselnd.

### Umwandeln

```bash
# CSV mit question/answer
python scripts/prepare_dataset.py --input faq.csv --format qa --out data/train.jsonl

# Alpaca-Style (instruction / input / output)
python scripts/prepare_dataset.py --input alpaca.json --format alpaca --out data/train.jsonl

# ShareGPT-Style (conversations mit from/value)
python scripts/prepare_dataset.py --input chat.json --format sharegpt --out data/train.jsonl

# Nur prüfen, nichts schreiben
python scripts/prepare_dataset.py --input data/train.jsonl --validate-only
```

Erkannte Formate: `messages`, `sharegpt`, `alpaca`, `qa`, `prompt-completion`,
`flat-vl`. Ohne `--format` wird geraten und das Ergebnis ausgegeben.

## Vision — flache Zeilen

VLM-Zeilen bleiben flach, weil Bilder nicht ins JSONL passen. Erwartet werden die
Spalten `image`, `question`, `answer`; `image` ist ein Pfad (relativ zu
`data.image_root`) oder ein PIL-Bild:

```json
{"image": "screenshots/panel-01.png", "question": "Welcher Wert steht bei Temperatur?", "answer": "63 °C"}
{"image": "screenshots/panel-02.png", "question": "Ist die Warnung aktiv?", "answer": "Nein."}
```

`build_vlm_conversations()` lädt die Bilder, konvertiert nach RGB und baut daraus
die Chat-Template-Eingaben. Bilder **immer** nach RGB konvertieren — der SigLIP2-
Encoder macht das nicht nachträglich.

## Audio — `liquid-audio`

Der Audio-Track hat eine eigene Kette (kein TRL):

1. Rohdaten auf `list[ChatMessage]` abbilden, mit `TextSegment`, `AudioSegment`
   oder `InterleavedSegment`.
2. `LFM2AudioChatMapper` + `preprocess_dataset()` schreiben ein vorverarbeitetes
   Dataset nach `data/audio/train`.
3. `LFM2DataLoader` + `liquid_audio.trainer.Trainer` trainieren darauf.

Siehe [`notebooks/04_lfm_audio_ft.ipynb`](../notebooks/04_lfm_audio_ft.ipynb) und
[`examples/preprocess_jenny_tts.py`](https://github.com/Liquid4All/liquid-audio/blob/main/examples/preprocess_jenny_tts.py)
für ein vollständiges TTS-Beispiel.

## Mengen & Qualität

Liquid empfiehlt **500–5.000 Beispiele**. Wichtiger als Volumen:

* **Verteilung**: Trainingsdaten müssen aussehen wie die echten Produktiv-Eingaben.
* **Ausgabe-Stil**: Der gewünschte Stil muss in den `assistant`-Antworten stehen,
  nicht im System-Prompt.
* **Held-out-Set vor dem Training einfrieren** — sonst evaluiert man gegen Daten,
  die das Modell schon kennt.
* **Chat-Template identisch zur Produktion.** Abweichungen (Leerzeichen, anderer
  System-Prompt) kosten mehr Qualität als doppelt so viele Daten.
* Für DPO braucht es `prompt` / `chosen` / `rejected` statt `messages`; die
  Learning-Rate liegt dort drei Größenordnungen niedriger (1e-7 … 1e-6).
