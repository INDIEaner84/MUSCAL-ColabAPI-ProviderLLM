# Modellmatrix & GPU-Budget

Stand: LFM2.5-Generation. Alle LFM2.5-Checkpoints sind laut Liquid-Dokumentation
per TRL trainierbar; vom alten `LFM2-Audio-1.5B` wird abgeraten („Trainable: No").

## Text (dicht)

| Modell | Kontext | QLoRA 4-bit (T4 15 GB) | LoRA bf16 (A100 40 GB) |
|---|---|---|---|
| `LFM2.5-230M` / `350M` | 32K | bs 8, 2048 Tokens | bs 16, 4096 |
| `LFM2.5-1.2B-Instruct` | 32K | bs 4, 1024 Tokens | bs 8, 2048 |
| `LFM2.5-1.2B-Thinking` / `-JP` | 32K | bs 4, 1024 | bs 8, 2048 |
| `LFM2.5-2.6B` | 128K | bs 1–2, 1024 | bs 4, 2048 |

Faustregel QLoRA: **~0.7 GB pro Milliarde Parameter** für die Gewichte in 4-bit
(doppelte Quantisierung), plus Aktivierungen, die quadratisch mit der Sequenzlänge
wachsen. Gradient Checkpointing ist in allen Configs an — das kostet ~30 % Zeit,
spart aber den Großteil der Aktivierungen.

## MoE

`LFM2.5-8B-A1B`: 8.3B Gesamtparameter, **1.5B pro Token aktiv**.

Das ist der Punkt, den man easy unterschätzt: der Speicherbedarf richtet sich nach
den *Gesamt*-Parametern, nicht nach den aktiven. In 4-bit sind das ca. 5.5 GB,
dazu kommen LoRA-Parameter, Optimizer-States (8-bit Adam) und Aktivierungen für
*alle* Experten-Pfade, die der Batch antriggert.

* **T4 (15 GB)**: machbar mit `per_device_train_batch_size: 1`,
  `max_length: 512`, `gradient_accumulation_steps: 16`. Eng, aber es läuft.
* **A100 / L4 (40 GB)**: bs 4, 2048 Tokens, deutlich entspannter.
* LoRA-Ziele greifen per Suffix (`gate_proj`, `up_proj`, `down_proj`) in jeden
  Experten — das bläht die Adapter-Parameterzahl auf. Bei OOM erst `r` auf 8 senken,
  dann Sequenzlänge.

## Vision

| Modell | LM | Vision-Encoder | T4 15 GB |
|---|---|---|---|
| `LFM2.5-VL-450M` | 350M | SigLIP2 NaFlex base (86M) | bs 2, `max_image_tokens: 256` |
| `LFM2.5-VL-1.6B` | 1.2B | SigLIP2 shape-optimiert (400M) | bs 1, `max_image_tokens: 256` |
| `LFM2.5-VL-3B` | — | — | eher L4/A100 |

Bild-Token sind der Hebel: `max_image_tokens` von 256 auf 64 zu senken spart mehr
als jede Batch-Size-Änderung. Liquid empfiehlt „narrow use cases" — die kleinen
VLMs sind auf breite Allgemeinheit nicht ausgelegt, aber auf eine spezifische
Aufgabe feingetunt sehr stark.

## Audio

`LFM2.5-Audio-1.5B` (1.2B LM + 115M Audio-Encoder, 24 kHz Ausgabe).

Der offizielle Weg ist das Paket [`liquid-audio`](https://github.com/Liquid4All/liquid-audio)
(ab v1.2.0 mit Finetuning-Support). Dessen `Trainer` macht ein **volles Finetuning**
— es gibt keinen LoRA-Schalter. Damit ist der VRAM-Bedarf eine andere Liga als bei
den QLoRA-Tracks: **A100/L4 einplanen**, die freie T4 wird eng bis unmöglich.

System-Prompts sind funktional, nicht dekorativ — das Modell wurde darauf trainiert:

| Aufgabe | System-Prompt |
|---|---|
| ASR | `Perform ASR.` |
| TTS | `Perform TTS. Use the US female voice.` (US/UK × male/female) |
| Voice Chat | `Respond with interleaved text and audio.` |

## Was ist *nicht* trainierbar

* `LFM2-Audio-1.5B` — deprecated, „Trainable: No". Stattdessen `LFM2.5-Audio-1.5B`.
* Embedding/ColBERT-Modelle: eigene Ökosysteme (sentence-transformers, PyLate),
  nicht Teil dieses Repos.
* Die Liquid **Nanos** sind bereits feingetunte Spezialmodelle — wenn eines davon
  die Aufgabe abdeckt, ist das billiger als selbst zu trainieren.
