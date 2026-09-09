#!/usr/bin/env python3
"""Generate the Colab notebooks in notebooks/.

Kept as a generator so the four notebooks stay consistent (install cell, Drive
mount, config handling) instead of drifting apart. Run after editing:

    python tools/make_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/INDIEaner84/MUSCAL-ColabAPI-ProviderLLM.git"
ROOT = Path(__file__).resolve().parent.parent


def nb(cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
        for c in cells
    ]
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    }
    return notebook


# --- shared cells ----------------------------------------------------------- #

def install_cell(extra: str = "") -> tuple[str, str]:
    src = f"""# Colab brings torch; we only need the training stack. --upgrade on purpose:
# LFM2.5 checkpoints need a recent transformers.
!pip install -q -U transformers trl peft accelerate bitsandbytes datasets pyyaml{extra}

import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
    print("bf16 supported:", torch.cuda.is_bf16_supported())"""
    return "code", src


def drive_cell(project: str) -> tuple[str, str]:
    src = f"""from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

DRIVE_DIR = Path('/content/drive/MyDrive/muscal-lfm/{project}')
(DRIVE_DIR / 'data').mkdir(parents=True, exist_ok=True)
(DRIVE_DIR / 'outputs').mkdir(parents=True, exist_ok=True)
print("drive project dir:", DRIVE_DIR)
print("contents:", sorted(p.name for p in DRIVE_DIR.iterdir()))"""
    return "code", src


def code_cell(branch: str = "main") -> tuple[str, str]:
    src = f"""# Get the training code. If you already have the repo in Drive, skip this cell.
import os, sys
from pathlib import Path

REPO_URL = "{REPO_URL}"
REF = os.environ.get("MUSCAL_REF", "{branch}")   # set to your branch before pushing

if not Path('/content/MUSCAL-ColabAPI-ProviderLLM').exists():
    !git clone --depth 1 --branch $REF $REPO_URL /content/MUSCAL-ColabAPI-ProviderLLM

%cd /content/MUSCAL-ColabAPI-ProviderLLM
sys.path.insert(0, '/content/MUSCAL-ColabAPI-ProviderLLM')
print("cwd:", Path.cwd())"""
    return "code", src


def config_cell(config_name: str, overrides: str = "") -> tuple[str, str]:
    src = f"""from muscal_lfm.config import TrainConfig
from muscal_lfm.model import gpu_info
import yaml

CONFIG = "configs/{config_name}"
cfg = TrainConfig.from_yaml(CONFIG)

# --- your settings -------------------------------------------------------
cfg.data.train_file = str(DRIVE_DIR / "data" / "train.jsonl")
cfg.training.output_dir = str(DRIVE_DIR / "outputs" / "{config_name.replace('_qlora', '').replace('_ft', '')}")
{overrides}
# ------------------------------------------------------------------------

print("gpu:", gpu_info())
print()
print(cfg.to_yaml())"""
    return "code", src


def validate_cell() -> tuple[str, str]:
    src = """from muscal_lfm import data as data_utils
from datasets import Dataset

ds = data_utils.load_file(cfg.data.train_file)
print("rows:", len(ds), "| columns:", ds.column_names)
print()
print("example row:")
print(ds[0])

problems_exist = False
try:
    data_utils.validate_conversational(ds)
    print("\\n[ok] dataset shape looks good")
except data_utils.DatasetError as exc:
    problems_exist = True
    print("\\n[problem]", exc)

print("\\nstats:", data_utils.dataset_report(ds))"""
    return "code", src


def train_cell() -> tuple[str, str]:
    src = """from muscal_lfm.train import train
from pathlib import Path

Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.training.output_dir, "config.resolved.yaml").write_text(cfg.to_yaml())

adapter_dir = train(cfg)
print("adapter:", adapter_dir)
print(sorted(p.name for p in Path(adapter_dir).iterdir()))"""
    return "code", src


def test_cell(track: str) -> tuple[str, str]:
    if track == "vl":
        return "code", '''from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

processor = AutoProcessor.from_pretrained(cfg.model.id, trust_remote_code=True)
base = AutoModelForImageTextToText.from_pretrained(
    cfg.model.id, dtype="auto", device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, str(adapter_dir))

# point this at one of your own images
image_path = sorted(Path(cfg.data.image_root).rglob("*.png"))[0] if cfg.data.image_root else None
if image_path:
    conv = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": "Beschreibe, was auf diesem Bild zu sehen ist."},
        ],
    }]
    inputs = processor.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))'''

    return "code", '''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)
base = AutoModelForCausalLM.from_pretrained(cfg.model.id, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, str(adapter_dir))

PROBE = "Erklaere in einem Satz, was dieses Modell macht."

messages = [{"role": "user", "content": PROBE}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("--- BASE ---")
with torch.no_grad():
    out = base.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

print("\\n--- FINETUNED ---")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))'''


def export_cell() -> tuple[str, str]:
    return "code", '''from muscal_lfm.export import merge_adapter, export_gguf

merged = merge_adapter(
    cfg.model.id,
    adapter_dir,
    str(Path(cfg.training.output_dir) / "merged"),
    track=cfg.model.track,
)
print("merged:", merged)

# Optional: GGUF for llama.cpp / Ollama / LM Studio.
# Needs a current llama.cpp (LFM2.5 is a young architecture) and ~10 min.
EXPORT_GGUF = False
if EXPORT_GGUF:
    gguf = export_gguf(merged, quant="q4_k_m")
    !cp "$gguf" "$DRIVE_DIR/outputs/"'''


# --- notebooks ------------------------------------------------------------- #

def text_notebook() -> list[tuple[str, str]]:
    return [
        ("md", """# 01 — LFM2.5 Text: QLoRA-Finetuning auf Colab

Trainiert einen LoRA-Adapter auf einem **dichten LFM2.5-Textmodell**
(Default: `LiquidAI/LFM2.5-1.2B-Instruct`) mit 4-bit-QLoRA.

**Vorher:** `data/train.jsonl` im `messages`-Format nach
`MyDrive/muscal-lfm/text/data/` legen (lokal erzeugen mit
`scripts/prepare_dataset.py`).

**Runtime:** GPU. Eine freie T4 (15 GB) reicht für 350M/1.2B; für 2.6B
`per_device_train_batch_size` auf 1–2 senken."""),
        install_cell(),
        drive_cell("text"),
        code_cell(),
        config_cell("text_qlora.yaml", """cfg.training.num_train_epochs = 3
cfg.training.per_device_train_batch_size = 4
cfg.training.gradient_accumulation_steps = 4
cfg.training.learning_rate = 2e-4"""),
        ("md", """## Daten prüfen

Lieber hier scheitern als nach 20 Minuten Training."""),
        validate_cell(),
        ("md", """## Trainieren

Der erste Durchlauf lädt das Basis-Modell (~2.5 GB für 1.2B). Danach sind es
bei 500 Beispielen und 3 Epochen auf der T4 typischerweise unter 15 Minuten."""),
        train_cell(),
        ("md", """## Vorher / Nachher

Erst die Basis-Antwort, dann die des feingetunten Modells. Wenn beide gleich
sind: Loss-Kurve prüfen — meistens war die Learning-Rate zu klein oder die Daten
haben nicht das gelernt, was der Probe-Prompt abfragt."""),
        test_cell("text"),
        ("md", """## Adapter zusammenführen

Für den Einsatz in llama.cpp / Ollama / LM Studio wird der Adapter mit dem
Basis-Modell verschmolzen und nach GGUF gewandelt. Der Merge läuft auf CPU, um
den GPU-Speicher nicht zu sprengen."""),
        export_cell(),
        ("md", """## Nächste Schritte

* Mehr Daten / andere `r`-Werte: `cfg.lora.r` (16 → 32) erhöhen, dafür LR leicht senken.
* Präferenz-Training: `DPOTrainer` aus TRL, Datensatz mit `prompt`/`chosen`/`rejected`,
  Learning-Rate 1e-7 … 1e-6 statt 2e-4.
* MoE statt dicht: `notebooks/03_lfm_moe_qlora.ipynb`.
* Siehe `docs/gguf-export.md` für den Weg bis aufs Gerät."""),
    ]


def vl_notebook() -> list[tuple[str, str]]:
    return [
        ("md", """# 02 — LFM2.5-VL: Vision-Finetuning mit QLoRA

Feintunt ein **Vision-Language-Modell** (Default `LiquidAI/LFM2.5-VL-1.6B`) auf
eigene Bild-Text-Paare.

**Datenformat** — flache Zeilen statt Konversationen, weil Bilder nicht ins JSONL
passen. `image` ist ein Pfad relativ zu `data.image_root`:

```json
{"image": "screenshots/panel-01.png", "question": "Welcher Wert steht bei Temperatur?", "answer": "63 °C"}
```

**Layout in Drive:**

```
MyDrive/muscal-lfm/vl/data/vl_train.jsonl
MyDrive/muscal-lfm/vl/data/images/...      # <- data.image_root
```"""),
        install_cell(),
        drive_cell("vl"),
        code_cell(),
        ("md", """## Bilder nach Colab holen

Am einfachsten als gezippter Ordner in Drive, dann hier entpacken. Bei sehr
vielen Bildern statt dessen direkt aus Drive lesen — Entpacken kostet Zeit, das
Zippen spart sie."""),
        ("code", '''import zipfile
from pathlib import Path

ZIP = DRIVE_DIR / "data" / "images.zip"
IMAGE_ROOT = DRIVE_DIR / "data" / "images"

if ZIP.exists() and not IMAGE_ROOT.exists():
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(IMAGE_ROOT)
    print("extracted ->", IMAGE_ROOT)

print("images found:", len(list(IMAGE_ROOT.rglob("*.[pj][np][g]"))) if IMAGE_ROOT.exists() else 0)'''),
        config_cell("vl_qlora.yaml", """cfg.data.train_file = str(DRIVE_DIR / "data" / "vl_train.jsonl")
cfg.data.image_root = str(IMAGE_ROOT)
cfg.data.max_image_tokens = 256     # bei OOM auf 128 / 64 senken
cfg.training.per_device_train_batch_size = 1
cfg.training.gradient_accumulation_steps = 16"""),
        ("md", """## Daten prüfen

Der VLM-Pfad baut die Chat-Template-Eingaben selbst; geprüft wird, dass pro Zeile
`image`, `question` und `answer` vorhanden sind und die Bilder ladbar sind."""),
        ("code", '''from datasets import Dataset
from muscal_lfm import data as data_utils
import json
from pathlib import Path

rows = [json.loads(l) for l in Path(cfg.data.train_file).read_text(encoding="utf-8").splitlines() if l.strip()]
ds = Dataset.from_list(rows)
print("rows:", len(ds), "| columns:", ds.column_names)
print("example:", rows[0])

problems = data_utils.validate(ds, flat_vl=True)
print("[ok] shape valid" if not problems else problems[:10])

# teurer Check: alle Bilder wirklich laden
convs = data_utils.build_vlm_conversations(ds.select(range(min(16, len(ds)))), image_root=cfg.data.image_root)
print("built", len(convs), "conversations (probe)")'''),
        train_cell(),
        test_cell("vl"),
        export_cell(),
        ("md", """## Hinweise

* **Bild-Token sind der Speicherhebel.** `max_image_tokens` 256 → 64 spart mehr als
  jede Batch-Size-Änderung.
* Die kleinen VLMs sind laut Liquid für **enge Anwendungsfälle** gedacht — ein
  feingetuntes 450M schlägt ein generisches 3B auf *seiner* Aufgabe oft.
* Bilder müssen RGB sein; `build_vlm_conversations` konvertiert, aber CMYK- oder
  16-bit-TIFFs vorher selbst sauber machen."""),
    ]


def moe_notebook() -> list[tuple[str, str]]:
    return [
        ("md", """# 03 — LFM2.5-8B-A1B (MoE): QLoRA-Finetuning

Dieselbe Kette wie Notebook 01, aber für das **Mixture-of-Experts**-Modell
`LiquidAI/LFM2.5-8B-A1B` — 8.3B Gesamtparameter, 1.5B pro Token aktiv.

**Der Denkfehler, den man hier macht:** der Speicherbedarf richtet sich nach den
*Gesamt*parametern, nicht nach den aktiven. In 4-bit sind das rund 5.5 GB — auf
einer 15-GB-T4 bleibt wenig Luft, auf einer A100/L4 (40 GB) ist es entspannt.

Die Zelle unten prüft die GPU und passt die Defaults an."""),
        install_cell(),
        drive_cell("moe"),
        code_cell(),
        ("code", '''import torch
from muscal_lfm.model import gpu_info

info = gpu_info()
print(info)

VRAM = info.get("vram_gb", 0)
if VRAM and VRAM < 20:
    print("\\n[warn] unter 20 GB VRAM -- bleibe bei batch size 1 und 512 Tokens.")
    PRESET = dict(per_device_train_batch_size=1, gradient_accumulation_steps=16, max_length=512)
else:
    print("\\n[info] genug VRAM fuer groessere Batches.")
    PRESET = dict(per_device_train_batch_size=4, gradient_accumulation_steps=4, max_length=2048)
print(PRESET)'''),
        config_cell("moe_qlora.yaml", """cfg.data.max_length = PRESET["max_length"]
cfg.training.per_device_train_batch_size = PRESET["per_device_train_batch_size"]
cfg.training.gradient_accumulation_steps = PRESET["gradient_accumulation_steps"]
cfg.training.learning_rate = 1e-4"""),
        ("md", """## Hinweis zu LoRA an MoE-Schichten

`gate_proj` / `up_proj` / `down_proj` matchen per Suffix **jeden Experten** — die
Adapter-Parameterzahl ist deshalb deutlich höher als beim dichten Modell gleicher
Größe. Bei OOM in dieser Reihenfolge runtergehen:

1. `cfg.lora.r` 16 → 8
2. `cfg.data.max_length` halbieren
3. `per_device_train_batch_size` 1 und `gradient_accumulation_steps` erhöhen"""),
        validate_cell(),
        train_cell(),
        test_cell("text"),
        ("md", """## Merge

Bei 8.3B lohnt es sich, den Merge **nicht** in Colab zu machen, wenn du nur einen
Adapter weitergeben willst — lade den Adapter nach Drive und merg lokal mit
`scripts/merge_adapter.py`. Für GGUF braucht es eine aktuelle llama.cpp mit
LFM-MoE-Support."""),
        export_cell(),
    ]


def audio_notebook() -> list[tuple[str, str]]:
    return [
        ("md", """# 04 — LFM2.5-Audio-1.5B: Finetuning mit `liquid-audio`

Andere Baustelle als die anderen drei Notebooks: Audio läuft **nicht** über TRL,
sondern über Liquid's eigenes Paket
[`liquid-audio`](https://github.com/Liquid4All/liquid-audio) (Finetuning ab v1.2.0).

Ablauf:

1. Rohdaten auf `list[ChatMessage]` abbilden (`TextSegment`, `AudioSegment`,
   `InterleavedSegment`).
2. `LFM2AudioChatMapper` + `preprocess_dataset()` → vorverarbeitetes Dataset.
3. `LFM2DataLoader` + `liquid_audio.trainer.Trainer` → Training.

**Wichtig:** dieser Trainer macht ein **volles Finetuning**, er hat keinen
LoRA-Schalter. VRAM-Bedarf ist eine andere Liga als bei den QLoRA-Tracks —
**A100 / L4 einplanen**, die freie T4 wird eng."""),
        ("code", '''!pip install -q -U liquid-audio datasets soundfile
!nvidia-smi --query-gpu=name,memory.total --format=csv

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")'''),
        drive_cell("audio"),
        ("md", """## Aufgabe festlegen

Der System-Prompt ist bei diesem Modell **funktional**, nicht dekorativ — er
wählt den Modus. Die vier vom Modell gelernten Prompts:

| Aufgabe | System-Prompt |
|---|---|
| ASR | `Perform ASR.` |
| TTS | `Perform TTS. Use the US female voice.` (US/UK × male/female) |
| Voice Chat | `Respond with interleaved text and audio.` |"""),
        ("code", '''SYSTEM_PROMPT = "Perform ASR."          # <- fuer deine Aufgabe anpassen
MODEL_ID      = "LiquidAI/LFM2.5-Audio-1.5B"
DATA_DIR      = DRIVE_DIR / "data"
PREPROC_DIR   = DRIVE_DIR / "data" / "audio" / "train"
OUT_DIR       = DRIVE_DIR / "outputs" / "lfm25-audio"

CONTEXT_LENGTH = 256
BATCH_SIZE     = 8
MAX_STEPS      = 1000
LR             = 1e-4

print(SYSTEM_PROMPT, "|", MODEL_ID)'''),
        ("md", """## 1. Rohdaten → ChatMessages

Unten ein Iterator für ein ASR-Setup (`audio` + `transcription`). Für TTS wird
getauscht: `user` bekommt den `TextSegment`, `assistant` den `AudioSegment`.
Für Voice-Chat kommen `InterleavedSegment(text=..., audio=...)` in die
Assistenten-Nachricht.

`audio` muss **Bytes eines Audiocontainers** sein (wav/flac/ogg/…), keine
vorverarbeiteten Features — der Processor macht daraus log-mel."""),
        ("code", '''from pathlib import Path

# %%writefile interpoliert keine Variablen -- deshalb schreiben wir die Datei
# selbst, mit den oben gesetzten Werten eingesetzt.
TEMPLATE = \'\'\'
"""Angepasst von liquid-audio/examples/preprocess_jenny_tts.py"""
from __future__ import annotations

from collections.abc import Iterator

from datasets import Audio, load_dataset

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

SYSTEM_PROMPT = "{system_prompt}"
MODEL_ID = "{model_id}"
OUTPUT_PATH = "{preproc_dir}"


class MyDataIterator:
    """Passt diese drei Zeilen an dein Dataset an."""

    def __init__(self, split: str = "train") -> None:
        self.ds = load_dataset("mein/dataset", split=split)
        self.ds = self.ds.cast_column("audio", Audio(decode=False))

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        for row in self.ds:
            audio_bytes = row["audio"]["bytes"]        # Audio-Spalte
            transcript = row["transcription"]          # Text-Spalte
            yield [
                ChatMessage(role="system", content=[TextSegment(text=SYSTEM_PROMPT)]),
                ChatMessage(role="user", content=[AudioSegment(audio=audio_bytes)]),
                ChatMessage(role="assistant", content=[TextSegment(text=transcript)]),
            ]


if __name__ == "__main__":
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cuda").eval()
    mapper = LFM2AudioChatMapper(processor)
    preprocess_dataset(
        data=MyDataIterator(),
        output_path=OUTPUT_PATH,
        mapper=mapper,
        max_context_length=256,   # laengere Samples werden uebersprungen
    )
\'\'\'

Path("preprocess_my_data.py").write_text(
    TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        model_id=MODEL_ID,
        preproc_dir=str(PREPROC_DIR),
    ),
    encoding="utf-8",
)
print(Path("preprocess_my_data.py").read_text(encoding="utf-8")[:400])'''),
        ("md", """Passe `MyDataIterator` an dein Dataset an (Spaltennamen, Rollenverteilung), dann
Preprocessing starten. Das schreibt das vorverarbeitete Dataset nach
`data/audio/train` — bei großen Sets dauert das, also einmal laufen lassen und in
Drive lassen."""),
        ("code", '''!python preprocess_my_data.py
!ls -la "$PREPROC_DIR" | head -20'''),
        ("md", """## 2. Trainieren

`liquid_audio.trainer.Trainer` übernimmt Loop, Logging und Checkpoints
(`save_interval`, `val_interval`). Checkpoints landen in `output_dir` — auf Drive
zeigen lassen, damit ein Runtime-Reset nichts verliert."""),
        ("code", '''from pathlib import Path

from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.trainer import Trainer

assert Path(PREPROC_DIR).exists(), "erst preprocess_my_data.py laufen lassen"

train_data = LFM2DataLoader(dataset_path=str(PREPROC_DIR), context_length=CONTEXT_LENGTH)

trainer = Trainer(
    model_id=MODEL_ID,
    train_data=train_data,
    lr=LR,
    batch_size=BATCH_SIZE,
    max_steps=MAX_STEPS,
    warmup_steps=50,
    dataloader_num_workers=2,
    logging_interval=10,
    save_interval=250,
    val_interval=100,
    output_dir=str(OUT_DIR),
)
trainer.train()'''),
        ("md", """## 3. Testen

Der Processor wandelt Tokens zurück in Waveform (24 kHz). Bei ASR ist die Ausgabe
capitalized und interpungiert — genau so, wie es im Training stand."""),
        ("code", '''import torch, soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

processor = LFM2AudioProcessor.from_pretrained(MODEL_ID).eval()
model = LFM2AudioModel.from_pretrained(MODEL_ID).eval().cuda()

wav, sr = sf.read("mein_test_audio.wav", dtype="float32")
wav = torch.from_numpy(wav).unsqueeze(0)

chat = ChatState(processor)
chat.new_turn("system")
chat.add_text(SYSTEM_PROMPT)
chat.end_turn()
chat.new_turn("user")
chat.add_audio(wav, sr)
chat.end_turn()
chat.new_turn("assistant")

for t in model.generate_sequential(**chat, max_new_tokens=256):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)'''),
        ("md", """## Aufs Gerät

Audio-GGUFs brauchen eigene Runner (`llama-liquid-audio-cli` / `-server`) und
drei zusätzliche Dateien aus dem `-GGUF`-Repo: `mmproj-`, `vocoder-` und
`tokenizer-`. Details: <https://docs.liquid.ai/lfm/models/lfm25-audio-1.5b>"""),
    ]


def agent_notebook() -> list[tuple[str, str]]:
    return [
        ("md", """# 05 — Audio-Agent: Sprache → Tool-Call (Web, Browser, Desktop)

Feintunt `LFM2.5-Audio-1.5B` darauf, gesprochene Befehle **direkt** in
Funktionsaufrufe zu übersetzen — und verdrahtet das mit echter Web-Recherche,
Browser-Steuerung und Desktop-Aktionen.

```
 Mikrofon → LFM2.5-Audio (feingetunt) → "web_search|query=lfm2.5"
                                              ↓
                                        muscal_agent.executor
                                              ↓
                       JSON → LFM2.5-1.2B-Instruct (ein Satz) → TTS → Lautsprecher
```

**Drei Dinge, die hier anders sind als bei den anderen Notebooks:**

1. Der System-Prompt ist zwingend `"Perform ASR."` — der GGUF-Server erzwingt ihn
   zur Inferenz, ein anderer Prompt wird vom Fine-Tune überschrieben.
2. Es ist ein **volles Finetune**, kein LoRA → **A100 (Colab Pro) nötig**.
3. Das Tool-Call-Format wird vor der Datenerzeugung festgelegt und danach nie
   geändert.

Alles nachlesbar in [docs/audio-agent.md](../docs/audio-agent.md)."""),
        ("code", '''!pip install -q -U liquid-audio datasets soundfile
!nvidia-smi --query-gpu=name,memory.total --format=csv

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("gpu:", torch.cuda.get_device_name(0), "| vram:", round(vram, 1), "GB")
    if vram < 40:
        print("[warn] unter 40 GB -- das volle Audio-Finetune wird eng. A100 waehlen.")'''),
        ("md", """## Was kann diese Maschine?

`--doctor` prueft, welche Backends ueberhaupt verfuegbar sind -- bevor man dem
Modell die Schuld gibt. Auf Wayland-Rechnern z. B. funktionieren die
`desktop_*`-Werkzeuge mit pyautogui prinzipiell nicht."""),
        ("code", '''!python -m muscal_agent --doctor'''),
        drive_cell("audio-agent"),
        code_cell(),
        ("md", """## Werkzeuge und Format


Der Katalog liegt in `muscal_agent/catalog.py`. Klein halten: 3–4 Funktionen mit
je ~500 Beispielen schlagen 12 Funktionen mit je 80."""),
        ("code", '''import sys
sys.path.insert(0, "/content/MUSCAL-ColabAPI-ProviderLLM")

from muscal_agent.catalog import TOOLS, TOOLS_BY_NAME
from muscal_agent.toolcall import ToolCall, parse_tool_call

FORMAT = "pipe"   # "pipe" | "pythonic" -- vor der Datenerzeugung festlegen!

for tool in TOOLS:
    args = ", ".join(p.name for p in tool.params)
    print(f"  {tool.name}({args})  [{tool.risk}]")

call = ToolCall("web_search", {"query": "liquid ai lfm", "max_results": 5})
print("\\nserialised:", call.serialise(FORMAT))
print("round-trip :", parse_tool_call(call.serialise(FORMAT), fmt=FORMAT))'''),
        ("md", """## Profil wählen — nicht alle Werkzeuge auf einmal

Ein Profil bündelt zwei Dinge, die zusammengehören: welche Werkzeuge es für das
Modell gibt **und** welche Risikoklassen der Executor akzeptiert. Elf Werkzeuge
auf einmal freischalten macht das Modell nicht fähiger, sondern fehleranfälliger
— jedes zusätzliche ist eine neue Möglichkeit, sich zu verhören.

Reihenfolge: **`web` → Zahlen messen → `browser` → `desktop`** (in der VM)."""),
        ("code", '''from muscal_agent.profiles import describe, get as get_profile

PROFILE = "web"     # web | browser | desktop | all

print(describe())
profile = get_profile(PROFILE)
print("aktiv:", profile.name, "->", profile.tools)'''),
        ("md", """## Trockenübung für den Executor


Bevor ein Modell im Spiel ist: prüfen, dass die Policy greift. `dry_run` ist der
Default — es wird nichts ausgeführt, nur ausgegeben, was passieren würde."""),
        ("code", '''from muscal_agent.agent import AudioAgent
from muscal_agent.executor import Policy
from muscal_agent.toolcall import parse_tool_call

policy = Policy.from_profile(PROFILE, dry_run=True)
agent = AudioAgent(fmt=FORMAT, policy=policy, answer_model_id=None)

for text in [
    "web_search|query=liquid ai lfm",
    "desktop_launch|app=firefox",          # nicht auf der Allow-List
    "desktop_click|x=100|y=200",           # destructive + nicht erlaubt
    "web_read|url=https://docs.liquid.ai",
]:
    call = parse_tool_call(text, fmt=FORMAT)
    result = agent.run_tools([call])[0]
    status = "ok" if result.ok else ("skipped" if result.skipped else "error")
    print(f"{text:45s} -> {status:8s} {result.output or result.error}")'''),
        ("md", """## 1. Daten erzeugen — Umfang folgt dem Profil

Der Generator kombiniert **Satzschablonen x Slot-Werte** pro Werkzeug.
`--slots` + `--per-tool` erzeugt fuer *alle* Werkzeuge im Katalog Daten, eigene
Saetze kommen per `--utterances` dazu.

| Datei | Inhalt |
|---|---|
| `data/agent_slots_de.json` / `_en.json` | Werte pro Argument (URLs, Selektoren, Apps, Hotkeys …) |
| `data/agent_templates_de.json` / `_en.json` | Satzschablonen pro Werkzeug |
| `data/agent_utterances_web_de.jsonl` / `_en.json` | **web-Profil**: 400 Paare (200 je Werkzeug) |
| `data/agent_utterances_full_de.jsonl` / `_en.json` | **alle 11 Werkzeuge**: ~600 Paare (60 je Werkzeug) |

Weniger Werkzeuge bei mehr Beispielen schlaegt mehr Werkzeuge bei duenner
Deckung. Eigene Slot-Werte (deine URLs, Apps, Hotkeys) sind der groesste Hebel
nach dem Audio — Vorlage holen und anpassen:

```bash
python scripts/build_toolcall_dataset.py --profile web --language en --dump-slots > my_slots.json
```

**Sprache:** `LFM2.5-Audio-1.5B` ist auf Englisch trainiert. Deutsch
funktioniert eher als Absichtserkennung, Englisch ist treffsicherer. Unten
umschaltbar — die beiden Sprachen **nicht** in einem Datensatz mischen.

Eigene Aufnahmen (`--audio-map`) schlagen TTS (`--tts`) immer: beim TTS hat
jede Probe dieselbe Stimme, und das Modell lernt eine Sprecherin statt dich."""),
        ("code", '''from pathlib import Path

LANG     = "en"     # "de" | "en" -- entscheidet Schablonen und Slot-Werte
PER_TOOL = 200      # Paare pro Werkzeug (web-Profil hat nur 2 -> 400 Paare)

UTTERANCES  = DRIVE_DIR / "data" / "agent_utterances_full.jsonl"
DATASET_DIR = DRIVE_DIR / "data" / "agent_audio"
EVAL_FILE   = DRIVE_DIR / "data" / "agent_eval.jsonl"
UTTERANCES.parent.mkdir(parents=True, exist_ok=True)

# Paare erzeugen -- kein Audio noetig, --write-pairs stoppt danach
!python scripts/build_toolcall_dataset.py \
    --profile "$PROFILE" \
    --language "$LANG" \
    --slots "data/agent_slots_$LANG.json" \
    --per-tool "$PER_TOOL" \
    --fmt "$FORMAT" \
    --write-pairs "$UTTERANCES"

with open(UTTERANCES, encoding="utf-8") as fh:
    lines = fh.read().splitlines()
print("Zeilen:", len(lines))
print("Beispiel:", lines[0])'''),
        ("md", """Jetzt das Audio dazu.

> **Sobald du eigene Aufnahmen hast: `--tts` durch `--audio-map map.jsonl`
> ersetzen.** Das ist der groesste einzelne Qualitaetssprung."""),
        ("code", '''!python scripts/build_toolcall_dataset.py \
    --utterances "$UTTERANCES" \
    --tts \
    --fmt "$FORMAT" \
    --val-ratio 0.05 \
    --eval-out "$EVAL_FILE" \
    --out "$DATASET_DIR"'''),
        ("md", """## 2. Preprocessen

Wandelt Audio + Zieltext in das Tensor-Format, das `liquid_audio`'s Trainer
erwartet. Der System-Prompt ist im Skript auf `"Perform ASR."` gepinnt — mit
Kommentar, warum."""),
        ("code", '''PREPROC = DRIVE_DIR / "data" / "agent_audio" / "train"

!python scripts/preprocess_audio_toolcalls.py \\
    --dataset "$DATASET_DIR" \\
    --output-path "$PREPROC" \\
    --max-context-length 512 \\
    --device cuda'''),
        ("md", """## 3. Boden messen (Baseline)

**Diesen Schritt nicht überspringen.** Das untrainierte Modell transkribiert —
Erwartung sind 0 % über alle drei Metriken. Das ist der Beweis, dass Fine-Tuning
Voraussetzung ist und nicht Optimierung."""),
        ("code", '''import io, json
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_from_disk

from muscal_agent.agent import AudioAgent

VAL_DIR = DRIVE_DIR / "data" / "agent_audio_val"
val = load_from_disk(str(VAL_DIR))
print("val samples:", len(val))

def predict(model_id: str, rows, out_file: str, limit: int = 200):
    agent = AudioAgent(model_id=model_id, fmt=FORMAT, answer_model_id=None)
    preds = []
    for row in rows.select(range(min(limit, len(rows)))):
        audio = row["audio_chat"][1]["content"][0]["audio"]
        if isinstance(audio, dict):           # datasets kann Audio als dict geben
            audio = audio["bytes"]
        wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
        preds.append(agent.speech_to_text(torch.from_numpy(wav).unsqueeze(0), sr))
    Path(out_file).write_text("\\n".join(p.replace("\\n", " ") for p in preds), encoding="utf-8")
    return preds

baseline = predict("LiquidAI/LFM2.5-Audio-1.5B", val, "baseline_preds.txt")
for row in baseline[:5]:
    print("  ", row)'''),
        ("code", '''!python scripts/eval_toolcalls.py \\
    --gold "$EVAL_FILE" \\
    --pred baseline_preds.txt --pred-raw \\
    --fmt "$FORMAT"'''),
        ("md", """## 4. Trainieren

**Volles Finetune auf einer A100.** Hyperparameter aus Liquid's Referenzlauf
(55k Paare, 41 Funktionen): context 512, batch 32, warmup 250, lr 5e-5.
Bei einem kleineren Katalog tun es 1.000–2.000 Steps."""),
        ("code", '''from pathlib import Path

from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.trainer import Trainer

OUT_DIR = DRIVE_DIR / "outputs" / "audio-agent"

train_data = LFM2DataLoader(dataset_path=str(PREPROC), context_length=512)

trainer = Trainer(
    model_id="LiquidAI/LFM2.5-Audio-1.5B",
    train_data=train_data,
    lr=5e-5,
    batch_size=32,
    max_steps=2000,
    warmup_steps=250,
    dataloader_num_workers=2,
    logging_interval=10,
    save_interval=500,
    val_interval=200,
    output_dir=str(OUT_DIR),
)
trainer.train()'''),
        ("md", """## 5. Danach messen

Gleicher Lauf wie in Schritt 3, nur mit dem trainierten Checkpoint. Trage die
Zahlen in `configs/audio_agent_ft.yaml` unter `eval:` ein, damit der Fortschritt
festgehalten ist."""),
        ("code", '''# Pfad zum Checkpoint aus Schritt 4 (Trainer schreibt outputs/checkpoint/...)
CKPT = OUT_DIR / "checkpoint"
print("checkpoint:", CKPT, "| exists:", CKPT.exists())

# Entweder die Gewichte zurueckladen oder -- falls schon als GGUF exportiert --
# den llama-liquid-audio-server dagegen laufen lassen. Fuer den PyTorch-Pfad:
trained = predict(str(CKPT), val, "trained_preds.txt") if CKPT.exists() else []
for row in trained[:5]:
    print("  ", row)'''),
        ("code", '''!python scripts/eval_toolcalls.py \\
    --gold "$EVAL_FILE" \\
    --pred trained_preds.txt --pred-raw \\
    --fmt "$FORMAT"'''),
        ("md", """## 6. In Betrieb

Der trainierte Checkpoint muss in den GGUF-Vierersatz (Modell, `mmproj`,
`vocoder`, `tokenizer`) und läuft dann im `llama-liquid-audio-server`. Vorgehen
wie in Schritt 3 von Liquid's
[Voice-Assistant-Beispiel](https://github.com/Liquid4All/cookbook/tree/main/examples/voice-assistant).

**Sicherheit** — Desktop-Steuerung ist hier die einzige Komponente mit
Schadenspotenzial. Drei Bremsen, Default ist „ändert nichts":

| Ebene | Schalter |
|---|---|
| Risikoklasse | `MUSCAL_AGENT_WRITE=1`, `MUSCAL_AGENT_DESTRUCTIVE=1` |
| Dry-Run | `MUSCAL_AGENT_DRY_RUN=0` (sonst wird nur ausgegeben) |
| Allow-List / Region | `MUSCAL_AGENT_TOOLS=web_search,web_read`, `MUSCAL_SCREEN=x,y,w,h` |

Erste Läufe in einer VM. Ein 1.5B-Modell verhört sich — „strg c" statt „strg v"
ist ein reales Szenario."""),
    ]


def main() -> None:
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    notebooks = {
        "01_lfm_text_qlora.ipynb": text_notebook(),
        "02_lfm_vl_qlora.ipynb": vl_notebook(),
        "03_lfm_moe_qlora.ipynb": moe_notebook(),
        "04_lfm_audio_ft.ipynb": audio_notebook(),
        "05_audio_agent_toolcall.ipynb": agent_notebook(),
    }
    for name, cells in notebooks.items():
        nbf.write(nb(cells), str(out_dir / name))
        print("wrote", out_dir / name)


if __name__ == "__main__":
    main()
