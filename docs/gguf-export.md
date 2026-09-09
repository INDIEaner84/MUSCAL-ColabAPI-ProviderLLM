# Vom Adapter zum laufenden Modell

Ein LoRA-Adapter ist nur eine Handvoll MB, aber ohne Basis-Modell nutzlos. Für den
Einsatz (llama.cpp, Ollama, LM Studio, Atomic Chat, iOS/Android) führt man ihn mit
dem Basis-Modell zusammen und wandelt das Ergebnis nach GGUF.

## 1. Mergen

Lokal oder in Colab:

```bash
python scripts/merge_adapter.py \
    --base LiquidAI/LFM2.5-1.2B-Instruct \
    --adapter outputs/lfm25-1.2b-text/adapter \
    --out    outputs/lfm25-1.2b-text/merged \
    --track  text
```

Warum CPU? Das *quantisierte* Trainingsmodell darf nicht gemergt werden — der Merge
lädt die Basisgewichte daher noch einmal in bf16/fp16. Auf der GPU würde das eine
15-GB-T4 sprengen, auf der CPU dauert es bei 1.2B wenige Minuten.

Ergebnis ist ein normales HF-Modellverzeichnis, das man direkt mit
`AutoModelForCausalLM.from_pretrained(...)` laden kann.

## 2. GGUF

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
pip install -q -r llama.cpp/requirements.txt

python llama.cpp/convert_hf_to_gguf.py outputs/lfm25-1.2b-text/merged \
    --outfile lfm25-1.2b-muscal-f16.gguf --outtype f16

# danach quantisieren (Q4_K_M ist der beste Größe/Qualität-Kompromiss)
./llama.cpp/build/bin/llama-quantize lfm25-1.2b-muscal-f16.gguf \
    lfm25-1.2b-muscal-Q4_K_M.gguf Q4_K_M
```

Oder aus dem Notebook heraus: `muscal_lfm.export.export_gguf(...)` macht genau
diese Schritte (inkl. Clone), siehe `configs/*.yaml` → `export.save_gguf`.

**Wichtig:** LFM2.5 ist eine junge Architektur — der Converter braucht ein
aktuelles llama.cpp. Bei „unknown model architecture" erst `git pull` in
`llama.cpp`, bevor irgendwas anderes debuggt wird.

## 3. Betreiben

```bash
# llama.cpp
./llama.cpp/build/bin/llama-server -m lfm25-1.2b-muscal-Q4_K_M.gguf -c 8192

# Ollama: Modelfile
#   FROM ./lfm25-1.2b-muscal-Q4_K_M.gguf
#   PARAMETER temperature 0.1
#   PARAMETER min_p 0.15
ollama create muscal-lfm -f Modelfile
```

Empfohlene Sampling-Parameter laut Liquid: `temperature 0.1`, `min_p 0.15`,
`repetition_penalty 1.05`. Für die LFM2.5-VL-Modelle zusätzlich
`min_image_tokens=64`, `max_image_tokens=256`, `do_image_splitting=true`.

Für Audio-GGUFs gibt es eigene Runner (`llama-liquid-audio-cli`,
`llama-liquid-audio-server`), die zusätzlich `mmproj-`, `vocoder-` und
`tokenizer`-Dateien brauchen — Details in der
[Liquid-Doku](https://docs.liquid.ai/lfm/models/lfm25-audio-1.5b).

## 4. Ohne Merge: Adapter direkt laden

Zum Weitertrainieren oder für Serving mit Adapter-Support (vLLM, TRL) reicht der
Adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "outputs/lfm25-1.2b-text/adapter")
```
