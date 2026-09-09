# Audio-Agent: Sprache → Tool-Call

Ein Assistent, dem du zurufst, was er tun soll, und der direkt Funktionen aufruft:
Web-Recherche, Browser-Steuerung, Desktop-Aktionen. Kein STT → LLM → TTS-Dreisprung,
sondern **ein** feingetuntes `LFM2.5-Audio-1.5B`, das aus Audio direkt den
Funktionsaufruf macht.

```
 Mikrofon (24 kHz)
        │
        ▼
 LFM2.5-Audio-1.5B  (feingetunt, System-Prompt "Perform ASR.")
        │
        ▼
 "web_search|query=liquid ai lfm"
        │
        ▼
 muscal_agent.executor ── Policy ──► web / browser / desktop
        │
        ▼
 JSON-Ergebnis ──► LFM2.5-1.2B-Instruct (ein gesprochener Satz) ──► TTS ──► Lautsprecher
```

Warum die Antwort von einem *zweiten* Modell kommt: das Fine-Tune bringt dem
Audio-Modell genau ein Verhalten bei — Sprache zu Funktionsaufruf. Dieselben
Gewichte sollen nicht zusätzlich über Ergebnisse plaudern; das verwässert das
Fine-Tune. Einen JSON-Brocken in einen Satz zu fassen, kann ein kleines
Instruct-Modell besser.

## Die drei Dinge, die man falsch macht

### 1. Der System-Prompt ist „Perform ASR." — und das ist kein Versehen

`llama-liquid-audio-server` akzeptiert nur eine geschlossene Allow-List von
System-Prompts (`"Perform ASR."`, die `"Perform TTS. ..."`-Varianten, die
interleaved-Variante) und **erzwingt** zur Inferenz einen davon. Trainierst du
mit einem eigenen Prompt, schiebt der Server das Modell zurück in den
Transkriptionsmodus und das Fine-Tune ist wirkungslos.

Liquid hat das empirisch belegt: 500 Steps ohne System-Prompt lieferten in
PyTorch korrekte Funktionsaufrufe — über den GGUF-Server kamen nur
Transkriptionen zurück. Deshalb steht in
[`scripts/preprocess_audio_toolcalls.py`](../scripts/preprocess_audio_toolcalls.py)
der Prompt hart auf `"Perform ASR."`.

### 2. Audio-Finetuning ist ein volles Finetune, kein LoRA

`liquid_audio.trainer.Trainer` hat keinen LoRA-Schalter. Einzuplanen: A100-80GB
(Colab Pro) oder Modal. Referenz-Setup aus Liquid's Voice-Assistant-Rezept:

| Knopf | Wert |
|---|---|
| context length | 512 |
| batch size | 32 |
| max steps | 1.000 (Referenz) … 10.000 |
| warmup steps | 250 |
| lr | 5e-5 |

### 3. Das Tool-Call-Format wird *vor* der Datenerzeugung festgelegt

Zur Wahl stehen `pipe` (kompakt, Empfehlung) und `pythonic` (LFM2.5-Standard mit
`<\|tool_call_start\|>`):

```
pipe      web_search|query=liquid ai lfm|max_results=5
pythonic  <|tool_call_start|>[web_search(query="liquid ai lfm", max_results=5)]<|tool_call_end|>
```

Das Format ist die Trainingsziel-Syntax. Umstellen heißt: Daten neu erzeugen und
neu trainieren.

## Tool-Katalog

| Tool | Risiko | Funktion |
|---|---|---|
| `web_search` | read | Websuche (SearxNG, DuckDuckGo oder eigene API) |
| `web_read` | read | Seite holen, Text extrahieren |
| `browser_snapshot` | read | Aktuelle Seite als Text (Accessibility-Tree) |
| `browser_open` | write | URL im Browser öffnen |
| `browser_click` | write | Element per CSS-Selektor klicken |
| `browser_type` | write | Text eingeben, optional Enter |
| `desktop_screenshot` | read | Screenshot (mss/pyautogui) |
| `desktop_launch` | write | Anwendung starten |
| `desktop_type` | write | Tastatureingabe ins fokussierte Fenster |
| `desktop_hotkey` | write | Tastenkürzel, z. B. `ctrl+c` |
| `desktop_click` | **destructive** | Klick auf absolute Koordinaten |

`browser_snapshot` liefert den **Accessibility-Tree, keine Pixel** — ein 1.5B-Modell
kann keine Screenshots lesen, ein A11y-Baum ist billiger in Tokens und zum Handeln
verlässlicher.

## Trainingdaten

Der Generator kombiniert **Satzschablonen × Slot-Werte** — damit kommt man ohne
Handarbeit auf ein vollständiges Korpus für alle Werkzeuge:

```bash
# 1. Paare erzeugen (kein Audio nötig, --write-pairs stoppt danach)
python scripts/build_toolcall_dataset.py \
    --language en \
    --slots data/agent_slots_en.json \
    --per-tool 60 \
    --write-pairs data/agent_utterances_full_en.jsonl

# 2. Audio dazu + Dataset bauen. Empfohlen: eigene Aufnahmen (--audio-map).
python scripts/build_toolcall_dataset.py \
    --utterances data/agent_utterances_full_en.jsonl \
    --audio-map data/audio_map.jsonl \
    --val-ratio 0.05 --eval-out data/agent_eval.jsonl \
    --out data/agent_audio

# 3. In das Tensor-Format des Trainers
python scripts/preprocess_audio_toolcalls.py \
    --dataset data/agent_audio --output-path data/agent_audio/train
```

Mitgelieferte Bausteine und das Ergebnis:

| Datei | Inhalt |
|---|---|
| `data/agent_slots_de.json` / `_en.json` | Werte pro Argument (URLs, Selektoren, Apps, Hotkeys …) |
| `data/agent_templates_de.json` / `_en.json` | Satzschablonen pro Werkzeug |
| `data/agent_utterances_full_de.jsonl` | **601 Paare**, alle 11 Werkzeuge |
| `data/agent_utterances_full_en.jsonl` | **591 Paare**, alle 11 Werkzeuge |

Verteilung im deutschen Korpus (pro Werkzeug): `web_search` 61, `web_read` 62,
`browser_open` 61, `browser_click` 61, `browser_type` 61, `browser_snapshot` 27,
`desktop_launch` 61, `desktop_type` 60, `desktop_hotkey` 62, `desktop_click` 60,
`desktop_screenshot` 25.

Die zwei argumentlosen Werkzeuge (`browser_snapshot`, `desktop_screenshot`)
kommen nur auf ~25, weil ihre Variation allein aus Formulierungen stammt — bei
Bedarf Schablonen ergänzen.

**Eine Formatregel, die beim Erzeugen auffällt:** im `pipe`-Format darf ein Wert
kein `|` enthalten (das trennt Argumente). `=` ist erlaubt — es wird nur am
*ersten* `=` getrennt, deshalb überleben URLs mit Query-String
(`...?q=lfm`) und CSS-Attributselektoren (`input[name=q]`) den Round-Trip.

**Qualität vor Volumen — aber nicht zu wenig.** Liquid's Referenz: 55.302 Paare
auf 41 Funktionen (~1.350 pro Funktion). Für den Start reicht ein kleinerer,
scharfer Katalog: 3–4 Funktionen mit je ~500 Beispielen schlagen 12 Funktionen
mit je 80. Danach erweitern.

Zwei Fallen bei `--tts`: jede Probe hat dieselbe Stimme (das Modell lernt eine
Sprecherin, nicht deine), und der TTS-Prompt erzeugt Prosodie, die echte
Spontansprache nicht hat. Mindestens einen Teil real aufnehmen.

**Sprache:** `LFM2.5-Audio-1.5B` ist auf Englisch trainiert. Deutsche Kommandos
funktionieren eher als *Absichtserkennung* denn als Diktat — mit genug gepaarten
Daten brauchbar, aber die Fehlerrate ist höher als bei englischen Kommandos. Die
Beispieldatei liegt auf Deutsch; für maximale Trefferquote englische Kommandos
verwenden.

## Evaluieren — erst den Boden messen

```bash
# Modell VOR dem Fine-Tune durch die gehaltenen Beispiele jagen (Erwartung: 0 %)
python scripts/eval_toolcalls.py --gold data/agent_eval.jsonl --pred baseline.txt --pred-raw
```

Drei Metriken, jeweils eine Teilmenge der vorherigen:

1. **Format-Compliance** — parst, Funktion existiert, Pflicht-Argumente da.
2. **Funktionsname** — und es ist die richtige Funktion.
3. **Argumente** — und alle Argumente stimmen (normalisiert).

Liquid's Baseline-Wert für das untrainierte Modell: **0 / 0 / 0** — es
transkribiert brav. Genau das ist der Sinn der Messung: sie beweist, dass
Fine-Tuning nicht Optimierung, sondern Voraussetzung ist.

## Sicherheit — der Teil, den man nicht überspringt

Desktop-Steuerung ist die einzige Komponente hier, die echten Schaden anrichten
kann. Drei unabhängige Bremsen, alle in `muscal_agent/executor.py`:

1. **Risikoklassen.** `read` läuft immer, `write` braucht `MUSCAL_AGENT_WRITE=1`,
   `destructive` (blinde Koordinaten-Klicks) braucht `MUSCAL_AGENT_DESTRUCTIVE=1`.
2. **Dry-Run ist Default.** Ohne `MUSCAL_AGENT_DRY_RUN=0` wird nichts ausgeführt,
   sondern ausgegeben, was passieren *würde*.
3. **Allow-List + Region.** `MUSCAL_AGENT_TOOLS=web_search,web_read` begrenzt den
   Katalog, `MUSCAL_SCREEN=x,y,w,h` begrenzt Klick-Koordinaten.

Vorher prüfen, was die Maschine überhaupt kann:

```bash
python -m muscal_agent --doctor
```

Meldet Plattform, Sitzungstyp (x11/wayland) und welche Backends installiert sind
— inklusive Hinweis, dass `desktop_*` unter Wayland mit pyautogui nicht geht.

```bash
# Nur suchen und lesen, nichts ausfuehren:
python -m muscal_agent --call "web_search|query=liquid ai lfm"

# Alles erlaubt, nichts simuliert:
MUSCAL_AGENT_DRY_RUN=0 MUSCAL_AGENT_WRITE=1 MUSCAL_AGENT_DESTRUCTIVE=1 MUSCAL_DESKTOP=1 \
    python -m muscal_agent --mic
```

Erste Schritte immer in einer VM oder auf einem Wegwerf-Profil — ein
feingetuntes 1.5B-Modell verhört sich, und „strg c" in „strg v" verwechselt ist
ein echtes Szenario.

**Plattform-Wirklichkeit:** pyautogui funktioniert auf Windows, macOS und Linux/X11.
Unter **Wayland** nicht — dort `ydotool`/`wtype` oder eine X11-Session nutzen. Für
mehr als Spielerei lieber Accessibility-APIs statt Koordinaten: `uiautomation`
(Windows), `pyatspi` (Linux), AX über PyObjC (macOS).

## In Betrieb nehmen

Der feingetunte Checkpoint muss in den GGUF-Vierersatz (Modell, `mmproj`,
`vocoder`, `tokenizer`) und läuft dann in `llama-liquid-audio-server`. Liquid's
Referenz-Repo: [`Paulescu/LFM2.5-Audio-1.5B-OHF-Voice-GGUF`](https://huggingface.co/Paulescu/LFM2.5-Audio-1.5B-OHF-Voice-GGUF),
das Vorgehen in Schritt 3 ihres
[Voice-Assistant-Beispiels](https://github.com/Liquid4All/cookbook/tree/main/examples/voice-assistant).

Zum Testen ohne GGUF reicht der PyTorch-Pfad:

```bash
python -m muscal_agent --wav command.wav --speak
```
