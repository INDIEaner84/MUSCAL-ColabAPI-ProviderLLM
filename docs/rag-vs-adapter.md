# RAG oder Adapter? Und welcher lohnt sich wo

Die teuerste Fehlentscheidung ist nicht das falsche Modell, sondern ein Adapter
für ein Problem, das Retrieval billiger löst — oder ein RAG-System, das an einem
Verhaltensproblem nichts ändert.

## 1. Drei Aufgaben, drei Werkzeuge

| Was fehlt | Richtiges Werkzeug | Warum |
|---|---|---|
| **Wissen**, das sich ändert (Dokumente, Preise, Tickets, Code) | **RAG** | Ein Adapter friert Wissen ein und veraltet. Retrieval bleibt aktuell. |
| **Verhalten**: Format, Stil, Tool-Calls, Klassengrenzen | **Adapter (LoRA/QLoRA)** | Das ist das, was Fine-Tuning kann und Prompting irgendwann nicht mehr. |
| **Fähigkeit**, die das Basismodell nicht hat (z. B. Audio → Funktionsaufruf) | **Fine-Tuning, alternativlos** | Liquid's Baseline: 0 % über alle Metriken — kein Prompting erreicht das. |

Der Merksatz: **Retrieval liefert Wissen, ein Adapter erzwingt Verhalten.** Ein
Generator-Adapter macht schlechtes Retrieval nicht besser — wenn die richtige
Stelle nicht im Kontext landet, kann das Modell sie nicht zitieren.

## 2. Erster Check: gibt es das schon als Nano?

Liquid liefert fertig feingetunte Spezialmodelle. Ein Nano, das passt, kostet
**null Trainingszeit**.

| Nano | Aufgabe | Wann nehmen |
|---|---|---|
| `LFM2.5-Embedding-350M` | Dense Retrieval | Kleiner, schneller Vektorindex, 11 Sprachen (inkl. Deutsch) |
| `LFM2.5-ColBERT-350M` | Retrieval + Reranking | Qualität wichtiger als Indexgröße |
| `LFM2.5-Encoder-350M` / `-230M` | Klassifikation | Textklassifikation, kein Generieren nötig |
| `LFM2.5-VL-450M/1.6B-Extract` | Bild → JSON | Felder aus Bildern/Dokumenten extrahieren |
| `LFM2-2.6B-Transcript` | Zusammenfassen | Transkripte, Meetings |
| `LFM2-350M-ENJP-MT` | Übersetzung | Japanisch ↔ Englisch |
| `LFM2-350M-Math` | Rechnen | Kleine Reasoning-Aufgaben |

Erst wenn keins passt, lohnt ein eigener Adapter.

## 3. Der LFM-RAG-Stack

```
Dokumente → Chunks (≤512 Tokens)
              │
              ▼
   LFM2.5-Embedding-350M  (1024-dim, Cosine, 11 Sprachen)
              │
              ▼
        Vektorindex  ──► Top-k
              │
              ▼
   LFM2.5-ColBERT-350M  (MaxSim-Reranking, optional)
              │
              ▼
   LFM2.5-1.2B-Instruct  (Antwort nur aus dem Kontext)
```

Warum diese Bausteine zusammenpassen: Embedding und Generator kommen aus
derselben Modellfamilie, sind beide klein genug für on-device, und beide gibt es
als GGUF. Der Embedding-Encoder verträgt 512 Tokhen Dokumentlänge — das ist
gleichzeitig eine sinnvolle Chunkgröße.

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("LiquidAI/LFM2.5-Embedding-350M", trust_remote_code=True)
doc_vecs = encoder.encode(chunks, prompt_name="document", normalize_embeddings=True)
query_vec = encoder.encode([frage], prompt_name="query", normalize_embeddings=True)
scores = query_vec @ doc_vecs.T          # Cosine, weil normalisiert
top_k = scores[0].argsort()[-5:][::-1]
```

`prompt_name="query"` / `"document"` ist nicht optional — das Modell wurde mit
asymmetrischen Prompt-Präfixen trainiert.

**Der höchste ROI-Adapter im RAG-Kontext** ist nicht „mehr Wissen", sondern
*Zitierdisziplin*: ein 1.2B-Adapter darauf trainiert, nur aus dem Kontext zu
antworten, Fundstellen zu nennen und bei fehlender Information „weiß ich nicht"
zu sagen. Wissen kommt aus dem Index, Verhalten aus dem Adapter — genau die
Aufgabenteilung von oben.

## 4. Lohnt sich ein Adapter? Pro Modell

| Modell | Adapter sinnvoll? | Wofür / Wann |
|---|---|---|
| `LFM2.5-230M` / `350M` | **Ja, eng** | Routing, Klassifikation, feste Formate. Minuten auf der T4. Sprachqualität ist die Grenze. |
| `LFM2.5-1.2B-Instruct` | **Bester ROI** | Default: Stil, Format, Tool-Calling, Extraktion. QLoRA auf der freien T4. |
| `LFM2.5-1.2B-Thinking` | Bedingt | Nur bei echtem Reasoning; dann eher GRPO mit regelbasierten Rewards als SFT. |
| `LFM2.5-2.6B` | Wenn 1.2B nicht reicht | Der eigentliche Grund ist der **128K-Kontext** (lange Dokumente im RAG). |
| `LFM2.5-8B-A1B` (MoE) | **Vorsicht** — siehe unten | Nur wenn 2.6B *messbar* zu schwach ist. |
| `LFM2.5-VL-450M` / `1.6B` | **Ja** | Narrow use cases; Liquid empfiehlt Feintuning ausdrücklich. |
| `LFM2.5-VL-3B` | Mit L4/A100 | Nur wenn die kleinen VLMs an der Aufgabe scheitern. |
| `LFM2.5-Audio-1.5B` | Nur für Tool-Calling | Kein LoRA möglich → volles Finetune, A100. |
| Encoder / Embedding / ColBERT | Eigene Ökosysteme | Transformers / sentence-transformers / PyLate — **nicht** über TRL. |

## 5. Der MoE-Sonderfall (`LFM2.5-8B-A1B`)

Vier Punkte machen MoE-Adapter teurer, als „1.5B aktive Parameter" klingt:

**Speicher.** Alle 8.3B Gewichte müssen resident sein (~5.5 GB in 4-bit). Auf
einer 15-GB-T4 bleibt kaum Luft → A100/L4.

**Parameter-Explosion.** `gate_proj`/`up_proj`/`down_proj` matchen per Suffix
*jeden* der 32 Experten. Der Adapter ist dadurch deutlich größer als beim dichten
Modell gleicher Zielgröße.

**Router-Risiken.** Beim Feintunen von MoE-Modellen sind Router Collapse
(>90 % der Tokens auf 1–2 Experten), Oszillation bei zu hoher LR auf den
Router-Gewichten und Qualitätsverlust nach dem Merge durch Aux-Loss-Mismatch
bekannte Fehlerbilder. Gegenmittel: niedrigere LR (1e-5 … 1e-4 statt 2e-4),
Balance-Loss im Auge behalten, Daten nicht einseitig (Sprache/Domäne) mischen.
Nach dem Merge **immer** gegen das Basismodell gegenmessen.

**Serving ist der eigentliche Engpass.** Multi-LoRA für MoE gibt es in vLLM ab
0.15 über einen `fused_moe_lora`-Kernel — bestätigt für GPT-OSS, Qwen3-MoE,
DeepSeek und Llama MoE. **LFM2.5 steht nicht auf dieser Liste.** vLLM hat
historisch Adapter mit Expertengewichten für MoE abgelehnt („ensure that the
loaded LoRA model does not contain expert weights"). Praktische Konsequenz:

* Variante A (servefreundlich): LoRA nur auf **Attention** (`q/k/v/o`) — keine
  Expertengewichte im Adapter, hot-swap-fähig, etwas weniger Anpassung.
* Variante B (maximale Anpassung): auch Experten-MLPs, dann **mergen** statt
  hot-swappen und ein eigenes Modell deployen.

Empfehlung: mit A starten, B nur wenn A die Messlatte nicht erreicht.

## 6. Entscheidungsprozedur (in dieser Reihenfolge)

1. **Nano-Check** — erledigt ein fertiges Modell die Aufgabe? (Tabelle oben)
2. **Baseline ohne Training**: Prompt + RAG mit `Embedding-350M`. Auf **30–50
   echten Fällen** messen, nicht auf Gefühl.
3. **Retrieval isoliert messen** (recall@k). Ist das Retrieval schlecht, hilft
   kein Generator-Adapter — dann Chunking oder ColBERT-Reranking reparieren.
4. **Erst dann Adapter**, und zwar auf dem **kleinsten** Modell, das Schritt 2
   knapp verfehlt: 350M → 1.2B → 2.6B → 8B-A1B.
5. **Gegenmessung**: Adapter gegen Baseline auf dem eingefrorenen Held-out-Set.
   Ohne Zahlen ist „besser" eine Meinung.

## 7. Grobe Kosten

| Schritt | Hardware | Zeit |
|---|---|---|
| Retrieval aufbauen (Index + Eval) | CPU / T4 | 1–2 h |
| QLoRA 350M / 1.2B, 1–2k Beispiele | **T4 free** | 15–60 min |
| QLoRA 2.6B | T4 / L4 | 1–3 h |
| QLoRA 8B-A1B | A100 / L4 | 2–6 h (+ Merge-Risiko) |
| VLM QLoRA 450M / 1.6B | T4 | 1–3 h |
| Audio Tool-Calling (volles FT) | A100 | ~20 min Referenzlauf, Datenbeschaffung dominiert |

## 8. Für den Audio-Agent bedeutet das

RAG ersetzt das Audio-Finetune **nicht** (das ist Fähigkeit, nicht Wissen), aber
der *Web-Recherche*-Teil des Agenten ist letztlich Retrieval. Sobald die Fragen
sich auf **eigene** Dokumente beziehen statt aufs offene Web, ist
`Embedding-350M` + RAG billiger und treffsicherer als `web_search` + Seiten
auslesen. Der Agent behält dann `web_search` für Aktualität und bekommt einen
zweiten, lokalen Retrieval-Pfad dazu.

## Quellen

* [Liquid Nanos](https://docs.liquid.ai/lfm/models/liquid-nanos) — fertige Spezialmodelle
* [LFM2.5-Embedding-350M](https://docs.liquid.ai/lfm/models/lfm25-embedding-350m) — 1024-dim, 512 Tokens, 11 Sprachen
* [LFM2.5-ColBERT-350M](https://docs.liquid.ai/lfm/models/lfm25-colbert-350m) — MaxSim, PyLate, Reranking
* [Fine-Tuning-Übersicht](https://docs.liquid.ai/lfm/fine-tuning/overview) — „Before you fine-tune: exhaust the cheaper levers"
* [vLLM Multi-LoRA für MoE](https://blog.vllm.ai/2026/02/26/multi-lora.html) — ab 0.15, `fused_moe_lora`, gelistete MoE-Familien
* [SGLang-Issue zu MoE-LoRA](https://github.com/sgl-project/sglang/issues/9897) — Zitat der vLLM-Einschränkung bei Expertengewichten
