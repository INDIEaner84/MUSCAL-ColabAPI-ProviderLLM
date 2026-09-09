# RAG-Baseline messen

Ziel dieses Harness: **bevor** irgendetwas trainiert wird, mit Zahlen beantworten,
ob das Problem überhaupt am Modell liegt. Retrieval misst man in Minuten auf der
CPU — ein Adapter kostet Stunden und ist womöglich die falsche Baustelle.

## Schnellstart

```bash
pip install sentence-transformers

# 1. Index bauen (CPU reicht)
python -m muscal_rag build --config configs/rag_baseline.yaml

# 2. Retrieval messen
python -m muscal_rag eval --config configs/rag_baseline.yaml

# 3. Einzelne Frage ansehen
python -m muscal_rag search " Welcher System-Prompt ist beim Audio-Finetuning zwingend?" \
    --config configs/rag_baseline.yaml

# 4. Geerdete Antwort erzeugen (braucht GPU, LFM2.5-1.2B-Instruct)
python -m muscal_rag answer "Welche LoRA-Ziele sind empfohlen?" --config configs/rag_baseline.yaml
```

**Ohne Modell-Download und ohne GPU** — nur um Chunking, Index und Metriken zu
prüfen:

```bash
python -m muscal_rag eval --config configs/rag_baseline.yaml --backend hashing
```

Das `hashing`-Backend ist ein deterministischer Bag-of-Words-Hasher. Es hat keine
Semantik und dient ausschließlich dem Pipeline-Test; der CLI warnt
entsprechend. Zahlen daraus sind keine Qualitätsaussage.

## Eval-Set anlegen

`data/rag_eval_example.jsonl` zeigt das Format:

```json
{"question": "Was ist Router Collapse beim MoE-Finetuning?", "relevant": ["moe-8b-a1b"]}
{"question": "Wie lang darf ein Dokument maximal sein?", "relevant": ["lfm25-embedding", "lfm25-colbert"]}
```

* `relevant` enthält **Dokument-IDs** (Dateiname ohne Endung). Wer feiner
  auflösen will, hängt die Chunk-Nummer an: `"moe-8b-a1b#2"`. Ohne `#` zählt
  jeder Chunk des Dokuments als Treffer.
* 30–50 Fragen sind genug. Sie müssen aus **echten** Anfragen stammen, nicht
  ausgedacht sein — sonst misst man die eigene Fantasie.
* Einmal erzeugt, wird das Set eingefroren. Spätere Änderungen machen Vergleiche
  wertlos.
* Die Chunk-Nummern findet man mit `python -m muscal_rag search "<frage>"`.

## Zahlen lesen

Die Ausgabe:

```
Fragen: 16

   k    recall@k
   1      62.5%
   3      93.8%
   5     100.0%

MRR      0.807
hit rate 100.0%
```

* **recall@k** — Anteil der Fragen, bei denen mindestens eine relevante Stelle
  unter den ersten k Treffern ist.
* **MRR** — mittlerer Kehrwert der Trefferposition. 1.0 = immer auf Platz 1.
* **hit rate** — Anteil der Fragen mit überhaupt einem Treffer.

Warnung, die der Harness selbst ausgibt: ist `k` größer als die Zahl der Chunks,
ist recall@k aussagelos (man findet einfach alles). Das Beispielkorpus hat nur
6 Chunks und dient ausschließlich als Format-Demo.

### Faustregeln für die Entscheidung

| recall@5 | MRR | Bedeutung | Nächster Schritt |
|---|---|---|---|
| ≥ 0.90 | ≥ 0.7 | Retrieval ist nicht das Problem | Antworten trotzdem falsch? → **Verhaltensproblem**: Prompt oder Reader-Adapter (Zitierdisziplin) |
| 0.70–0.90 | 0.4–0.7 | Retrieval verbesserungsfähig | Chunkgröße, `top_k`, ColBERT-Reranking |
| < 0.70 | < 0.4 | Retrieval ist die Baustelle | Dokumentqualität, Chunking, Sprachpassung des Embeddings |

Ein Adapter am Generator repariert **kein** schlechtes Retrieval. Wenn die
richtige Passage nicht im Kontext landet, kann das Modell sie nicht zitieren.

## Stellschrauben (in dieser Reihenfolge)

1. **Chunkgröße** (`chunking.chunk_size`, Default 512 = Dokumentlänge des
   Embedding-Modells). Zu groß → verwässerte Vektoren; zu klein → Kontext fehlt.
2. **Überlappung** (`overlap`, Default 64) — verhindert verlorene Aussagen an
   Chunk-Grenzen.
3. **top_k** (`retrieval.top_k`) — mehr Kontext, mehr Rauschen.
4. **Reranking** (`retrieval.rerank_model: LiquidAI/LFM2.5-ColBERT-350M`) —
   der stärkste Hebel, wenn die Metriken mittelmäßig sind:

   ```bash
   pip install pylate
   ```

## Vom Retrieval zum Adapter

Erst wenn das Retrieval sitzt und die Antworten trotzdem nicht taugen, ist es ein
Verhaltensproblem. Dann ist der höchste ROI-Adapter nicht „mehr Wissen", sondern
**Zitierdisziplin**: nur aus dem Kontext antworten, Quellen-IDs nennen, bei
Lücken „weiß ich nicht" sagen.

Der Prompt dafür liegt in `configs/rag_baseline.yaml` unter
`generation.system_prompt`, die gerenderte Version erzeugt
`muscal_rag.pipeline.grounded_prompt()` — genau diese (Frage, Kontext, Antwort)-
Paare sind später die Trainingsdaten für den Reader-Adapter.

## Dateien

| Pfad | Zweck |
|---|---|
| `muscal_rag/chunking.py` | Chunks mit Überlappung, paragraph-basiert |
| `muscal_rag/documents.py` | md/txt/html, pdf mit `pypdf` |
| `muscal_rag/embed.py` | sentence-transformers-Backend + hashing-Notbehelf |
| `muscal_rag/index.py` | numpy-Cosine-Index, als npz + json gespeichert |
| `muscal_rag/metrics.py` | recall@k, MRR, hit rate |
| `muscal_rag/rerank.py` | ColBERT/MaxSim-Reranking über PyLate |
| `muscal_rag/pipeline.py` | build / search / evaluate / grounded_prompt |
