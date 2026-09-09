# LFM2.5-ColBERT-350M

LFM2.5-ColBERT-350M ist ein Late-Interaction-Retrieval-Modell. Anders als ein
Bi-Encoder erzeugt es nicht einen Vektor pro Dokument, sondern einen
128-dimensionalen Vektor pro Token. Die Ähnlichkeit zwischen Anfrage und
Dokument berechnet sich über MaxSim: für jeden Query-Token wird der beste
Treffer unter den Dokument-Token gesucht und aufsummiert.

Das verbessert die Trefferqualität und die Generalisierung deutlich, kostet aber
Indexgröße: statt eines Vektors pro Dokument werden so viele Vektoren gespeichert,
wie das Dokument Token hat. Für sehr große Korpora ist deshalb ein zweistufiger
Aufbau üblich: ein dichter Retriever liefert Kandidaten, ColBERT sortiert sie neu.

Die Dokumentlänge beträgt 512 Tokens, Anfragen werden auf 32 Tokens begrenzt.
Auch dieses Modell deckt elf Sprachen ab, darunter Deutsch.

Für Indexierung, Suche und Reranking wird PyLate verwendet. Die Klasse
models.ColBERT übernimmt das Kodieren, indexes.PLAID baut den Index, und
rank.rerank sortiert eine Kandidatenliste neu. Wer nur Reranking braucht, kann
ColBERT allein dafür einsetzen und den ersten Retriever unverändert lassen.
