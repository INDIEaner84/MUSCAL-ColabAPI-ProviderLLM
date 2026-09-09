# LFM2.5-Embedding-350M

LFM2.5-Embedding-350M ist ein dichter Bi-Encoder für mehrsprachige semantische
Suche. Das Modell erzeugt für jede Anfrage und jedes Dokument genau einen
1024-dimensionalen Vektor aus dem CLS-Token. Die Ähnlichkeit wird über Kosinus
berechnet, weshalb die Vektoren vor dem Indexieren normalisiert werden sollten.

Die maximale Dokumentlänge beträgt 512 Tokens. Längere Texte müssen in Chunks
zerlegt werden; ein Überlappungsbereich verhindert, dass Aussagen an einer
Chunk-Grenze verloren gehen. Für Chunks, die deutlich kürzer als 512 Tokens
sind, verschenkt man Kontext, für längere wird abgeschnitten.

Das Modell unterstützt elf Sprachen: Englisch, Spanisch, Deutsch, Französisch,
Italienisch, Portugiesisch, Arabisch, Schwedisch, Norwegisch, Japanisch und
Koreanisch. Die Suche funktioniert auch sprachübergreifend, eine deutsche
Anfrage findet also englische Dokumente.

Wichtig bei der Nutzung: Das Modell wurde mit asymmetrischen Prompt-Präfixen
trainiert. Anfragen werden mit dem Prompt "query" kodiert, Dokumente mit dem
Prompt "document". Wer beide gleich behandelt, verliert messbar Qualität.

Der empfohlene Zugang ist sentence-transformers. Die GGUF-Variante läuft in
llama.cpp-Builds mit Unterstützung für LFM2.5-Embedding-Modelle.
