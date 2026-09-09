# LFM2.5-8B-A1B (MoE)

LFM2.5-8B-A1B ist ein Mixture-of-Experts-Modell mit 8,3 Milliarden
Gesamtparametern, von denen pro Token nur 1,5 Milliarden aktiv sind. Es nutzt
das hybride LFM2-Backbone aus gated Convolutions und Grouped-Query-Attention und
ersetzt die dichten MLPs in fast allen Schichten durch 32 Experten, von denen
ein Router pro Token vier auswählt.

Der häufigste Denkfehler betrifft den Speicher: Er richtet sich nach den
Gesamtparametern, nicht nach den aktiven. In 4-bit sind das rund 5,5 GB. Eine
freie Colab-T4 mit 15 GB hat damit nur noch wenig Luft, empfohlen werden L4 oder
A100 mit 40 GB.

Beim Fine-Tuning von MoE-Modellen sind drei Fehlerbilder bekannt. Router Collapse
bedeutet, dass der Großteil der Token auf ein oder zwei Experten entfällt, oft
nach einem Domänenwechsel. Bei zu hoher Lernrate auf den Router-Gewichten
oszilliert das Training. Und nach dem Zusammenführen von Adapter und Basis kann
die Qualität sinken, wenn die Balance-Verluste von Training und Inferenz nicht
zusammenpassen. Gegenmittel sind eine niedrigere Lernrate, ausgewogene
Trainingsdaten und eine Gegenmessung nach jedem Merge.

Beim Serving ist Vorsicht geboten. vLLM unterstützt Multi-LoRA für MoE-Modelle ab
Version 0.15 über einen fused_moe_lora-Kernel, bestätigt für GPT-OSS, Qwen3-MoE,
DeepSeek und Llama MoE. LFM2.5 steht nicht auf dieser Liste. vLLM hat Adapter mit
Expertengewichten für MoE historisch abgelehnt. Deshalb ist die
servefreundliche Variante ein LoRA nur auf die Attention-Projektionen, während
Adapter auf den Experten-MLPs gemergt statt heiß getauscht werden sollten.
