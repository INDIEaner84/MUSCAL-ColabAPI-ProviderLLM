# LoRA-Ziele und Hyperparameter

LFM2.5 ist ein hybrides Backbone: gated Short-Range-Convolutions kombiniert mit
wenigen Grouped-Query-Attention-Blöcken. Für LoRA-Ziele bedeutet das, dass die
klassischen Aufmerksamkeits-Projektionen nicht alles sind — die MLP-Blöcke
tragen einen erheblichen Teil der Anpassung.

Bewährt hat sich die Kombination aus q_proj, k_proj, v_proj, o_proj, gate_proj,
up_proj und down_proj. Wichtig bei MoE-Modellen: Die Namen der MLP-Projektionen
matchen per Suffix jeden Experten. Dieselbe Zielliste funktioniert deshalb für
dichte und für MoE-Checkpoints, erzeugt bei MoE aber deutlich mehr
Adapterparameter, weil jeder Experte eigene Matrizen bekommt.

Bei Vision-Modellen kommen die Connector-Schichten hinzu. Liquid verwendet in
seinem Rezept für LFM2.5-VL zusätzlich die linearen Projektionen fc1, fc2 und
linear, bei Rang 8 und Alpha 16.

Als Startpunkt gelten Rang 16 und Alpha 32. Die Regel Alpha gleich zweimal Rang
hat sich als robust erwiesen. Höhere Ränge helfen bei komplexen Aufgaben,
vergrößern aber Adapter und Overfitting-Risiko. Für die Lernrate ist 2e-4 bei
SFT ein gängiger Wert, bei DPO liegt sie drei Größenordnungen niedriger.

Ein praktischer Hinweis zum Debugging: Wenn die Zielliste Schichten enthält, die
es im Checkpoint nicht gibt, passiert oft nichts Sichtbares. Die Zahl der
trainierbaren Parameter sollte deshalb immer ausgegeben und gegen die Erwartung
geprüft werden.
