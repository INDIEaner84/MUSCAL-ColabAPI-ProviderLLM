# data/

Trainingsdaten liegen hier — lokal, damit nichts versehentlich in Git landet
(siehe `.gitignore`: alles außer diesem Ordner-Hinweis und dem Beispieldatensatz
wird ignoriert).

```
data/
├── example_train.jsonl   # 5 Beispiel-Zeilen im messages-Format
├── train.jsonl           # deine Textdaten
├── vl_train.jsonl        # deine Vision-Daten (image / question / answer)
└── images/               # Bilder fuer vl_train.jsonl (data.image_root)
```

Erzeugen mit `scripts/prepare_dataset.py`, Format-Details in
[docs/datasets.md](../docs/datasets.md).

Für Colab: den Ordner nach `MyDrive/muscal-lfm/<track>/data/` hochladen (oder
direkt in Drive ablegen) — die Notebooks erwarten ihn dort.
