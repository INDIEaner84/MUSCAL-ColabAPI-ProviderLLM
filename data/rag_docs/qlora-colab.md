# QLoRA auf Colab

QLoRA quantisiert die Basisgewichte auf 4-bit und trainiert nur LoRA-Adapter
darauf. Für die Gewichte gilt die Faustregel von etwa 0,7 GB pro Milliarde
Parameter bei NF4 mit doppelter Quantisierung. Ein 1.2B-Modell belegt damit rund
1 GB, ein 8B-Modell rund 5,5 GB.

Der wichtigste Colab-spezifische Fallstrick ist die fehlende bf16-Unterstützung
der T4. Wer bf16 hart setzt, erhält kryptische bitsandbytes-Fehler. Die
Computation-dtype-Einstellung muss auf "auto" stehen und auf fp16 zurückfallen.
Bei den neueren L4- und A100-Karten ist bf16 verfügbar und meist schneller.

Als Optimierer hat sich paged_adamw_8bit bewährt, weil er die Optimizer-States
klein hält. Dazu kommen Gradient Checkpointing mit use_reentrant=False und das
Vorbereiten des Modells für k-bit-Training, sonst liefert Gradient
Checkpointing stillschweigend keine Gradienten.

Bei LoRA gilt: Rang 16 mit Alpha 32 ist ein solider Startpunkt. Höhere Ränge
bringen mehr Kapazität, aber auch mehr Adapterparameter und Overfitting-Risiko.
Die Lernrate für SFT liegt typischerweise zwischen 1e-5 und 2e-4, während DPO
drei Größenordnungen niedriger arbeitet.

Für den Speicher während des Trainings sind die Aktivierungen der zweite Faktor
nach den Gewichten. Sie wachsen quadratisch mit der Sequenzlänge, deshalb ist
max_length der wirksamste Hebel, wenn der Speicher knapp wird — noch vor der
Batchgröße.
