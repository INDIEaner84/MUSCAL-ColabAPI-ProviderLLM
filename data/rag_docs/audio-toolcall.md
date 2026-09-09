# Audio-Agent: Sprache zu Funktionsaufruf

Das Fine-Tuning-Ziel für den Audio-Agenten ist ungewöhnlich: Das Modell soll
gesprochene Befehle direkt in Funktionsaufrufe übersetzen, ohne den Umweg über
eine Transkription. Das untrainierte Modell transkribiert lediglich, Liquid
misst dafür 0 Prozent über alle drei Metriken.

Der System-Prompt ist dabei zwingend "Perform ASR." zu setzen. Das wirkt
zunächst falsch, hat aber einen konkreten Grund: llama-liquid-audio-server
akzeptiert nur eine geschlossene Liste von System-Prompts und erzwingt zur
Inferenz einen davon. Wird mit einem abweichenden Prompt trainiert, schiebt der
Server das Modell zurück in den Transkriptionsmodus und das Fine-Tune ist
wirkungslos.

Ein zweiter Unterschied zu den Text-Tracks: Audio-Finetuning läuft als volles
Finetune, nicht als LoRA. Der Trainer aus dem Paket liquid-audio hat keinen
LoRA-Schalter, entsprechend höher ist der Speicherbedarf. Eingeplant werden
sollte eine A100.

Das Zielformat wird vor der Datenerzeugung festgelegt und danach nicht mehr
geändert. Zur Wahl stehen ein kompaktes Pipe-Format und das native
LFM2.5-Format mit speziellen Tool-Call-Tokens. Im Pipe-Format darf ein Wert kein
Pipe-Zeichen enthalten, Gleichheitszeichen sind erlaubt, weil nur am ersten
getrennt wird.
