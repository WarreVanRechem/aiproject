#AI project
Stap 1:
gebruik prepare_dataset.py om te gaan kiezen welke vogels je wilt gaan trainen in uw model. 
Je doet dit aan de hand van de nummers aan te passen in de array, deze kan je uit de classes.txt terug vinden.

Stap 2:
Laat voor de zekerheid fix_dataset.py runnen dan ben je zeker dat alle foto's aanwezig zijn.

Stap 3:
Laat fix_labels runnen. Zo krijg je een mooi labels.txt bestand met de geselecteerde vogels.

Stap 4:
Train het model aan de hand van train_model.py

Stap 5:
converteer dit model naar een edgetpu bestand: edgetpu_compiler [options] model...

Stap 6:
Tijd om dit model uit te testen op de coral dev board. Dit doe je in het bestand coral_vol_code.py

