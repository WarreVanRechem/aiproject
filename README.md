<!-- #AI project
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
 -->

# 🦅 Edge AI Vogelclassificatie: NABirds op Coral Dev Board

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Mendel Linux](https://img.shields.io/badge/OS-Mendel_Linux-purple.svg)](https://coral.ai/docs/dev-board/get-started/)
[![Coral Dev Board](https://img.shields.io/badge/Hardware-Dev_Board-green.svg)](https://coral.ai/products/dev-board/)

<div align="center">
<a href="https://github.com/WarreVanRechem/aiproject">
  <img src="./assets/Project_opstelling_AI.png" alt="Project Banner" style="width: 50vw;">
</a>
<br>
<small><em>(Opmerking: AI gegenereerde afbeelding)</em></small>
</div>

## 📖 Introductie

Dit project demonstreert de kracht van **Edge AI** door een geavanceerd Deep Learning model volledig lokaal te draaien op een **Coral Dev Board**. Het systeem is getraind om vogelsoorten te herkennen uit de **NABirds dataset** en voert deze herkenning in real-time uit zonder afhankelijk te zijn van een internetverbinding of cloud-services.

De workflow omvat het volledige spectrum van AI-development: van data-ingestie en opschoning, tot het trainen en kwantiseren van het model, en uiteindelijk de deployment op gespecialiseerde Edge TPU hardware.

---

## ⚙️ Hardware Architectuur

De kern van dit project is de **Coral Dev Board**, een Single Board Computer (SBC) die specifiek is ontworpen voor snelle ML-inferentie.

* **Host Systeem:** Een krachtige PC/Laptop (voor het trainen van het model).
* **Target Device:** Coral Dev Board (voor het uitvoeren van het model).
* **Accelerator:** De geïntegreerde Edge TPU coprocessor, die tot 4 biljoen operaties per seconde (TOPS) kan uitvoeren met een minimaal stroomverbruik.
<div align="center">
<a href="https://github.com/WarreVanRechem/aiproject">
  <img src="./assets/Coral_Dev_Board.jpg" alt="Coral_Dev_Board" style="width: 50vw;">
</a>
</div>

---

## 📂 Overzicht van deze Repository

Deze repository bevat scripts voor zowel de trainingsfase (Host PC) als de inferentiefase (Dev Board).

## 🦜 Dataset Ophalen

Voor dit project maken we gebruik van de **NABirds V1** dataset van het Cornell Lab of Ornithology.

1. **Downloaden:**
   Ga naar de [NABirds downloadpagina](https://dl.allaboutbirds.org/nabirds) en vraag de dataset aan.
   
2. **Plaatsen:**
   Plaats het gedownloade bestand (`nabirds.tar.gz`) in de root van dit project of in een map genaamd `data/`.

3. **Uitpakken:**
   Gebruik de terminal of 7-ZIP om het archief uit te pakken:
   <a href="https://github.com/WarreVanRechem/aiproject">
  <img src="./assets/7zip.png" alt="7zip" style="width: 5vw;">
   </a>

   of     
   ```bash
   # Maak een map aan op de gewenste locatie (optioneel)
   mkdir -p dataset
   
   # Verplaats het bestand (indien nodig) en pak uit
   tar -xvzf nabirds.tar.gz -C data/

### 🛠 Fase 1: Data Voorbereiding & Training (Host PC)

Deze scripts worden uitgevoerd op een krachtige computer om het model te bouwen:

* **`prepare_dataset.py`**
    De originele NABirds dataset is "rommelig" voor een AI-model: alle afbeeldingen staan in diepe mappenstructuren met ID-nummers, en er zijn duizenden vogels die je niet nodig hebt. Dit script pakt alleen de 8 vogels die wij gebruiken en zet ze netjes klaar dit komt omdat de volledige dataset te groot is.
    <br>
* **`fix_dataset.py`**
  Dit script fungeert als kwaliteitscontrole. Het scant de gedownloade afbeeldingen op corruptie en verwijdert bestanden die niet leesbaar zijn voor TensorFlow, wat crashes tijdens het trainen voorkomt. Daarnaast kun je dit script gebruiken om de selectie van vogels aan te passen, zonder dat je de eerdere voorbereidingsstappen (`prepare_dataset.py`) volledig opnieuw hoeft te doorlopen.
  <br>
* **`fix_labels.py`**
    Dit eenvoudige, maar cruciale script zorgt voor de correcte mapping tussen de mapnamen (de vogelsoorten) en de numerieke uitgangen van het AI-model.<br><br>

    1. Leest Mappen: Het script scant de map coral_dataset/train en verzamelt de namen van alle vogelsoorten die je hebt geselecteerd.

    2. Sorteert: Het sorteert de vogelnamen alfabetisch. Dit is de volgorde die het machine learning-raamwerk (TensorFlow/TensorFlow Lite) automatisch zal gebruiken.

    3. Genereert labels.txt: Het schrijft de gesorteerde namen naar het bestand labels.txt.<br><br>

    Dit bestand dient als de "vertaler" voor je getrainde model. Als het model tijdens de detectie een output van bijvoorbeeld '3' geeft, vertelt `labels.txt` dat '3' overeenkomt met de vierde vogelsoort in de alfabetische lijst (index start bij 0). Dit voorkomt een 'label-mix-up' waarbij het model wel de juiste categorie voorspelt, maar het systeem de verkeerde naam toont.
    <br>
* **`train_model.py`**
    Dit is het hoofdscript dat de machine learning uitvoert en het model voor de Coral optimaliseert.<br><br>

    1. Data Inladen en Verharden: Het laadt de reeds gesorteerde vogelafbeeldingen in. Het past strenge data-augmentatie toe (zoomen, draaien, contrast) om het model robuust te maken voor wisselende omstandigheden buiten.

    2. Transfer Learning: Het gebruikt het snelle, lichte MobileNetV2-model, dat al is getraind op miljoenen afbeeldingen (ImageNet). Alleen de laatste laag wordt aangepast om te classificeren tussen jouw 8 vogels.

    3. Training: Het model wordt getraind (6 epochs) op de geaugmenteerde data om de vogelsoorten te leren herkennen.

    4. Optimalisatie voor Coral (Kwantisatie): Dit is de cruciale stap. Het getrainde model wordt geconverteerd naar het TensorFlow Lite (TFLite)-formaat en geoptimaliseerd met Int8 Kwantisatie.

    5. Deze kwantisatie maakt het model veel kleiner en zorgt ervoor dat de Edge TPU (de AI-chip op de Coral) het model honderden keren sneller kan uitvoeren.<br><br>

    **Output**: Het resultaat is het bestand nabirds_strict_quant.tflite, het geoptimaliseerde model dat klaar is voor inzet op de Coral Dev Board.

    **Kortom**: Dit script traint een slim, robuust vogelherkenningsmodel en maakt het ultrasnel door het te optimaliseren voor de Edge TPU van de Coral.

### 🧪 Fase 2: Validatie (Host PC)

* **`test_flite_model_pc.py`**
    Draait het geconverteerde `.tflite` model op de CPU van je PC om de nauwkeurigheid te verifiëren vóór deployment.
* **`web_test.py`**
    Een web-interface (bijv. Flask) om het model lokaal in de browser te testen met geüploade afbeeldingen.

### 🚀 Fase 3: Productie (Coral Dev Board)

Deze bestanden draaien op de Dev Board zelf:

* **`nabirds_strict_quant_edgetpu.tflite`**
    Het eindproduct: het gekwantiseerde model, gecompileerd voor de Edge TPU.
* **`coral_vol_code.py`**
    De productie-code. Dit script laadt het model in de TPU, leest de camerabeelden of bestanden in, en voert de detectie uit.
* **`labels.txt`**
    De lijst met vogelnamen die overeenkomen met de output van het model.

---

## 🚀 Installatie & Gebruik

### Stap 1: Omgeving opzetten (Host PC)

Zorg dat Python 3 geïnstalleerd is en installeer de benodigde libraries voor het trainen:

```bash
pip install tensorflow numpy pillow matplotlib