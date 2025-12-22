# 🦅 Edge AI Vogelclassificatie  
## NABirds op Coral Dev Board  
### Volledige README & Projectguide (incl. originele projectinfo)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Mendel Linux](https://img.shields.io/badge/OS-Mendel_Linux-purple.svg)](https://coral.ai/docs/dev-board/get-started/)
[![Coral Dev Board](https://img.shields.io/badge/Hardware-Coral_Dev_Board-green.svg)](https://coral.ai/products/dev-board/)

---

## 🎯 Projectopdracht & Doel

Dit project focust op het ontwikkelen van een **intelligent embedded Edge AI-systeem** dat in staat is om **vogelsoorten in real-time te herkennen** op basis van camerabeelden.  
Het project doorloopt de **volledige machine learning pipeline**, van ruwe dataset tot deployment op gespecialiseerde hardware.

### Concrete doelstellingen
- Ontwikkelen en trainen van een deep learning model (TensorFlow)
- Optimaliseren naar TensorFlow Lite (Int8 kwantisatie)
- Compileren voor de **Google Edge TPU**
- Real-time inferentie op een **Coral Dev Board**
- Live visualisatie via **webinterface (Flask)** en **LCD-scherm**
- Robuuste foutafhandeling bij onzekere voorspellingen

De AI-logica draait volledig **on-device**, zonder cloudverbinding.

---

## 👥 Team & Rolverdeling

### Model & Software Backend
- **Daan**
  - Modelarchitectuur & training
  - Camera setup & drivers
  - TF Lite conversie
  - Flask webserver & multithreading
  - Datasetselectie
- **Warre**
  - Modeltraining & parameter-tuning
  - Preprocessing scripts
  - TF Lite optimalisatie
  - Flask applicatie
  - GPU-training onderzoek

### Hardware & Integratie
- **Quinten**
  - Coral Dev Board configuratie
  - I2C-communicatie
  - LCD-integratie
- **Robin**
  - Hardware support
  - Model-evaluatie
  - Validatie & testing
- **Warre**
  - Integratie software ↔ hardware

### Documentatie
- **Daan** – README & projectguide  
- **Robin** – Weekrapporten  
- **Quinten** – Onderzoek & verificatie  

---

# 📘 Projectguide

## 1️⃣ Benodigde Hardware

| Component | Beschrijving |
|---------|-------------|
| Coral Dev Board | 1GB / 4GB met Edge TPU |
| USB Webcam | Logitech C270 (UVC compatible) |
| LCD-scherm | I2C LCD (optioneel) |
| MicroSD-kaart | ≥ 8 GB |
| USB-C voeding | 5V – 2 à 3A |
| Host PC | Training & deployment |

---

## 2️⃣ Software & Dependencies

### Host PC
- Python 3.7+
- TensorFlow 2.x
- NumPy, Pillow, OpenCV, Matplotlib
- Google Coral MDT
- Edge TPU Compiler (Linux / Colab)

```bash
pip install tensorflow numpy pillow matplotlib opencv-python
```

### Coral Dev Board (Mendel Linux)
```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-flask libedgetpu1-std
```

---

## 3️⃣ Hardware Architectuur

- **Host PC**: modeltraining
- **Coral Dev Board**: inferentie
- **Edge TPU**: hardwareversnelde AI

Het systeem scheidt AI-inferentie en visualisatie voor maximale performantie en stabiliteit.

---

## 4️⃣ Dataset

- **Dataset**: NABirds V1 (Cornell Lab of Ornithology)
- Enkel **8 geselecteerde vogelsoorten** (volledige dataset te groot)
- Dataset opgeschoond en herstructureerd via scripts

---

## 5️⃣ Trainings- & Voorbereidingsscripts

| Script | Functie |
|------|--------|
| prepare_dataset.py | Selecteert en structureert data |
| fix_dataset.py | Verwijdert corrupte afbeeldingen |
| fix_labels.py | Genereert correcte labels.txt |
| train_model.py | Training + kwantisatie |
| test_flite_model_pc.py | Validatie op PC |
| web_test.py | Lokale webtest |

Output:
- `nabirds_strict_quant.tflite`

---

## 6️⃣ Model & ML-details

- **Modeltype**: CNN – MobileNetV2
- **Methode**: Transfer Learning (ImageNet)
- **Optimalisatie**: Int8 kwantisatie
- **Aantal klassen**: 8 vogelsoorten

### Classificatiedrempel
- Confidence ≥ **0.5**
- Vermijdt fout-positieven
- Geschikt voor real-time edge toepassingen

---

## 7️⃣ Compilatie & Deployment

```bash
edgetpu_compiler nabirds_strict_quant.tflite
```

Bestanden overzetten:
```bash
mdt push nabirds_strict_quant_edgetpu.tflite
mdt push labels.txt
mdt push top3_project.py
```

---

## 8️⃣ Opstartprocedure (Edge Device)

```bash
mdt shell
python3 top3_project.py
```

Webinterface:
```
http://<IP_CORAL>:5000
```

---

## 9️⃣ Testinstructies & Praktijktests

### Installatiecontrole
- `import tflite_runtime` werkt
- Webcam zichtbaar als `/dev/video0`
- Flask bereikbaar op poort 5000

### Concreet testscenario
| Situatie | Verwacht resultaat |
|--------|-------------------|
| Bald Eagle in beeld | Bald Eagle (top-1) |
| Onbekende vogel | Geen classificatie |
| Slechte belichting | Lage confidence |

---

## 🔬 Modelperformantie

| Metric | Waarde |
|------|--------|
| Accuracy | ± 85% |
| Precision | ± 83% |
| Recall | ± 82% |

Bron:
- Training logs
- Testset
- PC-validatie

Latency:
- ~10–15 ms per frame op Edge TPU

---

## ⚠️ Faalgedrag & Beperkingen

- Verwarring bij slechte belichting
- Vergelijkbare vogelsoorten
- Kleine objecten op afstand

Foutafhandeling:
- Confidence < 0.5 → geen output
- Systeem blijft stabiel

---

## 🔧 Bekende Problemen

- Edge TPU compiler enkel native op Linux
- Camera-compatibiliteit afhankelijk van model
- Correcte I2C-adressering vereist voor LCD

---

## 📌 Conclusie

Dit project demonstreert een **volledig autonoom Edge AI-systeem** met:
- End-to-end ML pipeline
- Reële edge-performantie
- Robuuste foutafhandeling
- Hardware-integratie

Repository:  
https://github.com/WarreVanRechem/aiproject
