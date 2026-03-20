# 📄 OCR Gatekeeper

**CPU-optimierte Dokumenten-Pipeline mit OCR-Qualitätsbewertung, Retry-Logik und Live-Dashboard**

---

## 🧠 Überblick

**OCR Gatekeeper** ist ein leichtgewichtiges, aber leistungsfähiges Dokumentenverarbeitungssystem für Scan-Workflows.

Der Fokus liegt nicht auf Bildoptimierung an sich, sondern auf der entscheidenden Frage:

> **Ist das Dokument maschinell lesbar?**

Dazu kombiniert das System:

* adaptive Bildverarbeitung
* OCR-basierte Qualitätsbewertung
* Retry-Strategien
* Queue-basierte Verarbeitung
* Live-Dashboard & API

---

## ⚙️ Features

* 🔁 **Multi-Strategy Preprocessing**

  * Histogram Equalization
  * CLAHE
  * Adaptive Thresholding

* 🧠 **OCR-Driven Quality Control**

  * automatische Bewertung via Tesseract Confidence

* 🔄 **Retry-Mechanismus**

  * alternative Strategien bei schlechter Qualität

* 🧵 **Worker Queue System**

  * parallele Verarbeitung auf CPU

* 📊 **Live Dashboard**

  * Echtzeit-Statistiken im Browser

* 🌐 **REST API**

  * Steuerung und Monitoring

---

## 🏗️ Architektur

```
Scan Input → Queue → Worker → Pipeline → OCR → Decision
                                    ↓
                          OK / Retry / Reject
                                    ↓
                             Dashboard + API
```

---

## 📦 Installation

### Voraussetzungen

* Python **3.12+**
* Tesseract OCR installiert

👉 Tesseract installieren:
https://github.com/tesseract-ocr/tesseract

---

### Setup

```bash
pip install fastapi uvicorn opencv-python numpy pytesseract
```

---

## 🚀 Nutzung

### 1. Verzeichnisstruktur

```
project/
├── main.py
├── scan_input/
├── scan_output/
├── scan_reject/
```

---

### 2. Start

```bash
python main.py
```

---

### 3. Dashboard öffnen

```
http://localhost:8000
```

---

### 4. Verarbeitung starten

```bash
POST /scan
```

→ Fügt alle Dateien aus `scan_input/` zur Queue hinzu

---

## 🌐 API

### GET `/status`

Gibt aktuelle Systemstatistiken zurück:

```json
{
  "processed": 120,
  "ok": 95,
  "retry": 15,
  "reject": 10,
  "last_files": [...]
}
```

---

### POST `/scan`

Startet Verarbeitung für alle Dateien im Input-Ordner.

---

## 🧪 Pipeline-Logik

### 1. Preprocessing

Mehrere Strategien werden getestet:

* Mode 0 → Histogram Equalization
* Mode 1 → CLAHE
* Mode 2 → Adaptive Threshold

---

### 2. OCR-Bewertung

```python
score = mean(confidence) / 100
```

---

### 3. Entscheidung

| Zustand           | Aktion    |
| ----------------- | --------- |
| Score ≥ Threshold | speichern |
| Score < Threshold | Retry     |
| Retry exhausted   | Reject    |

---

## ⚡ Performance

* CPU-only (keine GPU erforderlich)
* skaliert mit Anzahl der CPU-Kerne
* ideal für Büro- und Produktionsumgebungen

---

## 📊 Dashboard

* Live-Update alle 2 Sekunden
* zeigt:

  * verarbeitete Dokumente
  * Erfolgsrate
  * Retries
  * Rejects
  * letzte Dateien

---

## 🧠 Design-Prinzipien

* deterministisch
* reproduzierbar
* fehlertolerant
* modular erweiterbar

---

## 🔧 Erweiterungsmöglichkeiten

* PDF (Multi-Page) Support
* Deskew & Auto-Cropping
* Dokumentklassifikation
* Audit Logging (CSV / DB)
* Redis Queue (Distributed Processing)
* Benutzerverwaltung & Auth

---

## 🧠 Einsatzbereiche

* Scanstraßen
* Dokumentenarchivierung
* Posteingang-Digitalisierung
* OCR-Vorverarbeitung
* Behörden / Enterprise Workflows

---

## ⚠️ Hinweise

* OCR-Qualität hängt stark von Eingabedaten ab
* schlechte Scans können trotz Retry nicht gerettet werden
* Tesseract sollte korrekt installiert und im PATH sein

---

## 📜 Lizenz

MIT License 

