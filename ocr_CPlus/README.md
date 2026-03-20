# OCR Gatekeeper – C++ Version (`ocr_CPlus`)

Diese Version bildet die Kernlogik aus `main.py` in C++ nach:

- Worker-Queue mit paralleler Verarbeitung
- 3 Preprocessing-Strategien (EqualizeHist, CLAHE, Adaptive Threshold)
- OCR-Qualitätsbewertung via Tesseract-Confidence
- Retry-Logik und Reject-Handling
- Ausgabe in `scan_output/`, Rejects in `scan_reject/`

## Voraussetzungen

- C++20 Compiler (g++, clang++)
- CMake 3.16+
- OpenCV
- Tesseract + Leptonica
- pkg-config

## Build

```bash
cd ocr_CPlus
cmake -S . -B build
cmake --build build -j
```

## Ausführen

Lege Eingabedateien in `scan_input/` (im Repo-Root) und starte:

```bash
./ocr_CPlus/build/ocr_gatekeeper_cpp
```

Die Anwendung verarbeitet alle Dateien einmalig und gibt anschließend eine Zusammenfassung aus.

## Hinweise

- Die ursprüngliche Python-Version enthält zusätzlich FastAPI + Dashboard.
- Diese C++-Version konzentriert sich auf den Verarbeitungskern (Queue/Pipeline/OCR/Retry).
