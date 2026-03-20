# OCR Gatekeeper – C++ Version (`ocr_CPlus`)

Diese Version bildet die Kernlogik aus `main.py` in C++ nach und enthält jetzt ein **IBML-Scanner-Profil**.

## Was ist umgesetzt?

- Worker-Queue mit paralleler Verarbeitung
- 3 Preprocessing-Strategien
- OCR-Qualitätsbewertung via Tesseract-Confidence
- Retry-Logik und Reject-Handling
- Multi-Page-TIFF-Unterstützung (`.tif/.tiff`)
- Scanner-Profile:
  - `generic` (Default)
  - `ibml` (für IBML-Scans optimierte Vorverarbeitung + strengere Qualitätsgrenze)

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

## IBML-Anpassung aktivieren

```bash
export OCR_PROFILE=ibml
export OCR_LANG=deu
./ocr_CPlus/build/ocr_gatekeeper_cpp
```

### Relevante Umgebungsvariablen

- `OCR_PROFILE=generic|ibml`
- `OCR_LANG` (z. B. `eng`, `deu`, `deu+eng`)
- `OCR_INPUT_DIR`
- `OCR_OUTPUT_DIR`
- `OCR_REJECT_DIR`

## Hinweise

- Die ursprüngliche Python-Version enthält zusätzlich FastAPI + Dashboard.
- Diese C++-Version konzentriert sich auf den Verarbeitungskern (Queue/Pipeline/OCR/Retry).
- Bei `OCR_PROFILE=ibml` werden u. a. stärkere Binarisierung, Denoising, Deskew-Versuch sowie erhöhte Qualitätsanforderung (`min_score=0.62`) verwendet.
