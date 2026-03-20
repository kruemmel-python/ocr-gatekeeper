# OCR Gatekeeper – C++ Version (`ocr_CPlus`)

Diese Version bildet die Kernlogik aus `main.py` in C++ nach, inklusive **IBML-Scanner-Profil** sowie **API + Dashboard**.

## Was ist umgesetzt?

- Worker-Queue mit paralleler Verarbeitung
- 3 Preprocessing-Strategien
- OCR-Qualitätsbewertung via Tesseract-Confidence
- Retry-Logik und Reject-Handling
- Multi-Page-TIFF-Unterstützung (`.tif/.tiff`)
- Scanner-Profile:
  - `generic` (Default)
  - `ibml` (für IBML-Scans optimierte Vorverarbeitung + strengere Qualitätsgrenze)
- HTTP-API + Live-Dashboard:
  - `GET /`
  - `GET /status`
  - `POST /scan`

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

## Batch-Modus (einmalige Verarbeitung)

```bash
./ocr_CPlus/build/ocr_gatekeeper_cpp
```

## API + Dashboard starten

```bash
export OCR_ENABLE_API=1
export OCR_API_PORT=8000
./ocr_CPlus/build/ocr_gatekeeper_cpp
```

Dann im Browser öffnen:

- `http://localhost:8000/` (Dashboard)
- `http://localhost:8000/status` (JSON-Status)

Manuell einen Scan-Lauf starten:

```bash
curl -X POST http://localhost:8000/scan
```

## IBML-Anpassung aktivieren

```bash
export OCR_PROFILE=ibml
export OCR_LANG=deu
export OCR_ENABLE_API=1
./ocr_CPlus/build/ocr_gatekeeper_cpp
```

### Relevante Umgebungsvariablen

- `OCR_PROFILE=generic|ibml`
- `OCR_LANG` (z. B. `eng`, `deu`, `deu+eng`)
- `OCR_INPUT_DIR`
- `OCR_OUTPUT_DIR`
- `OCR_REJECT_DIR`
- `OCR_ENABLE_API=0|1`
- `OCR_API_HOST` (aktuell nur Anzeigezweck)
- `OCR_API_PORT`

## Hinweise

- Diese C++-Version enthält jetzt ebenfalls API + Dashboard, analog zur Python-Variante.
- Bei `OCR_PROFILE=ibml` werden u. a. stärkere Binarisierung, Denoising, Deskew-Versuch sowie erhöhte Qualitätsanforderung (`min_score=0.62`) verwendet.
