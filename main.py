from __future__ import annotations

import cv2
import numpy as np
import pytesseract

import threading
import queue
import os

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# =========================
# CONFIG
# =========================

@dataclass(slots=True)
class Config:
    min_score: float = 0.55
    workers: int = max(1, os.cpu_count() - 1)


CONFIG = Config()


# =========================
# GLOBAL STATE
# =========================

job_queue: queue.Queue = queue.Queue()
stats = {
    "processed": 0,
    "ok": 0,
    "retry": 0,
    "reject": 0,
    "last_files": []
}


# =========================
# JOB MODEL
# =========================

@dataclass(slots=True)
class Job:
    path: Path
    retries: int = 0
    history: list[str] = field(default_factory=list)


# =========================
# OCR
# =========================

def ocr_score(image: np.ndarray) -> float:
    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    confs = [float(c) for c in data["conf"] if c != "-1"]

    if not confs:
        return 0.0

    return float(np.mean(confs) / 100.0)


# =========================
# PREPROCESS
# =========================

def preprocess(image: np.ndarray, mode: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match mode:
        case 0:
            return cv2.equalizeHist(gray)

        case 1:
            clahe = cv2.createCLAHE(2.5, (8, 8))
            return clahe.apply(gray)

        case 2:
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 10
            )

        case _:
            return gray


# =========================
# PIPELINE
# =========================

def process_job(job: Job) -> None:
    global stats

    image = cv2.imread(str(job.path))
    if image is None:
        stats["reject"] += 1
        return

    best_score = 0
    best_img = None

    for mode in range(3):
        img = preprocess(image, mode)
        score = ocr_score(img)

        job.history.append(f"{mode}:{score:.2f}")

        if score > best_score:
            best_score = score
            best_img = img

    stats["processed"] += 1
    stats["last_files"].append(f"{job.path.name} ({best_score:.2f})")

    if len(stats["last_files"]) > 10:
        stats["last_files"].pop(0)

    if best_score >= CONFIG.min_score:
        stats["ok"] += 1
        out = Path("scan_output") / job.path.name
        cv2.imwrite(str(out), best_img)
        return

    if job.retries < 1:
        job.retries += 1
        stats["retry"] += 1
        job_queue.put(job)
        return

    stats["reject"] += 1
    job.path.rename(Path("scan_reject") / job.path.name)


# =========================
# WORKER
# =========================

def worker():
    while True:
        job = job_queue.get()
        try:
            process_job(job)
        finally:
            job_queue.task_done()


# =========================
# START WORKERS
# =========================

def start_workers():
    for _ in range(CONFIG.workers):
        threading.Thread(target=worker, daemon=True).start()


# =========================
# API
# =========================

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/status")
def get_status():
    return stats


@app.post("/scan")
def add_scan():
    input_dir = Path("scan_input")

    for f in input_dir.glob("*.*"):
        job_queue.put(Job(path=f))

    return {"message": "Jobs added"}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return f"""
    <html>
    <head>
        <title>Scan Pipeline Dashboard</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ font-family: Arial; background:#111; color:#eee; }}
            .card {{ padding:20px; margin:10px; background:#222; border-radius:10px; }}
        </style>
    </head>
    <body>
        <h1>📄 Scan Pipeline Dashboard</h1>

        <div class="card">Processed: {stats["processed"]}</div>
        <div class="card">OK: {stats["ok"]}</div>
        <div class="card">Retry: {stats["retry"]}</div>
        <div class="card">Reject: {stats["reject"]}</div>

        <h2>Last Files</h2>
        <div class="card">
            {"<br>".join(stats["last_files"])}
        </div>
    </body>
    </html>
    """


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    Path("scan_output").mkdir(exist_ok=True)
    Path("scan_reject").mkdir(exist_ok=True)

    start_workers()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
