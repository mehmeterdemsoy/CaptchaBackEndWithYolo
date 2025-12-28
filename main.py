from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from detection import run_yolo_detection, run_spoof_checks
from config import settings
import uuid
import time
import csv
import os

app = FastAPI(title="YOLO Human Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# CSV for experiment logs
CSV_PATH = "data/logs/results.csv"
os.makedirs("data/logs", exist_ok=True)

# Ensure CSV headers exist
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "request_id", "human_detected",
            "spoof_flagged", "confidence", "processing_ms"
        ])

def log_result(res):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            res["request_id"],
            res["human_detected"],
            res["spoof_flagged"],
            res["confidence"],
            res["processing_ms"]
        ])


@app.post("/verify")
async def verify(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start = time.time()
    request_id = str(uuid.uuid4())

    img_bytes = await image.read()

    # YOLO detection
    person_detected, confidence = run_yolo_detection(img_bytes)

    # Spoof check (GAN, printed photos)
    spoof_flagged = run_spoof_checks(img_bytes)

    result = {
        "request_id": request_id,
        "human_detected": person_detected and not spoof_flagged,
        "spoof_flagged": spoof_flagged,
        "confidence": confidence,
        "processing_ms": int((time.time() - start) * 1000)
    }

    # Log for research analysis
    log_result(result)

    return JSONResponse(result)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.MODEL_PATH}
