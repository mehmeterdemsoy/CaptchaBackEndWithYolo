from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import cv2
import random

# Load YOLO model once
model = YOLO("models/yolov8n.pt")

def run_yolo_detection(image_bytes):
    """Runs YOLOv8 person detection and returns (detected, confidence)."""

    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil)

    results = model.predict(img, conf=0.25)

    # Person class = 0
    persons = [
        (float(box.conf[0]))
        for box in results[0].boxes
        if int(box.cls[0]) == 0
    ]

    if len(persons) == 0:
        return False, 0.0

    # Return highest confidence person
    return True, max(persons)


def run_spoof_checks(image_bytes):
    """
    Light anti-spoof checks used in the research:
    - Detect flatness (printed image detection)
    - Detect low-frequency GAN inconsistencies
    - Optional: placeholder for blink/motion detection (if video used)
    """

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(img)

    # --- 1) Flatness detection (printed photo)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Low variance => suspiciously flat (printed photo)
    if laplacian_var < 18:
        return True

    # --- 2) GAN detection heuristic
    # GAN faces often have abnormal high-frequency noise
    high_freq_noise = np.std(cv2.Canny(gray, 50, 150))

    if high_freq_noise > 80:  # threshold determined experimentally
        return True

    return False
