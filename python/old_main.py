# main.py - Phase 2 combined, ONNX edition
# Obstacle avoidance (distance sensor over Bridge) + person detection (ONNX YOLOv8n).
# No PyTorch — uses onnxruntime, which is small enough to fit the container comfortably.
#
# Motors are PRINT-ONLY for now (test board). Swap act() for set_motors on the car.

from arduino.app_utils import *
import cv2, time, os
import numpy as np
import onnxruntime as ort
from datetime import datetime

# ── Tunables ────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")
INFER_SIZE   = 320          # must match the imgsz used at export
CONF_THRESH  = 0.5
IOU_THRESH   = 0.45         # for non-max suppression
PERSON_CLASS = 0            # COCO class id for "person"
CAMERA_INDEX = 0
BLOCKED_MM   = 300
CLEAR_MM     = 500
SAVE_DIR     = os.path.expanduser("~/detections")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load ONNX model ─────────────────────────────────────────────────────
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: camera not available"); raise SystemExit

person_present = False

# ── ONNX helpers (what YOLO() used to do for us) ────────────────────────
def preprocess(frame):
    """Resize to model size, BGR->RGB, HWC->CHW, normalize, add batch dim."""
    img = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[None]

def detect_people(frame):
    """Run ONNX inference and decode YOLOv8 output into person boxes.
    Returns list of (x1,y1,x2,y2,conf) in original frame coords."""
    h, w = frame.shape[:2]
    inp = preprocess(frame)
    out = session.run(None, {input_name: inp})[0]   # shape (1, 84, N)

    # YOLOv8 output: 84 rows = 4 box coords + 80 class scores, N predictions.
    preds = out[0].T                                  # -> (N, 84)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Keep only confident "person" detections
    keep = (class_ids == PERSON_CLASS) & (confidences > CONF_THRESH)
    boxes_xywh = boxes_xywh[keep]
    confidences = confidences[keep]
    if len(boxes_xywh) == 0:
        return []

    # Convert center-x,center-y,w,h (in model space) to corner coords in frame space
    sx, sy = w / INFER_SIZE, h / INFER_SIZE
    detections = []
    rects, scores = [], []
    for (cx, cy, bw, bh), conf in zip(boxes_xywh, confidences):
        x1 = (cx - bw / 2) * sx; y1 = (cy - bh / 2) * sy
        x2 = (cx + bw / 2) * sx; y2 = (cy + bh / 2) * sy
        rects.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(float(conf))

    # Non-max suppression to remove overlapping duplicate boxes
    idxs = cv2.dnn.NMSBoxes(rects, scores, CONF_THRESH, IOU_THRESH)
    for i in np.array(idxs).flatten():
        x, y, bw, bh = rects[i]
        detections.append((x, y, x + bw, y + bh, scores[i]))
    return detections

def check_for_people(frame):
    global person_present
    people = detect_people(frame)
    if people and not person_present:
        conf = max(p[4] for p in people)
        print(f"\n🔴 PERSON DETECTED — {len(people)} person(s), {conf:.0%}")
        annotated = frame.copy()
        for (x1, y1, x2, y2, c) in people:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_DIR, f"person_{ts}.jpg")
        cv2.imwrite(path, annotated)
        print(f"   saved {path}")
        person_present = True
    elif not people and person_present:
        print("   ...area clear of people")
        person_present = False

# ── Motors (print-only on test board) ───────────────────────────────────
def act(decision):
    print(f"   >> ACTION: {decision}")

def read_distance_mm():
    mm = Bridge.call("read_distance")
    return mm if (mm is not None and mm >= 0) else None

# ── Warm up ─────────────────────────────────────────────────────────────
for _ in range(3):
    ret, frame = cap.read()
    if ret: detect_people(frame)
print("Phase 2 (ONNX) running: scanning for obstacles and people.")

# ── Main loop: move-stop-scan ───────────────────────────────────────────
def main():
    while True:
        act("STOP (scanning)")
        dist = read_distance_mm()
        dist_str = f"{dist}mm" if dist is not None else "no reading"

        ret, frame = cap.read()
        if ret:
            check_for_people(frame)

        if dist is None:
            print(f"[scan] distance: {dist_str} -> can't confirm clear, holding")
            act("STOP (no distance reading)")
        elif dist < BLOCKED_MM:
            print(f"[scan] distance: {dist_str} -> BLOCKED, need to turn")
            act("PIVOT to find opening")
        elif dist > CLEAR_MM:
            print(f"[scan] distance: {dist_str} -> CLEAR, advance")
            act("FORWARD (short nudge)")
        else:
            print(f"[scan] distance: {dist_str} -> caution zone, small nudge")
            act("FORWARD (slow)")

        time.sleep(1.0)

main()