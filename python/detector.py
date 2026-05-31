"""detector.py — MobileNet-SSD person detection via OpenCV DNN. No internal camera."""

import threading, time, math, os, subprocess
from dataclasses import dataclass
from typing import List
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH, PERSON_HEIGHT_METERS

CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
           "cat","chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_ssd.caffemodel")
DETECT_CLASSES = {15}

@dataclass
class Detection:
    label: str; confidence: float; bbox: tuple
    distance: float; bearing: float
    world_x: float = 0.0; world_y: float = 0.0

class Detector:
    def __init__(self):
        self._detections: List[Detection] = []
        self._lock = threading.Lock()
        self._net = None
        self._pending = None
        self._plock = threading.Lock()
        self._running = False
        self.fx = CAMERA_FOCAL_LENGTH
        self.cx = CAMERA_WIDTH / 2.0

    def start(self):
        self._ensure_model()
        if os.path.exists(PROTO_PATH) and os.path.exists(MODEL_PATH):
            try:
                import cv2
                self._net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[detector] MobileNet-SSD loaded")
            except Exception as e:
                print(f"[detector] Model load failed: {e}"); self._net = None
        else:
            print("[detector] Model files not found"); self._net = None
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self): self._running = False

    def feed_frame(self, frame):
        with self._plock: self._pending = frame

    def get_detections(self):
        with self._lock: return list(self._detections)

    def _loop(self):
        import cv2
        while self._running:
            with self._plock: frame = self._pending; self._pending = None
            if frame is None or self._net is None:
                time.sleep(0.1); continue
            try:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843,
                    (300,300), (127.5,127.5,127.5), swapRB=False)
                self._net.setInput(blob)
                raw = self._net.forward()
                results = []
                for i in range(raw.shape[2]):
                    conf = float(raw[0,0,i,2]); cls = int(raw[0,0,i,1])
                    if conf < 0.45 or cls not in DETECT_CLASSES: continue
                    x1,y1 = int(raw[0,0,i,3]*w), int(raw[0,0,i,4]*h)
                    x2,y2 = int(raw[0,0,i,5]*w), int(raw[0,0,i,6]*h)
                    bh = y2 - y1
                    dist = (self.fx*PERSON_HEIGHT_METERS)/bh if bh > 10 else 5.0
                    bearing = math.atan2((x1+x2)/2.0 - self.cx, self.fx)
                    results.append(Detection(CLASSES[cls],conf,(x1,y1,x2,y2),dist,bearing))
                with self._lock: self._detections = results
            except Exception as e:
                print(f"[detector] Error: {e}")
            time.sleep(0.05)

    def project_to_world(self, det, rx, ry, rtheta):
        a = rtheta + det.bearing
        det.world_x = rx + det.distance * math.cos(a)
        det.world_y = ry + det.distance * math.sin(a)
        return det

    def _ensure_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(PROTO_PATH):
            print("[detector] Downloading prototxt...")
            subprocess.run(["wget","-q","-O",PROTO_PATH,
                "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"],
                timeout=30, check=False)
        if not os.path.exists(MODEL_PATH):
            print("[detector] Downloading model (~23MB)...")
            subprocess.run(["wget","-q","-O",MODEL_PATH,
                "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"],
                timeout=120, check=False)
