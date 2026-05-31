"""sensors.py — Bridge interface to the MCU: read sensors, drive motors, pan servo."""

import time, threading
from dataclasses import dataclass
from arduino.app_utils import Bridge
from config import ACCEL_BIAS_X, ACCEL_BIAS_Y, GYRO_BIAS_Z

# Bridge.call() is not thread-safe. The sensor loop (50 Hz) and the control loop
# (30 Hz) both call it concurrently, which causes heap corruption
# ("malloc(): invalid next size"). Serialize every call with this lock.
_bridge_lock = threading.Lock()

@dataclass
class SensorFrame:
    timestamp: float = 0.0
    ax: float = 0.0; ay: float = 0.0; az: float = 0.0
    gx: float = 0.0; gy: float = 0.0; gz: float = 0.0
    tof_distance: float = 0.0; us_distance: float = 0.0
    temperature: float = 0.0; valid: bool = False

class SensorReader:
    def __init__(self):
        self._latest = SensorFrame()
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[sensors] Reading via Bridge.call('read_sensors')")

    def stop(self): self._running = False

    def get_latest(self):
        with self._lock:
            return self._latest

    def _loop(self):
        fails = 0
        while self._running:
            try:
                with _bridge_lock:
                    csv = Bridge.call("read_sensors")
                p = [float(x) for x in csv.split(",")]
                ax, ay, az, gx, gy, gz, tof_mm, us_cm, temp_c = p
                f = SensorFrame(
                    timestamp=time.time(),
                    ax=ax - ACCEL_BIAS_X, ay=ay - ACCEL_BIAS_Y, az=az,
                    gx=gx, gy=gy, gz=gz - GYRO_BIAS_Z,
                    tof_distance=tof_mm / 1000.0,
                    us_distance=us_cm / 100.0,
                    temperature=temp_c, valid=True,
                )
                with self._lock:
                    self._latest = f
                fails = 0
                time.sleep(0.02)
            except Exception as e:
                fails += 1
                if fails == 1: print(f"[sensors] Bridge not ready: {e}")
                elif fails % 200 == 0: print(f"[sensors] Waiting for bridge... ({fails})")
                time.sleep(0.1)

    def send_command(self, left, right):
        try:
            with _bridge_lock:
                Bridge.call("set_motors", int(left), int(right))
        except Exception: pass

    def set_servo(self, deg):
        try:
            with _bridge_lock:
                Bridge.call("set_servo", int(deg))
        except Exception: pass

    def get_servo(self):
        try:
            with _bridge_lock:
                return int(Bridge.call("get_servo"))
        except Exception: return 90

    def set_indicator(self, mode):
        try:
            with _bridge_lock:
                Bridge.call("set_indicator", int(mode))
        except Exception: pass

    def play_tune(self, tune_id):
        try:
            with _bridge_lock:
                Bridge.call("play_tune", int(tune_id))
        except Exception: pass
