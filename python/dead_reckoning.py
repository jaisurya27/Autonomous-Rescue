"""dead_reckoning.py — Motion-model + gyro pose tracking and breadcrumb trail.

Why not visual odometry for position? VO *translation* drifts badly on the
Uno Q at a few fps, which wrecked return-to-base. Instead we drive the pose from
two trustworthy things:

  - HEADING: integrate the pitch-axis gyro (gy, deg/s). Bias is zeroed at the
    start of exploration while the car sits still, so heading stays usable.
  - TRANSLATION: a simple motion model. When the navigator commands FORWARD we
    advance along the current heading at CRUISE_SPEED_MPS; when pivoting in
    place or stopped we do NOT translate. No VO drift enters the position.

The navigator reports what it is doing each loop via a motion flag
("forward" | "pivot" | "stop"), and main.py calls update() with that flag, the
raw gyro reading, and dt.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple
from config import (START_X, START_Y, GRID_RESOLUTION, BREADCRUMB_INTERVAL,
                    CRUISE_SPEED_MPS, GYRO_BIAS_SAMPLES, GYRO_SIGN,
                    ACCEL_VAR_WINDOW, ACCEL_STUCK_CV, ACCEL_STUCK_SAMPLES)

@dataclass
class Pose:
    x: float = 0.0; y: float = 0.0; theta: float = 0.0; speed: float = 0.0

class DeadReckoning:
    def __init__(self):
        self.pose = Pose()
        self.breadcrumbs: List[Tuple[float,float,float]] = [(0.0, 0.0, 0.0)]
        self._last_breadcrumb_dist = 0.0
        self._total_distance = 0.0
        # gyro bias estimation
        self._gyro_bias = 0.0
        self._bias_samples = []
        self._bias_locked = False
        # accelerometer-based stuck detection
        self._accel_buf: List[float] = []   # rolling window of |accel| magnitude
        self._stuck_count = 0               # consecutive low-variation samples
        self.physically_moving = True       # False when IMU says car is not moving

    def reset(self):
        """Zero the pose and trail; re-estimate gyro bias. Called at EXPLORE start."""
        self.pose = Pose()
        self.breadcrumbs = [(0.0, 0.0, 0.0)]
        self._last_breadcrumb_dist = 0.0
        self._total_distance = 0.0
        self._gyro_bias = 0.0
        self._bias_samples = []
        self._bias_locked = False
        self._accel_buf = []
        self._stuck_count = 0
        self.physically_moving = True

    def update(self, motion, gy, dt, ax=None, ay=None, az=None):
        """Advance the pose. motion is 'forward' | 'pivot' | 'stop'.
        gy is the raw pitch-axis gyro (deg/s); dt seconds.
        ax/ay/az are raw accelerometer readings (any unit: g or m/s²).
        When provided they gate the forward position update so a stuck car
        (wheels spinning, no actual movement) does not corrupt the 2D map."""
        dt = max(0.0, min(dt, 0.2))

        # ── Gyro bias estimation (first N samples while car sits still) ──
        if not self._bias_locked:
            self._bias_samples.append(gy)
            if len(self._bias_samples) >= GYRO_BIAS_SAMPLES:
                self._gyro_bias = sum(self._bias_samples) / len(self._bias_samples)
                self._bias_locked = True

        # ── Heading from de-biased gyro ──
        rate = (gy - self._gyro_bias) * GYRO_SIGN   # deg/s
        self.pose.theta += math.radians(rate) * dt
        self.pose.theta = math.atan2(math.sin(self.pose.theta),
                                     math.cos(self.pose.theta))

        # ── Accelerometer-based movement detection ──
        # Track sum of per-axis variances for (ax, az) — the two horizontal axes
        # (ay ≈ g is the gravity axis on this vertical-mount IMU and is excluded).
        # Normalize by gravity² so the threshold is unit-agnostic (g or m/s²).
        # A moving car has motor/floor vibration → high var_ax + var_az.
        # Wheels spinning in place produce near-zero horizontal variation.
        if motion == "forward" and ax is not None:
            self._accel_buf.append((ax, az))
            if len(self._accel_buf) > ACCEL_VAR_WINDOW:
                self._accel_buf.pop(0)
            if len(self._accel_buf) >= ACCEL_VAR_WINDOW:
                n   = len(self._accel_buf)
                mu_x = sum(s[0] for s in self._accel_buf) / n
                mu_z = sum(s[1] for s in self._accel_buf) / n
                var_x = sum((s[0]-mu_x)**2 for s in self._accel_buf) / n
                var_z = sum((s[1]-mu_z)**2 for s in self._accel_buf) / n
                # gravity estimate for normalisation (|ay| ≈ g on vertical mount)
                g_est = max(abs(ay) if ay is not None else 1.0, 0.1)
                # normalised total horizontal variance (unit-agnostic)
                cv = (var_x + var_z) / (g_est * g_est)
                if cv < ACCEL_STUCK_CV:
                    self._stuck_count += 1
                else:
                    self._stuck_count = 0
                self.physically_moving = self._stuck_count < ACCEL_STUCK_SAMPLES
                if not self.physically_moving and self._stuck_count == ACCEL_STUCK_SAMPLES:
                    print(f"[dr] IMU: stuck (cv={cv:.5f} < {ACCEL_STUCK_CV})")
                # Periodic log every ~2 m so you can tune ACCEL_STUCK_CV to your surface
                if self._total_distance > 0 and (self._total_distance % 2.0) < (CRUISE_SPEED_MPS * 0.02 * 2):
                    print(f"[dr] accel cv={cv:.5f} moving={self.physically_moving}")
        elif motion != "forward":
            self._accel_buf = []
            self._stuck_count = 0
            self.physically_moving = True

        # ── Translation: only advance when physically moving ──
        if motion == "forward" and self.physically_moving:
            step = CRUISE_SPEED_MPS * dt
            self.pose.x += step * math.cos(self.pose.theta)
            self.pose.y += step * math.sin(self.pose.theta)
            self.pose.speed = CRUISE_SPEED_MPS
            self._total_distance += step
            if self._total_distance - self._last_breadcrumb_dist >= BREADCRUMB_INTERVAL:
                self.breadcrumbs.append((self.pose.x, self.pose.y, self.pose.theta))
                self._last_breadcrumb_dist = self._total_distance
        else:
            self.pose.speed = 0.0

    def get_grid_position(self) -> Tuple[int,int]:
        return (int(START_X + self.pose.x / GRID_RESOLUTION),
                int(START_Y + self.pose.y / GRID_RESOLUTION))

    def get_return_path(self):
        return list(reversed(self.breadcrumbs))

    def distance_to_start(self):
        return math.sqrt(self.pose.x**2 + self.pose.y**2)

    @property
    def total_distance(self):
        return self._total_distance
