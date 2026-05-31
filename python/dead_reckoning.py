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
                    CRUISE_SPEED_MPS, GYRO_BIAS_SAMPLES, GYRO_SIGN)

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

    def reset(self):
        """Zero the pose and trail; re-estimate gyro bias. Called at EXPLORE start."""
        self.pose = Pose()
        self.breadcrumbs = [(0.0, 0.0, 0.0)]
        self._last_breadcrumb_dist = 0.0
        self._total_distance = 0.0
        self._gyro_bias = 0.0
        self._bias_samples = []
        self._bias_locked = False

    def update(self, motion, gy, dt):
        """Advance the pose. motion is 'forward' | 'pivot' | 'stop'.
        gy is the raw pitch-axis gyro (deg/s); dt seconds."""
        dt = max(0.0, min(dt, 0.2))

        # Lock a gyro bias from the first samples (car should be still then).
        if not self._bias_locked:
            self._bias_samples.append(gy)
            if len(self._bias_samples) >= GYRO_BIAS_SAMPLES:
                self._gyro_bias = sum(self._bias_samples) / len(self._bias_samples)
                self._bias_locked = True
            # still translate/rotate using raw value meanwhile (bias ~ small)

        # Heading from de-biased gyro.
        rate = (gy - self._gyro_bias) * GYRO_SIGN          # deg/s
        self.pose.theta += math.radians(rate) * dt
        self.pose.theta = math.atan2(math.sin(self.pose.theta),
                                     math.cos(self.pose.theta))

        # Translation only when driving forward.
        if motion == "forward":
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
