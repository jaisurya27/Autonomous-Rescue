"""navigator.py — Explore / classify / return state machine.

Hardware note: the ToF distance sensor and the camera are BOTH mounted on the
pan servo head, so panning the servo aims both the camera and the rangefinder.
That makes a left/right scan give *real* left/right clearances.

Explore behaviour:
  - Drive forward while the path ahead is clear.
  - On an obstacle: stop, pan the head LEFT and read clearance, pan RIGHT and
    read clearance, recenter, then pivot toward the more open side.
  - On a confirmed person: pause (CLASSIFY) so main.py can log the location.
RETURN replays breadcrumbs (drive back toward each prior point).
"""

import math, time
from enum import Enum
from config import (MOTOR_BASE_SPEED, MOTOR_TURN_SPEED, MOTOR_STOP,
                    OBSTACLE_THRESHOLD, EXPLORATION_TIMEOUT,
                    SERVO_CENTER, SERVO_LEFT, SERVO_RIGHT, SERVO_SETTLE_S)

CLASSIFY_PAUSE = 1.5

class NavState(Enum):
    IDLE="idle"; EXPLORE="explore"; CLASSIFY="classify"; RETURN="return"; ARRIVED="arrived"

class Navigator:
    def __init__(self, sensors=None):
        self.state = NavState.IDLE
        self._explore_start = None
        self._classify_start = None
        self._return_path = []
        self._return_idx = 0
        self.sensors = sensors          # for servo scanning
        # scan state machine
        self._scan_phase = None
        self._scan_t = 0
        self._scan_left = 0.0
        self._scan_right = 0.0
        self._turn_dir = 0              # +1 left, -1 right
        self._turn_until = 0
        # current commanded head angle (degrees). main.py reads this so the
        # detector/grid can offset bearing by how far the head is panned.
        self.servo_angle = SERVO_CENTER

    # ---- transitions ----
    def start_exploration(self):
        self._reset_scan()
        self.state = NavState.EXPLORE
        self._explore_start = time.time()

    def start_return(self, breadcrumbs):
        self._reset_scan()
        self.state = NavState.RETURN
        self._return_path = list(reversed(breadcrumbs))
        self._return_idx = 0

    def stop(self):
        self._reset_scan()
        self.state = NavState.IDLE

    def _reset_scan(self):
        """Abort any in-progress scan/turn and recenter the head."""
        self._scan_phase = None
        self._turn_dir = 0
        self._turn_until = 0
        self._set_head(SERVO_CENTER)

    def _set_head(self, deg):
        self.servo_angle = deg
        if self.sensors:
            self.sensors.set_servo(deg)

    @property
    def state_name(self): return self.state.value
    @property
    def is_exploring(self): return self.state in (NavState.EXPLORE, NavState.CLASSIFY)
    @property
    def exploration_elapsed(self):
        return time.time() - self._explore_start if self._explore_start else 0.0

    def compute_command(self, tof, us, rx, ry, rtheta, has_det):
        if self.state in (NavState.IDLE, NavState.ARRIVED):
            return MOTOR_STOP, MOTOR_STOP

        if self.state == NavState.EXPLORE and self.exploration_elapsed > EXPLORATION_TIMEOUT:
            self.state = NavState.ARRIVED
            self._reset_scan()
            return MOTOR_STOP, MOTOR_STOP

        # Only interrupt to classify when the head is centered and we're not
        # mid-maneuver — avoids logging a person at a panned/garbage bearing.
        if (has_det and self.state == NavState.EXPLORE
                and self._scan_phase is None and self._turn_dir == 0):
            self.state = NavState.CLASSIFY
            self._classify_start = time.time()
            self._set_head(SERVO_CENTER)

        if self.state == NavState.CLASSIFY:
            if time.time() - self._classify_start < CLASSIFY_PAUSE:
                return MOTOR_STOP, MOTOR_STOP
            self.state = NavState.EXPLORE

        if self.state == NavState.RETURN:
            return self._return_drive(rx, ry, rtheta)

        # ── EXPLORE with servo-scan deflection ──
        return self._explore_drive(tof)

    # ---- explore with smart scan ----
    def _explore_drive(self, tof):
        now = time.time()

        # mid-turn after a scan: keep pivoting until the timer expires
        if self._turn_dir != 0:
            if now < self._turn_until:
                return ((-MOTOR_TURN_SPEED, MOTOR_TURN_SPEED) if self._turn_dir > 0
                        else (MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED))
            self._turn_dir = 0
            self._set_head(SERVO_CENTER)
            return MOTOR_STOP, MOTOR_STOP

        blocked = (0.01 < tof < OBSTACLE_THRESHOLD)

        # start a scan when blocked and head is centered (forward-facing tof valid)
        if blocked and self._scan_phase is None:
            self._scan_phase = "look_left"
            self._scan_t = now
            self._set_head(SERVO_LEFT)
            return MOTOR_STOP, MOTOR_STOP

        if self._scan_phase == "look_left":
            # wait for the head to physically reach LEFT before trusting tof
            if now - self._scan_t < SERVO_SETTLE_S:
                return MOTOR_STOP, MOTOR_STOP
            # head is now pointing left; tof reads LEFT clearance
            self._scan_left = tof if tof > 0.01 else 99.0
            self._scan_phase = "look_right"
            self._scan_t = now
            self._set_head(SERVO_RIGHT)
            return MOTOR_STOP, MOTOR_STOP

        if self._scan_phase == "look_right":
            if now - self._scan_t < SERVO_SETTLE_S:
                return MOTOR_STOP, MOTOR_STOP
            self._scan_right = tof if tof > 0.01 else 99.0
            self._scan_phase = None
            self._set_head(SERVO_CENTER)
            # pivot toward the more open side
            self._turn_dir = 1 if self._scan_left >= self._scan_right else -1
            self._turn_until = now + 0.5
            return MOTOR_STOP, MOTOR_STOP

        # path clear -> advance
        return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

    # ---- breadcrumb return-to-base ----
    def _return_drive(self, rx, ry, rtheta):
        if self._return_idx >= len(self._return_path):
            self.state = NavState.ARRIVED
            return MOTOR_STOP, MOTOR_STOP

        tx, ty, _ = self._return_path[self._return_idx]
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)

        if dist < 0.12:                 # reached this breadcrumb
            self._return_idx += 1
            return MOTOR_STOP, MOTOR_STOP

        target_th = math.atan2(dy, dx)
        err = math.atan2(math.sin(target_th - rtheta), math.cos(target_th - rtheta))

        if abs(err) > 0.35:             # face the target first
            return ((-MOTOR_TURN_SPEED, MOTOR_TURN_SPEED) if err > 0
                    else (MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED))
        return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED
