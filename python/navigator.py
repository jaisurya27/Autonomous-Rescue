"""navigator.py — Explore / classify / return state machine.

Upgrades over the original:
  - SERVO SCAN deflection: on obstacle, look left & right with the pan head and
    turn toward the more open side (instead of always pivoting right blindly).
  - RETURN actually replays breadcrumbs (drive back toward each prior point).
"""

import math, time
from enum import Enum
from config import (MOTOR_BASE_SPEED, MOTOR_TURN_SPEED, MOTOR_STOP,
                    OBSTACLE_THRESHOLD, EXPLORATION_TIMEOUT,
                    SERVO_CENTER, SERVO_LEFT, SERVO_RIGHT)

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

    def start_exploration(self):
        self.state = NavState.EXPLORE
        self._explore_start = time.time()

    def start_return(self, breadcrumbs):
        self.state = NavState.RETURN
        self._return_path = list(reversed(breadcrumbs))
        self._return_idx = 0

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
            return MOTOR_STOP, MOTOR_STOP

        if has_det and self.state == NavState.EXPLORE:
            self.state = NavState.CLASSIFY
            self._classify_start = time.time()

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

        # mid-turn after a scan: keep pivoting briefly
        if self._turn_dir != 0:
            if now < self._turn_until:
                return ((-MOTOR_TURN_SPEED, MOTOR_TURN_SPEED) if self._turn_dir > 0
                        else (MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED))
            self._turn_dir = 0
            if self.sensors: self.sensors.set_servo(SERVO_CENTER)

        blocked = (0.01 < tof < OBSTACLE_THRESHOLD)

        if blocked and self._scan_phase is None:
            self._scan_phase = "look_left"
            self._scan_t = now
            if self.sensors: self.sensors.set_servo(SERVO_LEFT)
            return MOTOR_STOP, MOTOR_STOP

        if self._scan_phase == "look_left":
            if now - self._scan_t < 0.4:
                return MOTOR_STOP, MOTOR_STOP
            self._scan_left = tof if tof > 0.01 else 99.0
            self._scan_phase = "look_right"
            self._scan_t = now
            if self.sensors: self.sensors.set_servo(SERVO_RIGHT)
            return MOTOR_STOP, MOTOR_STOP

        if self._scan_phase == "look_right":
            if now - self._scan_t < 0.4:
                return MOTOR_STOP, MOTOR_STOP
            self._scan_right = tof if tof > 0.01 else 99.0
            self._scan_phase = None
            # turn toward the more open side
            self._turn_dir = 1 if self._scan_left >= self._scan_right else -1
            self._turn_until = now + 0.5
            if self.sensors: self.sensors.set_servo(SERVO_CENTER)
            return MOTOR_STOP, MOTOR_STOP

        # clear -> advance
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
