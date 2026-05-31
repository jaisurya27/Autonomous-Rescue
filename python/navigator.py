"""navigator.py — Explore / classify / return state machine with obstacle avoidance."""

import math, time
from enum import Enum
from config import (MOTOR_BASE_SPEED, MOTOR_TURN_SPEED, MOTOR_STOP,
                    OBSTACLE_THRESHOLD, EXPLORATION_TIMEOUT)

# How long to pause and log when a person is spotted (seconds)
CLASSIFY_PAUSE = 1.5

class NavState(Enum):
    IDLE="idle"; EXPLORE="explore"; CLASSIFY="classify"; RETURN="return"; ARRIVED="arrived"

class Navigator:
    def __init__(self):
        self.state = NavState.IDLE
        self._explore_start = None
        self._classify_start = None
        self._return_idx = 0
        self._return_path = []

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
        """Return (left_pwm, right_pwm). Differential drive:
        forward=(+,+), pivot-right=(+,-), pivot-left=(-,+)."""

        # Idle / arrived -> stop
        if self.state in (NavState.IDLE, NavState.ARRIVED):
            return MOTOR_STOP, MOTOR_STOP

        # Auto-stop after the exploration timeout
        if self.state == NavState.EXPLORE and self.exploration_elapsed > EXPLORATION_TIMEOUT:
            self.state = NavState.ARRIVED
            return MOTOR_STOP, MOTOR_STOP

        # Person seen while exploring -> enter CLASSIFY (pause to log it)
        if has_det and self.state == NavState.EXPLORE:
            self.state = NavState.CLASSIFY
            self._classify_start = time.time()

        if self.state == NavState.CLASSIFY:
            if time.time() - self._classify_start < CLASSIFY_PAUSE:
                return MOTOR_STOP, MOTOR_STOP          # hold still while detector logs position
            self.state = NavState.EXPLORE              # resume after the pause

        # RETURN: breadcrumb-replay driving is a later step; hold for now.
        if self.state == NavState.RETURN:
            return MOTOR_STOP, MOTOR_STOP

        # ── EXPLORE: obstacle avoidance ──
        # Blocked if EITHER sensor sees something within the threshold.
        # tof/us are in METERS; 0 means "no valid reading" -> treat as clear.
        blocked = False
        if 0.01 < tof < OBSTACLE_THRESHOLD:
            blocked = True
        if 0.01 < us < OBSTACLE_THRESHOLD:
            blocked = True

        if blocked:
            # One fixed forward sensor can't tell which way is open, so pivot
            # in place (right) until the path ahead clears. (Servo sweep later
            # will replace this with a smarter "look then turn".)
            return MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED
        else:
            # Clear path -> advance
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED