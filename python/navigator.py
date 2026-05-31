"""navigator.py — Explore / classify / return state machine.

Recon logic (rewritten to stop the "stuck sweeping at a wall" bug):

  CRUISE   drive forward while the front is clear.
  SWEEP    on an obstacle: rotate the WHOLE CAR in SWEEP_STEP_DEG steps through a
           full 360 deg, reading front clearance (ultrasonic + centered ToF) at
           each heading. The front ultrasonic is body-fixed so it always reads
           straight ahead; that is the trusted front-obstacle source.
  COMMIT   turn in place to the most-open heading (gyro closed-loop).
  ADVANCE  drive forward; only SWEEP again once actually blocked again — never
           re-scan in place (that was the infinite loop).
  DEAD-END if every swept heading is below MIN_CLEARANCE, back up briefly then
           head to the least-bad direction. Always makes progress.

RETURN replays breadcrumbs using the motion-model pose (see dead_reckoning.py),
which no longer drifts the way visual odometry did.

The navigator reports what it is commanding each loop via `self.motion`
("forward" | "pivot" | "stop") so the odometry can advance the pose correctly.
"""

import math, time
from enum import Enum
from config import (MOTOR_BASE_SPEED, MOTOR_TURN_SPEED, MOTOR_STOP,
                    US_STOP_DISTANCE, TOF_STOP_DISTANCE, SWEEP_STEP_DEG,
                    SWEEP_SETTLE_S, MIN_CLEARANCE, PIVOT_STEP_TIMEOUT,
                    BACKUP_TIME, EXPLORATION_TIMEOUT, SERVO_CENTER)

CLASSIFY_PAUSE = 1.5
HEADING_TOLERANCE = math.radians(12)     # "reached" a target heading
SWEEP_SAMPLES = max(2, round(360 / SWEEP_STEP_DEG))

class NavState(Enum):
    IDLE="idle"; EXPLORE="explore"; CLASSIFY="classify"; RETURN="return"; ARRIVED="arrived"

def _ang_diff(a, b):
    """Smallest signed angle a-b in (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))

class Navigator:
    def __init__(self, sensors=None):
        self.state = NavState.IDLE
        self._explore_start = None
        self._classify_start = None
        self._return_path = []
        self._return_idx = 0
        self.sensors = sensors
        self.servo_angle = SERVO_CENTER     # kept for detector/grid bearing
        self.motion = "stop"                # forward | pivot | stop (for odometry)

        # recon sub-state
        self._phase = "cruise"              # cruise | sweep | commit | advance | backup
        self._sweep_samples = []            # list of (heading, clearance)
        self._sweep_start_heading = None
        self._sweep_target = None           # next heading to pivot to during sweep
        self._phase_t = 0.0
        self._settle_t = None               # set when a sweep step starts settling
        self._commit_target = None          # heading to commit-turn toward

    # ---- transitions ----
    def start_exploration(self):
        self._reset_recon()
        self.state = NavState.EXPLORE
        self._explore_start = time.time()

    def start_return(self, breadcrumbs):
        self._reset_recon()
        self.state = NavState.RETURN
        self._return_path = list(reversed(breadcrumbs))
        self._return_idx = 0

    def stop(self):
        self._reset_recon()
        self.state = NavState.IDLE

    def _reset_recon(self):
        self._phase = "cruise"
        self._sweep_samples = []
        self._sweep_start_heading = None
        self._sweep_target = None
        self._settle_t = None
        self._commit_target = None
        self.motion = "stop"
        if self.sensors:
            self.sensors.set_servo(SERVO_CENTER)
        self.servo_angle = SERVO_CENTER

    @property
    def state_name(self): return self.state.value
    @property
    def is_exploring(self): return self.state in (NavState.EXPLORE, NavState.CLASSIFY)
    @property
    def exploration_elapsed(self):
        return time.time() - self._explore_start if self._explore_start else 0.0

    # ---- main entry ----
    def compute_command(self, tof, us, rx, ry, rtheta, has_det):
        if self.state in (NavState.IDLE, NavState.ARRIVED):
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if self.state == NavState.EXPLORE and self.exploration_elapsed > EXPLORATION_TIMEOUT:
            self.state = NavState.ARRIVED
            self._reset_recon()
            return MOTOR_STOP, MOTOR_STOP

        # Pause to log a person only while cruising (head forward, bearing valid).
        if (has_det and self.state == NavState.EXPLORE and self._phase == "cruise"):
            self.state = NavState.CLASSIFY
            self._classify_start = time.time()

        if self.state == NavState.CLASSIFY:
            self.motion = "stop"
            if time.time() - self._classify_start < CLASSIFY_PAUSE:
                return MOTOR_STOP, MOTOR_STOP
            self.state = NavState.EXPLORE

        if self.state == NavState.RETURN:
            return self._return_drive(rx, ry, rtheta)

        return self._explore_drive(tof, us, rtheta)

    # ---- helpers ----
    @staticmethod
    def _front_clear_dist(tof, us, head_centered):
        """Best estimate of forward clearance. US is body-fixed (always fwd);
        ToF only counts when the head is centered. 0 readings mean 'no return'
        which we treat as far/open (99)."""
        vals = []
        vals.append(us if us > 0.01 else 99.0)
        if head_centered:
            vals.append(tof if tof > 0.01 else 99.0)
        return min(vals)

    def _pivot(self, direction):
        """direction +1 = left (CCW), -1 = right (CW). Sets motion=pivot."""
        self.motion = "pivot"
        return ((-MOTOR_TURN_SPEED, MOTOR_TURN_SPEED) if direction > 0
                else (MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED))

    # ---- explore FSM ----
    def _explore_drive(self, tof, us, rtheta):
        now = time.time()
        head_centered = (self.servo_angle == SERVO_CENTER)
        front = self._front_clear_dist(tof, us, head_centered)

        # ---------------- CRUISE ----------------
        if self._phase == "cruise":
            blocked = (us > 0.01 and us < US_STOP_DISTANCE) or \
                      (head_centered and 0.01 < tof < TOF_STOP_DISTANCE)
            if blocked:
                # begin a fresh 360 sweep from the current heading
                self._phase = "sweep"
                self._sweep_samples = []
                self._sweep_start_heading = rtheta
                self._sweep_target = rtheta                 # sample here first
                self._phase_t = now
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP
            self.motion = "forward"
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

        # ---------------- SWEEP ----------------
        # Per step: (1) pivot to the target heading, (2) settle, (3) record the
        # front clearance, then advance to the next heading or finish.
        if self._phase == "sweep":
            if self._settle_t is None:
                err = _ang_diff(self._sweep_target, rtheta)
                reached = abs(err) <= HEADING_TOLERANCE
                timed_out = (now - self._phase_t) > PIVOT_STEP_TIMEOUT
                if not reached and not timed_out:
                    return self._pivot(+1)          # rotate CCW toward target
                self._settle_t = now                 # begin settle window
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP

            # settling
            if (now - self._settle_t) < SWEEP_SETTLE_S:
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP

            # record this heading's clearance
            self._sweep_samples.append((rtheta, front))
            self._settle_t = None
            if len(self._sweep_samples) >= SWEEP_SAMPLES:
                return self._finish_sweep(rtheta, now)
            self._sweep_target = _wrap(rtheta + math.radians(SWEEP_STEP_DEG))
            self._phase_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ---------------- COMMIT (turn to chosen heading) ----------------
        if self._phase == "commit":
            err = _ang_diff(self._commit_target, rtheta)
            if abs(err) > HEADING_TOLERANCE and (now - self._phase_t) < PIVOT_STEP_TIMEOUT * 2:
                return self._pivot(+1 if err > 0 else -1)
            self._phase = "advance"
            self._phase_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ---------------- ADVANCE (drive until blocked again) ----------------
        if self._phase == "advance":
            blocked = (us > 0.01 and us < US_STOP_DISTANCE) or \
                      (head_centered and 0.01 < tof < TOF_STOP_DISTANCE)
            if blocked:
                self._phase = "cruise"          # will re-trigger a sweep next call
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP
            self.motion = "forward"
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

        # ---------------- BACKUP (dead-end escape) ----------------
        if self._phase == "backup":
            if (now - self._phase_t) < BACKUP_TIME:
                self.motion = "stop"            # reversing isn't tracked as forward
                return -MOTOR_BASE_SPEED, -MOTOR_BASE_SPEED
            # after backing up, commit toward the least-bad heading found
            self._phase = "commit"
            self._phase_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # safety fallback
        self._phase = "cruise"
        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    def _finish_sweep(self, rtheta, now):
        """Pick the most-open heading from the sweep; commit or escape dead-end."""
        best_h, best_c = max(self._sweep_samples, key=lambda s: s[1])
        self._commit_target = best_h
        self._phase_t = now
        self.motion = "stop"
        if best_c < MIN_CLEARANCE:
            # nowhere is open -> back up first, then turn to least-bad heading
            self._phase = "backup"
        else:
            self._phase = "commit"
        return MOTOR_STOP, MOTOR_STOP

    # ---- breadcrumb return-to-base ----
    def _return_drive(self, rx, ry, rtheta):
        if self._return_idx >= len(self._return_path):
            self.state = NavState.ARRIVED
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        tx, ty, _ = self._return_path[self._return_idx]
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)

        if dist < 0.12:
            self._return_idx += 1
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        target_th = math.atan2(dy, dx)
        err = _ang_diff(target_th, rtheta)
        if abs(err) > 0.35:
            return self._pivot(+1 if err > 0 else -1)
        self.motion = "forward"
        return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))
