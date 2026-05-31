"""navigator.py — Explore / approach-person / return state machine.

Recon logic:

  CRUISE      Drive forward while front is clear.

  SWEEP       On an obstacle: use the servo to look LEFT / CENTER / RIGHT
              (±90 deg) without moving the car — that covers the front 180 deg.
              Then pivot the car 180 deg and repeat the servo sweep for the rear
              180 deg. Pick the most open heading from all 6 samples and commit.
              Much faster than rotating the whole car 8 × 45 deg.

  COMMIT      Turn in place to the chosen heading (gyro closed-loop).
  ADVANCE     Drive forward; only SWEEP again when blocked.
  BACKUP      Dead-end escape: reverse then commit to least-bad heading.

  APPROACH    A confirmed person was seen. Drive forward slowly, using the front
              ultrasonic AND the camera bbox size to know when close enough.
              Do NOT stop immediately — mark the position and come closer.
              Once close, log the confirmed position and resume CRUISE.

RETURN replays breadcrumbs from the motion-model pose.

nav.motion flag ("forward"|"pivot"|"stop") lets dead_reckoning.py advance
the pose only when the robot is actually translating.
"""

import math, time
from enum import Enum
from config import (MOTOR_BASE_SPEED, MOTOR_TURN_SPEED, MOTOR_STOP,
                    US_STOP_DISTANCE, TOF_STOP_DISTANCE,
                    SWEEP_SETTLE_S, MIN_CLEARANCE, PIVOT_STEP_TIMEOUT,
                    BACKUP_TIME, EXPLORATION_TIMEOUT,
                    SERVO_CENTER, SERVO_LEFT, SERVO_RIGHT, SERVO_SWEEP_ANGLES,
                    PERSON_APPROACH_DIST, PERSON_BBOX_CLOSE_PX)

HEADING_TOLERANCE = math.radians(12)
MOTOR_APPROACH_SPEED = 35   # slower than cruise when closing on a person

class NavState(Enum):
    IDLE="idle"; EXPLORE="explore"; APPROACH="approach"; RETURN="return"; ARRIVED="arrived"

def _ang_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

# Servo angle (deg from center, e.g. -90/0/+90) -> absolute servo position
def _servo_pos(offset_deg):
    return max(20, min(160, SERVO_CENTER + offset_deg))

class Navigator:
    def __init__(self, sensors=None):
        self.state = NavState.IDLE
        self._explore_start = None
        self._return_path = []
        self._return_idx = 0
        self.sensors = sensors
        self.servo_angle = SERVO_CENTER
        self.motion = "stop"

        # recon sub-state
        self._phase = "cruise"
        self._sweep_samples = []       # [(world_heading, clearance), ...]
        self._sweep_step = 0           # index into the current sweep sequence
        self._sweep_half = 0           # 0 = front half, 1 = rear half
        self._settle_t = None
        self._phase_t = 0.0
        self._commit_target = None
        # APPROACH state
        self._approach_logged = False

    # ── transitions ──────────────────────────────────────────────────────────

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
        self._sweep_step = 0
        self._sweep_half = 0
        self._settle_t = None
        self._commit_target = None
        self._approach_logged = False
        self.motion = "stop"
        self._set_servo(SERVO_CENTER)

    def _set_servo(self, deg):
        self.servo_angle = deg
        if self.sensors:
            self.sensors.set_servo(deg)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def state_name(self): return self.state.value

    @property
    def exploration_elapsed(self):
        return time.time() - self._explore_start if self._explore_start else 0.0

    # ── main entry ───────────────────────────────────────────────────────────

    def compute_command(self, tof, us, rx, ry, rtheta, detections):
        """detections: list of confirmed Detection objects (may be empty).
        Returns (left_speed, right_speed)."""

        if self.state in (NavState.IDLE, NavState.ARRIVED):
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if self.state == NavState.EXPLORE and self.exploration_elapsed > EXPLORATION_TIMEOUT:
            self.state = NavState.ARRIVED
            self._reset_recon()
            return MOTOR_STOP, MOTOR_STOP

        if self.state == NavState.RETURN:
            return self._return_drive(rx, ry, rtheta)

        if self.state == NavState.APPROACH:
            return self._approach_drive(tof, us, detections, rx, ry, rtheta)

        # EXPLORE
        return self._explore_drive(tof, us, rtheta, detections)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _front_dist(self, tof, us):
        """Trusted forward clearance. US is always forward; ToF only when
        the servo is centered. 0 means no reading → treat as open (99)."""
        vals = [us if us > 0.01 else 99.0]
        if abs(self.servo_angle - SERVO_CENTER) < 5:
            vals.append(tof if tof > 0.01 else 99.0)
        return min(vals)

    def _pivot(self, direction):
        self.motion = "pivot"
        return ((-MOTOR_TURN_SPEED, MOTOR_TURN_SPEED) if direction > 0
                else (MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED))

    def _is_blocked(self, tof, us):
        return ((us > 0.01 and us < US_STOP_DISTANCE) or
                (abs(self.servo_angle - SERVO_CENTER) < 5 and 0.01 < tof < TOF_STOP_DISTANCE))

    # ── APPROACH: come closer to a detected person ────────────────────────────

    def _approach_drive(self, tof, us, detections, rx, ry, rtheta):
        """Drive slowly toward the person using US + bbox size as proximity.
        Once close enough (or blocked), log 'close confirm' and resume explore."""
        now = time.time()

        # check if close enough by US distance or bbox height
        us_close = (0.01 < us < PERSON_APPROACH_DIST)
        bbox_close = any(
            (d.bbox[3] - d.bbox[1]) >= PERSON_BBOX_CLOSE_PX
            for d in detections
        )

        if us_close or bbox_close:
            # we're close — resume exploration
            self.state = NavState.EXPLORE
            self._phase = "cruise"
            self._set_servo(SERVO_CENTER)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # if front is physically blocked (wall not person), sweep instead
        if self._is_blocked(tof, us):
            self.state = NavState.EXPLORE
            self._start_sweep(rtheta, now)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # if person lost from camera, just resume cruise
        if not detections:
            self.state = NavState.EXPLORE
            self._phase = "cruise"
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        self.motion = "forward"
        return MOTOR_APPROACH_SPEED, MOTOR_APPROACH_SPEED

    # ── EXPLORE ───────────────────────────────────────────────────────────────

    def _explore_drive(self, tof, us, rtheta, detections):
        now = time.time()

        # ── CRUISE ──
        if self._phase == "cruise":
            # Person spotted → switch to APPROACH (don't stop, come closer)
            if detections:
                self.state = NavState.APPROACH
                self._approach_logged = False
                self._set_servo(SERVO_CENTER)
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP

            if self._is_blocked(tof, us):
                self._start_sweep(rtheta, now)
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP

            self.motion = "forward"
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

        # ── SWEEP (servo half) ──
        if self._phase == "sweep_servo":
            return self._sweep_servo_step(tof, us, rtheta, now)

        # ── SWEEP PIVOT (car rotates 180 for rear half) ──
        if self._phase == "sweep_pivot":
            return self._sweep_pivot_step(rtheta, now)

        # ── SWEEP REAR (servo sweep of rear half after 180 pivot) ──
        if self._phase == "sweep_rear":
            return self._sweep_servo_step(tof, us, rtheta, now, rear=True)

        # ── COMMIT ──
        if self._phase == "commit":
            err = _ang_diff(self._commit_target, rtheta)
            if abs(err) > HEADING_TOLERANCE and (now - self._phase_t) < PIVOT_STEP_TIMEOUT * 2:
                return self._pivot(+1 if err > 0 else -1)
            self._phase = "advance"
            self._set_servo(SERVO_CENTER)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── ADVANCE ──
        if self._phase == "advance":
            if detections:
                self.state = NavState.APPROACH
                self._approach_logged = False
                self._set_servo(SERVO_CENTER)
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP
            if self._is_blocked(tof, us):
                self._phase = "cruise"
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP
            self.motion = "forward"
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

        # ── BACKUP ──
        if self._phase == "backup":
            if (now - self._phase_t) < BACKUP_TIME:
                self.motion = "stop"
                return -MOTOR_BASE_SPEED, -MOTOR_BASE_SPEED
            self._phase = "commit"
            self._phase_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # fallback
        self._phase = "cruise"
        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    # ── SWEEP internals ───────────────────────────────────────────────────────

    def _start_sweep(self, rtheta, now):
        self._sweep_samples = []
        self._sweep_step = 0
        self._sweep_half = 0
        self._settle_t = None
        self._phase_t = now
        self._phase = "sweep_servo"
        # point servo at first angle of the front sweep
        self._set_servo(_servo_pos(SERVO_SWEEP_ANGLES[0]))

    def _sweep_servo_step(self, tof, us, rtheta, now, rear=False):
        """Servo sweeps through SERVO_SWEEP_ANGLES without moving the car.
        Each step: command servo angle → settle → read ToF → record.
        Clearance at a given servo angle = ToF distance (the head points that way).
        The world heading for each sample = robot heading + servo offset."""
        angles = SERVO_SWEEP_ANGLES

        if self._settle_t is None:
            # command the servo and wait for it to arrive
            target_servo = _servo_pos(angles[self._sweep_step])
            self._set_servo(target_servo)
            self._settle_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if (now - self._settle_t) < SWEEP_SETTLE_S:
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # read clearance: ToF is on the head so it now points at this angle
        clearance = tof if tof > 0.01 else 99.0
        # world heading = robot heading + servo offset (in radians)
        offset_rad = math.radians(angles[self._sweep_step])
        # note: servo LEFT (positive offset) = CCW = positive angle
        world_h = _wrap(rtheta + offset_rad)
        self._sweep_samples.append((world_h, clearance))

        self._sweep_step += 1
        self._settle_t = None

        if self._sweep_step < len(angles):
            # move to next servo angle
            self._set_servo(_servo_pos(angles[self._sweep_step]))
            return MOTOR_STOP, MOTOR_STOP

        # finished this half
        self._sweep_step = 0
        self._set_servo(SERVO_CENTER)

        if not rear:
            # front half done → pivot car 180 for rear half
            self._phase = "sweep_pivot"
            self._sweep_half = 1
            self._phase_t = now
            self._pivot_start_h = rtheta
        else:
            # rear half done → choose best heading
            self._finish_sweep(now)

        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    def _sweep_pivot_step(self, rtheta, now):
        """Pivot car 180 deg to face rear, then start rear servo sweep."""
        target_h = _wrap(self._pivot_start_h + math.pi)
        err = _ang_diff(target_h, rtheta)
        timed_out = (now - self._phase_t) > PIVOT_STEP_TIMEOUT * 2

        if abs(err) > HEADING_TOLERANCE and not timed_out:
            return self._pivot(+1)

        # reached 180° position → settle then do rear servo sweep
        if self._settle_t is None:
            self._settle_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if (now - self._settle_t) < SWEEP_SETTLE_S:
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        self._settle_t = None
        self._sweep_step = 0
        self._phase = "sweep_rear"
        self._set_servo(_servo_pos(SERVO_SWEEP_ANGLES[0]))
        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    def _finish_sweep(self, now):
        """Pick the most open heading from all samples; commit or back up."""
        if not self._sweep_samples:
            self._phase = "cruise"
            return
        best_h, best_c = max(self._sweep_samples, key=lambda s: s[1])
        self._commit_target = best_h
        self._phase_t = now
        if best_c < MIN_CLEARANCE:
            self._phase = "backup"
        else:
            self._phase = "commit"

    # ── RETURN ────────────────────────────────────────────────────────────────

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
