"""navigator.py — Explore / approach-person / return state machine.

Sweep logic (simplified):
  On an obstacle the car rotates in three steps — LEFT 90°, CENTER, RIGHT 90°.
  At each step it settles, reads front US + ToF, and records clearance.
  No rear pivot needed: 3 readings span the meaningful forward arc.
  Picks the clearest heading and turns to it (car rotation, not servo).

  If both sweeps fail (no clear heading found twice in a row) it reverses
  slowly, then sweeps again from the new position.

Return:
  Point toward (0,0) and drive. No breadcrumb chasing — breadcrumbs drift.
  If an obstacle appears, do a quick 90° dodge then re-aim at home.
  Gives up breadcrumb replay entirely since dead-reckoning pose drifts.

Camera signal: NOT used for obstacle stopping. Only US + ToF stop the car.
"""

import math, time
from enum import Enum
from config import (MOTOR_BASE_SPEED, MOTOR_TURN_SPEED, MOTOR_REVERSE_SPEED, MOTOR_STOP,
                    US_STOP_DISTANCE, TOF_STOP_DISTANCE,
                    SWEEP_SETTLE_S, MIN_CLEARANCE, PIVOT_STEP_TIMEOUT,
                    BACKUP_TIME, EXPLORATION_TIMEOUT,
                    STUCK_TIMEOUT, STUCK_MOVE_THRESHOLD, STUCK_REVERSE_TIME,
                    MAX_SWEEP_ATTEMPTS, RETURN_TURN_TIMEOUT, RETURN_US_STOP,
                    SERVO_CENTER, SERVO_LEFT, SERVO_RIGHT,
                    PERSON_APPROACH_DIST, PERSON_BBOX_CLOSE_PX)

HEADING_TOLERANCE    = math.radians(15)
MOTOR_APPROACH_SPEED = 35

# Car rotation steps during sweep (degrees from heading at sweep start).
CAR_SWEEP_OFFSETS = [90, 0, -90]   # LEFT 90, CENTER, RIGHT 90

# Servo pan angles for each car position (absolute servo degrees).
# At each car angle the head pans LEFT, CENTER, RIGHT to give 3 camera samples.
SERVO_SWEEP_POS = [SERVO_LEFT, SERVO_CENTER, SERVO_RIGHT]

class NavState(Enum):
    IDLE="idle"; EXPLORE="explore"; APPROACH="approach"; RETURN="return"; ARRIVED="arrived"

def _ang_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Navigator:
    def __init__(self, sensors=None):
        self.state    = NavState.IDLE
        self.sensors  = sensors
        self.servo_angle = SERVO_CENTER
        self.motion   = "stop"

        self._explore_start = None
        self._phase   = "cruise"
        self._phase_t = 0.0

        # sweep
        self._sweep_samples    = []   # [(world_heading, score)]
        self._sweep_step       = 0    # car rotation index: 0=LEFT90, 1=CENTER, 2=RIGHT90
        self._servo_step       = 0    # servo sub-step: 0=LEFT, 1=CENTER, 2=RIGHT
        self._sweep_start_h    = 0.0  # robot heading when sweep began
        self._sweep_target_h   = 0.0  # car heading target for current car step
        self._settle_t         = None
        self._step_start_t     = 0.0
        self._sweep_attempts   = 0
        self._sweep_ref_pos    = (0.0, 0.0)

        # commit
        self._commit_target    = 0.0

        # backup
        self._backup_until     = 0.0

        # stuck
        self._stuck_ref_pos    = (0.0, 0.0)
        self._stuck_ref_t      = time.time()
        self._stuck_rev_until  = 0.0
        self._stuck_reversing  = False

        # approach
        self._approach_logged  = False
        self._rtheta           = 0.0
        self._cam_left_score   = 0.5
        self._cam_right_score  = 0.5

        # return
        self._return_path        = []
        self._return_idx         = 0
        self._return_turn_start  = None
        self._return_dodge_until = 0.0
        self._return_dodge_dir   = 1

    # ── public transitions ───────────────────────────────────────────────────

    def start_exploration(self):
        self._reset()
        self.state = NavState.EXPLORE
        self._explore_start = time.time()

    def start_return(self, breadcrumbs=None):
        self._reset()
        self.state = NavState.RETURN
        # Replay breadcrumbs in reverse. Each crumb is (x, y, theta).
        # If no breadcrumbs given, we'll aim straight at (0,0) as fallback.
        if breadcrumbs and len(breadcrumbs) > 1:
            self._return_path = list(reversed(breadcrumbs))
        else:
            self._return_path = []
        self._return_idx = 0

    def stop(self):
        self._reset()
        self.state = NavState.IDLE

    def _reset(self):
        self._phase            = "cruise"
        self._phase_t          = 0.0
        self._sweep_samples    = []
        self._sweep_step       = 0
        self._servo_step       = 0
        self._settle_t         = None
        self._step_start_t     = 0.0
        self._sweep_attempts   = 0
        self._sweep_ref_pos    = (0.0, 0.0)
        self._commit_target    = 0.0
        self._backup_until     = 0.0
        self._stuck_ref_pos    = (0.0, 0.0)
        self._stuck_ref_t      = time.time()
        self._stuck_rev_until  = 0.0
        self._stuck_reversing  = False
        self._approach_logged  = False
        self._return_turn_start = None
        self._return_dodge_until = 0.0
        self.motion            = "stop"
        self._set_servo(SERVO_CENTER)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def state_name(self):
        return self.state.value

    @property
    def exploration_elapsed(self):
        return time.time() - self._explore_start if self._explore_start else 0.0

    # ── main entry ───────────────────────────────────────────────────────────

    def compute_command(self, tof, us, rx, ry, rtheta, detections,
                        cam_blocked=False, cam_left_score=0.5, cam_right_score=0.5):
        # store for use in _check_stuck which doesn't receive rtheta directly
        self._rtheta = rtheta
        # store camera scores so sweep can use them when reading servo positions
        self._cam_left_score  = cam_left_score
        self._cam_right_score = cam_right_score

        if self.state in (NavState.IDLE, NavState.ARRIVED):
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if self.state == NavState.EXPLORE and self.exploration_elapsed > EXPLORATION_TIMEOUT:
            self.state = NavState.ARRIVED
            self._reset()
            return MOTOR_STOP, MOTOR_STOP

        # stuck check (explore + approach only)
        if self.state not in (NavState.IDLE, NavState.ARRIVED, NavState.RETURN):
            cmd = self._check_stuck(rx, ry)
            if cmd is not None:
                return cmd

        if self.state == NavState.RETURN:
            return self._return_drive(tof, us, rx, ry, rtheta)

        if self.state == NavState.APPROACH:
            return self._approach_drive(tof, us, detections, rx, ry, rtheta)

        return self._explore_drive(tof, us, rx, ry, rtheta, detections, cam_blocked)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _set_servo(self, deg):
        self.servo_angle = deg
        if self.sensors:
            self.sensors.set_servo(deg)

    def _pivot(self, direction):
        self.motion = "pivot"
        return ((-MOTOR_TURN_SPEED, MOTOR_TURN_SPEED) if direction > 0
                else (MOTOR_TURN_SPEED, -MOTOR_TURN_SPEED))

    def _reverse(self):
        self.motion = "stop"
        return -MOTOR_REVERSE_SPEED, -MOTOR_REVERSE_SPEED

    def _is_blocked(self, tof, us):
        """Hard stops only — US and centered ToF. No camera signal."""
        us_hit  = (0.01 < us < US_STOP_DISTANCE)
        tof_hit = (abs(self.servo_angle - SERVO_CENTER) < 5 and 0.01 < tof < TOF_STOP_DISTANCE)
        return us_hit or tof_hit

    def _front_clearance(self, tof, us):
        """Best forward distance reading."""
        vals = [us if us > 0.01 else 99.0]
        if abs(self.servo_angle - SERVO_CENTER) < 5:
            vals.append(tof if tof > 0.01 else 99.0)
        return min(vals)

    # ── stuck detection ───────────────────────────────────────────────────────

    def _check_stuck(self, rx, ry):
        now = time.time()
        if self._stuck_reversing:
            if now < self._stuck_rev_until:
                return self._reverse()
            self._stuck_reversing = False
            self._start_sweep(self._rtheta, rx, ry)
            self._stuck_ref_pos = (rx, ry)
            self._stuck_ref_t   = now
            return None

        if self.motion != "forward":
            self._stuck_ref_pos = (rx, ry)
            self._stuck_ref_t   = now
            return None

        moved = math.hypot(rx - self._stuck_ref_pos[0], ry - self._stuck_ref_pos[1])
        if moved >= STUCK_MOVE_THRESHOLD:
            self._stuck_ref_pos = (rx, ry)
            self._stuck_ref_t   = now
            return None

        if (now - self._stuck_ref_t) > STUCK_TIMEOUT:
            print("[nav] stuck — reversing")
            self._stuck_reversing = True
            self._stuck_rev_until = now + STUCK_REVERSE_TIME
            self._stuck_ref_pos   = (rx, ry)
            self._stuck_ref_t     = now
            return self._reverse()

        return None

    # ── APPROACH ─────────────────────────────────────────────────────────────

    def _approach_drive(self, tof, us, detections, rx, ry, rtheta):
        us_close   = (0.01 < us < PERSON_APPROACH_DIST)
        bbox_close = any((d.bbox[3]-d.bbox[1]) >= PERSON_BBOX_CLOSE_PX for d in detections)

        if us_close or bbox_close:
            self.state  = NavState.EXPLORE
            self._phase = "cruise"
            self._set_servo(SERVO_CENTER)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if self._is_blocked(tof, us):
            self.state = NavState.EXPLORE
            self._start_sweep(rtheta, rx, ry)
            return MOTOR_STOP, MOTOR_STOP

        if not detections:
            self.state  = NavState.EXPLORE
            self._phase = "cruise"
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        self.motion = "forward"
        return MOTOR_APPROACH_SPEED, MOTOR_APPROACH_SPEED

    # ── EXPLORE ───────────────────────────────────────────────────────────────

    def _explore_drive(self, tof, us, rx, ry, rtheta, detections, cam_blocked=False):
        now = time.time()

        # ── CRUISE ──────────────────────────────────────────────────────────
        if self._phase == "cruise":
            if detections:
                self.state = NavState.APPROACH
                self._set_servo(SERVO_CENTER)
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP

            # Hard stop: US or ToF within threshold
            if self._is_blocked(tof, us):
                self._start_sweep(rtheta, rx, ry)
                return MOTOR_STOP, MOTOR_STOP

            # Pre-emptive: camera sees obstacle while still far enough to act
            if cam_blocked:
                self._start_sweep(rtheta, rx, ry)
                return MOTOR_STOP, MOTOR_STOP

            self._sweep_attempts = 0
            self.motion = "forward"
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

        # ── SWEEP ────────────────────────────────────────────────────────────
        # Step sequence: turn car to (start+LEFT90), settle+read,
        #                turn to (start+0),     settle+read,
        #                turn to (start+RIGHT90), settle+read  → pick best.
        if self._phase == "sweep":
            return self._sweep_step_exec(tof, us, rtheta, now, rx, ry)

        # ── COMMIT (turn car to chosen heading) ───────────────────────────────
        if self._phase == "commit":
            err = _ang_diff(self._commit_target, rtheta)
            timed_out = (now - self._phase_t) > PIVOT_STEP_TIMEOUT * 2
            if abs(err) > HEADING_TOLERANCE and not timed_out:
                return self._pivot(+1 if err > 0 else -1)
            # verify we can actually go forward from here
            if self._is_blocked(tof, us):
                # heading is still blocked — sweep again
                self._start_sweep(rtheta, rx, ry)
                return MOTOR_STOP, MOTOR_STOP
            self._phase = "advance"
            self._set_servo(SERVO_CENTER)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── ADVANCE ──────────────────────────────────────────────────────────
        if self._phase == "advance":
            if detections:
                self.state = NavState.APPROACH
                self._set_servo(SERVO_CENTER)
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP
            if self._is_blocked(tof, us):
                self._phase = "cruise"
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP
            self._sweep_attempts = 0
            self.motion = "forward"
            return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED

        # ── BACKUP ───────────────────────────────────────────────────────────
        if self._phase == "backup":
            if now < self._backup_until:
                return self._reverse()
            # after backing up: sweep from new position
            self._start_sweep(rtheta, rx, ry)
            return MOTOR_STOP, MOTOR_STOP

        # fallback
        self._phase = "cruise"
        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    # ── SWEEP internals ───────────────────────────────────────────────────────

    def _start_sweep(self, rtheta, rx, ry):
        now = time.time()
        self._sweep_attempts += 1
        moved = math.hypot(rx - self._sweep_ref_pos[0], ry - self._sweep_ref_pos[1])

        if self._sweep_attempts == 1:
            self._sweep_ref_pos = (rx, ry)
        elif self._sweep_attempts > MAX_SWEEP_ATTEMPTS and moved < 0.08:
            print(f"[nav] {self._sweep_attempts} sweeps no progress — backing up")
            self._sweep_attempts = 0
            self._sweep_ref_pos  = (rx, ry)
            self._phase          = "backup"
            self._backup_until   = now + BACKUP_TIME
            self.motion = "stop"
            self._set_servo(SERVO_CENTER)
            return

        self._sweep_samples  = []
        self._sweep_step     = 0    # which car rotation (0=LEFT90, 1=CENTER, 2=RIGHT90)
        self._servo_step     = 0    # which servo pan within current car step
        self._sweep_start_h  = rtheta
        self._settle_t       = None
        self._phase          = "sweep"
        self._phase_t        = now
        self._step_start_t   = now
        # start by turning car LEFT 90°
        self._sweep_target_h = _wrap(rtheta + math.radians(CAR_SWEEP_OFFSETS[0]))
        self.motion = "stop"

    def _sweep_step_exec(self, tof, us, rtheta, now, rx, ry):
        """3 car rotations × 3 servo positions = 9 viewpoints.
        Car rotates to LEFT-90 / CENTER / RIGHT-90.
        At each car position, servo pans LEFT / CENTER / RIGHT.
        US clearance is used when servo is centered; camera scores when panned.
        Picks the world heading with the best combined score.
        """
        if self._sweep_step >= len(CAR_SWEEP_OFFSETS):
            self._finish_sweep(rtheta, now)
            return MOTOR_STOP, MOTOR_STOP

        # ── A: rotate car to target heading (only at start of each car step) ──
        if self._servo_step == 0 and self._settle_t is None:
            err = _ang_diff(self._sweep_target_h, rtheta)
            timed_out = (now - self._step_start_t) > PIVOT_STEP_TIMEOUT
            if abs(err) > HEADING_TOLERANCE and not timed_out:
                return self._pivot(+1 if err > 0 else -1)
            # car reached angle — move servo to first position and start settle
            self._set_servo(SERVO_SWEEP_POS[0])
            self._settle_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── B: settle (servo moved or car just stopped) ──
        if self._settle_t is not None and (now - self._settle_t) < SWEEP_SETTLE_S:
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── C: record sample at current car+servo position ──
        servo_pos = SERVO_SWEEP_POS[self._servo_step]
        servo_offset_rad = math.radians(servo_pos - SERVO_CENTER)
        world_h = _wrap(rtheta + servo_offset_rad)

        if servo_pos == SERVO_CENTER:
            score = self._front_clearance(tof, us)
        elif servo_pos == SERVO_LEFT:
            score = self._cam_left_score * 3.0    # scale ~0-3 m equivalent
        else:
            score = self._cam_right_score * 3.0

        self._sweep_samples.append((world_h, score))

        self._servo_step += 1
        self._settle_t    = None

        if self._servo_step < len(SERVO_SWEEP_POS):
            # move servo to next pan position
            self._set_servo(SERVO_SWEEP_POS[self._servo_step])
            self._settle_t = now
            return MOTOR_STOP, MOTOR_STOP

        # all servo sub-steps done for this car position — recenter servo, advance car step
        self._set_servo(SERVO_CENTER)
        self._servo_step   = 0
        self._sweep_step  += 1
        self._step_start_t = now

        if self._sweep_step < len(CAR_SWEEP_OFFSETS):
            self._sweep_target_h = _wrap(
                self._sweep_start_h + math.radians(CAR_SWEEP_OFFSETS[self._sweep_step])
            )
            self._settle_t = now   # brief settle before next car rotation

        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    def _finish_sweep(self, rtheta, now):
        if not self._sweep_samples:
            self._phase = "cruise"
            return

        # exclude headings within 30° of where we came from (back direction)
        back_h = _wrap(self._sweep_start_h + math.pi)
        candidates = [
            (h, c) for h, c in self._sweep_samples
            if abs(_ang_diff(h, back_h)) > math.radians(30)
        ]
        if not candidates:
            candidates = self._sweep_samples  # nothing excluded, use all

        best_h, best_c = max(candidates, key=lambda s: s[1])
        self._commit_target = best_h
        self._phase_t = now

        if best_c < MIN_CLEARANCE:
            # nothing open anywhere — back up
            self._phase        = "backup"
            self._backup_until = now + BACKUP_TIME
        else:
            self._phase = "commit"

    # ── RETURN ────────────────────────────────────────────────────────────────

    def _return_drive(self, tof, us, rx, ry, rtheta):
        """Replay breadcrumbs in reverse toward (0,0).

        For each waypoint: face it first (gyro-closed-loop), then drive.
        If an obstacle blocks the path, stop and sweep for a clear heading
        (same as explore), then continue toward the next waypoint.
        Falls back to aiming straight at (0,0) if breadcrumbs run out.
        """
        now = time.time()

        # ── obstacle during return — use the same sweep logic as explore ──
        if self._phase in ("sweep", "commit", "backup"):
            cmd = self._explore_drive(tof, us, rx, ry, rtheta, [])
            # Once the sweep resolves and we've advanced a bit, resume return
            if self._phase == "cruise":
                self._phase = "return_drive"
            return cmd

        self._phase = "return_drive"

        # ── arrived? ──
        dist_home = math.hypot(rx, ry)
        if dist_home < 0.25:
            self.state = NavState.ARRIVED
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── obstacle check — trigger sweep ──
        if 0.01 < us < RETURN_US_STOP:
            self._start_sweep(rtheta, rx, ry)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── choose target waypoint ──
        # Skip breadcrumbs that are already behind us (already passed them).
        while self._return_idx < len(self._return_path):
            tx, ty, _ = self._return_path[self._return_idx]
            if math.hypot(tx - rx, ty - ry) < 0.15:
                self._return_idx += 1
                self._return_turn_start = None
            else:
                break

        if self._return_idx < len(self._return_path):
            tx, ty, _ = self._return_path[self._return_idx]
        else:
            # breadcrumbs exhausted — aim straight at origin
            tx, ty = 0.0, 0.0

        # ── face the target ──
        target_th = math.atan2(ty - ry, tx - rx)
        err = _ang_diff(target_th, rtheta)

        if abs(err) > math.radians(20):
            # start/continue turning
            if self._return_turn_start is None:
                self._return_turn_start = now
            if (now - self._return_turn_start) < RETURN_TURN_TIMEOUT:
                return self._pivot(+1 if err > 0 else -1)
            # turn timed out — skip this crumb and try next
            self._return_idx += 1
            self._return_turn_start = None
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── facing it — drive ──
        self._return_turn_start = None
        self.motion = "forward"
        return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED
