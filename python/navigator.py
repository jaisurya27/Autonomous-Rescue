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
from config import (MOTOR_BASE_SPEED, MOTOR_SLOW_SPEED, MOTOR_TURN_SPEED, MOTOR_REVERSE_SPEED, MOTOR_STOP,
                    TOF_STOP_DISTANCE, TOF_WARN_DISTANCE, RETURN_TOF_STOP,
                    SWEEP_SETTLE_S, MIN_CLEARANCE, PIVOT_STEP_TIMEOUT,
                    BACKUP_TIME, EXPLORATION_TIMEOUT,
                    STUCK_TIMEOUT, STUCK_MOVE_THRESHOLD, STUCK_REVERSE_TIME,
                    MAX_SWEEP_ATTEMPTS, RETURN_TURN_TIMEOUT,
                    SERVO_CENTER, SERVO_LEFT, SERVO_RIGHT, SERVO_BEARING_SIGN,
                    CAR_SURVEY_OFFSETS, SERVO_SCAN_STEP_DEG, SERVO_SCAN_INTERVAL,
                    GREEDY_COMMIT_SCORE, CAMERA_CLEAR_THRESHOLD,
                    PERSON_APPROACH_DIST, PERSON_BBOX_CLOSE_PX,
                    MANUAL_DRIVE_SPEED, MANUAL_TURN_SPEED, MANUAL_HEAD_STEP)

HEADING_TOLERANCE    = math.radians(15)
MOTOR_APPROACH_SPEED = 35

# On an obstacle the car should deflect at most this far from its original
# heading — a left/right turn within 90°, never a near-180° turn-around.
# Keeps the rover making forward progress instead of doubling back.
MAX_DEFLECTION = math.radians(90)

# Car rotation steps during sweep (degrees from heading at sweep start).
CAR_SWEEP_OFFSETS = [0, 90, -90]   # CENTER first, then LEFT 90, RIGHT 90
# Scanning at current heading first → greedy can fire in < 1 settle period.
# Only if the ±60° servo arc is all blocked does the car body rotate.

# Three servo positions per car orientation: right 60°, straight, left 60°.
# ToF is on the servo head so each reading gives a direct distance in that direction.
# Greedy commit fires as soon as any position reads clear — no camera score needed.
SERVO_SWEEP_POS = [SERVO_LEFT, SERVO_CENTER, SERVO_RIGHT]
# with SERVO_BEARING_SIGN = -1.0:
#   SERVO_LEFT (150) → world -60° (right 60°)
#   SERVO_CENTER (90) → world  0° (straight)
#   SERVO_RIGHT (30) → world +60° (left 60°)

class NavState(Enum):
    IDLE="idle"; EXPLORE="explore"; APPROACH="approach"; RETURN="return"; ARRIVED="arrived"
    MANUAL="manual"

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
        self._cam_center_score = 0.5

        # survey / sweep mode
        self._active_sweep_offsets = CAR_SWEEP_OFFSETS
        self._sweep_full           = False
        self._pending_survey       = False

        # cruise servo scan
        self._cruise_servo_dir = 1
        self._servo_scan_t     = 0.0

        # Forward ToF cache: only updated when servo is near centre.
        self._tof_forward = 99.0

        # Best direction seen during cruise servo scan.
        # Updated continuously while cruising; consumed by the next sweep.
        self._cruise_best_h     = None   # world heading of best recent scan reading
        self._cruise_best_score = 0.0    # its score

        # return
        self._return_path          = []
        self._return_idx           = 0
        self._return_turn_start    = None
        self._return_dodge_until   = 0.0
        self._return_dodge_dir     = 1
        self._return_advance_until = 0.0

        # manual RC control (final-demo-feature)
        self._manual_drive = "stop"   # forward / back / left / right / stop

    # ── public transitions ───────────────────────────────────────────────────

    def start_exploration(self):
        self._reset()
        self.state = NavState.EXPLORE
        self._explore_start = time.time()
        # No initial survey: car starts moving immediately.
        # Full 360° surveys happen after backup or stuck-reversal (see _start_sweep full=True).

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

    # ── manual RC control (final-demo-feature) ────────────────────────────────

    def start_manual(self):
        """Switch to manual mode. Autonomy stops driving; the operator's
        joypad commands drive the car directly. Pose/grid/detection/rendering
        all keep running in the control loop regardless of state."""
        self._reset()
        self.state = NavState.MANUAL
        self._manual_drive = "stop"
        self._set_servo(SERVO_CENTER)

    def set_manual_drive(self, cmd):
        """Set the current manual movement: forward/back/left/right/stop."""
        if cmd in ("forward", "back", "left", "right", "stop"):
            self._manual_drive = cmd

    def manual_head(self, direction):
        """Pan the servo head in manual mode. direction: 'left'/'right'/'center'."""
        if direction == "center":
            self._set_servo(SERVO_CENTER)
            return
        step = MANUAL_HEAD_STEP if direction == "left" else -MANUAL_HEAD_STEP
        new_angle = self.servo_angle + step
        # Clamp to the servo's usable arc (SERVO_RIGHT..SERVO_LEFT).
        lo, hi = min(SERVO_RIGHT, SERVO_LEFT), max(SERVO_RIGHT, SERVO_LEFT)
        self._set_servo(max(lo, min(hi, new_angle)))

    def _manual_drive_cmd(self):
        """Translate the latched manual command into motor speeds."""
        if self._manual_drive == "forward":
            self.motion = "forward"
            return MANUAL_DRIVE_SPEED, MANUAL_DRIVE_SPEED
        if self._manual_drive == "back":
            self.motion = "stop"
            return -MANUAL_DRIVE_SPEED, -MANUAL_DRIVE_SPEED
        if self._manual_drive == "left":
            self.motion = "pivot"
            return -MANUAL_TURN_SPEED, MANUAL_TURN_SPEED
        if self._manual_drive == "right":
            self.motion = "pivot"
            return MANUAL_TURN_SPEED, -MANUAL_TURN_SPEED
        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

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
        self._approach_logged      = False
        self._return_turn_start    = None
        self._return_dodge_until   = 0.0
        self._return_advance_until = 0.0
        self._active_sweep_offsets = CAR_SWEEP_OFFSETS
        self._sweep_full           = False
        self._pending_survey       = False
        self._cruise_servo_dir     = 1
        self._servo_scan_t         = 0.0
        self._tof_forward          = 99.0
        self._cruise_best_h        = None
        self._cruise_best_score    = 0.0
        self._manual_drive         = "stop"
        self.motion                = "stop"
        self._set_servo(SERVO_CENTER)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def state_name(self):
        return self.state.value

    @property
    def exploration_elapsed(self):
        return time.time() - self._explore_start if self._explore_start else 0.0

    # ── main entry ───────────────────────────────────────────────────────────

    def compute_command(self, tof, rx, ry, rtheta, detections,
                        cam_blocked=False, cam_left_score=0.5, cam_right_score=0.5,
                        cam_center_score=0.5):
        self._rtheta           = rtheta
        self._cam_left_score   = cam_left_score
        self._cam_right_score  = cam_right_score
        self._cam_center_score = cam_center_score

        # Cache forward ToF only when servo is near centre (within 20°).
        # When servo is panned during cruise scan, ToF reads sideways — using that
        # reading for obstacle detection causes constant false-positive sweeps.
        if abs(self.servo_angle - SERVO_CENTER) <= 20:
            self._tof_forward = tof

        if self.state in (NavState.IDLE, NavState.ARRIVED):
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # Manual RC: operator drives directly; no autonomy logic runs, but the
        # caller (control loop) still updates pose/grid/detection/dashboard.
        if self.state == NavState.MANUAL:
            return self._manual_drive_cmd()

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
            return self._return_drive(tof, rx, ry, rtheta)

        if self.state == NavState.APPROACH:
            return self._approach_drive(tof, detections, rx, ry, rtheta)

        return self._explore_drive(tof, rx, ry, rtheta, detections, cam_blocked)

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

    def _is_blocked(self, tof):
        """Use the cached forward ToF reading so panned servo readings don't
        cause false stops during cruise scan."""
        return 0.01 < self._tof_forward < TOF_STOP_DISTANCE

    def _front_clearance(self, tof):
        """Clearance in the servo's CURRENT pan direction (for sweep scoring)."""
        return tof if tof > 0.01 else 99.0

    # ── stuck detection ───────────────────────────────────────────────────────

    def _check_stuck(self, rx, ry):
        now = time.time()
        if self._stuck_reversing:
            if now < self._stuck_rev_until:
                return self._reverse()
            self._stuck_reversing = False
            self._start_sweep(self._rtheta, rx, ry)   # same L/C/R scan after reversal
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

    def _approach_drive(self, tof, detections, rx, ry, rtheta):
        tof_close  = (0.01 < tof < PERSON_APPROACH_DIST)
        bbox_close = any((d.bbox[3]-d.bbox[1]) >= PERSON_BBOX_CLOSE_PX for d in detections)

        if tof_close or bbox_close:
            self.state  = NavState.EXPLORE
            self._phase = "cruise"
            self._set_servo(SERVO_CENTER)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        if self._is_blocked(tof):
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

    def _explore_drive(self, tof, rx, ry, rtheta, detections, cam_blocked=False):
        now = time.time()

        # ── CRUISE ──────────────────────────────────────────────────────────
        if self._phase == "cruise":
            if self._pending_survey:   # reserved flag, currently unused
                self._pending_survey = False

            if detections:
                self.state = NavState.APPROACH
                self._set_servo(SERVO_CENTER)
                self.motion = "stop"
                return MOTOR_STOP, MOTOR_STOP

            # Hard stop: ToF within stop threshold (also guards with US when servo panned)
            if self._is_blocked(tof):
                self._start_sweep(rtheta, rx, ry)
                return MOTOR_STOP, MOTOR_STOP

            # Graduated slow-down using forward-cached ToF (not panned reading).
            if 0.01 < self._tof_forward < TOF_WARN_DISTANCE:
                factor = (self._tof_forward - TOF_STOP_DISTANCE) / (TOF_WARN_DISTANCE - TOF_STOP_DISTANCE)
                speed = max(MOTOR_SLOW_SPEED, int(MOTOR_BASE_SPEED * max(0.0, min(1.0, factor))))
            else:
                speed = MOTOR_BASE_SPEED

            # Keep the head locked FORWARD while cruising. The ToF rides on the
            # servo, so a centred head means _tof_forward is always a live
            # straight-ahead reading — the car reacts instantly to an obstacle
            # dead ahead instead of waiting for the head to swing back to centre.
            # The servo is only used to look around AFTER an obstacle stops us
            # (the sweep phase pans L/C/R to choose a turn).
            if self.servo_angle != SERVO_CENTER:
                self._set_servo(SERVO_CENTER)

            self._sweep_attempts = 0
            self.motion = "forward"
            return speed, speed

        # ── SWEEP ────────────────────────────────────────────────────────────
        # Step sequence: turn car to (start+LEFT90), settle+read,
        #                turn to (start+0),     settle+read,
        #                turn to (start+RIGHT90), settle+read  → pick best.
        if self._phase == "sweep":
            return self._sweep_step_exec(tof, rtheta, now, rx, ry)

        # ── COMMIT (turn car to chosen heading) ───────────────────────────────
        if self._phase == "commit":
            err = _ang_diff(self._commit_target, rtheta)
            timed_out = (now - self._phase_t) > PIVOT_STEP_TIMEOUT * 2
            if abs(err) > HEADING_TOLERANCE and not timed_out:
                return self._pivot(+1 if err > 0 else -1)
            # verify we can actually go forward from here
            if self._is_blocked(tof):
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
            if self._is_blocked(tof):
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
            # After backing up, do the same L/C/R scan from the new position.
            self._start_sweep(rtheta, rx, ry)
            return MOTOR_STOP, MOTOR_STOP

        # fallback
        self._phase = "cruise"
        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    # ── SWEEP internals ───────────────────────────────────────────────────────

    def _start_sweep(self, rtheta, rx, ry, full=False):
        """Start a sweep. full=True → full 360° survey (CAR_SURVEY_OFFSETS, no back-exclusion).
        Normal sweep is 3-position (L/C/R). Full survey is used at exploration start
        and after backing up, to properly understand the environment before committing."""
        now = time.time()
        self._sweep_attempts += 1
        moved = math.hypot(rx - self._sweep_ref_pos[0], ry - self._sweep_ref_pos[1])

        # If the cruise servo scan already found a clear direction before the obstacle
        # stop, use it immediately — skip the sweep entirely.
        if (self._cruise_best_h is not None and
                self._cruise_best_score >= GREEDY_COMMIT_SCORE):
            back_h   = _wrap(rtheta + math.pi)
            not_back = abs(_ang_diff(self._cruise_best_h, back_h)) > math.radians(30)
            if not_back:
                print(f"[nav] cruise scan → instant commit "
                      f"{math.degrees(self._cruise_best_h):.0f}° "
                      f"(score={self._cruise_best_score:.2f})")
                self._commit_target    = self._cruise_best_h
                self._phase_t          = now
                self._phase            = "commit"
                self._cruise_best_h    = None
                self._cruise_best_score = 0.0
                self.motion = "stop"
                return

        self._cruise_best_h    = None   # reset for next cruise leg
        self._cruise_best_score = 0.0

        if self._sweep_attempts == 1:
            self._sweep_ref_pos = (rx, ry)
        elif self._sweep_attempts >= MAX_SWEEP_ATTEMPTS and moved < 0.08:
            print(f"[nav] {self._sweep_attempts} sweeps, no progress → backing up")
            self._sweep_attempts = 0
            self._sweep_ref_pos  = (rx, ry)
            self._phase          = "backup"
            self._backup_until   = now + BACKUP_TIME
            self.motion = "stop"
            self._set_servo(SERVO_CENTER)
            return

        offsets = CAR_SURVEY_OFFSETS if full else CAR_SWEEP_OFFSETS
        self._active_sweep_offsets = offsets
        self._sweep_full  = full
        self._sweep_samples  = []
        self._sweep_step     = 0
        self._servo_step     = 0
        self._sweep_start_h  = rtheta
        self._settle_t       = None
        self._phase          = "sweep"
        self._phase_t        = now
        self._step_start_t   = now
        self._sweep_target_h = _wrap(rtheta + math.radians(offsets[0]))
        self.motion = "stop"

    def _sweep_step_exec(self, tof, rtheta, now, rx, ry):
        """N car rotations × 3 servo positions = 3N samples.
        Normal sweep: LEFT-90 / CENTER / RIGHT-90 (9 samples).
        Full survey:  0 / +90 / +180 / -90 (12 samples, covers full 360°).
        """
        if self._sweep_step >= len(self._active_sweep_offsets):
            self._finish_sweep(rtheta, now)
            return MOTOR_STOP, MOTOR_STOP

        # ── helpers shared by A and B ──────────────────────────────────────────
        def _fwd_clear():
            """Forward clearance, treating ToF=0 (out-of-range) as 99 m."""
            return self._tof_forward if self._tof_forward > 0.01 else 99.0

        def _try_opportunistic(heading):
            """Commit to `heading` immediately if it is clear and within the
            ±90° deflection cap (no turning around)."""
            within_90 = abs(_ang_diff(heading, self._sweep_start_h)) <= MAX_DEFLECTION
            if within_90 and _fwd_clear() >= GREEDY_COMMIT_SCORE:
                print(f"[nav] opportunistic → {math.degrees(heading):.0f}° "
                      f"(tof_fwd={_fwd_clear():.2f}m)")
                self._commit_target = heading
                self._phase_t       = now
                self._phase         = "commit"
                self._sweep_samples = []
                self.motion         = "stop"
                return True
            return False

        # ── A: rotate car to target heading ────────────────────────────────
        if self._servo_step == 0 and self._settle_t is None:
            err      = _ang_diff(self._sweep_target_h, rtheta)
            timed_out = (now - self._step_start_t) > PIVOT_STEP_TIMEOUT

            # While rotating, servo is at centre → _tof_forward reads the current
            # heading. If it's clear, stop rotating and commit NOW.
            if abs(err) > HEADING_TOLERANCE:
                if _try_opportunistic(rtheta):
                    return MOTOR_STOP, MOTOR_STOP

            if abs(err) > HEADING_TOLERANCE and not timed_out:
                return self._pivot(+1 if err > 0 else -1)
            # car reached target — move servo to first pan position and settle
            self._set_servo(SERVO_SWEEP_POS[0])
            self._settle_t = now
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── B: settle after servo moved ─────────────────────────────────────
        if self._settle_t is not None and (now - self._settle_t) < SWEEP_SETTLE_S:
            # Don't wait idle: if forward is already clearly open, commit now.
            # Servo may not be fully settled yet but ToF=0 (OOR) is unambiguous.
            if self._tof_forward == 0.0:   # OOR = definitely nothing ahead
                if _try_opportunistic(rtheta):
                    return MOTOR_STOP, MOTOR_STOP
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── C: record sample at current car+servo position ──
        servo_pos = SERVO_SWEEP_POS[self._servo_step]
        servo_offset_rad = math.radians((servo_pos - SERVO_CENTER) * SERVO_BEARING_SIGN)
        world_h = _wrap(rtheta + servo_offset_rad)

        # Score = clearance in this pan direction.
        # ToF is on the servo head → direct distance in whatever direction servo points.
        # Camera center_score (middle strip) shows what the head is pointing at after settle.
        # A direction is clear if EITHER:
        #   - ToF reads far / OOR (no obstacle detected by laser), OR
        #   - Camera sees open space (low variance in centre = no wall/object in shot)
        tof_score = self._front_clearance(tof)   # 0.01–2 m or 99 if OOR
        cam_clear = self._cam_center_score >= CAMERA_CLEAR_THRESHOLD

        if tof_score >= GREEDY_COMMIT_SCORE:
            # ToF says clearly open — trust it
            score = tof_score
        elif tof_score > TOF_STOP_DISTANCE and cam_clear:
            # ToF is in marginal zone (0.20–0.50m) — could be a weak echo or thin object.
            # Camera confirms open space → treat as clear.
            score = GREEDY_COMMIT_SCORE
        else:
            # ToF reads a real obstacle (< 0.20m) or camera also sees obstacle.
            # Camera cannot override a hard ToF block.
            score = tof_score
        self._sweep_samples.append((world_h, score))

        # ── Greedy commit: act immediately on the first clearly-open direction ──
        # Don't wait for all N samples — if this direction is definitively clear,
        # commit now. Only accept headings within ±90° of the original forward
        # heading so the car deflects left/right and keeps progressing, never a
        # near-180° turn-around.
        within_90 = abs(_ang_diff(world_h, self._sweep_start_h)) <= MAX_DEFLECTION
        if score >= GREEDY_COMMIT_SCORE and within_90:
            print(f"[nav] greedy commit → {math.degrees(world_h):.0f}° (score={score:.1f})")
            self._commit_target = world_h
            self._phase_t       = now
            self._phase         = "commit"
            self._sweep_samples = []   # discard partial samples
            self._set_servo(SERVO_CENTER)
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

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

        if self._sweep_step < len(self._active_sweep_offsets):
            self._sweep_target_h = _wrap(
                self._sweep_start_h + math.radians(self._active_sweep_offsets[self._sweep_step])
            )
        # Do NOT set _settle_t here. Section A requires _settle_t is None to trigger
        # the car rotation. The servo recenter is fast and will finish during rotation.
        # (The old _settle_t=now here was causing section A to be permanently skipped.)

        self.motion = "stop"
        return MOTOR_STOP, MOTOR_STOP

    def _finish_sweep(self, rtheta, now):
        if not self._sweep_samples:
            self._phase = "cruise"
            return

        # For a full 360° survey every direction was intentionally sampled, so use all.
        # For a 3-position obstacle sweep, keep only headings within ±90° of the
        # original forward heading: the car deflects left/right and keeps making
        # forward progress instead of turning around into where it came from.
        if self._sweep_full:
            candidates = self._sweep_samples
        else:
            candidates = [
                (h, c) for h, c in self._sweep_samples
                if abs(_ang_diff(h, self._sweep_start_h)) <= MAX_DEFLECTION
            ]
            if not candidates:
                candidates = self._sweep_samples

        # Score is ToF distance. Only consider headings with genuine clearance.
        # TOF_STOP_DISTANCE * 1.5 = 30 cm minimum — avoids committing to a direction
        # that will immediately re-trigger a stop on the first advance step.
        driveable = [(h, c) for h, c in candidates if c > TOF_STOP_DISTANCE * 1.5]
        if driveable:
            best_h, best_c = max(driveable, key=lambda s: s[1])
            self._commit_target = best_h
            self._phase_t = now
            self._phase = "commit"
        else:
            # Nothing open anywhere — back up and try from a new position
            self._phase        = "backup"
            self._backup_until = now + BACKUP_TIME

    # ── RETURN ────────────────────────────────────────────────────────────────

    def _return_drive(self, tof, rx, ry, rtheta):
        """Replay breadcrumbs in reverse toward (0,0).

        For each waypoint: face it first (gyro-closed-loop), then drive.
        If an obstacle blocks the path, stop and sweep for a clear heading
        (same as explore), then continue toward the next waypoint.
        Falls back to aiming straight at (0,0) if breadcrumbs run out.
        """
        now = time.time()

        # ── obstacle avoidance during return ──
        if self._phase in ("sweep", "commit", "backup"):
            cmd = self._explore_drive(tof, rx, ry, rtheta, [])
            if self._phase == "cruise":
                self._phase = "return_drive"
            return cmd

        # After committing to a clear heading, drive forward for BACKUP_TIME seconds
        # to physically clear the obstacle before resuming return navigation.
        # Previously missing: without this, _phase="advance" fell through to
        # _phase="return_drive" every loop and the car never moved after a sweep.
        if self._phase == "advance":
            if self._return_advance_until == 0.0:
                self._return_advance_until = now + BACKUP_TIME
            if now < self._return_advance_until:
                if self._is_blocked(tof):
                    # Hit something new mid-advance — sweep again from here
                    self._return_advance_until = 0.0
                    self._start_sweep(rtheta, rx, ry)
                    return MOTOR_STOP, MOTOR_STOP
                self.motion = "forward"
                return MOTOR_BASE_SPEED, MOTOR_BASE_SPEED
            # Advance time elapsed — resume return navigation
            self._return_advance_until = 0.0

        self._phase = "return_drive"

        # ── arrived? ──
        dist_home = math.hypot(rx, ry)
        if dist_home < 0.25:
            self.state = NavState.ARRIVED
            self.motion = "stop"
            return MOTOR_STOP, MOTOR_STOP

        # ── obstacle check — trigger sweep ──
        if 0.01 < tof < RETURN_TOF_STOP:
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
