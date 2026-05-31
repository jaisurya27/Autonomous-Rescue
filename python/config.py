"""config.py — Recon Rover tunables (their brain + your hardware)."""

# Sensor calibration
ACCEL_BIAS_X = 0.0
ACCEL_BIAS_Y = 0.0
GYRO_BIAS_Z  = 0.0

# Occupancy grid
GRID_RESOLUTION = 0.05
GRID_WIDTH = 200
GRID_HEIGHT = 200
START_X = GRID_WIDTH // 2
START_Y = GRID_HEIGHT // 2
L_FREE = -0.4
L_OCCUPIED = 0.85
L_PRIOR = 0.0
L_MAX = 5.0
L_MIN = -5.0

# Distance sensor (Modulino ToF, in metres)
TOF_MAX_RANGE = 2.0

# Navigation — calibrated to YOUR car (slow, your earlier finding)
BREADCRUMB_INTERVAL = 0.10
EXPLORATION_TIMEOUT = 3600  # 1 hour; was 180s which caused unexpected auto-stop
OBSTACLE_THRESHOLD = 0.25
MOTOR_BASE_SPEED = 48     # gentler cruise so the few-fps camera/CV keeps up (was 55;
                          # 45 stalls on start, 48 still rolls reliably once moving)
MOTOR_SLOW_SPEED = 44     # reduced speed in obstacle warning zone (just above stall)
MOTOR_TURN_SPEED = 65     # in-place pivot — needs more torque than forward; was 40 = stall
MOTOR_STOP = 0

# Manual RC control (final-demo-feature). Speeds for the dashboard joypad.
MANUAL_DRIVE_SPEED = 50    # forward/back when driving manually
MANUAL_TURN_SPEED  = 60    # in-place left/right pivot when driving manually
MANUAL_HEAD_STEP   = 20    # degrees the servo head moves per left/right tap

# --- Recon / obstacle avoidance ---
TOF_STOP_DISTANCE     = 0.20   # ToF hard stop (m). Sole obstacle sensor (US removed).
TOF_WARN_DISTANCE     = 0.45   # ToF slow-down zone (m): car decelerates between here and stop distance
RETURN_TOF_STOP       = 0.20   # ToF obstacle threshold during return
SWEEP_SETTLE_S        = 0.45   # wait after each servo/pivot step before reading
                               # 0.45s: servo physically settles + 1-2 camera frames captured
MIN_CLEARANCE         = 0.10   # if best clearance everywhere below this -> dead end
GREEDY_COMMIT_SCORE   = 0.50   # distance (m) above which a direction is "clearly clear"
CAMERA_CLEAR_THRESHOLD = 0.40  # center_score above this = camera sees open space
                                # (1.0 - variance/CAM_OBSTACLE_VARIANCE; 0=wall, 1=open floor)
PIVOT_STEP_TIMEOUT    = 4.0    # s, safety cap per pivot step
BACKUP_TIME           = 1.8    # s reverse on dead end / stuck (longer = more clearance gained)
MOTOR_REVERSE_SPEED   = 65     # reverse speed (needs more than stall ~45, more than forward)
STUCK_TIMEOUT         = 5.0    # s without meaningful movement -> force reverse+reroute
STUCK_MOVE_THRESHOLD  = 0.04   # m — below this over STUCK_TIMEOUT = stuck
STUCK_REVERSE_TIME    = 1.5    # s reverse when stuck is detected (longer = more clearance)
MAX_SWEEP_ATTEMPTS    = 2      # after this many sweeps without moving, force reverse
RETURN_TURN_TIMEOUT   = 4.0    # s max time to complete one turn step during return
# Camera path vision: only for person detection overlays, NOT for obstacle stop.
CAM_OBSTACLE_VARIANCE = 400    # kept for PathVision display, no longer triggers sweep
# Person approach
PERSON_APPROACH_DIST  = 0.30   # stop this close to a detected person (m)
PERSON_BBOX_CLOSE_PX  = 180    # bbox height in px = "close enough" (fallback)
# 360 sweep: servo covers ±90 deg (3 samples), car pivots 180 for the other side
SERVO_SWEEP_ANGLES    = [-90, 0, 90]   # degrees from center; servo pans these

# Full 360° survey: 4 car body rotations × 3 servo pans = 12 samples
# Used on exploration start and after stuck-backup to understand surroundings before moving.
CAR_SURVEY_OFFSETS = [0, 90, 180, -90]  # forward, left, back, right from start heading

# Servo scanning while cruising forward (builds occupancy grid)
SERVO_SCAN_STEP_DEG  = 15    # degrees per scan step
SERVO_SCAN_INTERVAL  = 0.18  # seconds between steps (≈5-6 Hz, smooth but not jittery)

# --- Motion-model odometry (replaces drifting VO translation) ---
CRUISE_SPEED_MPS   = 0.20   # forward speed at MOTOR_BASE_SPEED (CALIBRATE to your car)
GYRO_BIAS_SAMPLES  = 20     # rest samples to zero gyro drift at exploration start
GYRO_SIGN          = 1.0    # flip to -1.0 if map directions are mirrored

# Servo scan (your pan head) — used for smart deflection.
# Camera + ToF are both on this head, so panning aims both.
SERVO_CENTER = 90
SERVO_LEFT   = 150
SERVO_RIGHT  = 30
# Seconds to let the head physically swing before trusting the panned ToF read.
SERVO_SETTLE_S = 0.45
# Sign: looking LEFT (servo>90) should add a POSITIVE bearing (CCW). If your
# head is geared so 150 actually points right, set this to -1.
SERVO_BEARING_SIGN = -1.0   # flip to +1.0 if turns still go the wrong way

# Heading source: your Movement gyro is on the PITCH axis (vertical mount)
# gx=roll, gy=pitch, gz=yaw  -> we use gy as turn-rate.
GYRO_HEADING_AXIS = "gy"

# Detection / Camera
CAMERA_INDEX = 2          # set to your Brio's /dev/videoN (shifts 0/2)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOCAL_LENGTH = 300
PERSON_HEIGHT_METERS = 1.7

# Threat dedup (cells, manhattan). 10 cells * 0.05 m = 0.5 m
THREAT_DEDUP_DISTANCE = 10
# Require a person to be seen this many consecutive detector frames before it
# is logged as a confirmed threat (kills single-frame false positives).
DETECTION_CONFIRM_FRAMES = 3

# Web
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 7000          # your standard port

# Dead reckoning
DT = 0.02
IMU_SAMPLE_RATE = 50

# Accelerometer-based stuck detection
# When motors are commanded forward but |accel| shows near-zero variation,
# the car is physically not moving (stuck against obstacle).
# CoV = σ/μ is unit-agnostic: works for both g and m/s² IMU outputs.
ACCEL_VAR_WINDOW    = 20     # rolling window length (samples, ≈ 0.4 s at 50 Hz)
ACCEL_STUCK_CV      = 0.0    # set > 0 to enable IMU-based stuck detection
                              # (disabled by default — smooth floors trigger false positives)
                              # suggested starting value once you want to try it: 0.001
                              # watch [dr] log lines to see real cv on your surface first
ACCEL_STUCK_SAMPLES = 10     # consecutive low-CV readings needed to declare stuck
