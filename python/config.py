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
EXPLORATION_TIMEOUT = 180
OBSTACLE_THRESHOLD = 0.25
MOTOR_BASE_SPEED = 55     # your low-speed cruise (45 stalled; ~55 reliable)
MOTOR_TURN_SPEED = 40
MOTOR_STOP = 0

# --- Recon / obstacle avoidance ---
US_STOP_DISTANCE      = 0.15   # front ultrasonic hard stop (m) [15 cm]
TOF_STOP_DISTANCE     = 0.25   # head ToF stop when centered (m)
SWEEP_SETTLE_S        = 0.7    # wait after each servo/pivot step before reading (longer = camera stabilises)
MIN_CLEARANCE         = 0.10   # if best clearance everywhere below this -> dead end
PIVOT_STEP_TIMEOUT    = 3.0    # s, safety cap per pivot step (longer at low turn speed)
BACKUP_TIME           = 0.6    # s reverse when in a dead end
STUCK_TIMEOUT         = 5.0    # s without meaningful movement -> force reverse+reroute
STUCK_MOVE_THRESHOLD  = 0.04   # m — below this over STUCK_TIMEOUT = stuck
STUCK_REVERSE_TIME    = 0.8    # s reverse when stuck is detected
# Camera path vision: analyse bottom-center strip of frame for obstacles.
# Variance above this in the look-ahead strip = textured obstacle ahead.
CAM_OBSTACLE_VARIANCE = 400    # tune up if false positives on textured floors
# Person approach
PERSON_APPROACH_DIST  = 0.30   # stop this close to a detected person (m)
PERSON_BBOX_CLOSE_PX  = 180    # bbox height in px = "close enough" (fallback)
# 360 sweep: servo covers ±90 deg (3 samples), car pivots 180 for the other side
SERVO_SWEEP_ANGLES    = [-90, 0, 90]   # degrees from center; servo pans these

# --- Motion-model odometry (replaces drifting VO translation) ---
CRUISE_SPEED_MPS   = 0.20   # forward speed at MOTOR_BASE_SPEED (CALIBRATE to your car)
GYRO_BIAS_SAMPLES  = 20     # rest samples to zero gyro drift at exploration start
GYRO_SIGN          = 1.0    # flip to -1.0 if turns integrate the wrong way

# Servo scan (your pan head) — used for smart deflection.
# Camera + ToF are both on this head, so panning aims both.
SERVO_CENTER = 90
SERVO_LEFT   = 150
SERVO_RIGHT  = 30
# Seconds to let the head physically swing before trusting the panned ToF read.
SERVO_SETTLE_S = 0.45
# Sign: looking LEFT (servo>90) should add a POSITIVE bearing (CCW). If your
# head is geared so 150 actually points right, set this to -1.
SERVO_BEARING_SIGN = 1.0

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
