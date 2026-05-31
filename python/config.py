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
MOTOR_TURN_SPEED = 75
MOTOR_STOP = 0

# Servo scan (your pan head) — used for smart deflection
SERVO_CENTER = 90
SERVO_LEFT   = 150
SERVO_RIGHT  = 30

# Heading source: your Movement gyro is on the PITCH axis (vertical mount)
# gx=roll, gy=pitch, gz=yaw  -> we use gy as turn-rate.
GYRO_HEADING_AXIS = "gy"

# Detection / Camera
CAMERA_INDEX = 2          # set to your Brio's /dev/videoN (shifts 0/2)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOCAL_LENGTH = 300
PERSON_HEIGHT_METERS = 1.7

# Threat dedup
THREAT_DEDUP_DISTANCE = 10

# Web
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 7000          # your standard port

# Dead reckoning
DT = 0.02
IMU_SAMPLE_RATE = 50
