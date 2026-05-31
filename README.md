# Recon Rover — merged build

Their advanced autonomy brain, running on YOUR proven hardware. One app.

## What it does (all offline, on-device)
- **Visual odometry** — ORB feature tracking + optical flow, fused with your
  IMU's pitch-axis gyro, to estimate motion (forward/back detected from whether
  features expand or contract).
- **Occupancy-grid mapping** — Bayesian log-odds grid; each ToF reading is
  ray-cast into free/occupied cells. Live map in the dashboard.
- **Person detection** — MobileNet-SSD (OpenCV DNN), estimates each person's
  distance + bearing, drops them on the map as "threats" (deduplicated).
- **Breadcrumb return-to-base** — logs a trail while exploring; RETURN replays
  it to drive home. (Upgraded from the original stub.)
- **Servo scan deflection** — on an obstacle it looks left & right with your
  pan head and turns toward the more open side (upgrade over blind pivoting).
- **State machine** — IDLE -> EXPLORE -> CLASSIFY (pause to log a person)
  -> RETURN -> ARRIVED.
- **Dashboard** on :7000 — live camera (with VO/detection overlays), map,
  status (heading, distance, FPS, breadcrumbs), and EXPLORE/RETURN/STOP buttons.

## Your hardware (what the original had to be adapted to)
- Motors: TB6612, PWMA=5 PWMB=6 AIN1=7 BIN1=8 STBY=3 (single dir pin/side).
  Their ENA/ENB + 4-pin driver was replaced with yours.
- Servo: pin 11, library-free pulse (no Servo.h).
- Sensors: Modulino ToF (0x29) + Movement (0x6A) on Wire1. NO ultrasonic
  (theirs used HC-SR04; us_cm is sent as 0 and ignored).
- Heading: gyro is on getPitch() = config GYRO_HEADING_AXIS "gy".

## Run
1. Copy `recon-rover` into ~/ArduinoApps/.
2. Set CAMERA_INDEX in python/config.py to your Brio's /dev/videoN
   (check: v4l2-ctl --list-devices — it shifts between 0 and 2).
3. Battery on (motors + servo), camera connected, wheels UP for first test.
4. arduino-app-cli app start ~/ArduinoApps/recon-rover
   (first run installs flask/opencv/numpy and compiles the sketch — slow.)
5. Open http://<board-ip>:7000
6. Click EXPLORE. Drive it around. Click RETURN to retrace home. STOP anytime.

## Honest caveats
- Heavy CV (ORB + MobileNet) on the Uno Q's Cortex-A53 runs at MODEST fps.
  Expect a few-fps map/detection update, not real-time video game speed.
- Visual odometry + gyro dead-reckoning DRIFT — position is an estimate, so
  return-to-base gets you ROUGHLY home, not to the exact cm. Keep runs short.
- VO needs visual texture — a blank floor/wall gives few features and weak
  odometry. Works best in a feature-rich space.
- Detection model is bundled (model/), so it works with no internet.
- Test wheels-up first; tune MOTOR_BASE_SPEED in config.py if it stalls (55)
  or is too fast.

## Files
- sketch/sketch.ino — your hardware + read_sensors/set_motors/set_servo bridge
- python/config.py — all tunables (pins via sketch; speeds, grid, camera here)
- python/sensors.py — Bridge interface (+ servo)
- python/visual_odometry.py, occupancy_grid.py, dead_reckoning.py — ported
- python/detector.py — MobileNet-SSD person detection
- python/navigator.py — state machine + servo-scan + breadcrumb return (upgraded)
- python/main.py — orchestrator + Flask dashboard
- python/templates/index.html — dashboard
