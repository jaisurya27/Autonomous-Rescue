"""main.py — Recon Rover: their advanced brain on your hardware.

Visual odometry + occupancy-grid mapping + person detection + breadcrumb
return-to-base + servo-scan obstacle deflection. Camera via OpenCV, served
with the sensor/motor/servo Bridge to the MCU. All offline.
"""

import time, json as _json, threading, math, signal, atexit
import numpy as _np, cv2
from flask import Flask, render_template, jsonify, Response, request
from config import *
from sensors import SensorReader
from dead_reckoning import DeadReckoning
from occupancy_grid import OccupancyGrid
from detector import Detector
from navigator import Navigator, NavState
from visual_odometry import VisualOdometry

class _NpEnc(_json.JSONEncoder):
    def default(self, o):
        if isinstance(o, _np.integer): return int(o)
        if isinstance(o, _np.floating): return float(o)
        if isinstance(o, _np.ndarray): return o.tolist()
        return super().default(o)

sensor_reader = SensorReader()
dead_reck = DeadReckoning()
occ_grid = OccupancyGrid()
detector = Detector()
nav = Navigator(sensors=sensor_reader)        # navigator can pan the servo
vo = VisualOdometry(CAMERA_FOCAL_LENGTH, CAMERA_WIDTH//2, CAMERA_HEIGHT//2)

camera = None; camera_lock = threading.Lock()
latest_frame = None; frame_lock = threading.Lock()
nav_lock = threading.Lock()   # guards nav state changes between Flask + control loop
status = {"state":"idle","speed":0,"heading":0,"distance_traveled":0,
          "distance_to_start":0,"detections":0,"fps":0,"us":0,
          "vo_matches":0,"vo_inliers":0}

def _emergency_stop():
    """Called on app stop / SIGTERM — kills motors immediately so the car
    doesn't keep driving after the Python process exits."""
    try:
        sensor_reader.send_command(0, 0)
        sensor_reader.send_command(0, 0)   # send twice in case first is dropped
    except Exception:
        pass

atexit.register(_emergency_stop)
signal.signal(signal.SIGTERM, lambda *_: (_emergency_stop(), exit(0)))
signal.signal(signal.SIGINT,  lambda *_: (_emergency_stop(), exit(0)))

app = Flask(__name__)

def init_camera():
    global camera
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if camera.isOpened(): print(f"[camera] Opened /dev/video{CAMERA_INDEX}")
    else: print("[camera] WARNING: no camera (set CAMERA_INDEX)"); camera = None

def grab_frame():
    global latest_frame
    if not camera: return None
    with camera_lock: ret, f = camera.read()
    if ret and f is not None:
        with frame_lock: latest_frame = f.copy()
        return f
    return None

def control_loop():
    global status
    lc = 0; ft = time.time(); lt = time.time(); dc = 0
    print("[main] Control loop started")
    while True:
        t0 = time.time(); dt = min(t0 - lt, 0.2); lt = t0
        fs = sensor_reader.get_latest()
        img = grab_frame()

        # POSE comes from the motion-model + gyro odometry (NOT VO translation,
        # which drifts). Advance it using the gyro and what the navigator was
        # commanding (nav.motion is last loop's command — what actually ran).
        dead_reck.update(nav.motion, fs.gy if fs.valid else 0.0, dt)
        px, py, pth = dead_reck.pose.x, dead_reck.pose.y, dead_reck.pose.theta

        # VO still runs for the camera overlay + person distance estimate only.
        if img is not None:
            vo.process_frame(img, imu_gz=fs.gy if fs.valid else 0.0, dt=dt)
            dc += 1
            if dc % 5 == 0: detector.feed_frame(img)

        # The ToF + camera ride on the pan servo. Convert the head's commanded
        # angle into a bearing offset (rad) so the obstacle and any person get
        # placed at the direction the head was actually pointing.
        head_bearing = math.radians(nav.servo_angle - SERVO_CENTER) * SERVO_BEARING_SIGN

        if fs.valid and fs.tof_distance > 0.01:
            occ_grid.update_from_distance(px, py, pth, fs.tof_distance,
                                          sensor_angle_offset=head_bearing)
        occ_grid.update_robot_position(px, py)
        # Only log CONFIRMED people (seen several frames in a row) as threats.
        for det in detector.get_confirmed():
            det = detector.project_to_world(det, px, py, pth, head_bearing)
            occ_grid.add_threat(det.world_x, det.world_y, det.label, det.confidence)

        # Only react (pause to classify) to confirmed people, and only when the
        # head is centered so the bearing is meaningful.
        has_det = len(detector.get_confirmed()) > 0
        with nav_lock:
            left, right = nav.compute_command(
                fs.tof_distance if fs.valid else 0.0,
                fs.us_distance  if fs.valid else 0.0,
                px, py, pth, has_det)
        sensor_reader.send_command(left, right)

        lc += 1; now = time.time()
        if now - ft >= 1.0:
            status["fps"] = round(lc/(now-ft),1); lc = 0; ft = now
        status.update({"state":nav.state_name, "speed":round(dead_reck.pose.speed,3),
            "heading":round(math.degrees(pth) % 360, 1),
            "us":round(fs.us_distance * 100, 1) if fs.valid else 0,
            "distance_traveled":round(dead_reck.total_distance,2),
            "distance_to_start":round(dead_reck.distance_to_start(),2),
            "detections":len(occ_grid.threats),
            "tof":round(fs.tof_distance,3) if fs.valid else 0,
            "imu_gz":round(fs.gy,2) if fs.valid else 0,
            "breadcrumbs":len(dead_reck.breadcrumbs),
            "head_angle":nav.servo_angle,
            "persons_confirmed":len(detector.get_confirmed()),
            "vo_matches":vo.matches_count, "vo_inliers":vo.inliers_count})
        time.sleep(max(0, 0.033-(time.time()-t0)))

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/map")
def get_map():
    gx, gy = dead_reck.get_grid_position()
    md = occ_grid.get_map_data_for_web(gx, gy, dead_reck.pose.theta)
    crumbs = []
    for bx,by,bt in dead_reck.breadcrumbs:
        crumbs.append({"x":int(START_X+bx/GRID_RESOLUTION)-md["x_min"],
                       "y":int(START_Y+by/GRID_RESOLUTION)-md["y_min"]})
    md["breadcrumbs"] = crumbs
    return app.response_class(_json.dumps(md, cls=_NpEnc), mimetype="application/json")

@app.route("/api/status")
def get_status(): return jsonify(status)

@app.route("/api/persons")
def get_persons():
    """Confirmed people and where they were found (world metres from start)."""
    return jsonify({"persons": occ_grid.get_persons_world()})

@app.route("/api/camera")
def camera_frame():
    with frame_lock: f = latest_frame
    if f is not None:
        d = f.copy()
        cv2.putText(d,f"VO:{vo.matches_count}m {vo.inliers_count}in",(5,15),0,0.35,(0,255,0),1)
        cv2.putText(d,f"({vo.x:.2f},{vo.y:.2f}) {vo.get_heading_deg():.0f}deg",(5,30),0,0.35,(0,255,0),1)
        for det in detector.get_detections():
            x1,y1,x2,y2 = [int(v) for v in det.bbox]
            cv2.rectangle(d,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(d,f"{det.label} {det.confidence:.0%}",(x1,y1-5),0,0.35,(0,0,255),1)
        _, buf = cv2.imencode('.jpg', d, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return Response(buf.tobytes(), mimetype="image/jpeg")
    return Response(b'', mimetype="image/png")

@app.route("/api/action", methods=["POST"])
def action():
    act = request.get_json().get("action","")
    with nav_lock:
        if act == "explore":
            dead_reck.reset()
            vo.x = vo.y = vo.theta = 0.0
            nav.start_exploration()
        elif act == "return":
            if len(dead_reck.breadcrumbs) > 1:
                nav.start_return(dead_reck.breadcrumbs)
            else:
                nav.stop()               # nowhere to return to — just stop
        elif act == "stop":
            nav.stop()
    # always send a stop command immediately so MCU doesn't coast
    sensor_reader.send_command(0, 0)
    return jsonify({"ok":True, "state":nav.state_name})

# App Lab runs this module; start everything at import.
print("="*50); print("  RECON ROVER — merged build"); print("="*50)
init_camera(); sensor_reader.start(); detector.start()
threading.Thread(target=control_loop, daemon=True).start()
app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
