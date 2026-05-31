"""path_vision.py — Lightweight camera-based path assessment.

Uses no ML — just OpenCV image statistics on two regions of the frame:

  LOOK-AHEAD strip  (centre column, bottom third of frame)
    High variance   = textured wall / obstacle in the path
    Low variance    = smooth floor / open space

  LEFT / RIGHT strips  (left and right columns, bottom third)
    Compared to the look-ahead variance to judge which side looks more open.

Returns:
  path_blocked  bool  — centre path looks obstructed
  left_score    float — higher = more open on left
  right_score   float — higher = more open on right

These are SOFT signals. A single camera reading is noisy. The navigator
uses them to break ties between sensor-equal headings, and to slow down
before the US/ToF triggers, NOT to hard-stop on their own.
"""

import cv2
import numpy as np
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAM_OBSTACLE_VARIANCE


class PathVision:
    def __init__(self):
        self._path_blocked = False
        self._left_score   = 0.0
        self._right_score  = 0.0
        self._center_score = 0.5  # 1.0 = open, 0.0 = wall visible in camera centre

    def update(self, frame):
        """Analyse a BGR frame. Call each control loop iteration when a
        frame is available. Results readable immediately after."""
        if frame is None:
            return
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Regions: bottom third of frame, split into left / centre / right thirds
            y0 = h * 2 // 3
            xL, xC1, xC2, xR = 0, w // 3, 2 * w // 3, w

            strip_centre = gray[y0:, xC1:xC2]
            strip_left   = gray[y0:, xL:xC1]
            strip_right  = gray[y0:, xC2:xR]

            var_c = float(np.var(strip_centre))
            var_l = float(np.var(strip_left))
            var_r = float(np.var(strip_right))

            self._path_blocked = var_c > CAM_OBSTACLE_VARIANCE
            # 1/(1+var) left/right scores kept for backward compat (cruise soft signals)
            self._left_score  = 1.0 / (1.0 + var_l)
            self._right_score = 1.0 / (1.0 + var_r)
            # Linear 0-1 score relative to the obstacle variance threshold.
            # 0 = clearly a wall in shot, 1 = smooth open floor/space.
            # Used by sweep to rank directions the servo is pointing at.
            self._center_score = max(0.0, 1.0 - var_c / CAM_OBSTACLE_VARIANCE)
        except Exception:
            pass

    @property
    def path_blocked(self):
        """True if the centre camera strip looks like an obstacle."""
        return self._path_blocked

    @property
    def center_score(self):
        """0.0=wall visible in centre, 1.0=open floor/space. Linear relative to CAM_OBSTACLE_VARIANCE."""
        return self._center_score

    @property
    def left_score(self):
        """Higher = left side looks more open."""
        return self._left_score

    @property
    def right_score(self):
        """Higher = right side looks more open."""
        return self._right_score

    def annotate(self, frame):
        """Draw the analysis regions onto a copy of the frame for the dashboard."""
        if frame is None:
            return frame
        d = frame.copy()
        h, w = d.shape[:2]
        y0 = h * 2 // 3
        xC1, xC2 = w // 3, 2 * w // 3
        col = (0, 0, 255) if self._path_blocked else (0, 255, 0)
        cv2.rectangle(d, (xC1, y0), (xC2, h), col, 2)
        label = "BLOCKED" if self._path_blocked else "CLEAR"
        cv2.putText(d, label, (xC1 + 4, y0 + 16), cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)
        return d
