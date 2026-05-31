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

            # Two analysis zones:
            #   BOTTOM third  → close-range floor / cruise obstacle warning
            #   MIDDLE third  → medium-distance scene in the camera's pan direction
            #                   (used for sweep scoring: better represents whether the
            #                    corridor the servo is pointing at is actually clear)
            y_bot = h * 2 // 3         # bottom strip top edge
            y_mid_top = h // 3         # middle strip top edge
            y_mid_bot = h * 2 // 3     # middle strip bottom edge
            xL, xC1, xC2, xR = 0, w // 3, 2 * w // 3, w

            strip_bot_c = gray[y_bot:,          xC1:xC2]
            strip_bot_l = gray[y_bot:,          xL:xC1]
            strip_bot_r = gray[y_bot:,          xC2:xR]
            strip_mid_c = gray[y_mid_top:y_mid_bot, xC1:xC2]

            var_c  = float(np.var(strip_bot_c))
            var_l  = float(np.var(strip_bot_l))
            var_r  = float(np.var(strip_bot_r))
            var_mc = float(np.var(strip_mid_c))   # middle-centre variance

            self._path_blocked = var_c > CAM_OBSTACLE_VARIANCE
            self._left_score  = 1.0 / (1.0 + var_l)
            self._right_score = 1.0 / (1.0 + var_r)
            # center_score: uses MIDDLE strip so sweep correctly evaluates whether
            # the corridor in the servo's pan direction is clear at medium distance.
            # Bottom-strip was floor texture, which scored "blocked" in open corridors.
            self._center_score = max(0.0, 1.0 - var_mc / CAM_OBSTACLE_VARIANCE)
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
        """Draw analysis regions onto a copy of the frame for the dashboard."""
        if frame is None:
            return frame
        d = frame.copy()
        h, w = d.shape[:2]
        xC1, xC2 = w // 3, 2 * w // 3
        # Bottom strip — cruise obstacle warning
        y_bot = h * 2 // 3
        col = (0, 0, 255) if self._path_blocked else (0, 255, 0)
        cv2.rectangle(d, (xC1, y_bot), (xC2, h), col, 1)
        # Middle strip — sweep direction scoring (green=clear, red=blocked)
        y_mid_top, y_mid_bot = h // 3, h * 2 // 3
        cs_col = (0, 255, 0) if self._center_score > 0.3 else (0, 0, 255)
        cv2.rectangle(d, (xC1, y_mid_top), (xC2, y_mid_bot), cs_col, 2)
        lbl = f"cs:{self._center_score:.2f}"
        cv2.putText(d, lbl, (xC1 + 2, y_mid_top + 14), cv2.FONT_HERSHEY_PLAIN, 0.9, cs_col, 1)
        return d
