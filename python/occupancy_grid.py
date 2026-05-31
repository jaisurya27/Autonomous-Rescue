"""occupancy_grid.py — 2D Bayesian occupancy grid with threat dedup."""

import numpy as np, math
from typing import Tuple, List
from config import (
    GRID_WIDTH, GRID_HEIGHT, GRID_RESOLUTION, START_X, START_Y,
    L_FREE, L_OCCUPIED, L_PRIOR, L_MAX, L_MIN, TOF_MAX_RANGE,
    THREAT_DEDUP_DISTANCE
)

class OccupancyGrid:
    def __init__(self):
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), L_PRIOR, dtype=np.float32)
        self.visited = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        self.threats: List[Tuple[int,int,str,float]] = []

    def update_from_distance(self, rx, ry, rtheta, distance, sensor_angle_offset=0.0):
        if distance <= 0.01 or distance > TOF_MAX_RANGE:
            return
        gx = int(START_X + rx / GRID_RESOLUTION)
        gy = int(START_Y + ry / GRID_RESOLUTION)
        angle = rtheta + sensor_angle_offset
        ex = int(START_X + (rx + distance * math.cos(angle)) / GRID_RESOLUTION)
        ey = int(START_Y + (ry + distance * math.sin(angle)) / GRID_RESOLUTION)
        cells = self._bresenham(gx, gy, ex, ey)
        for cx, cy in cells[:-1]:
            if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                self.grid[cy, cx] = max(self.grid[cy, cx] + L_FREE, L_MIN)
                self.visited[cy, cx] = True
        if distance < TOF_MAX_RANGE * 0.95 and cells:
            ecx, ecy = cells[-1]
            if 0 <= ecx < GRID_WIDTH and 0 <= ecy < GRID_HEIGHT:
                self.grid[ecy, ecx] = min(self.grid[ecy, ecx] + L_OCCUPIED, L_MAX)
                self.visited[ecy, ecx] = True

    def update_robot_position(self, rx, ry):
        gx = int(START_X + rx / GRID_RESOLUTION)
        gy = int(START_Y + ry / GRID_RESOLUTION)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cx, cy = gx + dx, gy + dy
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                    self.grid[cy, cx] = L_MIN
                    self.visited[cy, cx] = True

    def add_threat(self, wx, wy, label="person", confidence=0.0):
        """Add threat only if no existing threat is within THREAT_DEDUP_DISTANCE cells."""
        if not (math.isfinite(wx) and math.isfinite(wy)):
            return
        gx = int(START_X + wx / GRID_RESOLUTION)
        gy = int(START_Y + wy / GRID_RESOLUTION)
        # A distant/edge detection used to be silently DROPPED when it projected
        # outside the grid — that's why confirmed persons never showed in the UI.
        # Clamp into bounds instead so the marker still appears (at the edge).
        gx = max(0, min(GRID_WIDTH - 1, gx))
        gy = max(0, min(GRID_HEIGHT - 1, gy))
        # Dedup: if a threat of the same label already exists nearby, treat this
        # as the same person. Keep the higher-confidence sighting (and its
        # position) instead of dropping a permanent marker at the first noisy hit.
        for i, (tx, ty, tl, tc) in enumerate(self.threats):
            if tl == label and abs(tx - gx) + abs(ty - gy) < THREAT_DEDUP_DISTANCE:
                if confidence > tc:
                    self.threats[i] = (gx, gy, label, confidence)
                return
        self.threats.append((gx, gy, label, confidence))

    def get_persons_world(self):
        """Confirmed person markers in world metres (relative to start), with
        distance from start and confidence — for the dashboard/API."""
        out = []
        for gx, gy, label, conf in self.threats:
            if label != "person":
                continue
            wx = (gx - START_X) * GRID_RESOLUTION
            wy = (gy - START_Y) * GRID_RESOLUTION
            out.append({
                "x": round(wx, 2), "y": round(wy, 2),
                "dist_from_start": round(math.hypot(wx, wy), 2),
                "confidence": round(float(conf), 2),
                "label": label,
            })
        return out

    def get_map_data_for_web(self, robot_gx, robot_gy, robot_theta):
        MIN_HALF = 25   # always show at least 50×50 cells centred on robot
        visited_coords = np.argwhere(self.visited)
        if len(visited_coords) == 0:
            x0 = max(0, robot_gx - MIN_HALF); x1 = min(GRID_WIDTH,  robot_gx + MIN_HALF)
            y0 = max(0, robot_gy - MIN_HALF); y1 = min(GRID_HEIGHT, robot_gy + MIN_HALF)
        else:
            m = 10
            y0 = max(0, int(visited_coords[:,0].min()) - m)
            y1 = min(GRID_HEIGHT, int(visited_coords[:,0].max()) + m)
            x0 = max(0, int(visited_coords[:,1].min()) - m)
            x1 = min(GRID_WIDTH, int(visited_coords[:,1].max()) + m)
            # enforce minimum size centred on robot so it's always visible
            if (x1 - x0) < MIN_HALF * 2:
                x0 = max(0, robot_gx - MIN_HALF); x1 = min(GRID_WIDTH, robot_gx + MIN_HALF)
            if (y1 - y0) < MIN_HALF * 2:
                y0 = max(0, robot_gy - MIN_HALF); y1 = min(GRID_HEIGHT, robot_gy + MIN_HALF)
        cropped = self.grid[y0:y1, x0:x1]
        # clip before exp() so a runaway log-odds value can't overflow to inf/NaN
        prob = 1.0 - 1.0 / (1.0 + np.exp(np.clip(cropped, L_MIN, L_MAX)))
        display = ((1.0 - prob) * 255).astype(np.uint8)
        # robot_theta can be NaN if pose math ever divided by zero — never ship NaN
        rtheta = float(robot_theta)
        if not math.isfinite(rtheta):
            rtheta = 0.0
        return {
            "cells": display.tolist(),
            "x_min": int(x0), "y_min": int(y0),
            "width": int(x1 - x0), "height": int(y1 - y0),
            "robot_x": int(robot_gx - x0), "robot_y": int(robot_gy - y0),
            "robot_theta": rtheta,
            "threats": [
                {"x": int(tx-x0), "y": int(ty-y0), "label": tl, "conf": float(tc)}
                for tx,ty,tl,tc in self.threats
                if x0 <= tx < x1 and y0 <= ty < y1
            ],
            "resolution": float(GRID_RESOLUTION)
        }

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        cells = []
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 < dx: err += dx; y0 += sy
        return cells
