"""dead_reckoning.py — Pose tracking and breadcrumb trail."""

import math, time
from dataclasses import dataclass
from typing import List, Tuple
from config import START_X, START_Y, GRID_RESOLUTION, BREADCRUMB_INTERVAL

@dataclass
class Pose:
    x: float = 0.0; y: float = 0.0; theta: float = 0.0; speed: float = 0.0

class DeadReckoning:
    def __init__(self):
        self.pose = Pose()
        self.breadcrumbs: List[Tuple[float,float,float]] = [(0.0, 0.0, 0.0)]
        self._last_breadcrumb_dist = 0.0
        self._total_distance = 0.0

    def get_grid_position(self) -> Tuple[int,int]:
        return (int(START_X + self.pose.x / GRID_RESOLUTION),
                int(START_Y + self.pose.y / GRID_RESOLUTION))

    def get_return_path(self):
        return list(reversed(self.breadcrumbs))

    def distance_to_start(self):
        return math.sqrt(self.pose.x**2 + self.pose.y**2)

    @property
    def total_distance(self):
        return self._total_distance
