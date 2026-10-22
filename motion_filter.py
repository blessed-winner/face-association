from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict
from math_utils import BBox

class BodyMotionFilter:
    """
    Constant Velocity Motion Filter for bounding box tracking [cx, cy, w, h, vx, vy].
    Smooths jitter, preserves trajectory through occlusions, and predicts next position.
    """
    def __init__(self, init_bbox: BBox):
        cx, cy = init_bbox.center
        self.cx = cx
        self.cy = cy
        self.w = init_bbox.width
        self.h = init_bbox.height
        self.vx = 0.0
        self.vy = 0.0

        # Tuning weights
        self.alpha_pos = 0.70  # Measurement confidence on position
        self.alpha_vel = 0.35  # Measurement velocity update
        self.alpha_size = 0.60 # Measurement confidence on size

    def predict(self, dt: float = 1.0) -> BBox:
        pred_cx = self.cx + self.vx * dt
        pred_cy = self.cy + self.vy * dt
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        return BBox(
            x1=pred_cx - half_w,
            y1=pred_cy - half_h,
            x2=pred_cx + half_w,
            y2=pred_cy + half_h
        )

    def update(self, meas_bbox: BBox, dt: float = 1.0):
        mcx, mcy = meas_bbox.center
        mw = meas_bbox.width
        mh = meas_bbox.height

        # Calculated instantaneous velocities
        raw_vx = (mcx - self.cx) / max(dt, 1e-4)
        raw_vy = (mcy - self.cy) / max(dt, 1e-4)

        # Smooth position and velocity
        self.vx = (1.0 - self.alpha_vel) * self.vx + self.alpha_vel * raw_vx
        self.vy = (1.0 - self.alpha_vel) * self.vy + self.alpha_vel * raw_vy

        self.cx = (1.0 - self.alpha_pos) * (self.cx + self.vx * dt) + self.alpha_pos * mcx
        self.cy = (1.0 - self.alpha_pos) * (self.cy + self.vy * dt) + self.alpha_pos * mcy

        self.w = (1.0 - self.alpha_size) * self.w + self.alpha_size * mw
        self.h = (1.0 - self.alpha_size) * self.h + self.alpha_size * mh

    @property
    def current_bbox(self) -> BBox:
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        return BBox(
            x1=self.cx - half_w,
            y1=self.cy - half_h,
            x2=self.cx + half_w,
            y2=self.cy + half_h
        )
