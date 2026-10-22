from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from math_utils import BBox

class TrackState(str, Enum):
    CONFIRMED = "Confirmed"
    TEMPORARY = "Temporary"
    UNKNOWN = "Unknown"

@dataclass
class PersonDetection:
    bbox: BBox
    confidence: float = 0.9
    detection_id: Optional[int] = None

@dataclass
class FaceDetection:
    bbox: BBox
    identity_label: Optional[str] = None
    embedding: Optional[List[float]] = None
    confidence: float = 0.9

@dataclass
class CenteringControlCommand:
    pan_error: float            # Normalized horizontal error from target to frame center [-1.0, 1.0]
    tilt_error: float           # Normalized vertical error from target to frame center [-1.0, 1.0]
    target_person_id: Optional[int]
    target_state: Optional[TrackState]
    active: bool
    recommended_pan_velocity: float
    recommended_tilt_velocity: float
    target_center: Tuple[float, float]
    frame_center: Tuple[float, float]

@dataclass
class EvaluationMetrics:
    total_frames: int = 0
    confirmed_state_count: int = 0
    temporary_state_count: int = 0
    unknown_state_count: int = 0
    id_switches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    centering_mean_error_px: float = 0.0
    temporal_preservation_ratio: float = 0.0
