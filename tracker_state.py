from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any
from math_utils import BBox, cosine_similarity
from models import TrackState, PersonDetection, FaceDetection
from motion_filter import BodyMotionFilter

class PersonTrack:
    """
    Maintains temporal identity and trajectory of a whole person.
    Implements the Honest Three-State Model:
      - CONFIRMED: Face is actively detected and mapped or recently verified with high confidence.
      - TEMPORARY: Face is temporarily lost (head turn, occluded, looking away), but whole-body tracking
                   is continuously maintained. Identity confidence decays exponentially over time.
      - UNKNOWN: Confidence has decayed below threshold tau_decay or body was initialized without face.
                 Does not guess or hallucinate identities.
    """
    def __init__(
        self,
        track_id: int,
        initial_body_bbox: BBox,
        initial_face: Optional[FaceDetection] = None,
        half_life_frames: float = 30.0,
        tau_confirmed: float = 0.70,
        tau_temporary: float = 0.30
    ):
        self.track_id = track_id
        self.motion_filter = BodyMotionFilter(initial_body_bbox)
        
        # Identity attributes
        self.identity_label: Optional[str] = None
        self.face_embedding: Optional[List[float]] = None
        self.last_face_bbox: Optional[BBox] = None

        # Exponential decay parameters: C(t) = C_0 * exp(-lambda * delta_t)
        # where lambda = ln(2) / half_life
        self.half_life_frames = max(1.0, half_life_frames)
        self.decay_lambda = math.log(2.0) / self.half_life_frames
        self.tau_confirmed = tau_confirmed
        self.tau_temporary = tau_temporary

        # Temporal confidence and state
        self.confidence: float = 0.0
        self.frames_since_face_seen: int = 0
        self.frames_since_body_seen: int = 0
        self.total_tracked_frames: int = 0
        self.state: TrackState = TrackState.UNKNOWN

        if initial_face is not None:
            self.update_face(initial_face)
        else:
            self.state = TrackState.UNKNOWN
            self.confidence = 0.0

    @property
    def current_bbox(self) -> BBox:
        return self.motion_filter.current_bbox

    def update_face(self, face: FaceDetection):
        self.identity_label = face.identity_label if face.identity_label else self.identity_label
        if face.embedding is not None:
            if self.face_embedding is None:
                self.face_embedding = list(face.embedding)
            else:
                # Running average update for embedding
                self.face_embedding = [
                    0.8 * old + 0.2 * new
                    for old, new in zip(self.face_embedding, face.embedding)
                ]
        self.last_face_bbox = face.bbox
        self.frames_since_face_seen = 0
        self.confidence = 1.0
        self.state = TrackState.CONFIRMED

    def step_temporal_decay(self):
        """
        Applies exponential confidence decay when no face is associated in current frame.
        C(t) = exp(-lambda * delta_t)
        Updates state transitions according to honest thresholds:
          C >= tau_confirmed  -> CONFIRMED
          tau_temporary <= C < tau_confirmed -> TEMPORARY
          C < tau_temporary   -> UNKNOWN
        """
        self.frames_since_face_seen += 1
        self.total_tracked_frames += 1

        if self.identity_label is not None:
            self.confidence = math.exp(-self.decay_lambda * self.frames_since_face_seen)
            if self.confidence >= self.tau_confirmed:
                self.state = TrackState.CONFIRMED
            elif self.confidence >= self.tau_temporary:
                self.state = TrackState.TEMPORARY
            else:
                self.state = TrackState.UNKNOWN
                # Note: Identity label is retained in history for evaluation, but state is honestly UNKNOWN
        else:
            self.confidence = 0.0
            self.state = TrackState.UNKNOWN

    def update_body(self, body_bbox: BBox, dt: float = 1.0):
        self.motion_filter.update(body_bbox, dt=dt)
        self.frames_since_body_seen = 0

    def predict_body(self, dt: float = 1.0) -> BBox:
        return self.motion_filter.predict(dt=dt)
