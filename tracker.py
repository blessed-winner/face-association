from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict
from math_utils import BBox, hungarian_algorithm
from models import TrackState, PersonDetection, FaceDetection
from spatial_mapper import SpatialFaceBodyMapper
from tracker_state import PersonTrack

class WholePersonTracker:
    """
    End-to-end Whole Person Tracking and Face Association Module.
    Combines:
      1. Whole-body IoU & motion tracking
      2. Upper-body spatial anthropometric face mapping
      3. Honest 3-state confidence decay
    """
    def __init__(
        self,
        max_disappeared_frames: int = 30,
        iou_threshold: float = 0.25,
        half_life_frames: float = 25.0,
        tau_confirmed: float = 0.70,
        tau_temporary: float = 0.30
    ):
        self.next_track_id = 1
        self.tracks: Dict[int, PersonTrack] = {}
        self.spatial_mapper = SpatialFaceBodyMapper()
        self.max_disappeared_frames = max_disappeared_frames
        self.iou_threshold = iou_threshold
        self.half_life_frames = half_life_frames
        self.tau_confirmed = tau_confirmed
        self.tau_temporary = tau_temporary

    def process_frame(
        self,
        person_detections: List[PersonDetection],
        face_detections: List[FaceDetection],
        dt: float = 1.0
    ) -> List[PersonTrack]:
        """
        Processes a single video/simulation frame:
          Step 1: Predict next positions of existing tracks using motion filter
          Step 2: Associate detected bodies with existing tracks (Hungarian IoU)
          Step 3: Associate detected faces to detected bodies using spatial anthropometrics
          Step 4: Update matched tracks with body and face observations
          Step 5: Apply exponential decay to tracks missing faces
          Step 6: Handle unconfirmed/disappeared tracks
        """
        # 1. Spatially match faces to detected bodies in the current observation frame
        body_face_pairs = self.spatial_mapper.associate(face_detections, person_detections)
        # body_face_pairs is List[Tuple[Optional[FaceDetection], PersonDetection]]

        track_ids = list(self.tracks.keys())
        predicted_boxes = [self.tracks[tid].predict_body(dt=dt) for tid in track_ids]

        # 2. Match detected bodies to existing tracks
        num_tracks = len(track_ids)
        num_dets = len(body_face_pairs)

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            # Build IoU cost matrix
            cost_matrix = []
            for i in range(num_tracks):
                row = []
                for j in range(num_dets):
                    _, body_det = body_face_pairs[j]
                    iou = predicted_boxes[i].iou(body_det.bbox)
                    row.append(1.0 - iou)
                cost_matrix.append(row)

            matches = hungarian_algorithm(cost_matrix)
            for t_idx, d_idx in matches:
                if cost_matrix[t_idx][d_idx] <= (1.0 - self.iou_threshold):
                    matched_tracks.add(track_ids[t_idx])
                    matched_dets.add(d_idx)
                    
                    # Update track with body and face
                    face_det, body_det = body_face_pairs[d_idx]
                    track = self.tracks[track_ids[t_idx]]
                    track.update_body(body_det.bbox, dt=dt)
                    if face_det is not None:
                        track.update_face(face_det)
                    else:
                        track.step_temporal_decay()

        # 3. Handle unmatched tracks (tracks without body detections in this frame)
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.frames_since_body_seen += 1
                track.step_temporal_decay()

        # 4. Handle unmatched detections (create new tracks)
        for j in range(num_dets):
            if j not in matched_dets:
                face_det, body_det = body_face_pairs[j]
                new_track = PersonTrack(
                    track_id=self.next_track_id,
                    initial_body_bbox=body_det.bbox,
                    initial_face=face_det,
                    half_life_frames=self.half_life_frames,
                    tau_confirmed=self.tau_confirmed,
                    tau_temporary=self.tau_temporary
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # 5. Prune tracks that exceeded max disappeared frames
        dead_track_ids = [
            tid for tid, track in self.tracks.items()
            if track.frames_since_body_seen > self.max_disappeared_frames
        ]
        for tid in dead_track_ids:
            del self.tracks[tid]

        return list(self.tracks.values())
