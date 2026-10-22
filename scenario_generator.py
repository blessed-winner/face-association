from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional, Dict
from math_utils import BBox
from models import PersonDetection, FaceDetection

class SyntheticScenarioGenerator:
    """
    High-fidelity realistic synthetic scenario generator.
    Generates deterministic or stochastic multi-person trajectories with real-world edge cases:
      - Scenario 1: Head Turn / Yaw Rotation (face turns 90-180 deg away from camera, vanishing face detection while body persists)
      - Scenario 2: Occlusion (target passes behind a pillar or another object, or partial body occlusion)
      - Scenario 3: Path Crossing / Dense Social Walk (two persons walking in opposite directions, crossing paths with near IoU overlap)
      - Scenario 4: Long Disappearance & Re-entry (face lost until decay to UNKNOWN, then recovered)
    """
    def __init__(self, frame_width: int = 1280, frame_height: int = 720, fps: int = 30):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    def scenario_head_turn(self, num_frames: int = 120) -> List[Tuple[List[PersonDetection], List[FaceDetection], Dict]]:
        """
        Subject walks across room.
        Frames 0-30: Facing camera (CONFIRMED)
        Frames 31-75: Subject turns head away / looks at phone (face detection drops completely)
        Frames 76-120: Subject turns head back to camera (face detection re-established)
        """
        frames_data = []
        cx = 300.0
        cy = 380.0
        bw, bh = 140.0, 360.0

        for f in range(num_frames):
            # Slow walking trajectory
            cx += 2.0
            body_box = BBox(cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2)
            person_det = PersonDetection(bbox=body_box, confidence=0.95, detection_id=1)

            # Face presence depends on yaw angle
            faces = []
            is_face_visible = not (30 <= f <= 75)
            if is_face_visible:
                fw, fh = 50.0, 65.0
                fcx = cx + math.sin(f * 0.05) * 2.0
                fcy = body_box.y1 + 0.18 * bh
                face_box = BBox(fcx - fw/2, fcy - fh/2, fcx + fw/2, fcy + fh/2)
                faces.append(FaceDetection(
                    bbox=face_box,
                    identity_label="Alice",
                    embedding=[0.8, -0.2, 0.5, 0.1],
                    confidence=0.92
                ))

            ground_truth = {
                "ground_truth_id": "Alice",
                "face_visible": is_face_visible,
                "event": "head_turn" if (30 <= f <= 75) else "frontal"
            }
            frames_data.append(([person_det], faces, ground_truth))
        return frames_data

    def scenario_occlusion(self, num_frames: int = 140) -> List[Tuple[List[PersonDetection], List[FaceDetection], Dict]]:
        """
        Target walks behind a static vertical column/partition (width 120px) at frame center (x=600 to 720).
        During frames 50-80: Face is occluded by the pillar, then body is partially occluded.
        Tests tracking through occlusion without false track destruction or identity swap.
        """
        frames_data = []
        cx = 250.0
        cy = 380.0
        bw, bh = 130.0, 350.0
        pillar_x1, pillar_x2 = 580.0, 700.0

        for f in range(num_frames):
            cx += 3.5
            body_box = BBox(cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2)
            
            # Check overlap with pillar
            is_pillar_occluded = (pillar_x1 <= cx <= pillar_x2)
            face_visible = not is_pillar_occluded

            persons = [PersonDetection(bbox=body_box, confidence=0.88, detection_id=1)]
            faces = []

            if face_visible:
                fw, fh = 48.0, 62.0
                fcx = cx
                fcy = body_box.y1 + 0.17 * bh
                face_box = BBox(fcx - fw/2, fcy - fh/2, fcx + fw/2, fcy + fh/2)
                faces.append(FaceDetection(
                    bbox=face_box,
                    identity_label="Bob",
                    embedding=[0.1, 0.9, -0.3, 0.4],
                    confidence=0.90
                ))

            ground_truth = {
                "ground_truth_id": "Bob",
                "face_visible": face_visible,
                "event": "pillar_occlusion" if is_pillar_occluded else "clear_view"
            }
            frames_data.append((persons, faces, ground_truth))
        return frames_data

    def scenario_path_crossing(self, num_frames: int = 150) -> List[Tuple[List[PersonDetection], List[FaceDetection], Dict]]:
        """
        Two people (Person 1 'Charlie', Person 2 'Diana') walking towards each other,
        crossing closely in the middle (frames 65-85), then continuing.
        During crossing, Diana's face is momentarily occluded by Charlie's shoulder/body.
        Evaluates track identity preservation without ID switching.
        """
        frames_data = []
        bw, bh = 130.0, 350.0

        for f in range(num_frames):
            # Charlie moves left to right
            c1_x = 200.0 + f * 4.0
            c1_y = 380.0
            b1 = BBox(c1_x - bw/2, c1_y - bh/2, c1_x + bw/2, c1_y + bh/2)

            # Diana moves right to left
            c2_x = 1000.0 - f * 4.0
            c2_y = 390.0
            b2 = BBox(c2_x - bw/2, c2_y - bh/2, c2_x + bw/2, c2_y + bh/2)

            persons = [
                PersonDetection(bbox=b1, confidence=0.92, detection_id=1),
                PersonDetection(bbox=b2, confidence=0.92, detection_id=2)
            ]
            faces = []

            # Charlie face is visible throughout
            f1_box = BBox(c1_x - 24, b1.y1 + 10, c1_x + 24, b1.y1 + 72)
            faces.append(FaceDetection(
                bbox=f1_box,
                identity_label="Charlie",
                embedding=[0.3, 0.3, 0.8, -0.2],
                confidence=0.93
            ))

            # Diana face is occluded during crossing (frames 65 to 85)
            diana_face_visible = not (65 <= f <= 85)
            if diana_face_visible:
                f2_box = BBox(c2_x - 24, b2.y1 + 10, c2_x + 24, b2.y1 + 72)
                faces.append(FaceDetection(
                    bbox=f2_box,
                    identity_label="Diana",
                    embedding=[-0.4, 0.7, 0.1, 0.6],
                    confidence=0.91
                ))

            ground_truth = {
                "charlie_x": c1_x,
                "diana_x": c2_x,
                "crossing": (65 <= f <= 85),
                "diana_face_visible": diana_face_visible
            }
            frames_data.append((persons, faces, ground_truth))
        return frames_data
