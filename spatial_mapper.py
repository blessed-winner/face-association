from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict
from math_utils import BBox, hungarian_algorithm
from models import FaceDetection, PersonDetection

class SpatialFaceBodyMapper:
    """
    Spatially associates detected faces to whole-person body bounding boxes.
    Enforces anthropomorphic spatial priors:
      1. Anatomical Head Zone: Face center must reside within the upper segment (top 45%) of the person.
      2. Face Containment: The face must substantially overlap with the upper body envelope.
      3. Anthropometric Proportion: The face size must be anatomically proportional to the body.
    """
    def __init__(self, upper_body_ratio: float = 0.45, min_containment: float = 0.45):
        self.upper_body_ratio = upper_body_ratio
        self.min_containment = min_containment

    def compute_spatial_affinity(self, face_bbox: BBox, body_bbox: BBox) -> float:
        fc_x, fc_y = face_bbox.center
        bc_x, _ = body_bbox.center

        # Relaxed horizontal boundary check (accounting for slight tilt/turn)
        margin_x = 0.15 * body_bbox.width
        if fc_x < body_bbox.x1 - margin_x or fc_x > body_bbox.x2 + margin_x:
            return 0.0

        # Anatomical vertical head zone check: face center should be in upper body section
        head_zone_bottom = body_bbox.y1 + self.upper_body_ratio * body_bbox.height
        head_zone_top = body_bbox.y1 - 0.15 * body_bbox.height
        if fc_y < head_zone_top or fc_y > head_zone_bottom:
            return 0.0

        # Overlap with upper body region
        upper_body_bbox = BBox(
            x1=body_bbox.x1,
            y1=body_bbox.y1,
            x2=body_bbox.x2,
            y2=head_zone_bottom
        )
        
        ix1 = max(face_bbox.x1, upper_body_bbox.x1)
        iy1 = max(face_bbox.y1, upper_body_bbox.y1)
        ix2 = min(face_bbox.x2, upper_body_bbox.x2)
        iy2 = min(face_bbox.y2, upper_body_bbox.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        face_area = face_bbox.area
        if face_area <= 0.0:
            return 0.0

        containment = inter / face_area
        if containment < self.min_containment:
            return 0.0

        # Centering score: face alignment with body horizontal axis
        norm_center_offset = abs(fc_x - bc_x) / (0.5 * body_bbox.width + 1e-6)
        alignment_score = max(0.0, 1.0 - norm_center_offset)

        # Scale proportion score
        scale_ratio = face_bbox.height / (body_bbox.height + 1e-6)
        if 0.10 <= scale_ratio <= 0.40:
            scale_score = 1.0
        elif 0.05 <= scale_ratio <= 0.50:
            scale_score = 0.7
        else:
            scale_score = 0.3

        total_affinity = (0.50 * containment) + (0.30 * alignment_score) + (0.20 * scale_score)
        return total_affinity

    def associate(
        self, faces: List[FaceDetection], bodies: List[PersonDetection]
    ) -> List[Tuple[Optional[FaceDetection], PersonDetection]]:
        """
        Solves optimal bipartite matching between detected faces and bodies.
        Returns a list of tuples: (matched_face_or_none, body).
        """
        if not bodies:
            return []
        if not faces:
            return [(None, b) for b in bodies]

        num_faces = len(faces)
        num_bodies = len(bodies)

        cost_matrix = []
        for i in range(num_faces):
            row = []
            for j in range(num_bodies):
                affinity = self.compute_spatial_affinity(faces[i].bbox, bodies[j].bbox)
                if affinity > 0.15:
                    cost = 1.0 - affinity
                else:
                    cost = 999.0  # Infeasible pair
                row.append(cost)
            cost_matrix.append(row)

        matches = hungarian_algorithm(cost_matrix)
        matched_dict = {}
        for f_idx, b_idx in matches:
            if cost_matrix[f_idx][b_idx] < 0.85: # Valid threshold
                matched_dict[b_idx] = faces[f_idx]

        results = []
        for b_idx, body in enumerate(bodies):
            results.append((matched_dict.get(b_idx, None), body))
        return results
