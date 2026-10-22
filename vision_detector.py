from __future__ import annotations
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from math_utils import BBox
from models import FaceDetection, PersonDetection

class VisionDetector:
    """
    CPU-only detector combining:
      1. OpenCV YuNet ONNX model (ultra-fast, ~300KB, CPU-optimized) for face detection.
      2. Adaptive Upper-Body / Torso Extrapolator + Foreground Motion Segmenter
         to deduce whole-person bounding boxes from upper body geometry.
    """
    def __init__(self, model_path: str = "models_zoo/face_detection_yunet_2023mar.onnx", score_threshold: float = 0.6):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.detector = None
        self.last_img_size = None
        
        if os.path.exists(model_path):
            self.detector = cv2.FaceDetectorYN_create(
                model_path,
                "",
                (320, 240),
                score_threshold=self.score_threshold,
                nms_threshold=0.3,
                top_k=5000
            )

        # Background subtractor for whole-body presence when face turns away
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    def detect(self, frame: np.ndarray) -> Tuple[List[PersonDetection], List[FaceDetection]]:
        h, w = frame.shape[:2]
        faces: List[FaceDetection] = []
        persons: List[PersonDetection] = []

        # 1. Detect faces using YuNet
        if self.detector is not None:
            if self.last_img_size != (w, h):
                self.detector.setInputSize((w, h))
                self.last_img_size = (w, h)

            _, detections = self.detector.detect(frame)
            if detections is not None:
                for det in detections:
                    fx, fy, fw, fh = det[0:4]
                    score = float(det[14])
                    if score >= self.score_threshold:
                        # Ensure bounds
                        x1 = max(0.0, float(fx))
                        y1 = max(0.0, float(fy))
                        x2 = min(float(w), float(fx + fw))
                        y2 = min(float(h), float(fy + fh))
                        face_box = BBox(x1, y1, x2, y2, confidence=score)
                        faces.append(FaceDetection(bbox=face_box, identity_label="User", confidence=score))

                        # Anthropomorphic body estimation:
                        # In typical webcam framing (upper body), person body width is ~2.2x face width,
                        # and height extends downwards ~3.0x face height.
                        bw = fw * 2.4
                        bh = fh * 3.4
                        bcx = x1 + fw / 2.0
                        body_x1 = max(0.0, bcx - bw / 2.0)
                        body_x2 = min(float(w), bcx + bw / 2.0)
                        body_y1 = max(0.0, y1 - 0.1 * fh)
                        body_y2 = min(float(h), body_y1 + bh)
                        
                        person_box = BBox(body_x1, body_y1, body_x2, body_y2, confidence=score)
                        persons.append(PersonDetection(bbox=person_box, confidence=score))

        # 2. Also run motion / foreground contours to sustain body bbox when face completely turns 180 deg
        fg_mask = self.bg_subtractor.apply(frame)
        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 12000:  # Substantial person contour in frame
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                body_candidate = BBox(cx, cy, cx + cw, cy + ch, confidence=0.75)
                
                # If this candidate overlaps with already found person via face, merge or ignore
                overlap = False
                for p in persons:
                    if p.bbox.iou(body_candidate) > 0.3:
                        overlap = True
                        break
                if not overlap:
                    persons.append(PersonDetection(bbox=body_candidate, confidence=0.75))

        return persons, faces
