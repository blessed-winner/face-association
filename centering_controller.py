from __future__ import annotations
import math
from typing import List, Tuple, Optional
from models import CenteringControlCommand, TrackState
from tracker_state import PersonTrack

class TargetCenteringController:
    """
    Target Centering Logic:
    Calculates pan/tilt servo or camera steering offsets to continuously keep the
    primary target centered in the frame.
    Features:
      - Exponential smoothing & deadzone to eliminate actuator chatter / limit cycling.
      - P-D (Proportional-Derivative) velocity recommendations.
      - Fallback logic: prioritizes CONFIRMED state, maintains smooth tracking in TEMPORARY state,
        gracefully halts or re-acquires in UNKNOWN state without violent jumps.
    """
    def __init__(
        self,
        frame_width: float = 1280.0,
        frame_height: float = 720.0,
        deadband_px: float = 12.0,
        kp_pan: float = 1.2,
        kd_pan: float = 0.15,
        kp_tilt: float = 1.0,
        kd_tilt: float = 0.10
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center = (frame_width / 2.0, frame_height / 2.0)
        self.deadband_px = deadband_px

        self.kp_pan = kp_pan
        self.kd_pan = kd_pan
        self.kp_tilt = kp_tilt
        self.kd_tilt = kd_tilt

        self.prev_pan_error = 0.0
        self.prev_tilt_error = 0.0
        self.smoothed_pan_vel = 0.0
        self.smoothed_tilt_vel = 0.0

    def compute_control_command(
        self,
        tracks: List[PersonTrack],
        preferred_identity: Optional[str] = None,
        dt: float = 1.0
    ) -> CenteringControlCommand:
        """
        Determines primary target:
          1. Matches preferred_identity in CONFIRMED or TEMPORARY state
          2. If none, selects track with highest identity confidence
          3. Returns control offset and recommended velocity
        """
        selected_track: Optional[PersonTrack] = None

        if preferred_identity:
            for t in tracks:
                if t.identity_label == preferred_identity and t.state in (TrackState.CONFIRMED, TrackState.TEMPORARY):
                    selected_track = t
                    break

        if selected_track is None and tracks:
            # Sort by state priority (CONFIRMED > TEMPORARY > UNKNOWN) and confidence
            sorted_tracks = sorted(
                tracks,
                key=lambda t: (
                    2 if t.state == TrackState.CONFIRMED else (1 if t.state == TrackState.TEMPORARY else 0),
                    t.confidence
                ),
                reverse=True
            )
            selected_track = sorted_tracks[0]

        if selected_track is None:
            # Idle/centered command
            return CenteringControlCommand(
                pan_error=0.0,
                tilt_error=0.0,
                target_person_id=None,
                target_state=None,
                active=False,
                recommended_pan_velocity=0.0,
                recommended_tilt_velocity=0.0,
                target_center=self.frame_center,
                frame_center=self.frame_center
            )

        # Target center calculation (focus on upper body/head region for natural framing)
        bbox = selected_track.current_bbox
        tc_x = (bbox.x1 + bbox.x2) / 2.0
        tc_y = bbox.y1 + 0.35 * bbox.height  # Aim at upper torso / head area

        raw_err_x = tc_x - self.frame_center[0]
        raw_err_y = tc_y - self.frame_center[1]

        # Apply deadzone
        if abs(raw_err_x) < self.deadband_px:
            err_x = 0.0
        else:
            err_x = raw_err_x - math.copysign(self.deadband_px, raw_err_x)

        if abs(raw_err_y) < self.deadband_px:
            err_y = 0.0
        else:
            err_y = raw_err_y - math.copysign(self.deadband_px, raw_err_y)

        # Normalized errors [-1.0, 1.0]
        norm_err_x = err_x / (self.frame_width / 2.0)
        norm_err_y = err_y / (self.frame_height / 2.0)

        # PD control velocities
        d_err_x = (norm_err_x - self.prev_pan_error) / max(dt, 1e-4)
        d_err_y = (norm_err_y - self.prev_tilt_error) / max(dt, 1e-4)

        cmd_pan_vel = self.kp_pan * norm_err_x + self.kd_pan * d_err_x
        cmd_tilt_vel = self.kp_tilt * norm_err_y + self.kd_tilt * d_err_y

        # Clamp max velocities [-1.0, 1.0]
        cmd_pan_vel = max(-1.0, min(1.0, cmd_pan_vel))
        cmd_tilt_vel = max(-1.0, min(1.0, cmd_tilt_vel))

        # Smooth velocity output
        alpha = 0.70
        self.smoothed_pan_vel = (1.0 - alpha) * self.smoothed_pan_vel + alpha * cmd_pan_vel
        self.smoothed_tilt_vel = (1.0 - alpha) * self.smoothed_tilt_vel + alpha * cmd_tilt_vel

        self.prev_pan_error = norm_err_x
        self.prev_tilt_error = norm_err_y

        return CenteringControlCommand(
            pan_error=norm_err_x,
            tilt_error=norm_err_y,
            target_person_id=selected_track.track_id,
            target_state=selected_track.state,
            active=True,
            recommended_pan_velocity=self.smoothed_pan_vel,
            recommended_tilt_velocity=self.smoothed_tilt_vel,
            target_center=(tc_x, tc_y),
            frame_center=self.frame_center
        )
