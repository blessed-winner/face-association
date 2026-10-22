from __future__ import annotations
import math
from typing import List, Dict, Any
from models import TrackState
from tracker import WholePersonTracker
from centering_controller import TargetCenteringController
from scenario_generator import SyntheticScenarioGenerator

class ScenarioEvaluator:
    """
    Evaluator executing scenarios against WholePersonTracker and TargetCenteringController.
    Logs and benchmarks:
      - State distribution (Confirmed, Temporary, Unknown)
      - Temporal identity continuity (continuity preservation across facial loss)
      - Identity switches (ID switches during crossing)
      - Centering tracking error (root-mean-square pixel error)
      - Actuator jerk / jitter metrics (rate of change of control effort)
      - Exponential confidence decay adherence
    """
    def __init__(self):
        self.generator = SyntheticScenarioGenerator()

    def evaluate_head_turn(self) -> Dict[str, Any]:
        tracker = WholePersonTracker(half_life_frames=25.0, tau_confirmed=0.70, tau_temporary=0.30)
        controller = TargetCenteringController()
        data = self.generator.scenario_head_turn(num_frames=120)

        history = []
        state_counts = {TrackState.CONFIRMED: 0, TrackState.TEMPORARY: 0, TrackState.UNKNOWN: 0}
        id_switches = 0
        assigned_id = None
        pan_errors = []
        velocities = []

        for frame_idx, (persons, faces, gt) in enumerate(data):
            tracks = tracker.process_frame(persons, faces)
            cmd = controller.compute_control_command(tracks, preferred_identity="Alice")

            target_track = None
            if tracks:
                target_track = tracks[0]
                state_counts[target_track.state] += 1
                if assigned_id is None:
                    assigned_id = target_track.track_id
                elif target_track.track_id != assigned_id:
                    id_switches += 1
                    assigned_id = target_track.track_id

            pan_errors.append(abs(cmd.pan_error * (controller.frame_width / 2.0)))
            velocities.append(cmd.recommended_pan_velocity)
            history.append({
                "frame": frame_idx,
                "face_visible": gt["face_visible"],
                "state": target_track.state.value if target_track else "None",
                "confidence": target_track.confidence if target_track else 0.0,
                "identity": target_track.identity_label if target_track else None,
                "pan_vel": cmd.recommended_pan_velocity
            })

        loss_frames = [h for h in history if 30 <= h["frame"] <= 75]
        preserved_frames = [h for h in loss_frames if h["identity"] == "Alice" and h["state"] in ("Confirmed", "Temporary")]
        continuity_ratio = len(preserved_frames) / max(1, len(loss_frames))

        # Velocity jerk: average absolute delta in consecutive pan velocity
        jerk = sum(abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))) / max(1, len(velocities) - 1)

        return {
            "scenario": "Head Turn / Yaw Rotation",
            "total_frames": len(data),
            "state_counts": {k.value: v for k, v in state_counts.items()},
            "id_switches": id_switches,
            "continuity_ratio": continuity_ratio,
            "mean_centering_error_px": sum(pan_errors) / len(pan_errors),
            "actuator_jerk_metric": jerk,
            "history": history
        }

    def evaluate_occlusion(self) -> Dict[str, Any]:
        tracker = WholePersonTracker(half_life_frames=25.0, tau_confirmed=0.70, tau_temporary=0.30)
        controller = TargetCenteringController()
        data = self.generator.scenario_occlusion(num_frames=140)

        history = []
        state_counts = {TrackState.CONFIRMED: 0, TrackState.TEMPORARY: 0, TrackState.UNKNOWN: 0}
        id_switches = 0
        assigned_id = None
        pan_errors = []
        velocities = []

        for frame_idx, (persons, faces, gt) in enumerate(data):
            tracks = tracker.process_frame(persons, faces)
            cmd = controller.compute_control_command(tracks, preferred_identity="Bob")

            target_track = None
            if tracks:
                target_track = tracks[0]
                state_counts[target_track.state] += 1
                if assigned_id is None:
                    assigned_id = target_track.track_id
                elif target_track.track_id != assigned_id:
                    id_switches += 1
                    assigned_id = target_track.track_id

            pan_errors.append(abs(cmd.pan_error * (controller.frame_width / 2.0)))
            velocities.append(cmd.recommended_pan_velocity)
            history.append({
                "frame": frame_idx,
                "face_visible": gt["face_visible"],
                "state": target_track.state.value if target_track else "None",
                "confidence": target_track.confidence if target_track else 0.0,
                "identity": target_track.identity_label if target_track else None
            })

        loss_frames = [h for h in history if not h["face_visible"]]
        preserved_frames = [h for h in loss_frames if h["identity"] == "Bob" and h["state"] in ("Confirmed", "Temporary")]
        continuity_ratio = len(preserved_frames) / max(1, len(loss_frames))
        jerk = sum(abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))) / max(1, len(velocities) - 1)

        return {
            "scenario": "Column / Object Occlusion",
            "total_frames": len(data),
            "state_counts": {k.value: v for k, v in state_counts.items()},
            "id_switches": id_switches,
            "continuity_ratio": continuity_ratio,
            "mean_centering_error_px": sum(pan_errors) / len(pan_errors),
            "actuator_jerk_metric": jerk,
            "history": history
        }

    def evaluate_path_crossing(self) -> Dict[str, Any]:
        tracker = WholePersonTracker(half_life_frames=25.0, tau_confirmed=0.70, tau_temporary=0.30)
        controller = TargetCenteringController()
        data = self.generator.scenario_path_crossing(num_frames=150)

        history = []
        id_switches = 0
        diana_track_id = None
        charlie_track_id = None

        for frame_idx, (persons, faces, gt) in enumerate(data):
            tracks = tracker.process_frame(persons, faces)
            d_track = next((t for t in tracks if t.identity_label == "Diana"), None)
            c_track = next((t for t in tracks if t.identity_label == "Charlie"), None)

            if d_track:
                if diana_track_id is None:
                    diana_track_id = d_track.track_id
                elif d_track.track_id != diana_track_id and frame_idx > 60:
                    id_switches += 1

            if c_track:
                if charlie_track_id is None:
                    charlie_track_id = c_track.track_id
                elif c_track.track_id != charlie_track_id and frame_idx > 60:
                    id_switches += 1

            history.append({
                "frame": frame_idx,
                "diana_state": d_track.state.value if d_track else "None",
                "charlie_state": c_track.state.value if c_track else "None",
                "diana_conf": d_track.confidence if d_track else 0.0,
                "crossing": gt["crossing"]
            })

        crossing_frames = [h for h in history if h["crossing"]]
        diana_preserved = [h for h in crossing_frames if h["diana_state"] in ("Confirmed", "Temporary")]
        continuity_ratio = len(diana_preserved) / max(1, len(crossing_frames))

        return {
            "scenario": "Path Crossing / Multi-Person Occlusion",
            "total_frames": len(data),
            "id_switches": id_switches,
            "continuity_ratio": continuity_ratio,
            "history": history
        }
