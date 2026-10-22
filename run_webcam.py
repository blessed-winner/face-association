#!/usr/bin/env python3
"""
Live Webcam Application for Whole-Person Tracking and Face Association.
Features:
  - Honest Three-State Model with exponential confidence decay
  - Real-time HUD overlay:
      * Bounding boxes color-coded by state:
          - Green:  CONFIRMED
          - Amber:  TEMPORARY (face lost, body continuously tracked)
          - Gray:   UNKNOWN
      * Real-time confidence decay progress bar
      * Target centering crosshairs and virtual servo steering indicator
      * Actuator deadband visualization
  - Interactive Hotkeys:
      * 'q' or ESC: Quit application
      * 'r': Reset tracker state
      * 's': Toggle simulated head-turn / face dropout
"""
import sys
import time
import math
import cv2
import numpy as np

from models import TrackState
from tracker import WholePersonTracker
from centering_controller import TargetCenteringController
from vision_detector import VisionDetector

def draw_hud(
    frame: np.ndarray,
    tracks,
    cmd,
    fps: float,
    manual_dropout: bool
):
    h, w = frame.shape[:2]
    cx, cy = int(w / 2), int(h / 2)

    # 1. Draw Camera Center & Deadband Crosshairs
    deadband_px = int(cmd.frame_width * 0.015) if hasattr(cmd, 'frame_width') else 15
    cv2.circle(frame, (cx, cy), 6, (200, 200, 200), 1)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (120, 120, 120), 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (120, 120, 120), 1)
    # Deadband rectangle
    cv2.rectangle(
        frame,
        (cx - deadband_px, cy - deadband_px),
        (cx + deadband_px, cy + deadband_px),
        (80, 80, 80),
        1
    )

    # 2. Draw Tracks & Honest State Machine
    colors = {
        TrackState.CONFIRMED: (0, 220, 0),     # Bright Green
        TrackState.TEMPORARY: (0, 180, 255),   # Orange / Amber
        TrackState.UNKNOWN:   (130, 130, 130)  # Gray
    }

    for track in tracks:
        bbox = track.current_bbox
        state = track.state
        color = colors.get(state, (255, 255, 255))
        
        # Bounding box
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
        thickness = 2 if state == TrackState.CONFIRMED else (2 if state == TrackState.TEMPORARY else 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label banner
        state_label = f"ID:{track.track_id} {state.value.upper()} [{track.confidence*100:.0f}%]"
        if track.identity_label and state != TrackState.UNKNOWN:
            state_label = f"{track.identity_label} | {state_label}"
            
        (tw, th), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, state_label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Draw Face box if present
        if track.last_face_bbox and state == TrackState.CONFIRMED:
            fb = track.last_face_bbox
            fx1, fy1 = max(0, int(fb.x1)), max(0, int(fb.y1))
            fx2, fy2 = min(w, int(fb.x2)), min(h, int(fb.y2))
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 1)

    # 3. Draw Centering Steering Vectors
    if cmd.active and cmd.target_center:
        tx, ty = int(cmd.target_center[0]), int(cmd.target_center[1])
        cv2.circle(frame, (tx, ty), 6, (0, 255, 255), -1)
        # Vector from center to target
        cv2.arrowedLine(frame, (cx, cy), (tx, ty), (0, 255, 255), 2, tipLength=0.15)

    # 4. Top Telemetry Dashboard (Glassmorphic HUD bar)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (370, 140), (60, 60, 60), 1)

    cv2.putText(frame, "WHOLE-PERSON TRACKER (CPU-ONLY)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Active Tracks: {len(tracks)}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Centering Steering Indicator
    pan_bar_len = int(cmd.recommended_pan_velocity * 60)
    pan_color = (0, 220, 0) if abs(cmd.pan_error) < 0.1 else (0, 180, 255)
    cv2.putText(frame, f"Pan Servo Cmd: {cmd.recommended_pan_velocity:+.2f}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.line(frame, (220, 72), (220 + pan_bar_len, 72), pan_color, 4)
    cv2.circle(frame, (220, 72), 3, (120, 120, 120), -1)

    # Confidence bar for primary target
    if tracks:
        primary = tracks[0]
        cv2.putText(frame, f"Conf Decay C(t): {primary.confidence*100:.1f}%", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        bar_w = int(primary.confidence * 120)
        st_color = colors.get(primary.state, (100, 100, 100))
        cv2.rectangle(frame, (200, 88), (200 + bar_w, 98), st_color, -1)
        cv2.rectangle(frame, (200, 88), (320, 98), (100, 100, 100), 1)

    # Drop-out Simulation Indicator
    if manual_dropout:
        cv2.putText(frame, "[SIMULATED FACE LOSS ACTIVE: Press 's' to toggle]", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    else:
        cv2.putText(frame, "Hotkeys: 'q' Exit | 's' Simulate Face Loss | 'r' Reset", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

def main():
    print("=" * 70)
    print("  INITIALIZING WEBCAM WHOLE-PERSON TRACKER & FACE ASSOCIATION")
    print("  - Honest 3-State Model (Confirmed -> Temporary -> Unknown)")
    print("  - Exponential Decay Law: C(t) = exp(-lambda * delta_t)")
    print("  - Target Centering: Smooth PD Steering with Deadband Elimination")
    print("=" * 70)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] ERROR: Unable to open webcam (device index 0). Please check camera permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = VisionDetector()
    tracker = WholePersonTracker(half_life_frames=25.0, tau_confirmed=0.70, tau_temporary=0.30)
    controller = TargetCenteringController(frame_width=640, frame_height=480, deadband_px=14.0)

    manual_dropout = False
    fps_ema = 30.0
    prev_time = time.time()

    print("[+] Camera stream started! Focus the OpenCV window to interact.")
    print("    Press 'q' or ESC to exit.")
    print("    Press 's' to simulate temporary face loss (tests decay from Confirmed -> Temporary).")
    print("    Press 'r' to reset all tracks.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame grab failed. Exiting loop.")
                break

            curr_time = time.time()
            dt = max(1e-4, curr_time - prev_time)
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
            prev_time = curr_time

            # Mirror view for intuitive user experience
            frame = cv2.flip(frame, 1)

            # Detect bodies and faces
            persons, faces = detector.detect(frame)

            # If user pressed 's' to simulate facial loss (e.g. head turn without moving away)
            if manual_dropout:
                faces = []

            # Update Tracker with Honest 3-State Logic
            tracks = tracker.process_frame(persons, faces, dt=dt)

            # Compute Target Centering Steering Command
            cmd = controller.compute_control_command(tracks, dt=dt)

            # Draw HUD
            draw_hud(frame, tracks, cmd, fps_ema, manual_dropout)

            cv2.imshow("Whole-Person Tracking & Honest Face Association (Webcam)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                manual_dropout = not manual_dropout
                status = "SIMULATED FACE LOSS ACTIVATED" if manual_dropout else "SIMULATED FACE LOSS DEACTIVATED"
                print(f"[>] {status}")
            elif key == ord('r'):
                tracker = WholePersonTracker(half_life_frames=25.0, tau_confirmed=0.70, tau_temporary=0.30)
                print("[>] Tracker reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[+] Webcam closed successfully.")

if __name__ == "__main__":
    main()
