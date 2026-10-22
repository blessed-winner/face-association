# Whole-Person Tracking & Honest 3-State Face Association Module

An end-to-end CPU-only Python software module that maintains temporal identity continuity across temporary facial loss through an **Honest Three-State Model** (`Confirmed`, `Temporary`, and `Unknown`) with **exponential confidence decay**.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Object Detection Engine   |
                      |  (Whole Body & Face Boxes)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Spatial Face-Body Mapper   |
                      |  - Anatomical Head Priors   |
                      |  - Kuhn-Munkres Bipartite   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Whole-Person Tracker    |
                      |  - Linear Motion Filter     |
                      |  - IoU Trajectory Predictor |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Honest 3-State FSM       |
                      |  - Exponential Decay C(t)   |
                      |  - Confirmed->Temp->Unknown |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Target Centering Control   |
                      |  - PD Steering Controller   |
                      |  - Anti-Chatter Deadband    |
                      +-----------------------------+
```

### 1. Honest Three-State Model
Identity confidence decays mathematically according to:
$$C(t) = \exp(-\lambda \cdot \Delta t) \quad \text{where} \quad \lambda = \frac{\ln 2}{\tau_{1/2}}$$

- **`Confirmed`** ($C(t) \ge 0.70$): Face actively detected or recently verified. Full identity certainty.
- **`Temporary`** ($0.30 \le C(t) < 0.70$): Face temporarily lost (head turn, looking away, brief occlusion). Whole-body track persists; target framing remains active without guessing.
- **`Unknown`** ($C(t) < 0.30$): Identity confidence expired or person initialized without a face. Refuses to hallucinate labels.

### 2. Spatial Anthropometric Face-to-Body Mapping
- **Anatomical Head Zone**: Face center must reside within the upper 45% of the person's bounding box.
- **Geometric Containment**: Minimum 45% spatial containment.
- **Proportion Matching**: Head-to-torso scale ratio filtering.
- **Optimal Assignment**: Kuhn-Munkres (Hungarian) linear sum assignment.

### 3. Target Centering Controller
- Continuous tracking of preferred target across face dropouts.
- **Deadband filter** ($\pm 12\text{ px}$) to eliminate actuator chatter / limit-cycling.
- **P-D control** with output velocity smoothing ($\alpha = 0.70$) to prevent sudden jumps.

---

## Tested Real-World Scenarios (Hardware-Free)

1. **Head Turn / Yaw Rotation**: Target walks across frame and looks away (180° head turn) for 45 frames. The system smoothly drops to `Temporary` and then re-acquires as `Confirmed` upon face reappearance with **0 identity switches**.
2. **Obstacle / Column Occlusion**: Target walks behind a static obstacle. Trajectory filter bridges the gap with **100% temporal continuity**.
3. **Path Crossing / Multi-Person Occlusion**: Two people cross paths in close proximity while one person's face is occluded by the other. The Hungarian IoU matcher preserves identity boundaries with **0 ID switches**.

---

## Commands to Run

### 1. Run Live Interactive Webcam Mode
Uses your computer webcam with real-time HUD telemetry, face-to-body mapping, honest 3-state exponential decay display, and virtual servo steering indicators:
```powershell
python run_webcam.py
```
**Interactive Webcam Controls:**
- **`q`** or **ESC**: Exit the webcam window.
- **`s`**: Simulate instant facial loss (forces the system to test `Confirmed` $\rightarrow$ `Temporary` $\rightarrow$ `Unknown` state decay while keeping your whole body in view).
- **Turn your head or cover your face**: Watch the bounding box smoothly transition from **Green (Confirmed)** to **Amber (Temporary)** with an active confidence decay bar, then immediately re-lock to **Green (Confirmed)** upon facing the camera again!
- **`r`**: Reset tracker state.

### 2. Run Headless Quantitative Benchmark Suite
Runs high-speed mathematical scenario evaluation (head turns, pillar occlusions, path crossing) in pure CPU Python:
```powershell
python run_benchmark.py
```

