<div align="center">
<h1>Neuer</h1>
<p><b>A humanoid goalkeeper that sees shots with its own head camera and blocks them</b></p>

Felix Ogon · AMBER Lab, Caltech (Prof. Aaron D. Ames)
</div>

---

## Overview

Most humanoid ball experiments rely on an external **motion capture system** to tell the robot where the ball is. That works in a lab, but nowhere else.

**Neuer** is a goalkeeper for the Unitree G1 that does it alone: the robot **sees a shot coming with its own head camera**, predicts where the ball will cross the goal line, and moves to block it. No external sensors.

Blocking shots is the first step. Catching and throwing come next.

## Approach

```
Head RGB-D camera
      │
      ▼
1. Camera calibration ─── intrinsics + camera-to-head extrinsics (OpenCV)
      │
      ▼
2. Ball detection ─────── color thresholding → learned detector on NVIDIA Jetson
      │
      ▼
3. 3D tracking ────────── depth + head-motion compensation (robot head pose)
      │
      ▼
4. Shot prediction ────── Kalman filter → predicted crossing point and time
      │
      ▼
5. Save policy ────────── RL policy trained in MuJoCo (MJlab) → joint targets → save
```

1. **Calibration.** Estimate the camera intrinsics and the transform from the camera to the robot's head, so detections are expressed in the robot's frame.
2. **Detection.** Start with a fast color-based detector as a baseline, then move to a learned detector running on the onboard Jetson.
3. **3D tracking.** Use depth to place the ball in 3D, and compensate for the robot's own head motion so the estimate stays stable while it moves.
4. **Prediction.** A Kalman filter tracks the ball and predicts where and when it will reach the goal. This becomes the target for the save.
5. **Save policy.** Train the policy in simulation with realistic perception noise and latency, so it tolerates the imperfect estimates a real camera produces, then deploy it on hardware.

## Hardware

The G1 needs a camera mount that stays stable during fast movements. I am designing a custom **head for the G1 in SolidWorks** with an integrated depth camera, using FEA to validate the parts.

## Evaluation

- **Metric:** save rate over repeated shots at increasing ball speeds.
- **Ground truth:** OptiTrack motion capture runs in parallel to measure perception error, but is never given to the policy.
- **Baseline:** the same policy driven by motion capture instead of the camera.

## Roadmap

**Blocking**
- [ ] Camera calibration and camera-to-head extrinsics
- [ ] Color-based ball detection baseline
- [ ] 3D tracking with head-motion compensation, validated against OptiTrack
- [ ] Kalman filter shot prediction
- [ ] Learned detector on Jetson
- [ ] Save policy trained in simulation with perception noise and latency
- [ ] Custom camera head (SolidWorks, FEA)
- [ ] Hardware experiments on the G1

**Next**
- [ ] Catching the ball
- [ ] Throwing it back

## Acknowledgments

This project uses tools and methods from the AMBER Lab, including the humanoid training pipeline developed by Blake Werner and colleagues for [TaskNPoint](https://ilonadem.github.io/tasknpoint_website/).
