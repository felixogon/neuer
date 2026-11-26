# Computer vision primer for Neuer

What you need to understand the perception benchmark, in the order the pipeline uses it. Each section points to the code that implements it.

## 1. An image is a measurement

A colour image is a `H × W × 3` array of numbers 0–255. OpenCV stores channels as **B, G, R** (not RGB). Each pixel is the light that hit one sensor cell during the **exposure time**. Two consequences:

- If the ball moves during the exposure, its light is spread over several pixels: **motion blur**. Blur length (px) ≈ speed in the image (px/s) × exposure (s). In the dark, the camera lengthens the exposure, so the blur gets worse.
- Most webcams have a **rolling shutter**: rows are read top to bottom, a few ms apart, so a fast ball can come out slightly skewed.

## 2. The pinhole camera and intrinsics (`geometry.py`, `calibrate.py`)

A 3D point (X, Y, Z) in front of the camera lands at pixel

```
u = fx · X/Z + cx        v = fy · Y/Z + cy
```

- **fx, fy**: focal length *in pixels* (how strongly the lens magnifies).
- **cx, cy**: principal point (where the optical axis hits the image, roughly the centre).
- Together they form the **camera matrix K**. Real lenses also bend straight lines (**distortion**, coefficients k1, k2, p1, p2, k3).

These are **intrinsics**, a property of the camera alone. **Extrinsics** are where the camera sits in the world, e.g. camera → robot head; that is a later stage of Neuer.

**Calibration** finds K and distortion by showing a checkerboard with known square size from many angles and minimising the **reprojection error** (detected corner vs where the model predicts it). Below 0.5 px RMS is good.

**Depth from size.** Turning the equation around, a ball of diameter D with radius r pixels is at depth Z = fx · D / (2r). With a tennis ball at 4 m and fx ≈ 1000 px, r ≈ 8 px, so a 1 px error in r is a 12 % depth error. That is why the robot gets a depth camera.

## 3. Colour spaces and thresholding (`methods/color.py`)

In BGR, a yellow ball in shadow and in sunlight has very different numbers in all three channels. **HSV** separates *which colour* (Hue, 0–179 in OpenCV), *how pure* (Saturation) and *how bright* (Value). A yellow ball keeps roughly the same Hue in shade and light, so a box in HSV (`cv2.inRange`) selects it more robustly.

The output is a **binary mask**. **Morphology** cleans it: *opening* (erode, then dilate) removes speckles; *closing* (dilate, then erode) fills holes such as the seam. **Contours** outline each blob, and `minEnclosingCircle` fits a circle. The fill ratio (blob area / circle area) says how disc-like a blob is.

This is "classical" CV: every step is a rule you wrote. It is fast and explainable, and it breaks on anything your rules did not anticipate.

## 4. Learned detection (`methods/yolo.py`)

A **convolutional neural network** learns its own filters from labelled examples. A detector like YOLO outputs, for the whole image in one pass, a set of **boxes**, each with a **class** (one of COCO's 80, e.g. 32 = sports ball) and a **confidence**. Important ideas:

- **Input size.** The image is resized (default 640 px long side) before the network sees it. Small objects can shrink to a few pixels and disappear. Running at 1280 helps, at ~4× the compute.
- **Confidence threshold.** Lower = finds more balls *and* more false ones. Precision/recall trade-off.
- **NMS** (non-maximum suppression) used to remove duplicate boxes after the network; YOLO26 is trained to not need it ("end-to-end").
- **Pretrained vs fine-tuned.** COCO's sports balls are mostly large and sharp. A small, blurred ball 6 m away is "out of distribution". Fine-tuning on your own labelled frames is the usual fix, and is a natural next step after this benchmark.

## 5. Segmentation and tracking (`methods/sam_family.py`)

- **Detection** answers "where is a ball?" with a box.
- **Segmentation** answers "which pixels are the ball?" with a mask.
- **Tracking** answers "where is *that same* ball now?" across frames.

**SAM 2** is a *promptable* segmenter: you click or draw a box, it returns the mask. In video, it keeps a **memory bank** of past frames and masks; each new frame attends to that memory (**memory attention**), which is how it follows the object, and how it can recover after a short occlusion. It never decides on its own *what* to track. **EfficientTAM** keeps the design but uses a much lighter image encoder.

## 6. Measuring (`metrics.py`)

- **Ground truth** = your hand labels. Every metric is only as good as these.
- **Recall** = found / should have found. **Precision** = correct / reported.
- **Centre error** in pixels, measured on hits only.
- **Latency** = time from frame to answer. Report the **median and p95**: a keeper is hurt by slow frames, not by the average.
- **Throughput (FPS)** ≠ latency. A pipeline can process 30 frames per second and still deliver each answer 100 ms late.

## 7. Where this goes next

Detections per frame → **3D position** (depth camera + intrinsics) → **filter** (Kalman: combine a motion model with noisy measurements to estimate position *and velocity*, bridge missed frames) → **predicted crossing point** at the goal line → policy. The benchmark's noise and dropout numbers are what the Kalman filter's measurement noise and the simulator's perception noise should be set to.

Good references: Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., free online), chapters on image formation and calibration; the OpenCV-Python tutorials; Hartley & Zisserman for the geometry once you want proofs.
