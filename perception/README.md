# Neuer perception: ball detection benchmark

Compares an OpenCV colour baseline, YOLO, SAM 2 and EfficientTAM on webcam clips of a ball at different distances, speeds, lighting, occlusion and clutter. The protocol and the reasoning are in [`docs/perception-benchmark.md`](../docs/perception-benchmark.md); the concepts are in [`docs/cv-primer.md`](../docs/cv-primer.md).

## Setup (macOS, Apple Silicon)

```bash
cd perception
python3.12 -m venv .venv         # Python 3.10-3.12
source .venv/bin/activate
pip install -r requirements.txt
bash setup_models.sh            # SAM 2 + EfficientTAM code and weights (~0.5 GB)
```

Give your terminal camera access: System Settings → Privacy & Security → Camera. Turn Center Stage off. If camera 0 is your iPhone (Continuity Camera), use `--camera 1`.

## Workflow

```bash
python scripts/make_checkerboard.py                       # print or show on an iPad; measure a square
python scripts/calibrate.py --square-mm 24.0              # -> configs/camera_macbook.yaml
python scripts/tune_color.py                              # HSV sliders, s to save
python scripts/live_demo.py --methods color yolo26n       # build intuition

python scripts/record.py --name dist_2m --test distance --distance 2 --pattern still
python scripts/record.py --name motion_lr_slow --test motion --pattern left-right --motion slow --distance 1.5
python scripts/label.py data/clips/dist_2m --every 5
python scripts/benchmark.py                                # every clip x every method
python scripts/summarize.py                                # -> results/RESULTS.md + plots
python scripts/render.py data/clips/dist_2m                # overlay video for review
python scripts/render.py data/clips/shake --gif --crop     # small cropped GIF for the docs (~1 MB)
# optional: YOLO threshold-sensitivity analysis (separate folder, main results untouched)
python scripts/benchmark.py --methods yolo26n yolo26n@1280 --yolo-conf 0.01 --out results/threshold_sweep/raw
python scripts/threshold_sweep.py                          # -> results/threshold_sweep/pr_curve.png
```

Sanity check the harness first: `python scripts/make_synthetic_clip.py --distractor`, then benchmark and summarize. The colour method should get every frame of the plain clip and show false positives on the distractor clip.

## Layout

```
neuer_perception/
  clip.py          recorded clip: frames, timestamps, conditions, labels
  geometry.py      intrinsics, depth from apparent size, back-projection
  metrics.py       recall, precision, centre/depth error, latency, miss streaks
  methods/
    color.py       HSV threshold + circle fit
    yolo.py        Ultralytics YOLO, COCO sports-ball class
    sam_family.py  SAM 2 / EfficientTAM video predictors (box-prompted)
scripts/           record, label, calibrate, tune, benchmark, summarize, render, live demo
configs/           colour thresholds, camera calibration
data/clips/        recordings (frames are git-ignored; labels and meta are committed)
results/           raw per-frame predictions, RESULTS.md, plots
```

## Adding a method

Subclass `FrameDetector` and implement `detect(frame_bgr) -> list[Detection]` (best first), then register a name in `methods/__init__.py`. Everything else (timing, scoring, tables, plots) works automatically.

## Known quirks

- SAM 2 on Apple's MPS backend is officially "preliminary". `benchmark.py` enables CPU fallback for unsupported ops; if a model still errors, run it with `--device cpu`.
- All six benchmark methods run on MPS; none needed `--device cpu` (checked on an M1 Pro, torch 2.15, with `PYTORCH_ENABLE_MPS_FALLBACK=0` to make sure no op silently falls back to the CPU).
- `sam2.1-tiny` takes ~1 s per frame on an M1 Pro (EfficientTAM-Ti/512: ~0.1 s). This is real compute, not a fallback: the latency ramps up over the first 7 frames as the memory bank fills, then plateaus, and a single SAM 2-sized memory cross-attention (4096 × ~28 700 tokens) alone takes ~180 ms on MPS in fp32, ×4 layers. The CPU is slower (~1.4 s). Cutting that cost is exactly what EfficientTAM's efficient memory attention is for.
- EfficientTAM's install pins older `pandas`, `pillow` and `imageio`; everything here works with those versions (`pip check` is clean).
- SAM 2 / EfficientTAM load the whole clip into memory before tracking. Keep clips around 5–10 s.
- The "post-processing step" warning from SAM 2 at start-up is harmless (it is the optional CUDA extension).
