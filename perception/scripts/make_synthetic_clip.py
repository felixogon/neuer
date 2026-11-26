"""Make a fake clip with perfectly known ground truth, to test the pipeline itself.

    python scripts/make_synthetic_clip.py

Draws a yellow-green ball flying toward the camera over a noisy background,
with motion blur and (optionally) a same-coloured distractor. Labels are
written for every frame from the known trajectory.

Why: before trusting any number in the results table, check that the
benchmark code is right on a case where you know the answer. If the colour
method does not score ~100 % here, the bug is in the harness, not the method.
(Pretrained YOLO / SAM may not recognise a flat painted disc as a ball;
that is expected and says nothing about real footage.)
"""

import argparse
import json

import _paths
import cv2
import numpy as np

from neuer_perception.clip import Clip, Label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synthetic_flight")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--distractor", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = _paths.CLIPS_DIR / args.name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    W, H = args.width, args.height
    bg = cv2.GaussianBlur(rng.integers(40, 160, (H // 8, W // 8, 3), dtype=np.uint8), (0, 0), 1.5)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    labels = {}
    fps = 30.0
    for i in range(args.frames):
        s = i / (args.frames - 1)
        cx = 150 + s * (W - 300)
        cy = 0.6 * H - 300 * s * (1 - s) * 2
        r = 6 + 40 * s**2  # approaches the camera
        vel = np.array([(W - 300) / (args.frames - 1), 0.0])
        img = bg.copy()
        if args.distractor:
            cv2.rectangle(img, (W - 220, 80), (W - 140, 120), (40, 230, 200), -1)  # yellow highlighter
        # motion blur: average several sub-frame positions
        layer = np.zeros_like(img, dtype=np.float32)
        alpha = np.zeros((H, W), np.float32)
        n_sub = 6
        for k in range(n_sub):
            off = (k / (n_sub - 1) - 0.5) * vel * 0.5
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (int(cx + off[0]), int(cy + off[1])), int(round(r)), 255, -1, cv2.LINE_AA)
            alpha += m.astype(np.float32) / 255.0 / n_sub
        layer[:] = (40, 230, 200)  # BGR optic yellow
        img = (img * (1 - alpha[..., None]) + layer * alpha[..., None]).astype(np.uint8)
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "frames" / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels[i] = Label(True, float(cx), float(cy), float(r))
    with open(out / "timestamps.csv", "w") as f:
        f.write("frame,t_s\n")
        for i in range(args.frames):
            f.write(f"{i},{i / fps:.6f}\n")
    meta = {"test": "other", "motion": "fast", "lighting": "normal", "n_balls": 1, "ball": "synthetic",
            "ball_diameter_m": 0.067, "notes": "synthetic sanity check", "n_frames": args.frames, "fps": fps,
            "width": W, "height": H}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    Clip(out).save_labels(labels)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
