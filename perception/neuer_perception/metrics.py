"""Scoring a method's output against hand-labelled ground truth.

Only labelled frames are scored (you label every k-th frame).

Definitions (per clip, per method):

* A prediction is a hit (true positive) if the ball is visible and the
  predicted centre is within one ball radius of the true centre (min 5 px).
* recall    = hits / frames where the ball is visible    ("did it find the ball?")
* precision = hits / frames where the method reported a ball ("when it says ball, is it right?")
  A detection in the wrong place counts as both a miss and a false positive.
* center error (px), radius error (%) are measured on hits only.
* depth error (%) converts radius to distance with the calibrated focal length
  (Z = fx * D / 2r) and compares against the distance implied by the
  ground-truth radius, and, for static clips, against the tape measure.
* latency is compute time per frame. p95 = 95th percentile: the slow frames
  matter for a goalkeeper, not only the average.
* max miss streak = longest run of consecutive labelled visible frames the
  method missed. Long streaks = track lost and not recovered.
* dropout = 1 - recall: the fraction of frames a Kalman filter has to bridge.
* 3D error (metres, hits only): predicted and labelled centres are both
  back-projected to camera coordinates (X right, Y down, Z forward), each with
  the depth from its own radius. The per-axis standard deviation of the error
  is the measurement noise of a position sensor. Note the reference is the
  hand label, so ~1-2 px of label noise is included. Frames labelled
  "occluded" are left out of depth and 3D error, but still count for recall:
  the label there is the estimated full ball, while a method only sees the
  visible part, so its predicted radius (and so its depth) is biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clip import Clip
from .geometry import TENNIS_BALL_DIAMETER_M, Intrinsics, backproject, depth_from_radius

MIN_GATE_PX = 5.0


def score_clip(clip: Clip, preds: pd.DataFrame, intr: Intrinsics | None = None) -> dict:
    labels = clip.labels()
    preds = preds.set_index("frame")
    diameter = float(clip.meta.get("ball_diameter_m", TENNIS_BALL_DIAMETER_M))
    fx = None
    intr = _intr_for(clip, intr)
    if intr is not None:
        fx = intr.fx

    hits = misses = fps_ = 0
    center_err, radius_err, depth_err, tape_err, err3d = [], [], [], [], []
    streak = max_streak = 0
    for f, lab in sorted(labels.items()):
        if f not in preds.index:
            continue
        p = preds.loc[f]
        found = bool(p["found"])
        if lab.visible:
            err = np.hypot(p["cx"] - lab.cx, p["cy"] - lab.cy) if found else np.inf
            if err <= max(lab.r, MIN_GATE_PX):
                hits += 1
                streak = 0
                center_err.append(err)
                radius_err.append(abs(p["r"] - lab.r) / lab.r * 100.0)
                if fx and not lab.occluded:  # occluded: label = full ball, prediction = visible part only -> biased radius
                    z_pred = depth_from_radius(p["r"], fx, diameter)
                    z_ref = depth_from_radius(lab.r, fx, diameter)
                    depth_err.append(abs(z_pred - z_ref) / z_ref * 100.0)
                    if np.isfinite(z_pred) and np.isfinite(z_ref):
                        err3d.append(backproject(p["cx"], p["cy"], z_pred, intr) - backproject(lab.cx, lab.cy, z_ref, intr))
                    if clip.meta.get("motion") == "static" and clip.meta.get("distance_m"):
                        z_tape = float(clip.meta["distance_m"])
                        tape_err.append(abs(z_pred - z_tape) / z_tape * 100.0)
            else:
                misses += 1
                streak += 1
                max_streak = max(max_streak, streak)
                if found:
                    fps_ += 1
        elif found:
            fps_ += 1

    lat = preds["latency_ms"].to_numpy(dtype=float)
    lat = lat[np.isfinite(lat)]
    n_vis = hits + misses
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")  # noqa: E731
    e3 = np.array(err3d).reshape(-1, 3)
    sd = e3.std(axis=0) if len(e3) >= 2 else np.full(3, np.nan)
    return {
        "n_labelled": len(labels),
        "n_visible": n_vis,
        "n_hits": hits,
        "recall": hits / n_vis if n_vis else float("nan"),
        "dropout": 1.0 - hits / n_vis if n_vis else float("nan"),
        "precision": hits / (hits + fps_) if (hits + fps_) else float("nan"),
        "false_pos": fps_,
        "center_err_px": med(center_err),
        "radius_err_pct": med(radius_err),
        "depth_err_pct": med(depth_err),
        "tape_depth_err_pct": med(tape_err),
        "max_miss_streak": max_streak,
        "latency_med_ms": med(lat),
        "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else float("nan"),
        "mean_dets_per_frame": float(preds["n_dets"].mean()),
        "sigma_x_m": float(sd[0]),
        "sigma_y_m": float(sd[1]),
        "sigma_z_m": float(sd[2]),
        "sigma_3d_m": float(np.sqrt(np.sum(sd**2))),
        "bias_z_m": float(e3[:, 2].mean()) if len(e3) else float("nan"),
        "_err3d": e3,  # per-hit error vectors, pooled across clips by summarize.py (not written to CSV)
    }


def _intr_for(clip: Clip, intr: Intrinsics | None) -> Intrinsics | None:
    if intr is None:
        return None
    w, h = clip.meta.get("width"), clip.meta.get("height")
    return intr.scaled_to(w, h) if w and h else intr


def clip_stats(clip: Clip, intr: Intrinsics | None = None, n_samples: int = 10) -> dict:
    """Properties of the clip itself (not of any method): how fast, how bright, how big.

    Ball speed comes from the hand labels. Between two consecutive labelled
    frames a, b (ball visible in both), the image displacement in pixels is
    converted to metres with the depth from the ball's apparent size,
    Z = fx * D / 2r (metres per pixel = Z / fx), and divided by the real time
    between the frames. This is the speed *across* the image only: motion
    toward or away from the camera is ignored, so toward-away clips read low.
    """
    import cv2

    all_labels = clip.labels()
    labels = {f: l for f, l in all_labels.items() if l.visible}
    frames = sorted(labels)
    intr = _intr_for(clip, intr)
    diameter = float(clip.meta.get("ball_diameter_m", TENNIS_BALL_DIAMETER_M))
    speeds, speeds_ms = [], []
    labelled = sorted(all_labels)
    for a, b in zip(labelled, labelled[1:]):
        if not (all_labels[a].visible and all_labels[b].visible):
            continue  # do not measure across a gap where the ball was hidden or out of frame
        la, lb = all_labels[a], all_labels[b]
        px = np.hypot(lb.cx - la.cx, lb.cy - la.cy)
        speeds.append(px / (b - a))
        if intr is not None and b < len(clip.timestamps):
            dt = float(clip.timestamps[b] - clip.timestamps[a])
            z = depth_from_radius((la.r + lb.r) / 2.0, intr.fx, diameter)
            if dt > 0 and np.isfinite(z):
                speeds_ms.append(px * z / intr.fx / dt)
    idx = np.linspace(0, len(clip) - 1, min(n_samples, len(clip))).astype(int)
    brightness = [cv2.cvtColor(clip.read(i), cv2.COLOR_BGR2HSV)[..., 2].mean() for i in idx]
    return {
        "gt_speed_px_per_frame": float(np.median(speeds)) if speeds else float("nan"),
        "gt_speed_m_s": float(np.median(speeds_ms)) if speeds_ms else float("nan"),
        "gt_speed_p90_m_s": float(np.percentile(speeds_ms, 90)) if speeds_ms else float("nan"),
        "gt_radius_px": float(np.median([labels[f].r for f in frames])) if frames else float("nan"),
        "brightness_0_255": float(np.mean(brightness)),
    }
