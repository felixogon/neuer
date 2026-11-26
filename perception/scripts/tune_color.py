"""Tune the colour baseline's HSV thresholds live, with sliders.

    python scripts/tune_color.py                     # webcam
    python scripts/tune_color.py --clip data/clips/dist_2m   # a recorded clip

Left: camera + detected circle. Right: the binary mask (white = "ball colour").
Goal: the ball is a solid white disc, everything else black. Start with the
S and V minimums: raise them until the background disappears.
Press s to save to configs/color_tennis.yaml, q to quit.

Tune in normal light, then *do not retune per test* - the benchmark should show
how a fixed threshold copes when the lighting changes.
"""

import argparse

import _paths  # noqa: F401
import cv2
import numpy as np
import yaml

from neuer_perception.camera import open_camera
from neuer_perception.clip import Clip
from neuer_perception.methods.color import DEFAULT_CONFIG, ColorDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    det = ColorDetector.from_yaml()
    names = ["H min", "S min", "V min", "H max", "S max", "V max"]
    init = list(det.hsv_lo) + list(det.hsv_hi)
    cv2.namedWindow("tune", cv2.WINDOW_NORMAL)
    for n, v in zip(names, init):
        cv2.createTrackbar(n, "tune", int(v), 179 if n.startswith("H") else 255, lambda _: None)

    cap = None if args.clip else open_camera(args.camera)
    clip = Clip(args.clip) if args.clip else None
    i = 0
    while True:
        if clip:
            frame = clip.read(i % len(clip))
            i += 1
        else:
            ok, frame = cap.read()
            if not ok:
                break
        vals = [cv2.getTrackbarPos(n, "tune") for n in names]
        det.hsv_lo, det.hsv_hi = np.array(vals[:3], np.uint8), np.array(vals[3:], np.uint8)
        mask = det.mask(frame)
        view = frame.copy()
        for k, d in enumerate(det.detect(frame)[:3]):
            cv2.circle(view, (int(d.cx), int(d.cy)), int(d.r), (0, 255, 0) if k == 0 else (0, 165, 255), 2)
        both = np.hstack([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("tune", cv2.resize(both, None, fx=0.6, fy=0.6))
        key = cv2.waitKey(30 if clip else 1) & 0xFF
        if key == ord("s"):
            cfg = yaml.safe_load(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.exists() else {}
            cfg["hsv_lo"], cfg["hsv_hi"] = vals[:3], vals[3:]
            DEFAULT_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f"saved {vals} to {DEFAULT_CONFIG}")
        elif key == ord("q"):
            break
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
