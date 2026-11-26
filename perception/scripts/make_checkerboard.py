"""Make a checkerboard image for camera calibration.

    python scripts/make_checkerboard.py            # 10x7 squares = 9x6 inner corners

Either print data/checkerboard.png at 100 % scale and tape it to something
flat and rigid (cardboard, a clipboard), or show it full-screen on an iPad.
Then MEASURE one square with a ruler (in mm) and pass it to calibrate.py:
the measured size is what gives calibration its real-world scale.
"""

import argparse

import _paths
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=10, help="squares across")
    ap.add_argument("--rows", type=int, default=7, help="squares down")
    ap.add_argument("--px", type=int, default=150, help="pixels per square")
    args = ap.parse_args()
    margin = args.px // 2
    img = np.full((args.rows * args.px + 2 * margin, args.cols * args.px + 2 * margin), 255, np.uint8)
    for r in range(args.rows):
        for c in range(args.cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * args.px, margin + c * args.px
                img[y : y + args.px, x : x + args.px] = 0
    out = _paths.DATA_DIR / "checkerboard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}: {args.cols - 1}x{args.rows - 1} inner corners")


if __name__ == "__main__":
    main()
