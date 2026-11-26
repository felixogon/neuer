"""Measure the camera's intrinsics (focal length, principal point, lens distortion).

    python scripts/calibrate.py --square-mm 24.0                 # live, from the webcam
    python scripts/calibrate.py --square-mm 24.0 --from-dir data/calib   # re-solve saved views

How it works (Zhang's method, what OpenCV implements): the checkerboard's
corners are points whose 3D positions we know exactly (a flat grid, spacing =
square size). Seeing that grid from many angles gives enough equations to
solve for the camera matrix K and the distortion coefficients, by minimising
the *reprojection error*: the pixel distance between where corners were
detected and where the fitted model says they should be. RMS < 0.5 px is good.

Tips for good views (~15-25): tilt the board in all directions (up to ~45°),
cover the corners and edges of the image, vary the distance, hold it still
(motion blur ruins corner accuracy), and keep the whole board in view.

Keys: SPACE capture (only when corners are found, shown in colour)
      c     calibrate with the views so far and save
      q     quit
"""

import argparse
import time
from pathlib import Path

import _paths
import cv2
import numpy as np

from neuer_perception.camera import open_camera
from neuer_perception.geometry import Intrinsics

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def find_corners(gray, pattern, fast=False):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags += cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ok and not fast:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)  # refine to sub-pixel
    return ok, corners


def solve(images, pattern, square_m, out_path):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    obj_pts, img_pts = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, pattern)
        if ok:
            obj_pts.append(objp)
            img_pts.append(corners)
    h, w = images[0].shape[:2]
    print(f"solving with {len(obj_pts)} views at {w}x{h} ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    per_view = []
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)  # shapes differ across OpenCV versions
        per_view.append(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))))
    intr = Intrinsics(K, dist.ravel(), w, h, rms)
    intr.save(out_path)
    hfov = 2 * np.degrees(np.arctan(w / (2 * intr.fx)))
    print(f"RMS reprojection error: {rms:.3f} px  (worst view {max(per_view):.2f} px)")
    print(f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}  ->  horizontal FOV {hfov:.1f}°")
    print(f"distortion k1,k2,p1,p2,k3 = {np.round(dist.ravel(), 4).tolist()}")
    print(f"saved {out_path}")
    if max(per_view) > 3 * rms:
        print("tip: some views are much worse than average; delete blurry ones from data/calib and re-run --from-dir")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-mm", type=float, required=True, help="measured side of one square, mm")
    ap.add_argument("--from-dir", default=None, help="solve from saved images instead of the camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=str(_paths.CALIB_FILE))
    args = ap.parse_args()
    pattern = (args.cols, args.rows)
    square_m = args.square_mm / 1000.0
    save_dir = _paths.DATA_DIR / "calib"

    if args.from_dir:
        images = [cv2.imread(str(p)) for p in sorted(Path(args.from_dir).glob("*.png"))]
        solve(images, pattern, square_m, args.out)
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height)
    images = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern, fast=True)
        view = frame.copy()
        cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"views: {len(images)}  (SPACE capture, c calibrate, q quit)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            images.append(frame.copy())
            cv2.imwrite(str(save_dir / f"view_{int(time.time() * 1000)}.png"), frame)
        elif key == ord("c") and len(images) >= 8:
            solve(images, pattern, square_m, args.out)
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
