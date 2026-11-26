"""Live side-by-side view of the per-frame detectors on the webcam.

    python scripts/live_demo.py --methods color yolo26n
    python scripts/live_demo.py --methods color yolo26n --save data/demo.mp4   # for the portfolio

Shows each method's circle, its compute time, and (if calibrated) the distance
estimated from the ball's apparent size. This is for building intuition and
for demo footage; the numbers in the results table come from benchmark.py.
SAM 2 / EfficientTAM are not included: their official predictors work on
recorded clips, not live streams.
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import _paths  # noqa: E402
import cv2  # noqa: E402

from neuer_perception.camera import open_camera  # noqa: E402
from neuer_perception.geometry import Intrinsics, backproject, depth_from_radius  # noqa: E402
from neuer_perception.methods import FrameDetector, build  # noqa: E402

BGR = [(224, 125, 37), (48, 114, 244), (148, 184, 0), (0, 166, 242)]  # same order as the plots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=["color", "yolo26n"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--save", default=None, help="optional .mp4 path")
    ap.add_argument("--diameter", type=float, default=0.067, help="ball diameter, m")
    args = ap.parse_args()

    methods = [build(m) for m in args.methods]
    assert all(isinstance(m, FrameDetector) for m in methods), "live demo supports per-frame detectors only"
    cap = open_camera(args.camera)
    intr = Intrinsics.load(_paths.CALIB_FILE) if _paths.CALIB_FILE.exists() else None
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if intr and (intr.width, intr.height) != (frame.shape[1], frame.shape[0]):
            intr = intr.scaled_to(frame.shape[1], frame.shape[0])
        view = frame.copy()
        for k, m in enumerate(methods):
            t0 = time.perf_counter()
            dets = m.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            col = BGR[k % len(BGR)]
            line = f"{m.name}: {ms:5.1f} ms"
            if dets:
                d = dets[0]
                cv2.circle(view, (int(d.cx), int(d.cy)), int(d.r), col, 2)
                if intr:
                    z = depth_from_radius(d.r, intr.fx, args.diameter)
                    x, y, _ = backproject(d.cx, d.cy, z, intr)
                    line += f"   ball at x={x:+.2f} y={y:+.2f} z={z:.2f} m"
            else:
                line += "   no ball"
            cv2.putText(view, line, (15, 35 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if args.save:
            if writer is None:
                h, w = view.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer.write(view)
        cv2.imshow("live", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
