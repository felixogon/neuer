"""Record a test clip from the webcam, tagged with its test conditions.

    python scripts/record.py --name dist_2m --test distance --distance 2.0 --motion static
    python scripts/record.py --name motion_lr_fast --test motion --pattern left-right --motion fast --distance 1.5

Hand-held patterns (--pattern), with the ball at roughly chest height:
    still          hold the ball as still as you can
    left-right     sweep across the image and back
    up-down        sweep vertically
    circles        draw circles about 50 cm across
    toward-away    walk from 3 m to 0.5 m and back to 3 m (--distance = start)
    shake          shake the ball in place (short, fast, blurry)
    out-and-back   move out of the frame, pause, come back
    behind-object  move behind a chair/box and back out

Keys in the preview window:
    SPACE  start / stop recording (also stops automatically after --seconds)
    q      quit

Frames are saved as JPEGs from a background thread so that writing to disk
does not slow the capture loop down (dropped frames would distort the timing).
"""

import argparse
import json
import queue
import threading
import time
from datetime import datetime

import _paths  # noqa: F401
import cv2

from neuer_perception.camera import describe, open_camera

PATTERNS = ["still", "left-right", "up-down", "circles", "toward-away", "shake", "out-and-back", "behind-object"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="clip folder name, e.g. dist_2m")
    ap.add_argument("--test", required=True,
                    choices=["distance", "motion", "speed", "lighting", "occlusion", "clutter", "multi", "other"])
    ap.add_argument("--pattern", default="still", choices=PATTERNS, help="how the ball is moved by hand")
    ap.add_argument("--distance", type=float, default=None, help="camera-to-ball distance in metres (tape measure)")
    ap.add_argument("--motion", default=None, choices=["static", "slow", "medium", "fast"],
                    help="speed of the movement (default: static for --pattern still, else slow)")
    ap.add_argument("--lighting", default="normal", choices=["bright", "normal", "dim", "dark"])
    ap.add_argument("--occlusion", default="none", help="none / 25 / 50 / 75 / passes-behind")
    ap.add_argument("--clutter", default="none", help="none / describe distractors, e.g. 'yellow highlighter'")
    ap.add_argument("--n-balls", type=int, default=1)
    ap.add_argument("--ball", default="tennis", help="tennis / soccer / ...")
    ap.add_argument("--ball-diameter", type=float, default=0.067, help="metres")
    ap.add_argument("--notes", default="")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds between SPACE and recording start")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    if args.motion is None:
        args.motion = "static" if args.pattern == "still" else "slow"

    out = _paths.CLIPS_DIR / args.name
    if out.exists():
        raise SystemExit(f"{out} already exists - pick another --name or delete it.")
    cap = open_camera(args.camera, args.width, args.height)
    cam = describe(cap)
    print("camera:", cam)

    q: queue.Queue = queue.Queue()

    def writer():
        while True:
            item = q.get()
            if item is None:
                return
            path, img = item
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    recording, frames, stamps, t_start = False, 0, [], 0.0
    th, countdown_end = None, None
    while True:
        ok, frame = cap.read()
        t = time.perf_counter()
        if not ok:
            print("camera read failed")
            break
        if countdown_end is not None and t >= countdown_end:
            (out / "frames").mkdir(parents=True)
            th = threading.Thread(target=writer, daemon=True)
            th.start()
            recording, t_start, countdown_end = True, t, None
        if recording:
            q.put((out / "frames" / f"{frames:05d}.jpg", frame))
            stamps.append(t - t_start)
            frames += 1
            if stamps[-1] >= args.seconds:
                recording = False
        preview = frame.copy()
        if recording:
            status = f"REC {stamps[-1]:.1f}s"
        elif countdown_end is not None:
            status = f"starting in {countdown_end - t:.1f}s"
        else:
            status = "SPACE to record, q to quit"
        cv2.putText(preview, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.imshow("record", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not recording and frames == 0 and countdown_end is None:
                countdown_end = time.perf_counter() + args.countdown
            elif recording:
                recording = False
        elif key == ord("q"):
            break
        if frames and not recording and th is not None:
            break  # done after one take

    cap.release()
    cv2.destroyAllWindows()
    if not frames:
        print("nothing recorded")
        return
    q.put(None)
    th.join()

    with open(out / "timestamps.csv", "w") as f:
        f.write("frame,t_s\n")
        for i, s in enumerate(stamps):
            f.write(f"{i},{s:.6f}\n")
    duration = stamps[-1] if len(stamps) > 1 else 0
    meta = {
        "test": args.test,
        "distance_m": args.distance,
        "pattern": args.pattern,
        "motion": args.motion,
        "lighting": args.lighting,
        "occlusion": args.occlusion,
        "clutter": args.clutter,
        "n_balls": args.n_balls,
        "ball": args.ball,
        "ball_diameter_m": args.ball_diameter,
        "notes": args.notes,
        "n_frames": frames,
        "fps": round((frames - 1) / duration, 2) if duration else None,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "camera": cam,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {frames} frames ({meta['fps']} fps measured) to {out}")
    print(f"next: python scripts/label.py {out}")


if __name__ == "__main__":
    main()
