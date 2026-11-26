"""Click ground truth for a clip: where is the ball, and how big is it?

    python scripts/label.py data/clips/dist_2m --every 5

For each shown frame:
    left click #1   ball centre
    left click #2   any point on the ball's edge  (sets the radius)
    n               ball not visible in this frame
    o               toggle "partly occluded" for this frame
    c               copy the previous frame's label (fast for static clips)
    ENTER           accept the suggestion (only with --assist)
    d / SPACE       next frame        a   previous frame
    r               reset this frame's clicks
    s               save              q   save and quit

A 6x magnifier around the cursor (top-left) helps with far, tiny balls.

--assist pre-fills each frame with the colour detector's guess, which you
accept with ENTER or correct by clicking. It is much faster, but check every
frame: unchecked suggestions would bias the benchmark toward the colour method.
"""

import argparse

import _paths  # noqa: F401
import cv2
import numpy as np

from neuer_perception.clip import Clip, Label

ZOOM, HALF = 6, 30  # magnifier: 60x60 px region shown 6x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--every", type=int, default=5, help="label every k-th frame")
    ap.add_argument("--assist", action="store_true", help="pre-fill with the colour detector")
    args = ap.parse_args()

    clip = Clip(args.clip)
    labels = clip.labels()
    todo = list(range(0, len(clip), args.every))
    detector = None
    if args.assist:
        from neuer_perception.methods.color import ColorDetector

        detector = ColorDetector.from_yaml()

    state = {"clicks": [], "mouse": (0, 0)}

    def on_mouse(event, x, y, flags, param):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicks"] = (state["clicks"] + [(x, y)])[-2:] if len(state["clicks"]) < 2 else [(x, y)]

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_mouse)
    k = 0
    suggestion = None
    while 0 <= k < len(todo):
        f = todo[k]
        img = clip.read(f)
        if detector is not None and f not in labels:
            dets = detector.detect(img)
            suggestion = dets[0] if dets else None
        while True:
            view = img.copy()
            lab = labels.get(f)
            if len(state["clicks"]) == 2:
                (x0, y0), (x1, y1) = state["clicks"]
                lab = Label(True, float(x0), float(y0), float(np.hypot(x1 - x0, y1 - y0)), lab.occluded if lab else False)
                labels[f] = lab
            if suggestion is not None and f not in labels:
                cv2.circle(view, (int(suggestion.cx), int(suggestion.cy)), int(suggestion.r), (255, 0, 255), 1)
            if state["clicks"]:
                cv2.drawMarker(view, state["clicks"][0], (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            if lab is not None and lab.visible:
                cv2.circle(view, (int(lab.cx), int(lab.cy)), max(int(round(lab.r)), 1), (0, 255, 0), 1)
            text = f"frame {f}/{len(clip) - 1}  [{k + 1}/{len(todo)}]  "
            text += "NOT VISIBLE" if (lab and not lab.visible) else ("labelled" if lab else "unlabelled")
            text += "  OCCLUDED" if (lab and lab.occluded) else ""
            cv2.putText(view, text, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # magnifier
            mx, my = state["mouse"]
            pad = cv2.copyMakeBorder(view, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT)
            crop = pad[my : my + 2 * HALF, mx : mx + 2 * HALF]
            if crop.shape[:2] == (2 * HALF, 2 * HALF):
                zoom = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
                c = HALF * ZOOM
                cv2.drawMarker(zoom, (c, c), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
                zh, zw = zoom.shape[:2]
                x_off = view.shape[1] - zw if mx < view.shape[1] // 2 else 0  # keep it away from the cursor
                view[:zh, x_off : x_off + zw] = zoom
            cv2.imshow("label", view)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("d"), ord(" ")):
                k += 1
                break
            if key == ord("a"):
                k -= 1
                break
            if key == ord("n"):
                labels[f] = Label(False)
                k += 1
                break
            if key == ord("o") and f in labels:
                labels[f].occluded = not labels[f].occluded
            if key == ord("c") and k > 0 and todo[k - 1] in labels:
                prev = labels[todo[k - 1]]
                labels[f] = Label(prev.visible, prev.cx, prev.cy, prev.r, prev.occluded)
                k += 1
                break
            if key == 13 and suggestion is not None:  # ENTER
                labels[f] = Label(True, suggestion.cx, suggestion.cy, suggestion.r)
                k += 1
                break
            if key == ord("r"):
                labels.pop(f, None)
                state["clicks"] = []
            if key == ord("s"):
                clip.save_labels(labels)
                print("saved")
            if key == ord("q"):
                k = -1
                break
        state["clicks"] = []
        suggestion = None

    clip.save_labels(labels)
    n_vis = sum(l.visible for l in labels.values())
    print(f"saved {len(labels)} labels ({n_vis} visible) to {clip.labels_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
