"""Render a clip with every method's prediction drawn on it (for review and demos).

    python scripts/render.py data/clips/speed_fast_1
    python scripts/render.py data/clips/speed_fast_1 --slowmo 4 --gif
    python scripts/render.py data/clips/shake --gif --crop

The GIF is meant for docs and the README: it is cropped to the region where the
ball and the predictions are (--crop), scaled to --gif-width and subsampled
to --gif-fps, which keeps it around 1-2 MB. The MP4 keeps the full frame.

White dashed circle = ground truth (on labelled frames). Coloured circles =
methods, same colours as the plots. Watching these is the fastest way to
understand *why* a method fails (blur? wrong object? lost track?).
"""

import argparse
from pathlib import Path

import _paths
import cv2
import numpy as np
import pandas as pd

from neuer_perception.clip import Clip

BGR = [(224, 125, 37), (48, 114, 244), (148, 184, 0), (0, 166, 242), (172, 127, 238), (23, 138, 0), (214, 95, 143)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--raw", default=str(_paths.RESULTS_DIR / "raw"))
    ap.add_argument("--slowmo", type=float, default=1.0, help="playback slow-down factor")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--gif", action="store_true", help="also write a GIF (needs Pillow)")
    ap.add_argument("--gif-width", type=int, default=480, help="GIF width in pixels")
    ap.add_argument("--gif-fps", type=float, default=15, help="GIF frame rate (frames are subsampled)")
    ap.add_argument("--gif-colors", type=int, default=48, help="GIF palette size")
    ap.add_argument("--gif-hold", type=int, default=16, help="ignore pixel changes below this (0-255) to keep the GIF small")
    ap.add_argument("--crop", action="store_true", help="crop the GIF to the ball / prediction region")
    ap.add_argument("--margin", type=float, default=0.12, help="crop margin, fraction of the frame width")
    args = ap.parse_args()

    clip = Clip(args.clip)
    labels = clip.labels()
    preds = {p.stem: pd.read_csv(p).set_index("frame") for p in sorted((Path(args.raw) / clip.name).glob("*.csv"))}
    out_dir = _paths.RESULTS_DIR / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{clip.name}.mp4"
    fps = float(clip.meta.get("fps") or 30) / args.slowmo
    writer, frames_rgb = None, []
    crop = crop_box(clip, labels, preds, args.margin) if args.crop else None
    src_fps = float(clip.meta.get("fps") or 30)
    gif_step = max(1, round(src_fps / args.slowmo / args.gif_fps))
    for i in range(len(clip)):
        img = clip.read(i)
        lab = labels.get(i)
        if lab and lab.visible:
            for a in range(0, 360, 30):  # dashed circle
                cv2.ellipse(img, (int(lab.cx), int(lab.cy)), (int(lab.r) + 4,) * 2, 0, a, a + 15, (255, 255, 255), 2)
        for k, (name, df) in enumerate(preds.items()):
            col = BGR[k % len(BGR)]
            if i in df.index and bool(df.loc[i, "found"]):
                cx, cy, r = df.loc[i, ["cx", "cy", "r"]]
                cv2.circle(img, (int(cx), int(cy)), max(int(r), 2), col, 2)
        if args.gif and i % gif_step == 0:
            frames_rgb.append(gif_frame(img, crop, args.gif_width, list(preds), i))
        for k, name in enumerate(preds):
            cv2.putText(img, name, (15, 35 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BGR[k % len(BGR)], 2)
        cv2.putText(img, f"frame {i}", (15, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    print(f"wrote {out}")
    if args.gif:
        from PIL import Image

        gif = out.with_suffix(".gif")
        # One shared palette: method colours + white/black reserved, the rest fitted to the footage.
        fixed = [c[::-1] for c in BGR] + [(255, 255, 255), (0, 0, 0)]
        sample = np.concatenate(frames_rgb[:: max(1, len(frames_rgb) // 8)], axis=0)
        adaptive = Image.fromarray(sample).quantize(colors=args.gif_colors - len(fixed), method=Image.Quantize.MEDIANCUT)
        pal = [v for c in fixed for v in c] + adaptive.getpalette()[: 3 * (args.gif_colors - len(fixed))]
        pal_img = Image.new("P", (1, 1))
        pal_img.putpalette(pal + [0] * (768 - len(pal)))
        # Webcam noise changes every pixel every frame, which GIF compresses badly. Hold a palette index
        # unless the pixel really changed, so unchanged background stays identical between frames.
        idx = [np.asarray(Image.fromarray(f).quantize(palette=pal_img, dither=Image.Dither.NONE)) for f in frames_rgb]
        held, prev_rgb = [idx[0]], frames_rgb[0].astype(np.int16)
        for f, q in zip(frames_rgb[1:], idx[1:]):
            changed = np.abs(f.astype(np.int16) - prev_rgb).max(axis=2) > args.gif_hold
            out_q = np.where(changed, q, held[-1])
            prev_rgb = np.where(changed[..., None], f, prev_rgb)
            held.append(out_q)
        ims = []
        for q in held:
            im = Image.fromarray(q.astype(np.uint8), mode="P")
            im.putpalette(pal_img.getpalette())
            ims.append(im)
        ms = round(1000 * gif_step * args.slowmo / src_fps)
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True, disposal=1)
        print(f"wrote {gif}  ({len(ims)} frames, {1000 / ms:.1f} fps, {gif.stat().st_size / 1e6:.2f} MB)")


def crop_box(clip, labels, preds, margin):
    """Bounding box (x0, y0, x1, y1) of every labelled ball and every prediction, plus a margin."""
    xs, ys = [], []
    for lab in labels.values():
        if lab.visible:
            xs += [lab.cx - lab.r, lab.cx + lab.r]
            ys += [lab.cy - lab.r, lab.cy + lab.r]
    for df in preds.values():
        f = df[df["found"].astype(bool)]
        xs += list(f.cx - f.r) + list(f.cx + f.r)
        ys += list(f.cy - f.r) + list(f.cy + f.r)
    W, H = int(clip.meta.get("width") or 1280), int(clip.meta.get("height") or 720)
    if not xs:
        return None
    m = margin * W
    x0, x1 = max(0, int(min(xs) - m)), min(W, int(max(xs) + m))
    y0, y1 = max(0, int(min(ys) - m)), min(H, int(max(ys) + m))
    # keep at least a 4:3-ish box so the legend fits
    if (x1 - x0) < 1.2 * (y1 - y0):
        cx, half = (x0 + x1) / 2, 0.6 * (y1 - y0)
        x0, x1 = max(0, int(cx - half)), min(W, int(cx + half))
    return x0, y0, x1, y1


def gif_frame(img, crop, width, names, i):
    """Crop, resize to `width`, then draw a small legend (so it is never cropped away)."""
    if crop:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    for k, name in enumerate(names):
        y = 14 + 14 * k
        tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.rectangle(img, (3, y - 10), (9 + tw, y + 3), (0, 0, 0), -1)
        cv2.putText(img, name, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BGR[k % len(BGR)], 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {i}", (6, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
