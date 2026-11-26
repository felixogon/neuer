"""Run methods over recorded clips and save every per-frame prediction.

    python scripts/benchmark.py --methods color yolo26n yolo26n@1280 sam2.1-tiny efficienttam-ti-512
    python scripts/benchmark.py --clips data/clips/dist_* --methods color
    python scripts/benchmark.py --methods yolo26n yolo26n@1280 --yolo-conf 0.01 --out results/threshold_sweep/raw

Writes results/raw/<clip>/<method>.csv (one row per frame). Scoring happens in
scripts/summarize.py, so you can re-score without re-running slow models.
Each model is loaded once and then run over all clips.
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back to CPU on Mac

import _paths  # noqa: E402
import pandas as pd  # noqa: E402

from neuer_perception.clip import Clip  # noqa: E402
from neuer_perception.methods import build  # noqa: E402

DEFAULT_METHODS = ["color", "yolo26n", "yolo26n@1280", "sam2.1-tiny", "efficienttam-ti-512"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", help="clip folders (default: all in data/clips)")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default: best available)")
    ap.add_argument("--out", default=str(_paths.RESULTS_DIR / "raw"))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--yolo-conf", type=float, default=None,
        help="override YOLO's confidence threshold (default 0.15); use with a separate --out, e.g. for a threshold sweep",
    )
    args = ap.parse_args()

    clip_dirs = args.clips or sorted(str(p) for p in _paths.CLIPS_DIR.iterdir() if (p / "frames").is_dir())
    clips = [Clip(c) for c in clip_dirs]
    print(f"{len(clips)} clips, methods: {args.methods}")

    for mname in args.methods:
        print(f"\n== {mname}: loading")
        try:
            method = build(mname, device=args.device, yolo_conf=args.yolo_conf)
        except Exception as e:  # keep going if one model is not installed
            print(f"   skipped ({type(e).__name__}: {e})")
            continue
        for clip in clips:
            out = os.path.join(args.out, clip.name, f"{method.name}.csv")
            if args.skip_existing and os.path.exists(out):
                continue
            t0 = time.time()
            rows = []
            try:
                for r in method.run(clip):
                    d = r.det
                    rows.append(
                        {
                            "frame": r.frame,
                            "t_s": float(clip.timestamps[r.frame]),
                            "found": d is not None,
                            "cx": d.cx if d else float("nan"),
                            "cy": d.cy if d else float("nan"),
                            "r": d.r if d else float("nan"),
                            "score": d.score if d else float("nan"),
                            "n_dets": r.n_dets,
                            "latency_ms": r.latency_ms,
                        }
                    )
            except Exception as e:
                if type(e).__name__ == "NoPrompt":  # e.g. a dark clip where the ball is never visible
                    print(f"   {clip.name:<28} n/a ({e})")
                else:
                    print(f"   {clip.name}: FAILED ({type(e).__name__}: {e})")
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_csv(out, index=False)
            print(
                f"   {clip.name:<28} {len(df):4d} frames  found {df.found.mean():5.1%}  "
                f"median {df.latency_ms.median():7.1f} ms/frame  ({time.time() - t0:.0f}s)"
            )
        del method
    print(f"\nnext: python scripts/summarize.py")


if __name__ == "__main__":
    main()
