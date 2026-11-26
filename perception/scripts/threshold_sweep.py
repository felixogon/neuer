"""Threshold-sensitivity analysis for YOLO: precision and recall over confidence thresholds.

    python scripts/benchmark.py --methods yolo26n yolo26n@1280 --yolo-conf 0.01 --out results/threshold_sweep/raw
    python scripts/threshold_sweep.py

The benchmark runs YOLO at its preset threshold (0.15). This script re-scores a
run made at a very low threshold (0.01) as if the threshold had been t, for a
range of t: a frame counts as "ball found" only if the best detection's score
is >= t. Scoring is identical to summarize.py (same hit rule, same labels,
pooled over all clips, dark clips included so false positives count).

This is a sensitivity analysis, not a tuning step: the thresholds are chosen
by looking at the test clips, so the best point here is optimistic. The
main results keep the preset 0.15.

Writes results/threshold_sweep/sweep.csv and results/threshold_sweep/pr_curve.png.
"""

import argparse
from pathlib import Path

import _paths
import numpy as np
import pandas as pd

from neuer_perception.clip import Clip
from neuer_perception.geometry import Intrinsics
from neuer_perception.metrics import score_clip

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#8f5fd6", "#7a7a7a"]
PRESET = 0.15
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(_paths.RESULTS_DIR / "threshold_sweep" / "raw"))
    ap.add_argument("--calib", default=str(_paths.CALIB_FILE))
    args = ap.parse_args()
    raw = Path(args.raw)
    out = raw.parent
    intr = Intrinsics.load(args.calib) if Path(args.calib).exists() else None

    # Same colour per method as the main plots (order of results/per_clip.csv).
    main_csv = _paths.RESULTS_DIR / "per_clip.csv"
    order = list(dict.fromkeys(pd.read_csv(main_csv)["method"])) if main_csv.exists() else []

    rows = []
    for clip_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        clip = Clip(_paths.CLIPS_DIR / clip_dir.name)
        if not clip.labels():
            continue
        for csv in sorted(clip_dir.glob("*.csv")):
            preds = pd.read_csv(csv)
            for t in THRESHOLDS:
                p = preds.copy()
                p["found"] = p["found"].astype(bool) & (p["score"] >= t)
                s = score_clip(clip, p, intr)
                rows.append({
                    "method": csv.stem, "threshold": t, "clip": clip.name,
                    "hits": s["n_hits"], "n_visible": s["n_visible"], "false_pos": s["false_pos"],
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "threshold"], sort=False)[["hits", "n_visible", "false_pos"]].sum().reset_index()
    agg["recall"] = agg.hits / agg.n_visible
    agg["precision"] = agg.hits / (agg.hits + agg.false_pos)
    agg.to_csv(out / "sweep.csv", index=False)
    per_clip = df[df.threshold == min(THRESHOLDS)].copy()
    per_clip["recall"] = per_clip.hits / per_clip.n_visible
    tab = agg.assign(recall=lambda d: (100 * d.recall).round(1), precision=lambda d: (100 * d.precision).round(1))
    tab = tab.rename(columns={"recall": "recall %", "precision": "precision %", "false_pos": "false pos."})
    fp = per_clip[per_clip.false_pos > 0][["method", "clip", "false_pos"]]
    (out / "README.md").write_text(
        "# YOLO threshold-sensitivity analysis\n\n"
        f"**Not the chosen setting.** The main results use the preset confidence threshold {PRESET:g}. "
        f"Here YOLO was re-run at {min(THRESHOLDS):g} (`raw/`) and re-scored as if the threshold were t, "
        "pooled over all clips with the same hit rule as `../RESULTS.md`. "
        "The thresholds are judged on the test clips themselves, so the best point is optimistic.\n\n"
        "Made by `scripts/benchmark.py --methods yolo26n yolo26n@1280 --yolo-conf 0.01 --out results/threshold_sweep/raw` "
        "and `scripts/threshold_sweep.py`.\n\n![precision-recall](pr_curve.png)\n\n"
        f"{tab[['method', 'threshold', 'hits', 'n_visible', 'false pos.', 'recall %', 'precision %']].to_markdown(index=False)}\n\n"
        f"False positives at {min(THRESHOLDS):g}, by clip:\n\n{fp.to_markdown(index=False)}\n"
    )
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for k, (m, g) in enumerate(agg.groupby("method", sort=False)):
        col = SERIES[order.index(m) % len(SERIES)] if m in order else SERIES[-1]
        g = g.sort_values("threshold")
        ax.plot(100 * g.recall, 100 * g.precision, "-o", lw=2, ms=5, color=col, label=m)
        dy = 9 if k % 2 == 0 else -14  # one series labelled above, the other below
        for t in (min(THRESHOLDS), PRESET, max(THRESHOLDS)):  # a few threshold labels, not every point
            r = g[g.threshold == t].iloc[0]
            ax.annotate(f"{t:g}", (100 * r.recall, 100 * r.precision), textcoords="offset points",
                        xytext=(-6, dy), fontsize=8, color="#555555")
        pre = g[g.threshold == PRESET].iloc[0]
        ax.plot(100 * pre.recall, 100 * pre.precision, "o", ms=11, mfc="none", mec="#222222", mew=1.5)
    ax.plot([], [], "o", ms=11, mfc="none", mec="#222222", mew=1.5, label=f"preset used in results (conf {PRESET:g})")
    if main_csv.exists():  # reference: the colour baseline's pooled recall at its own (fixed) setting
        pc = pd.read_csv(main_csv)
        c = pc[pc["method"] == "opencv-color"]
        if len(c):
            rc = 100 * c.n_hits.sum() / c.n_visible.sum()
            ax.axvline(rc, color="#9c9c9c", lw=1, ls="--")
            ax.text(rc - 1.2, 90, f"opencv-color recall ({rc:.0f} %)", rotation=90, ha="right", va="center", fontsize=8, color="#555555")
    ax.set_xlabel("Recall (%)  – share of visible-ball frames found")
    ax.set_ylabel("Precision (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(80, 101)  # precision stays high; zoomed so the curves are readable
    ax.grid(color="#e6e6e6", lw=0.8)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Threshold-sensitivity analysis (not the chosen setting):\n"
                 "YOLO precision vs recall as the confidence threshold varies", loc="left", fontsize=10)
    fig.text(0.01, 0.01, "Numbers next to points = confidence threshold. Pooled over all 21 clips (dark clips count for false positives).\n"
             "Precision axis starts at 80 %. Thresholds picked on the test data, so the best point here is optimistic.", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "pr_curve.png", dpi=150)
    print(f"wrote {out / 'pr_curve.png'}")


if __name__ == "__main__":
    main()
