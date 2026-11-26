"""Score all predictions against the labels and write the results table.

    python scripts/summarize.py

Reads  results/raw/<clip>/<method>.csv  + each clip's labels.json / meta.json
Writes results/per_clip.csv              every metric, every clip x method
       results/RESULTS.md                the tables to show Blake / put in the README
       results/recall_vs_distance.png
       results/latency.png
"""

import argparse
import platform
import subprocess
from datetime import date
from pathlib import Path

import _paths
import numpy as np
import pandas as pd

from neuer_perception.clip import Clip
from neuer_perception.geometry import Intrinsics
from neuer_perception.metrics import clip_stats, score_clip

# Fixed colour per method (colour follows the method, whatever subset is plotted).
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#8f5fd6", "#7a7a7a"]
RELIABLE_RECALL = 0.9


PATTERNS = ["still", "left-right", "up-down", "circles", "toward-away", "shake", "out-and-back", "behind-object"]


def pattern_of(meta: dict) -> str:
    """Hand-held movement pattern; older clips without one are 'still' if static."""
    if meta.get("pattern"):
        return meta["pattern"]
    return "still" if meta.get("motion") == "static" else "–"


def speed_txt(meta: dict, stats: dict) -> str:
    """Measured lateral ball speed; blank for still clips (that is only hand tremor)."""
    if pattern_of(meta) == "still" or pd.isna(stats.get("gt_speed_m_s")):
        return ""
    return f"~{stats['gt_speed_m_s']:.2f} m/s (p90 {stats['gt_speed_p90_m_s']:.2f})"


def condition(meta: dict, stats: dict) -> str:
    t = meta.get("test")
    pat = pattern_of(meta)
    d = f"{meta['distance_m']:g} m" if meta.get("distance_m") else ""
    v = speed_txt(meta, stats)
    parts: list[str]
    if t == "distance":
        parts = [d, pat]
    elif t in ("motion", "speed"):
        parts = [f"{pat} {meta.get('motion')}", v, d]
    elif t == "lighting":
        parts = [f"{meta.get('lighting')} (V={stats['brightness_0_255']:.0f})", pat, v, d]
    elif t == "occlusion":
        occ = meta.get("occlusion")
        parts = [pat if pat != "still" else f"still, occlusion {occ}", v, d]
    elif t == "clutter":
        parts = [f"{pat} in front of {meta.get('clutter')}", v, d]
    elif t == "multi":
        parts = [f"{meta.get('n_balls')} balls, {pat}", v, d]
    else:
        parts = [meta.get("notes") or "", d]
    return ", ".join(p for p in parts if p)


def machine() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def fmt_pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.0f}%"


def fmt_num(x, digits=1):
    return "–" if pd.isna(x) else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(_paths.RESULTS_DIR / "raw"))
    ap.add_argument("--calib", default=str(_paths.CALIB_FILE))
    ap.add_argument("--out", default=str(_paths.RESULTS_DIR))
    args = ap.parse_args()
    out = Path(args.out)

    intr = Intrinsics.load(args.calib) if Path(args.calib).exists() else None
    if intr is None:
        print(f"(no calibration at {args.calib}: depth errors will be blank - run scripts/calibrate.py)")

    rows, errs = [], {}  # errs[(clip, method)] = per-hit 3D error vectors (metres)
    for clip_dir in sorted(Path(args.raw).iterdir()):
        src = _paths.CLIPS_DIR / clip_dir.name
        if not (src / "frames").is_dir():
            print(f"skip {clip_dir.name}: clip folder not found")
            continue
        clip = Clip(src)
        if not clip.labels():
            print(f"skip {clip.name}: no labels yet (python scripts/label.py {src})")
            continue
        stats = clip_stats(clip, intr)
        for csv in sorted(clip_dir.glob("*.csv")):
            preds = pd.read_csv(csv)
            m = score_clip(clip, preds, intr)
            errs[(clip.name, csv.stem)] = m.pop("_err3d")
            rows.append(
                {
                    "clip": clip.name,
                    "method": csv.stem,
                    "test": clip.meta.get("test", "other"),
                    "pattern": pattern_of(clip.meta),
                    "motion": clip.meta.get("motion"),
                    "lighting": clip.meta.get("lighting", "normal"),
                    "condition": condition(clip.meta, stats),
                    "distance_m": clip.meta.get("distance_m"),
                    **stats,
                    **m,
                }
            )
    if not rows:
        raise SystemExit("nothing to summarize - run benchmark.py and label your clips first")
    df = pd.DataFrame(rows)
    # Clips where the ball is never visible (e.g. lights off: the webcam itself cannot see it) have no
    # recall; precision there would be 0 % from false positives alone. Score them by false positives only.
    df["no_ball"] = df.n_visible == 0
    df.loc[df.no_ball, "precision"] = np.nan
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_clip.csv", index=False)

    methods = list(dict.fromkeys(df["method"]))
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)}

    # ---- headline table -------------------------------------------------------
    def max_reliable_distance(g):
        """Farthest tested distance such that every distance up to it has recall >= 90 %.

        Requiring all nearer distances to pass as well keeps a lucky far clip from counting when a
        nearer one fails (pretrained YOLO's score on a hand-held ball sits near its threshold, so its
        recall is all-or-nothing per clip rather than falling off smoothly with distance).
        """
        d = g[g["test"] == "distance"].sort_values("distance_m")
        best = np.nan
        for dist, rec in zip(d.distance_m, d.recall):
            if not rec >= RELIABLE_RECALL:
                break
            best = dist
        return best

    def mean_recall(g):
        return g.recall.mean() if len(g) else np.nan

    def pooled_sigma(g):
        """Per-axis std of the 3D error over every hit of the clips in g (metres)."""
        e = [errs[(c, m)] for c, m in zip(g["clip"], g["method"]) if len(errs[(c, m)])]
        e = np.vstack(e) if e else np.empty((0, 3))
        return (e.std(axis=0) if len(e) >= 2 else np.full(3, np.nan)), len(e)

    def pooled_bias_z(g):
        e = [errs[(c, m)] for c, m in zip(g["clip"], g["method"]) if len(errs[(c, m)])]
        return float(np.vstack(e)[:, 2].mean()) if e else np.nan

    def fmt_m(x):
        return "–" if pd.isna(x) else f"{x:.3f}"

    moving = df["test"].isin(["motion", "speed"])
    head, noise = [], []
    for m in methods:
        g = df[df["method"] == m]
        lat = g.latency_med_ms.median()
        sd, _ = pooled_sigma(g)
        reacq = g[g["pattern"].isin(["out-and-back", "behind-object"])]
        head.append(
            {
                "Method": f"`{m}`",
                "Recall (all)": fmt_pct(mean_recall(g)),
                "Dropout": fmt_pct(1 - mean_recall(g)),
                "Precision (all)": fmt_pct(g.precision.mean()),
                "Centre err (px)": fmt_num(g.center_err_px.median()),
                "Pos. noise σ (m)": fmt_m(np.sqrt(np.sum(sd**2))),
                "Max dist ≥90% recall": fmt_num(max_reliable_distance(g), 1) + (" m" if not pd.isna(max_reliable_distance(g)) else ""),
                "Recall: fast": fmt_pct(mean_recall(g[moving.loc[g.index] & (g["motion"] == "fast") & (g["pattern"] != "shake")])),
                "Recall: shake": fmt_pct(mean_recall(g[g["pattern"] == "shake"])),
                "Recall: dim": fmt_pct(mean_recall(g[g["lighting"].isin(["dark", "dim"]) & ~g.no_ball])),
                "False pos. (dark, no ball visible)": str(int(g[g.no_ball].false_pos.sum())) if g.no_ball.any()
                else ("n/a" if df.no_ball.any() else "–"),
                "Recall: occluded": fmt_pct(mean_recall(g[g["test"] == "occlusion"])),
                "Max miss streak: out/behind": str(int(reacq.max_miss_streak.max())) if len(reacq) else "–",
                "False pos. (clutter)": str(int(g[g["test"] == "clutter"].false_pos.sum())) if (g["test"] == "clutter").any() else "–",
                "Latency median / p95 (ms)": f"{fmt_num(lat)} / {fmt_num(g.latency_p95_ms.median())}",
                "≈ FPS": fmt_num(1000 / lat, 0) if lat and not pd.isna(lat) else "–",
            }
        )
        gm = g[moving.loc[g.index]]
        sd_m, n_m = pooled_sigma(gm)
        n_hits = int(g.n_hits.sum())
        noise.append(
            {
                "Method": f"`{m}`",
                "Hits": n_hits,
                "σX (m)": fmt_m(sd[0]),
                "σY (m)": fmt_m(sd[1]),
                "σZ (m)": fmt_m(sd[2]),
                "Z bias (m)": "–" if pd.isna(pooled_bias_z(g)) else f"{pooled_bias_z(g):+.3f}",
                "Z bias, still ≥ 2.4 m (m)": (lambda b: "–" if pd.isna(b) else f"{b:+.3f}")(
                    pooled_bias_z(g[(g["test"] == "distance") & (g.distance_m >= 2.4)])),
                "σ 3D, all clips (m)": fmt_m(np.sqrt(np.sum(sd**2))),
                "σ 3D, moving clips (m)": fmt_m(np.sqrt(np.sum(sd_m**2))) if n_m else "–",
                "Dropout, all": fmt_pct(1 - mean_recall(g)),
                "Dropout, moving": fmt_pct(1 - mean_recall(gm)),
                "Latency median / p95 (ms)": f"{fmt_num(lat)} / {fmt_num(g.latency_p95_ms.median())}",
            }
        )
    head_md = pd.DataFrame(head).to_markdown(index=False)
    noise_md = pd.DataFrame(noise).to_markdown(index=False)

    # ---- by pattern: rows = hand-held movement, columns = methods -------------
    pat_order = [p for p in PATTERNS if p in set(df["pattern"])] + sorted(set(df["pattern"]) - set(PATTERNS))
    by_pat = []
    for pat in pat_order:
        g = df[df["pattern"] == pat]
        sp = g.drop_duplicates("clip").gt_speed_m_s
        row = {
            "Pattern": pat,
            "Clips": g["clip"].nunique(),
            "Median speed (m/s)": "–" if pat == "still" or sp.isna().all() else f"{sp.median():.2f}",
        }
        for m in methods:
            row[f"`{m}`"] = fmt_pct(mean_recall(g[g["method"] == m]))
        by_pat.append(row)
    pat_md = pd.DataFrame(by_pat).to_markdown(index=False)

    # ---- per-test tables: rows = conditions, columns = methods ----------------
    sections = []
    order = ["distance", "motion", "speed", "lighting", "occlusion", "clutter", "multi", "other"]
    for test in [t for t in order if t in set(df["test"])]:
        g = df[df["test"] == test].copy()
        g["cell"] = [
            f"{int(r.false_pos)} FP in {int(r.n_labelled)} labelled frames (ball not visible)" if r.no_ball else
            f"{fmt_pct(r.recall)} · {fmt_num(r.center_err_px)} px"
            + (f" · σ {r.sigma_3d_m:.3f} m" if test == "distance" and not pd.isna(r.sigma_3d_m) else "")
            + (f" · {int(r.false_pos)} FP" if r.false_pos else "")
            + (f" · {r.mean_dets_per_frame:.1f} dets/frame" if test == "multi" else "")
            + (f" · miss streak {int(r.max_miss_streak)}" if r.pattern in ("out-and-back", "behind-object") else "")
            for r in g.itertuples()
        ]
        g["pat_rank"] = [pat_order.index(p) for p in g["pattern"]]
        g = g.sort_values(["distance_m", "clip"] if test == "distance" else ["pat_rank", "clip"])
        piv = g.pivot_table(index=["clip", "condition"], columns="method", values="cell", aggfunc="first", sort=False)
        piv = piv[[m for m in methods if m in piv.columns]].fillna("n/a (no GT prompt)").reset_index()
        piv.columns = ["Clip", "Condition"] + [f"`{c}`" for c in piv.columns[2:]]
        sections.append(f"### {test.capitalize()}\n\nCell = recall · median centre error · false positives\n\n{piv.to_markdown(index=False)}\n")

    # ---- plots -----------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plots = []
    dist = df[df["test"] == "distance"]
    if len(dist):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in methods:
            g = dist[dist["method"] == m].sort_values("distance_m")
            if len(g):
                ax.plot(g.distance_m, 100 * g.recall, "-o", lw=2, ms=6, color=color[m], label=m)
        ax.axhline(100 * RELIABLE_RECALL, color="#9c9c9c", lw=1, ls="--")
        ax.set_xlabel("Ball distance from camera (m)")
        ax.set_ylabel("Recall (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.legend(frameon=False, loc="lower left")
        ax.set_title("How far away can each method still find the ball?", loc="left")
        fig.tight_layout()
        fig.savefig(out / "recall_vs_distance.png", dpi=150)
        plots.append("![recall vs distance](recall_vs_distance.png)")
        perfect = [m for m in methods if len(dist[dist["method"] == m]) and (dist[dist["method"] == m].recall >= 0.999).all()]
        if len(perfect) > 1:  # identical lines hide each other
            plots.append(f"*At 100 % at every distance (lines drawn on top of each other): {', '.join(f'`{m}`' for m in perfect)}.*\n")
        plt.close(fig)

    lat = df.groupby("method", sort=False)[["latency_med_ms", "latency_p95_ms"]].median().reindex(methods)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(methods) + 1.4))
    y = np.arange(len(methods))
    for yi, m in zip(y, methods):  # dot = median, whisker to p95 (log scale, so no bars)
        med, p95 = lat.loc[m, "latency_med_ms"], lat.loc[m, "latency_p95_ms"]
        ax.plot([med, p95], [yi, yi], color=color[m], lw=2)
        ax.plot(med, yi, "o", ms=8, color=color[m])
        ax.text(max(med, p95) * 1.15, yi, f"{med:.0f} ms (p95 {p95:.0f})", va="center", fontsize=9)
    ax.axvline(33.3, color="#9c9c9c", lw=1, ls="--")
    ax.text(33.3 * 1.05, -0.6, "one frame at 30 fps", color="#6c6c6c", fontsize=8, va="bottom")
    ax.set_yticks(y, methods)
    ax.set_ylim(len(methods) - 0.5, -0.8)
    ax.set_xscale("log")
    lo, hi = np.nanmin(lat.values), np.nanmax(lat.values)
    ax.set_xlim(lo / 2, hi * 6)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_xlabel("Compute time per frame, ms (log scale; dot = median, line to p95)")
    ax.set_title("Latency on this laptop", loc="left")
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=150)
    plots.append("![latency](latency.png)")
    plt.close(fig)

    # ---- write markdown -------------------------------------------------------
    any_meta = Clip(_paths.CLIPS_DIR / df["clip"].iloc[0]).meta
    calib_txt = f"calibrated, reprojection RMS {intr.rms_px:.2f} px, fx = {intr.fx:.0f} px" if intr else "not calibrated"
    md = f"""# Ball perception benchmark: results

*Generated by `scripts/summarize.py` on {date.today()}.*

**Setup.** {machine()} · camera {any_meta.get('width')}×{any_meta.get('height')} @ ~{any_meta.get('fps')} fps ({calib_txt}) ·
{df["clip"].nunique()} clips · {int(df.groupby('clip').n_labelled.first().sum())} hand-labelled frames · ball: {any_meta.get('ball', 'tennis')}.
SAM 2 / EfficientTAM rows prompted with the first ground-truth box unless the name says `detector+tracker`.
Protocol: [`docs/perception-benchmark.md`](../../docs/perception-benchmark.md).

## Summary

{head_md}

Dropout = 1 − recall. Recall, dropout and precision are means over clips (each condition counts equally).
Max dist ≥90% recall = farthest distance with ≥ 90 % recall *at every distance up to it*.
Pos. noise σ = √(σX² + σY² + σZ²) of the 3D error on hits, pooled over all clips (see below).
Clips where the ball is never visible (lights off) are left out of recall, dropout and precision and scored by false positives only;
the GT-prompted trackers cannot run on them at all (n/a: there is no visible ball to prompt with).
Max miss streak is counted in labelled frames (×5 camera frames when labelled with `--every 5`).

{chr(10).join(plots)}

## Noise model (for the Kalman filter and policy training)

{noise_md}

3D error = back-projected prediction − back-projected hand label, in camera coordinates (X right, Y down, Z forward),
each point at the depth implied by its own radius (Z = fx · D / 2r). σ is the per-axis standard deviation over every hit.
The reference is a hand click, so ~1–2 px of label noise is included. Frames labelled occluded are excluded: the label is the
estimated full ball, but a method only sees the visible part, so its predicted radius (and depth) is biased. Z bias = mean Z error; negative = the method sees the ball too big, i.e. too close.
σZ comes mostly from the moving clips (motion blur changes the apparent radius, and blurred balls are harder to label) and,
for the SAM-family masks, from a radius bias that grows with distance.
Dropout is the fraction of visible frames with no hit (missed or wrong place): the frames the filter must bridge.

## By pattern

{pat_md}

Cell = mean recall over the clips with that hand-held pattern.
Speed is lateral speed from the labels: pixel displacement × (Z / fx) / Δt, with Z from the ball's apparent size.
It ignores motion toward or away from the camera, so `toward-away` reads low.
Labels are every 5th frame (~167 ms), so fast oscillation is aliased: `shake` reads ~3× too low (a 30 fps colour track gives ~2 m/s median,
p90 ~3.4 m/s). Fast sweeps are short bursts between pauses, so their p90 (in the per-test tables) is the better "fast" number.

## By test

{chr(10).join(sections)}
"""
    (out / "RESULTS.md").write_text(md)
    print(head_md)
    print(f"\nwrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
