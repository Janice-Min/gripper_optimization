"""Figures of disturbance experiment B (after disturb_impulse.py).

  impulse_heatmap.png     share of success cells that held the pulse, per direction and pulse
                          size, one row per selected design, env and tip side by side
  impulse_vs_ramp.png     share of trials held at each pulse size against A's ramp F_max of the
                          same cell and direction (binned): is a short pulse survived above the
                          slowly-ramped limit (curve still high right of the dashed line k) or
                          does it break grasps that hold k statically (low left of it)?
  impulse_pareto.png      every design: workspace J_stable against B's grasp quality (--pareto-y:
                          share of success cells holding the --k pulse in every direction by
                          default), Pareto front marked
  impulse_vs_A.png        every design: A's median F_min against B's median level_min (do the
                          two experiments rank designs the same way? Spearman in the title)
  impulse_trace_<label>_<cell>.png   (--trace LABEL CELL) displacement and rotation over time
                          of one cell, every direction, one row per pulse size

  python B/plot_impulse.py                                                      (from on_rb5/)
  python B/plot_impulse.py --labels design075 candidate_02 design001 --k 2
  python B/plot_impulse.py --tag pilot --trace design075 r40_x0_y15
Default selection: design001, candidate_02 and the top 6 by impact_area.
Writes B/results/<run name>/impulse[_<tag>]/plots/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from gripper_core import read_json
from eval_design_rb5 import read_rows
from disturb_impulse import C, impulse_dir, sub_dir, tkey
from plot_disturbance import (EMPTY_CELL, GRID, INK, INK_2, MUTED, SEQUENTIAL, SERIES, SURFACE,
                              pareto_front, style)


def load(run, sub):
    """{label: summary} of every design with B results."""
    out = {}
    base = impulse_dir(run, sub)
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        p = d/"impulse_summary.json"
        if p.is_file():
            out[d.name] = read_json(p)
    return out


def trials(run, label, sub):
    p = impulse_dir(run, sub, label)/"impulse_trials.csv"
    return read_rows(p) if p.is_file() and p.stat().st_size else []


def level(sm, k):
    return next((L for L in sm["levels"] if abs(L["mag_mg"]-k) < 1e-9), None)


def select(sms, labels, n=6):
    if labels:
        return [l for l in labels if l in sms]
    top = sorted(sms, key=lambda l: -sms[l]["impact_area"])[:n]
    return list(dict.fromkeys([l for l in ("design001", "candidate_02") if l in sms] + top))


def plot_heatmap(path, sms, labels):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    sm0 = sms[labels[0]]
    dirs, mags = sm0["dirs"], sm0["mags"]
    cols = [(d, k) for d in dirs for k in mags]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", *SEQUENTIAL])
    cmap.set_bad(EMPTY_CELL)
    fig, axes = plt.subplots(1, 2, figsize=(2.2+0.34*len(cols)*2, 1.2+0.32*len(labels)), sharey=True)
    for ax, g in zip(axes, ("env", "tip")):
        M = np.full((len(labels), len(cols)), np.nan)
        for i, l in enumerate(labels):
            per = sms[l]["held_fraction"].get(g, {})
            for j, (d, k) in enumerate(cols):
                v = per.get(d, {}).get(f"{k:g}")
                M[i, j] = np.nan if v is None else v
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, "1" if v >= 0.995 else "0" if v < 0.005 else f"{v:.2f}".lstrip("0"), ha="center",
                            va="center", fontsize=6, color=SURFACE if v > 0.6 else INK_2)
        for j in range(len(mags), len(cols), len(mags)):
            ax.axvline(j-0.5, color=SURFACE, lw=2)
        ax.set_xticks(range(len(cols)), [f"{k:g}" for _, k in cols], fontsize=7)
        top = ax.secondary_xaxis("top")
        top.set_xticks([j*len(mags)+(len(mags)-1)/2 for j in range(len(dirs))], dirs, fontsize=8, color=INK)
        top.tick_params(length=0)
        top.spines["top"].set_visible(False)
        ax.set_title(f"{g} cells", loc="left", pad=18)
        ax.set_xlabel("pulse size [m g] per direction")
        ax.set_yticks(range(len(labels)), [f"{l} (n={sms[l][g]})" for l in labels], fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="share of success cells held")
    fig.suptitle(f"Experiment B: {sm0['pulse_ms']:g} ms {sm0['shape']} pulse, then {sm0['after_s']:g} s",
                 x=0.01, y=1.0+0.35/(1.2+0.32*len(labels)), ha="left", fontsize=10)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vs_ramp(path, run, sub, labels, sms):
    import matplotlib.pyplot as plt
    mags = sms[labels[0]]["mags"]
    rows = [t for l in labels for t in trials(run, l, sub) if t["group"] in ("env", "tip") and t["A_F_ramp_mg"] is not None]
    if not rows:
        return None
    edges = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10.01])
    mid = np.sqrt(edges[:-1]*edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, g in zip(axes, ("env", "tip")):
        for i, k in enumerate(mags):
            sel = [t for t in rows if t["group"] == g and abs(t["mag_mg"]-k) < 1e-9]
            if not sel:
                continue
            F = np.clip([t["A_F_ramp_mg"] for t in sel], 0.1, None)  # F_max 0 (no hold at all) on the left edge.
            h = np.array([bool(t["held"]) for t in sel])
            b = np.digitize(F, edges)-1
            x, y, n = [], [], []
            for j in range(len(mid)):
                m = b == j
                if m.sum() >= 3:
                    x.append(mid[j]); y.append(h[m].mean()); n.append(m.sum())
            col = SERIES[i % len(SERIES)]
            ax.plot(x, y, color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"{k:g} m g pulse")
            ax.axvline(k, color=col, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 11)
        ticks = [0.25, 0.5, 1, 2, 3, 5, 10]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("A: ramp F_max of the same cell and direction [m g]")
        ax.set_title(f"{g} cells", loc="left")
    axes[0].set_ylabel("share of pulses held")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Short pulse vs slowly ramped force (dashed: pulse size = ramp limit)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def b_quality(sm, y, k):
    if y == "cond":
        L = level(sm, k)
        return (L["cond"] or 0.0) if L else 0.0
    return sm["level_min_median_mg"] if y == "median" else sm["impact_area"]


def plot_pareto(path, sms, highlight, y, k):
    import matplotlib.pyplot as plt
    names = {"cond": f"share of success cells holding the {k:g} m g pulse in every direction",
             "median": "median level_min of the success cells [m g]", "area": "impact_area [m g]"}
    rows = [(l, sm["A_J_stable"], b_quality(sm, y, k)) for l, sm in sms.items() if sm.get("A_J_stable") is not None]
    if not rows:
        return None
    labels, J, Q = zip(*rows)
    front = set(pareto_front(list(zip(J, Q))))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    rest = [i for i in range(len(labels)) if i not in front]
    ax.scatter([J[i] for i in rest], [Q[i] for i in rest], s=22, color=MUTED, alpha=0.7, edgecolors="none",
               label="design")
    f = sorted(front, key=lambda i: J[i])
    ax.plot([J[i] for i in f], [Q[i] for i in f], color=SERIES[0], lw=2, zorder=3)
    ax.scatter([J[i] for i in f], [Q[i] for i in f], s=40, color=SERIES[0], edgecolors=SURFACE, linewidths=2,
               zorder=4, label="Pareto front")
    for i, l in enumerate(labels):
        if l in highlight or i in front:
            ax.annotate(l, (J[i], Q[i]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("workspace J_stable (experiment A)")
    ax.set_ylabel(names[y], fontsize=8)
    ax.set_title("Workspace vs impact robustness (experiment B)", loc="left")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(v)).astype(float) for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def plot_vs_a(path, sms, highlight):
    import matplotlib.pyplot as plt
    rows = [(l, sm["A_F_min_median_mg"], sm["level_min_median_mg"]) for l, sm in sms.items()
            if sm.get("A_F_min_median_mg") is not None and sm["success"]]
    if len(rows) < 2:
        return None
    labels, a, b = zip(*rows)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.04, 0.04, len(b))  # level_min takes only the pulse sizes.
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(a, np.array(b)+jitter, s=24, color=SERIES[0], alpha=0.8, edgecolors=SURFACE, linewidths=1)
    for i, l in enumerate(labels):
        if l in highlight:
            ax.annotate(l, (a[i], b[i]+jitter[i]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("A: median F_min of the success cells (ramp) [m g]")
    ax.set_ylabel("B: median level_min of the success cells [m g]")
    ax.set_title(f"Experiment A vs B per design (Spearman {spearman(a, b):.2f}, n={len(a)})", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_trace(path, run, sub, label, cell):
    import matplotlib.pyplot as plt
    p = impulse_dir(run, sub, label)/"traces"/f"{cell}.npz"
    if not p.is_file():
        raise SystemExit(f"No {p}")
    tr = np.load(p)
    meta = read_json(impulse_dir(run, sub, label)/"impulse_meta.json")
    s, crit = meta["settings"], meta.get("criteria", meta["criteria_at_simulation"])
    dirs, mags = [str(d) for d in tr["dirs"]], [float(k) for k in tr["mags"]]
    fig, axes = plt.subplots(len(mags), 2, figsize=(10, 2.2*len(mags)+0.6), sharex=True, squeeze=False)
    for i, k in enumerate(mags):
        for j, (col, lim, unit) in enumerate((("disp", crit["slip_mm"], "displacement [mm]"),
                                              ("rot_deg", crit["rot_deg"], "rotation [deg]"))):
            ax = axes[i, j]
            if "z0_trace" in tr.files:
                z = tr["z0_trace"].astype(float)
                yz = np.linalg.norm(z[:, 2:5], axis=1) if col == "disp" else z[:, C[col]]
                ax.plot(np.maximum(z[:, 0], 1e-3), yz, color=MUTED, lw=1.5, ls=":", label="no pulse")
            for n, d in enumerate(dirs):
                key = tkey(d, k)
                if str(tr[f"{key}_ended"]) == "skipped":
                    continue
                x = tr[f"{key}_trace"].astype(float)
                y = np.linalg.norm(x[:, 2:5], axis=1) if col == "disp" else x[:, C[col]]
                ax.plot(np.maximum(x[:, 0], 1e-3), y, color=SERIES[n % len(SERIES)], lw=1.5, label=d)
            if lim:
                ax.axhline(lim, color=INK_2, lw=1, ls="--")
            ax.axvspan(1e-3, s["pulse_s"], color=GRID, lw=0)
            ax.set_xscale("log")
            ax.set_ylim(0, 3*lim if lim else None)
            ax.set_ylabel(unit if j == 0 else unit, fontsize=8)
            ax.set_title(f"{k:g} m g pulse", loc="left", fontsize=9) if j == 0 else None
    for ax in axes[-1]:
        ax.set_xlabel("time from pulse start [s] (shaded: pulse)")
    axes[0, 1].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(f"{label} {cell}: {s['pulse_s']*1000:g} ms {s['shape']} pulse (dashed: failure limit)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=layout.DEFAULT_RUN)
    p.add_argument("--tag", default="", help="Results of disturb_impulse.py --tag")
    p.add_argument("--labels", nargs="+", help="Designs for the heatmap and impulse_vs_ramp")
    p.add_argument("--top", type=int, default=6, help="Default selection: top N by impact_area (default 6)")
    p.add_argument("--k", type=float, help="Pulse size for --pareto-y cond (default: middle size)")
    p.add_argument("--pareto-y", choices=("cond", "median", "area"), default="cond")
    p.add_argument("--trace", nargs=2, metavar=("LABEL", "CELL"), help="Time traces of one cell, e.g. design075 r40_x0_y15")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    run, sub = args.run.resolve(), sub_dir(args.tag)
    out = impulse_dir(run, sub)/"plots"
    out.mkdir(parents=True, exist_ok=True)
    if args.trace:
        print("Saved:", plot_trace(out/f"impulse_trace_{args.trace[0]}_{args.trace[1]}.png", run, sub, *args.trace))
        return
    sms = load(run, sub)
    if not sms:
        raise SystemExit(f"No */impulse_summary.json in {impulse_dir(run, sub)}. Run B/disturb_impulse.py first.")
    labels = select(sms, args.labels, args.top)
    mags = sms[labels[0]]["mags"]
    k = args.k if args.k is not None else mags[len(mags)//2]
    for path in (plot_heatmap(out/"impulse_heatmap.png", sms, labels),
                 plot_vs_ramp(out/"impulse_vs_ramp.png", run, sub, labels, sms),
                 plot_pareto(out/"impulse_pareto.png", sms, set(labels), args.pareto_y, k),
                 plot_vs_a(out/"impulse_vs_A.png", sms, set(labels))):
        if path:
            print("Saved:", path)


if __name__ == "__main__":
    main()
