"""Figures of disturbance experiment A (after disturb_force.py / batch_disturbance.py).

  pareto.png       every design: workspace J against grasp robustness (--pareto-y: median
                   F_min of the success cells by default; robust_area grows with the
                   workspace itself); Pareto front marked. Is the workspace-optimal design
                   also the robust one, or is there a trade-off?
  directions.png   F of every direction, env vs tip, for the selected designs
  survival.png     number of success cells whose weakest direction holds >= F, env / tip
  direction_heatmap.png  median F of every direction, env and tip side by side, one row per
                   selected design (hold-test values for designs run with --refine)
  selected_designs.csv   the selected designs: design values, stiffness, preload, workspace
                   J / J_stable, success cells, pass@1 mg, robust_area, F_min medians, and
                   the median F of every direction for env and tip (ramp and hold)
  fmin_map_<label>.png  (--map) weakest-direction F of every cell of one design's map

  python plot_disturbance.py --run runs/rb5_cmaes_final
  python plot_disturbance.py --run runs/rb5_cmaes_final --labels design001 candidate_02 --map candidate_02
  python plot_disturbance.py --run runs/rb5_cmaes_final --top-union 10   # top 10 of both rankings
Default selection: design001, candidate_02 (if analyzed) and the top 3 by robust_area.
--top-union N: the union of the top N by robust_area and the top N by J_robust at 1 m g.
directions.png shows at most 6 designs; the heatmap and the CSV take any number.
Writes <run>/plots/disturbance/.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from gripper_core import read_json
from eval_design_rb5 import read_rows
from disturb_collect import OUT_DIR, ROOT, load_label
from disturb_force import DIRECTIONS
from batch_disturbance import DESIGN_KEYS, HAND_KEYS, design_info

# Reference palette (dataviz skill): categorical slots in fixed order, sequential blue.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GROUP_COLOR = {"env": SERIES[0], "tip": SERIES[1]}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1", "#fcfcfb"
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
EMPTY_CELL = "#efeeeb"
ZERO_FAIL_CELL = "#c3c2b7"


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "axes.titlesize": 10, "legend.frameon": False})


def load_summaries(run):
    base = Path(run)/OUT_DIR
    out = {}
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if (d/"force_summary.json").is_file():
            sm = read_json(d/"force_summary.json")
            sm["J"] = design_info(run, sm["label"]).get("J")
            # x of pareto.png: the map's J without the cells that fail the zero-force control.
            sm["J_x"] = sm["J_stable"] if sm.get("zero_rule") == "fail" and "J_stable" in sm else sm["J"]
            out[sm["label"]] = sm
    return out


def pareto_front(points):
    """Indices of points not dominated in (x, y), both maximized."""
    idx = []
    for i, (x, y) in enumerate(points):
        if not any((x2 >= x and y2 >= y) and (x2 > x or y2 > y) for j, (x2, y2) in enumerate(points) if j != i):
            idx.append(i)
    return idx


PARETO_Y = {  # --pareto-y: (summary key, axis label)
    "median": ("F_min_median_mg", "median F_min of the success cells [m g]  (grasp quality)"),
    "area": ("robust_area", "robust_area [m g]  (sum of F_min over success cells)"),
    "cond": ("cond_1", "share of success cells holding 1 m g in every direction"),
}


def pareto_value(sm, key):
    if key == "cond_1":
        L = next((L for L in sm["levels"] if abs(L["level_mg"]-1.0) < 1e-9), None)
        return (L["conditional_robustness"] or 0.0) if L else 0.0
    if key == "F_min_median_mg" and key not in sm:  # Summary of an older disturb_force.py.
        return 0.0
    return sm[key]


def plot_pareto(path, sms, highlight, y="median"):
    import matplotlib.pyplot as plt
    key, ylabel = PARETO_Y[y]
    rows = [(l, sm["J_x"], pareto_value(sm, key)) for l, sm in sms.items() if sm.get("J_x") is not None]
    if not rows:
        return None
    labels, J, area = zip(*rows)
    front = set(pareto_front(list(zip(J, area))))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    other = [i for i in range(len(labels)) if i not in front]
    zero = [i for i in other if area[i] <= 0]
    rest = [i for i in other if area[i] > 0]
    ax.scatter([J[i] for i in rest], [area[i] for i in rest], s=22, color=MUTED, alpha=0.7,
               edgecolors="none", label="design")
    if zero:
        ax.scatter([J[i] for i in zero], [0]*len(zero), s=22, facecolors="none", edgecolors=MUTED,
                   linewidths=1, label="0 (nothing graspable / holds)")
    f = sorted(front, key=lambda i: J[i])
    ax.plot([J[i] for i in f], [area[i] for i in f], color=SERIES[0], lw=2, zorder=3)
    ax.scatter([J[i] for i in f], [area[i] for i in f], s=40, color=SERIES[0], edgecolors=SURFACE,
               linewidths=2, zorder=4, label="Pareto front")
    for i, l in enumerate(labels):
        if l in highlight or i in front:
            ax.annotate(l, (J[i], area[i]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
            if l in highlight and i not in front:
                ax.scatter([J[i]], [area[i]], s=40, color=INK_2, edgecolors=SURFACE, linewidths=2, zorder=4)
    metric = next(iter(sms.values())).get("metric", "ramp")
    stable = any(sm.get("zero_rule") == "fail" for sm in sms.values())
    ax.set_xlabel("workspace J_stable (cells that also hold with no load)" if stable
                  else "workspace J (env + w_tip tip + force term)")
    ax.set_ylabel(f"{ylabel}, {metric}", fontsize=8)
    ax.set_title("Workspace vs disturbance robustness", loc="left")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def dir_rows(run, label):
    path = Path(run)/OUT_DIR/label/"force_dirs.csv"
    return read_rows(path) if path.is_file() and path.stat().st_size else []


def cell_rows(run, label):
    path = Path(run)/OUT_DIR/label/"force_cells.csv"
    return read_rows(path) if path.is_file() and path.stat().st_size else []


def plot_directions(path, run, labels, cap):
    import matplotlib.pyplot as plt
    dirs = list(DIRECTIONS)
    fig, axes = plt.subplots(len(labels), 1, figsize=(7.6, 2.3*len(labels)+0.6), sharey=True, squeeze=False)
    for ax, label in zip(axes[:, 0], labels):
        rows = dir_rows(run, label)
        for k, g in enumerate(("env", "tip")):
            data = [[float(r["F_mg"]) for r in rows if r["group"] == g and r["dir"] == d] for d in dirs]
            pos = np.arange(len(dirs))+(k-0.5)*0.34
            keep = [i for i, v in enumerate(data) if v]
            if not keep:
                continue
            n = max(len(v) for v in data)
            ax.plot([], [], color=GROUP_COLOR[g], lw=6, label=f"{g} (n={n})")
            if n < 5:  # Too few cells for a box: show the values.
                for i in keep:
                    ax.scatter([pos[i]]*len(data[i]), data[i], s=26, color=GROUP_COLOR[g],
                               edgecolors=SURFACE, linewidths=1.5, zorder=3)
                continue
            bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=0.28, patch_artist=True,
                            showfliers=True, medianprops=dict(color=SURFACE, lw=1.5),
                            whiskerprops=dict(color=GROUP_COLOR[g], lw=1), capprops=dict(color=GROUP_COLOR[g], lw=1),
                            flierprops=dict(marker="o", markersize=3, markerfacecolor=GROUP_COLOR[g],
                                            markeredgecolor="none", alpha=0.6))
            for b in bp["boxes"]:
                b.set(facecolor=GROUP_COLOR[g], edgecolor=GROUP_COLOR[g])
        ax.axhline(1.0, color=INK_2, lw=0.8, ls="--")
        ax.set_xticks(range(len(dirs)), dirs)
        ax.set_ylim(0, cap)
        ax.set_ylabel("F [m g]")
        ax.set_title(label, loc="left")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("direction (hand frame: x jaw, y out of hand, z cylinder axis/up)    dashed: 1 m g")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_survival(path, run, labels, cap):
    import matplotlib.pyplot as plt
    levels = np.linspace(0, cap, 201)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=False)
    for ax, g in zip(axes, ("env", "tip")):
        for k, label in enumerate(labels):
            rows = cell_rows(run, label)
            f = np.array([float(r["F_min_mg"]) for r in rows if r["group"] == g])
            ax.plot(levels, [(f >= lv-1e-9).sum() for lv in levels], color=SERIES[k % len(SERIES)], lw=2,
                    label=label, drawstyle="steps-post")
        ax.set_title(f"{g} cells holding >= F in every direction", loc="left")
        ax.set_xlabel("F [m g]")
        ax.set_ylabel("cells")
        ax.set_xlim(0, cap)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_map(path, run, label, vmax=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _, _, _, ws = load_label(run, label)
    p = Path(run)/OUT_DIR/label/"force_cells.csv"
    cells = {(float(r["r_mm"]), float(r["x_mm"]), float(r["y_mm"])): r
             for r in (read_rows(p) if p.is_file() and p.stat().st_size else [])}
    radii = sorted({float(r["r_mm"]) for r in ws})
    if not vmax:  # 95th percentile of the design's F_min, rounded up to 0.5 m g.
        f = [float(c["F_min_mg"]) for c in cells.values() if c["group"] in ("env", "tip")]
        vmax = max(0.5, math.ceil(np.percentile(f, 95)*2)/2) if f else 1.0
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    cols = min(4, len(radii))
    rows_n = int(np.ceil(len(radii)/cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.0*cols, 2.9*rows_n+0.5), squeeze=False)
    xs = sorted({float(r["x_mm"]) for r in ws})
    ys = sorted({float(r["y_mm"]) for r in ws})
    step = min(np.diff(xs).min() if len(xs) > 1 else 10, np.diff(ys).min() if len(ys) > 1 else 10)
    im = None
    for ax, r in zip(axes.flat, radii):
        for w in ws:
            if float(w["r_mm"]) != r:
                continue
            x, y = float(w["x_mm"]), float(w["y_mm"])
            c = cells.get((r, x, y))
            if c is not None and c["group"] == "zero_fail":  # Lifted, but lost it with no load.
                ax.add_patch(plt.Rectangle((x-step/2+0.6, y-step/2+0.6), step-1.2, step-1.2, color=ZERO_FAIL_CELL, lw=0))
                continue
            if c is None or c["group"] not in ("env", "tip"):
                ax.add_patch(plt.Rectangle((x-step/2+0.6, y-step/2+0.6), step-1.2, step-1.2, color=EMPTY_CELL, lw=0))
                continue
            col = cmap(min(float(c["F_min_mg"]), vmax)/vmax)
            ax.add_patch(plt.Rectangle((x-step/2+0.6, y-step/2+0.6), step-1.2, step-1.2, facecolor=col,
                                       edgecolor=GROUP_COLOR["tip"] if c["group"] == "tip" else col, lw=1.2))
        ax.set_xlim(min(xs)-step, max(xs)+step)
        ax.set_ylim(min(ys)-step, max(ys)+step)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"r = {r:g} mm", loc="left")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    for ax in list(axes.flat)[len(radii):]:
        ax.axis("off")
    im = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label(f"weakest-direction F [m g] (capped at {vmax:g})")
    fig.suptitle(f"{label}: F_min per cell (orange edge = tip grasp, light gray = not graspable, "
                 "dark gray = lifted but fails with no load)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def level(sm, lv, key):
    L = next((L for L in sm["levels"] if abs(L["level_mg"]-lv) < 1e-9), None)
    return L[key] if L else None


def top_union(sms, n):
    by_area = sorted(sms, key=lambda l: -sms[l]["robust_area"])[:n]
    by_jr = sorted(sms, key=lambda l: -(level(sms[l], 1.0, "J_robust") or 0.0))[:n]
    return list(dict.fromkeys(by_area+by_jr))


def direction_value(sm, g, d, value):
    st = sm.get("direction_stats", {}).get(g, {}).get(d)
    if not st or not st.get("n"):
        return None
    return st.get("hold_median") if value == "hold" else st.get("median")


def plot_heatmap(path, sms, labels, value="auto", vmax=None):
    """Median F per direction, env | tip, one row per design."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    labels = [l for l in labels if l in sms]
    if not labels:
        return None
    if value == "auto":
        value = "hold" if all(sms[l].get("hold") for l in labels) else "ramp"
    dirs = list(DIRECTIONS)
    grid = np.full((len(labels), 2*len(dirs)), np.nan)
    for i, l in enumerate(labels):
        for k, g in enumerate(("env", "tip")):
            for j, d in enumerate(dirs):
                v = direction_value(sms[l], g, d, value)
                if v is not None:
                    grid[i, k*len(dirs)+j] = v
    finite = grid[np.isfinite(grid)]
    # Censored env directions (up to the 10 m g cap) would wash out the range that matters.
    vmax = vmax or (min(3.0, max(0.5, math.ceil(np.percentile(finite, 90)*2)/2)) if finite.size else 1.0)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(9.2, 0.38*len(labels)+1.8))
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            x = j+(0.35 if j >= len(dirs) else 0)  # Gap between the env and the tip block.
            if not np.isfinite(v):
                ax.add_patch(plt.Rectangle((x+0.04, i+0.04), 0.92, 0.92, color=EMPTY_CELL, lw=0))
                continue
            u = min(v, vmax)/vmax
            ax.add_patch(plt.Rectangle((x+0.04, i+0.04), 0.92, 0.92, color=cmap(u), lw=0))
            ax.text(x+0.5, i+0.5, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if u > 0.55 else INK)
    ax.set_xlim(0, grid.shape[1]+0.35)
    ax.set_ylim(len(labels), 0)
    ax.set_xticks([j+0.5+(0.35 if j >= len(dirs) else 0) for j in range(grid.shape[1])], dirs*2)
    ax.set_yticks([i+0.5 for i in range(len(labels))],
                  [f"{l}  ({sms[l]['env']} env, {sms[l]['tip']} tip)" for l in labels])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(dirs)/2, -0.35, "env cells", ha="center", va="bottom", color=INK_2)
    ax.text(len(dirs)+0.35+len(dirs)/2, -0.35, "tip cells", ha="center", va="bottom", color=INK_2)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm_, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(f"median F [m g] ({'hold test' if value == 'hold' else 'ramp'}; capped at {vmax:g})")
    ax.set_title(f"Median F per direction ({value}); gray = no such cells", loc="left", pad=36)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_selected(path, run, sms, labels):
    """One row per selected design: design values and the experiment's numbers."""
    rows = []
    for l in labels:
        if l not in sms:
            continue
        sm, info = sms[l], design_info(run, l)
        row = dict(label=l, **{k: info.get(k) for k in DESIGN_KEYS}, **{k: info.get(k) for k in HAND_KEYS.values()},
                   J=info.get("J"), J_stable=sm.get("J_stable"), success=sm["success"], env=sm["env"], tip=sm["tip"],
                   removed_zero_fail=sum((sm.get("zero_fail_removed") or {}).values()),
                   pass_1mg=level(sm, 1.0, "passed"), cond_1mg=level(sm, 1.0, "conditional_robustness"),
                   J_robust_1mg=level(sm, 1.0, "J_robust"), robust_area=sm["robust_area"],
                   F_min_median_mg=sm.get("F_min_median_mg"))
        hold = sm.get("hold") or {}
        row.update(F_min_hold_median_mg=hold.get("F_min_median_mg"), pass_hold_1mg=(hold.get("passed") or {}).get("1mg"))
        for g in ("env", "tip"):
            for d in DIRECTIONS:
                row[f"F_{d}_{g}_mg"] = direction_value(sm, g, d, "ramp")
                row[f"F_{d}_{g}_hold_mg"] = direction_value(sm, g, d, "hold")
        rows.append(row)
    if not rows:
        return None
    import csv
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=ROOT/"runs"/"rb5_cmaes_final")
    p.add_argument("--labels", nargs="+", help="Designs for directions/survival (default: see above)")
    p.add_argument("--top", type=int, default=3, help="Top designs by robust_area added by default")
    p.add_argument("--top-union", type=int, help="Select the union of the top N by robust_area and by J_robust@1mg")
    p.add_argument("--value", choices=("auto", "ramp", "hold"), default="auto",
                   help="Heatmap values: hold test if every selected design has one (auto), ramp, or hold")
    p.add_argument("--map", nargs="*", default=[], help="Labels to draw an F_min map for")
    p.add_argument("--pareto-y", choices=sorted(PARETO_Y), default="median",
                   help="Robustness axis of pareto.png: median F_min of the success cells (default, "
                        "independent of workspace size), robust_area, or share holding 1 m g")
    p.add_argument("--vmax", type=float, help="Color scale end of the map [m g] (default: 95th percentile)")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    run = args.run.resolve()
    sms = load_summaries(run)
    if not sms:
        raise SystemExit(f"No force_summary.json under {run/OUT_DIR}. Run disturb_force.py first.")
    ranked = sorted(sms, key=lambda l: -sms[l]["robust_area"])
    if args.top_union:
        labels = top_union(sms, args.top_union)
    else:
        labels = args.labels or list(dict.fromkeys([l for l in ("design001", "candidate_02") if l in sms]
                                                   + ranked[:args.top]))
    missing = [l for l in labels if l not in sms]
    if missing:
        print(f"No force_summary.json for {missing}; left out.")
        labels = [l for l in labels if l in sms]
    print(f"Selected {len(labels)} designs: {' '.join(labels)}")
    out = run/"plots"/"disturbance"
    out.mkdir(parents=True, exist_ok=True)
    cap = max(sm["cap_mg"] for sm in sms.values())
    top_f = max([float(r["F_mg"]) for l in labels for r in dir_rows(run, l)] or [1.0])
    top_min = max([float(r["F_min_mg"]) for l in labels for r in cell_rows(run, l)] or [1.0])
    if len(labels) > 6:
        print(f"directions.png: first 6 of the {len(labels)} designs (all of them are in direction_heatmap.png)")
    for path in (plot_pareto(out/"pareto.png", sms, set(labels), args.pareto_y),
                 plot_directions(out/"directions.png", run, labels[:6], min(cap, top_f*1.08)),
                 plot_heatmap(out/"direction_heatmap.png", sms, labels, args.value),
                 write_selected(out/"selected_designs.csv", run, sms, labels),
                 plot_survival(out/"survival.png", run, labels, min(cap, top_min*1.1)),
                 *(plot_map(out/f"fmin_map_{l}.png", run, l, args.vmax) for l in args.map)):
        if path:
            print("Saved:", path)


if __name__ == "__main__":
    main()
