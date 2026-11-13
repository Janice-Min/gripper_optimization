"""Per-direction F_max of chosen designs for the SAME object size (and optionally the same cells).

F depends strongly on the object radius r and on where the object sits in the hand, and
every design grasps a different set of map cells. For a design-to-design comparison, fix r
(and the grasp type), and if possible use only the cells every design grasps.

  1) Which radii do all designs grasp?
     python compare_force.py --labels design074 design078 candidate_02 --coverage
  2) Best grasp of every design over ALL its success cells (env and tip, any radius)
     python compare_force.py --labels design074 design078 candidate_02 --peak
     per design: the cell whose weakest direction holds most (max F_min, i.e. the strongest
     grasp in every direction at once) with its r, position and all six F; the mean of the
     top --top-k cells and the 90th percentile (the max of one cell favours designs with
     many cells and single lucky cells); and the max of each direction on its own, which may
     come from different cells. Values at the ramp cap are censored (true value higher).
     --peak also draws the directional maxima: peak_dirs_<group>_<value>.png (one panel per
     design, value and radius of the cell above each bar) and peak_by_direction_<group>_<value>.png
     (all designs side by side per direction). --peak-csv FILE redraws them from a saved CSV.
     add --plot-all for graphs of every cell (works on its own, too):
       all_cells_dirs_<group>_<value>.png  one panel per design: every cell's F per direction
                                           and its F_min (weakest direction), env/tip colored
       all_cells_fmin_<group>_<value>.png  F_min of every cell, designs side by side
     (log F axis; --linear for a linear one)
  3) Table and heatmap at one radius (env grasps, median over the design's cells at that r)
     python compare_force.py --labels design074 design078 candidate_02 --r 40
     ... --r 40 50          several radii side by side
     ... --common           only the (x, y) cells that every design grasps at that r
     ... --x 0              only cells centered in the jaw (x = 0): +x and -x are then symmetric
     ... --group tip|all    tip grasps, or env and tip together
     ... --value ramp|hold  F of the ramp, or of the hold test (default: hold if every design
                            was run with disturb_force.py --refine, else ramp)

Reads <run>/disturbance/<label>/force_dirs.csv and force_cells.csv (zero-force failures are
group 'zero_fail' there and never count as success cells).
Writes <run>/plots/disturbance/compare_<tag>.csv and compare_<tag>.png.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from eval_design_rb5 import read_rows
from disturb_collect import OUT_DIR, ROOT
from disturb_force import DIRECTIONS

GROUPS = {"env": ("env",), "tip": ("tip",), "all": ("env", "tip")}


def rows_of(run, label, name):
    path = Path(run)/OUT_DIR/label/name
    if not path.is_file():
        raise SystemExit(f"No {path}. Run disturb_force.py for {label} first.")
    return read_rows(path) if path.stat().st_size else []


def coverage(run, labels):
    """Success cells per radius and grasp type."""
    table, radii = {}, set()
    for l in labels:
        counts = {}
        for c in rows_of(run, l, "force_cells.csv"):
            if c["group"] in ("env", "tip"):
                counts.setdefault(c["r_mm"], {"env": 0, "tip": 0})[c["group"]] += 1
        table[l] = counts
        radii |= set(counts)
    radii = sorted(radii)
    w = max(len(l) for l in labels)+2
    print("Success cells per object radius r [mm] (env/tip; zero-force failures excluded):")
    print(" "*w + "".join(f"{f'r={r:g}':>10}" for r in radii))
    for l in labels:
        print(f"{l:<{w}}" + "".join(f"{table[l].get(r, {}).get('env', 0):>6}/{table[l].get(r, {}).get('tip', 0):<3}"
                                    for r in radii))
    for g in ("env", "all"):
        ok = [r for r in radii if all(sum(table[l].get(r, {}).get(k, 0) for k in GROUPS[g]) > 0 for l in labels)]
        best = sorted(ok, key=lambda r: -min(sum(table[l].get(r, {}).get(k, 0) for k in GROUPS[g]) for l in labels))
        print(f"Radii where every design has {'env' if g == 'env' else 'env or tip'} cells: "
              + (", ".join(f"{r:g}" for r in ok) if ok else "none")
              + (f"  (most cells for the weakest design: r={best[0]:g})" if best else ""))
    return table


def select(rows, r, groups, x=None):
    out = [d for d in rows if abs(d["r_mm"]-r) < 1e-6 and d["group"] in groups]
    if x is not None:
        out = [d for d in out if abs(d["x_mm"]-x) < 1e-6]
    return out


def resolve_value(dirs_rows, labels, value):
    """('hold'|'ramp', column): hold only if every design has hold-test values."""
    if value == "auto":
        has_hold = {l: any(d.get("F_hold_mg") is not None for d in dirs_rows[l]) for l in labels}
        value = "hold" if all(has_hold.values()) else "ramp"
        if not all(has_hold.values()):
            print(f"Using ramp F: no hold test for {[l for l, h in has_hold.items() if not h]} "
                  "(python disturb_force.py --labels ... --refine).")
    return value, ("F_hold_mg" if value == "hold" else "F_max_mg")


def collect_cells(dirs_rows, labels, group, key, cens_key):
    """{label: {(r, x, y): dict(group, F={dir: F}, cens={dir: bool})}} of the cells with all six directions."""
    out = {}
    for l in labels:
        cells = {}
        for d in dirs_rows[l]:
            if d["group"] not in GROUPS[group] or d.get(key) is None:
                continue
            c = cells.setdefault((d["r_mm"], d["x_mm"], d["y_mm"]), dict(group=d["group"], F={}, cens={}))
            c["F"][d["dir"]], c["cens"][d["dir"]] = float(d[key]), bool(d.get(cens_key))
        out[l] = {k: c for k, c in cells.items() if len(c["F"]) == len(DIRECTIONS)}
    return out


def plot_all(run, labels, group, value_opt, linear=False):
    """Every success cell of every design as one point: per direction and the weakest direction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_disturbance import GROUP_COLOR, INK, INK_2, MUTED, SURFACE, style
    style(plt)
    dirs_rows = {l: rows_of(run, l, "force_dirs.csv") for l in labels}
    value, key = resolve_value(dirs_rows, labels, value_opt)
    cens_key = "hold_censored" if value == "hold" else "censored"
    cells_of = collect_cells(dirs_rows, labels, group, key, cens_key)
    labels = [l for l in labels if cells_of[l]]
    if not labels:
        raise SystemExit(f"No {group} success cells in these designs.")
    cols = list(DIRECTIONS)+["F_min"]
    rng = np.random.default_rng(0)
    every = [v for l in labels for c in cells_of[l].values() for v in c["F"].values()]
    lo, hi = max(0.02, min(every)*0.8), max(every)*1.25
    kind = "env and tip" if group == "all" else group

    def points(ax, x0, cells, col, width=0.32):
        for g in ("env", "tip"):
            sel = [c for c in cells if c["group"] == g]
            if not sel:
                continue
            if col == "F_min":
                vals = [min(c["F"].values()) for c in sel]
                cens = [c["cens"][min(c["F"], key=c["F"].get)] for c in sel]
            else:
                vals = [c["F"][col] for c in sel]
                cens = [c["cens"][col] for c in sel]
            xs = x0+rng.uniform(-width, width, len(vals))
            v, cz = np.array(vals), np.array(cens, bool)
            ax.scatter(xs[~cz], v[~cz], s=14, color=GROUP_COLOR[g], alpha=0.75, edgecolors="none", zorder=3)
            if cz.any():  # Held up to the cap: true value higher.
                ax.scatter(xs[cz], v[cz], s=22, marker="^", facecolors="none", edgecolors=GROUP_COLOR[g],
                           linewidths=1.1, zorder=3)
        allv = [min(c["F"].values()) if col == "F_min" else c["F"][col] for c in cells]
        ax.plot([x0-0.38, x0+0.38], [np.median(allv)]*2, color=INK, lw=2, zorder=4, solid_capstyle="butt")

    def finish(ax):
        if not linear:
            ax.set_yscale("log")
            ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo <= t <= hi]
            ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
        ax.set_ylim(lo if not linear else 0, hi)
        ax.axhline(1.0, color=INK_2, lw=0.8, ls="--", zorder=1)

    # 1) Small multiples: one panel per design, all cells per direction and their F_min.
    n = len(labels)
    ncol = min(4, n)
    nrow = math.ceil(n/ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5*ncol, 3.0*nrow+0.9), sharey=True, squeeze=False)
    for ax, l in zip(axes.flat, labels):
        cells = list(cells_of[l].values())
        for j, col in enumerate(cols):
            points(ax, j, cells, col)
        ax.axvline(len(cols)-1.5, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(cols)), cols, fontsize=8)
        ne, nt = sum(c["group"] == "env" for c in cells), sum(c["group"] == "tip" for c in cells)
        ax.set_title(f"{l}  ({ne} env, {nt} tip)", loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
        finish(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"F [m g] ({value})")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], ls="", marker="o", color=GROUP_COLOR["env"], label="env cell"),
               plt.Line2D([], [], ls="", marker="o", color=GROUP_COLOR["tip"], label="tip cell"),
               plt.Line2D([], [], ls="", marker="^", markerfacecolor="none", color=INK_2, label="held to the cap (true F higher)"),
               plt.Line2D([], [], color=INK, lw=2, label="median of all cells"),
               plt.Line2D([], [], color=INK_2, lw=0.8, ls="--", label="1 m g")]
    fig.legend(handles=handles, loc="upper left", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"F of every {kind} success cell, per direction; F_min = weakest direction of the cell",
                 x=0.01, y=1.05, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(run)/"plots"/"disturbance"
    out.mkdir(parents=True, exist_ok=True)
    p1 = out/f"all_cells_dirs_{group}_{value}.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) One panel: F_min of every cell, designs side by side.
    fig, ax = plt.subplots(figsize=(1.05*n+2.2, 4.2))
    for j, l in enumerate(labels):
        points(ax, j, list(cells_of[l].values()), "F_min", width=0.3)
        best = max(min(c["F"].values()) for c in cells_of[l].values())
        ax.text(j, best*1.08 if not linear else best+0.05*hi, f"{best:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(n), [f"{l}\n(n={len(cells_of[l])})" for l in labels], fontsize=8)
    ax.grid(axis="x", visible=False)
    finish(ax)
    ax.set_ylabel(f"F_min of the cell [m g] ({value})")
    ax.set_title(f"Weakest-direction F of every {kind} success cell (number above = best cell)", loc="left", fontsize=10)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    p2 = out/f"all_cells_fmin_{group}_{value}.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}\nSaved: {p2}")


def peak(run, labels, group, value_opt, top_k):
    """Best cell per design over all its success cells; see the module docstring."""
    dirs_rows = {l: rows_of(run, l, "force_dirs.csv") for l in labels}
    value, key = resolve_value(dirs_rows, labels, value_opt)
    cens_key = "hold_censored" if value == "hold" else "censored"
    dirs = list(DIRECTIONS)
    rows, cells_of = [], collect_cells(dirs_rows, labels, group, key, cens_key)
    for l in labels:
        full = cells_of[l]
        if not full:
            rows.append(dict(label=l, cells=0))
            continue
        fmin = {k: min(c["F"].values()) for k, c in full.items()}
        best = max(fmin, key=fmin.get)
        bc = full[best]
        weak = min(bc["F"], key=bc["F"].get)
        ranked = sorted(fmin.values(), reverse=True)
        row = dict(label=l, cells=len(full), env=sum(c["group"] == "env" for c in full.values()),
                   tip=sum(c["group"] == "tip" for c in full.values()), best_F_min_mg=fmin[best],
                   best_censored=bc["cens"][weak], best_r_mm=best[0], best_x_mm=best[1], best_y_mm=best[2],
                   best_group=bc["group"], best_weakest_dir=weak,
                   top_k_mean_mg=float(np.mean(ranked[:top_k])), p90_mg=float(np.percentile(ranked, 90)),
                   median_mg=float(np.median(ranked)))
        row.update({f"best_cell_{d}_mg": bc["F"][d] for d in dirs})
        for d in dirs:
            k = max(full, key=lambda k: full[k]["F"][d])
            row.update({f"max_{d}_mg": full[k]["F"][d], f"max_{d}_censored": full[k]["cens"][d],
                        f"max_{d}_cell": f"r{k[0]:g} x{k[1]:g} y{k[2]:g} {full[k]['group']}"})
        rows.append(row)
    ok = [r for r in rows if r["cells"]]
    w = max(len(l) for l in labels)+2
    kind = "env and tip" if group == "all" else group
    print(f"\nBest grasp of every design over all its {kind} success cells ({value} F, m g; * = at the cap):")
    print(f"{'':{w}}{'cells':>11}{'max F_min':>11}  {'at (r, x, y)':<22}{'weakest':>8}{f'top{top_k} mean':>11}{'p90':>7}{'median':>8}")
    for r in rows:
        if not r["cells"]:
            print(f"{r['label']:<{w}}  no {group} success cells")
            continue
        at = f"r{r['best_r_mm']:g} x{r['best_x_mm']:g} y{r['best_y_mm']:g} {r['best_group']}"
        print(f"{r['label']:<{w}}{r['cells']:>4} ({r['env']:>2}/{r['tip']:>2}){r['best_F_min_mg']:>10.2f}{'*' if r['best_censored'] else ' '}"
              f" {at:<22}{r['best_weakest_dir']:>8}{r['top_k_mean_mg']:>11.2f}{r['p90_mg']:>7.2f}{r['median_mg']:>8.2f}")
    print(f"\nMax of each direction on its own (may come from different cells):")
    print(f"{'':{w}}" + "".join(f"{d:>8}" for d in dirs))
    for r in ok:
        print(f"{r['label']:<{w}}" + "".join(f"{r[f'max_{d}_mg']:>7.2f}{'*' if r[f'max_{d}_censored'] else ' '}" for d in dirs))

    def winner(col, cens_col=None):
        top = max(r[col] for r in ok)
        tied = [r["label"] for r in ok if r[col] >= top-1e-9]
        capped = cens_col and any(r[cens_col] for r in ok if r["label"] in tied)
        return top, tied, capped

    if ok:
        print("\nWinners:")
        for col, cens, text in (("best_F_min_mg", "best_censored", "strongest grasp in every direction (max F_min)"),
                                ("top_k_mean_mg", None, f"mean of the top {top_k} cells"),
                                ("p90_mg", None, "90th percentile of F_min")):
            top, tied, capped = winner(col, cens)
            print(f"  {text}: {', '.join(tied)} = {top:.2f} m g" + (" (at the cap: true value higher, tie undecided)"
                                                                     if capped else ""))
        for d in dirs:
            top, tied, capped = winner(f"max_{d}_mg", f"max_{d}_censored")
            print(f"  max {d}: {', '.join(tied)} = {top:.2f} m g" + (" (at the cap)" if capped else ""))
        if any(r["best_censored"] or any(r[f"max_{d}_censored"] for d in dirs) for r in ok):
            print("  Values at the cap cannot be ranked: rerun those designs with a higher cap, e.g. "
                  "python disturb_force.py --labels ... --max-mg 30 --refine")
    out = Path(run)/"plots"/"disturbance"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    path = out/f"peak_{group}_{value}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows([{k: r.get(k) for k in keys} for r in rows])
    print(f"\nSaved: {path}")
    for fig_path in plot_peak(out, [r for r in rows if r.get("cells")], value, group):
        print(f"Saved: {fig_path}")


def read_peak_csv(path):
    """Rows of a peak_<group>_<value>.csv with numbers and flags restored."""
    rows = []
    for r in read_rows(path):
        if not r.get("cells"):
            continue
        for d in DIRECTIONS:
            r[f"max_{d}_censored"] = r.get(f"max_{d}_censored") in (True, "True")
        rows.append(r)
    return rows


def plot_peak(out, rows, value, group):
    """Max F of every direction per design: one panel per design, and all designs per direction."""
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_disturbance import INK, INK_2, SERIES, SURFACE, style
    style(plt)
    dirs = list(DIRECTIONS)
    labels = [r["label"] for r in rows]
    top = max(float(r[f"max_{d}_mg"]) for r in rows for d in dirs)
    ymax = top*1.22
    kind = "env and tip" if group == "all" else group
    cens_note = "hatched = held up to the ramp cap, true value higher"

    # 1) One panel per design: the six directional maxima, with the radius of the cell they came from.
    n = len(rows)
    ncol = min(4, n)
    nrow = math.ceil(n/ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4*ncol, 2.9*nrow+0.8), sharey=True, squeeze=False)
    for ax, r in zip(axes.flat, rows):
        vals = [float(r[f"max_{d}_mg"]) for d in dirs]
        cens = [bool(r[f"max_{d}_censored"]) for d in dirs]
        bars = ax.bar(range(len(dirs)), vals, width=0.68, color=SERIES[0], edgecolor=SURFACE, linewidth=0)
        for b, c in zip(bars, cens):
            if c:
                b.set_hatch("////")
                b.set_edgecolor(SURFACE)
        for j, (v, c, d) in enumerate(zip(vals, cens, dirs)):
            cell = str(r[f"max_{d}_cell"]).split()
            ax.text(j, v+0.012*ymax, f"{'≥' if c else ''}{v:.1f}\n{cell[0] if cell else ''}", ha="center", va="bottom",
                    fontsize=6.5, color=INK, linespacing=1.0)
        ax.axhline(1.0, color=INK_2, lw=0.8, ls="--")
        ax.set_xticks(range(len(dirs)), dirs)
        ax.set_ylim(0, ymax)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{r['label']}  ({int(float(r['cells']))} cells)", loc="left", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"max F over cells [m g] ({value})")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"Largest F of each direction over all {kind} success cells (label: value, radius of that cell); "
                 f"{cens_note}; dashed: 1 m g", x=0.01, y=1.03, ha="left", fontsize=9.5, color=INK_2)
    fig.tight_layout()
    p1 = Path(out)/f"peak_dirs_{group}_{value}.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) All designs side by side, grouped by direction.
    w = 0.84/n
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25*len(dirs)*max(1, n/4)), 4.2))
    for k, r in enumerate(rows):
        xs = np.arange(len(dirs))-0.42+w*(k+0.5)
        vals = [float(r[f"max_{d}_mg"]) for d in dirs]
        bars = ax.bar(xs, vals, width=w*0.9, color=SERIES[k % len(SERIES)], label=r["label"], linewidth=0)
        for b, d in zip(bars, dirs):
            if r[f"max_{d}_censored"]:
                b.set_hatch("////")
                b.set_edgecolor(SURFACE)
    ax.axhline(1.0, color=INK_2, lw=0.8, ls="--")
    ax.set_xticks(range(len(dirs)), dirs)
    ax.set_ylim(0, ymax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"max F over cells [m g] ({value})")
    ax.set_title(f"Largest F of each direction, designs side by side ({kind} cells; {cens_note})", loc="left", fontsize=9.5)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SERIES[k % len(SERIES)], label=r["label"]) for k, r in enumerate(rows)]
    handles.append(Patch(facecolor=INK_2, hatch="////", edgecolor=SURFACE, label="held to the cap"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    p2 = Path(out)/f"peak_by_direction_{group}_{value}.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=ROOT/"runs"/"rb5_cmaes_final")
    p.add_argument("--labels", nargs="+", help="Designs to compare (not needed with --peak-csv)")
    p.add_argument("--coverage", action="store_true", help="Only print which radii every design grasps")
    p.add_argument("--r", type=float, nargs="+", help="Object radius/radii [mm] to compare at")
    p.add_argument("--group", choices=sorted(GROUPS), help="env (default with --r), tip or all (default with --peak)")
    p.add_argument("--peak", action="store_true", help="Best cell of every design over all its success cells")
    p.add_argument("--plot-all", action="store_true", help="Graphs of the F of every success cell, per design")
    p.add_argument("--peak-csv", type=Path, help="Only redraw the --peak graphs from this peak_<group>_<value>.csv")
    p.add_argument("--linear", action="store_true", help="Linear F axis for --plot-all (default log)")
    p.add_argument("--top-k", type=int, default=3, help="Cells averaged for the robust peak (default 3)")
    p.add_argument("--x", type=float, help="Only cells at this x [mm] (0 = centered in the jaw)")
    p.add_argument("--common", action="store_true", help="Only (x, y) cells that every design grasps")
    p.add_argument("--value", choices=("auto", "ramp", "hold"), default="auto")
    p.add_argument("--tag", help="Name of the output files (default from the options)")
    args = p.parse_args()
    run = args.run.resolve()
    if args.peak_csv:
        stem = args.peak_csv.stem.split("_")  # peak_<group>_<value>
        group, value = (stem[1], stem[2]) if len(stem) == 3 else ("all", "ramp")
        for path in plot_peak(args.peak_csv.resolve().parent, read_peak_csv(args.peak_csv), value, group):
            print(f"Saved: {path}")
        return
    if not args.labels:
        p.error("--labels is required")
    labels = list(dict.fromkeys(args.labels))
    group = args.group or ("all" if args.peak or args.plot_all else "env")
    if args.peak or args.plot_all:
        if args.peak:
            peak(run, labels, group, args.value, args.top_k)
        if args.plot_all:
            plot_all(run, labels, group, args.value, args.linear)
        return
    coverage(run, labels)
    if args.coverage or not args.r:
        if not args.r:
            print("\nAdd --r <radius> for the per-direction table.")
        return
    dirs_rows = {l: rows_of(run, l, "force_dirs.csv") for l in labels}
    groups = GROUPS[group]
    value, key = resolve_value(dirs_rows, labels, args.value)
    out_rows, grid, notes = [], {}, []
    for r in args.r:
        sel = {l: select(dirs_rows[l], r, groups, args.x) for l in labels}
        if args.common:
            cells = None
            for l in labels:
                s = {(d["x_mm"], d["y_mm"]) for d in sel[l]}
                cells = s if cells is None else cells & s
            notes.append(f"r={r:g}: {len(cells)} common cells {sorted(cells)}")
            sel = {l: [d for d in sel[l] if (d["x_mm"], d["y_mm"]) in cells] for l in labels}
        for l in labels:
            n_cells = len({(d["x_mm"], d["y_mm"]) for d in sel[l]})
            for dname in DIRECTIONS:
                v = [d[key] for d in sel[l] if d["dir"] == dname and d.get(key) is not None]
                cens = sum(bool(d["censored"]) for d in sel[l] if d["dir"] == dname)
                med = float(np.median(v)) if v else None
                grid[(r, l, dname)] = (med, n_cells)
                out_rows.append(dict(r_mm=r, label=l, group=group, dir=dname, value=value, cells=n_cells,
                                     median_mg=med, min_mg=min(v) if v else None, max_mg=max(v) if v else None,
                                     censored_at_cap=cens))
    for n in notes:
        print(n)
    # Console table: one block per radius.
    for r in args.r:
        print(f"\nr = {r:g} mm, {group} cells{'' if args.x is None else f', x = {args.x:g} mm'}"
              f"{', common cells only' if args.common else ''}: median F per direction [m g] ({value})")
        w = max(len(l) for l in labels)+2
        print(" "*w + f"{'cells':>6}" + "".join(f"{d:>8}" for d in DIRECTIONS) + f"{'weakest':>10}")
        for l in labels:
            vals = [grid[(r, l, d)][0] for d in DIRECTIONS]
            n = grid[(r, l, "+x")][1]
            weakest = min(((v, d) for v, d in zip(vals, DIRECTIONS) if v is not None), default=(None, "-"))
            print(f"{l:<{w}}{n:>6}" + "".join(f"{v:>8.2f}" if v is not None else f"{'-':>8}" for v in vals)
                  + f"{weakest[1]:>10}")
    tag = args.tag or (f"{group}_r{'_'.join(f'{r:g}' for r in args.r)}"
                       + ("" if args.x is None else f"_x{args.x:g}") + ("_common" if args.common else "") + f"_{value}")
    out = run/"plots"/"disturbance"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out/f"compare_{tag}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved: {out/f'compare_{tag}.csv'}")
    path = plot(out/f"compare_{tag}.png", grid, labels, args.r, value, group)
    print(f"Saved: {path}")


def plot(path, grid, labels, radii, value, group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from plot_disturbance import EMPTY_CELL, INK, INK_2, SEQUENTIAL, SURFACE, style
    style(plt)
    dirs = list(DIRECTIONS)
    vals = [v for (v, _) in grid.values() if v is not None]
    vmax = min(5.0, max(1.0, math.ceil(np.percentile(vals, 90)*2)/2)) if vals else 1.0
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    fig, axes = plt.subplots(1, len(radii), figsize=(5.2*len(radii)+1.6, 0.42*len(labels)+1.7), squeeze=False)
    for ax, r in zip(axes[0], radii):
        ax.grid(False)
        for i, l in enumerate(labels):
            for j, d in enumerate(dirs):
                v, n = grid[(r, l, d)]
                if v is None:
                    ax.add_patch(plt.Rectangle((j+0.04, i+0.04), 0.92, 0.92, color=EMPTY_CELL, lw=0))
                    continue
                u = min(v, vmax)/vmax
                ax.add_patch(plt.Rectangle((j+0.04, i+0.04), 0.92, 0.92, color=cmap(u), lw=0))
                ax.text(j+0.5, i+0.5, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if u > 0.55 else INK)
        ax.set_xlim(0, len(dirs))
        ax.set_ylim(len(labels), 0)
        ax.set_xticks([j+0.5 for j in range(len(dirs))], dirs)
        ax.set_yticks([i+0.5 for i in range(len(labels))],
                      [f"{l} (n={grid[(r, l, '+x')][1]})" for l in labels] if ax is axes[0][0]
                      else [f"n={grid[(r, l, '+x')][1]}" for l in labels])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"r = {r:g} mm ({group} cells)", loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cb.set_label(f"median F [m g] ({value}; capped at {vmax:g})")
    fig.suptitle("Median F per direction at a fixed object radius; gray = no such cells", x=0.01, y=1.08,
                 ha="left", fontsize=10, color=INK_2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()