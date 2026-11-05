"""Compare two designs mounted on the RB5: workspace maps, gained/lost cells, metrics.

Both results must come from the same optimize_rb5_cmaes.py run folder, so they share
config_used.json and the grid:
  * the final best (default for --b): best_design_fine.json + final/<label>_workspace.csv
  * a final candidate or the run's baseline: final/<label>.json, e.g. candidate_02
  * a design evaluated with eval_design_rb5.py: references/<label>.json, e.g. design001

  python compare_designs_rb5.py --run runs/rb5_cmaes_final                 # design001 vs best
  python compare_designs_rb5.py --run runs/rb5_cmaes_final --a design001 --b candidate_02

Writes to <run>/plots/:
  compare_<A>_vs_<B>_maps.png     category map per radius, one row per design
  compare_<A>_vs_<B>_diff.png     cells gained / lost by B against A (both hands drawn)
  compare_<A>_vs_<B>_summary.png  totals, success per radius, failure reasons, grip force
  compare_<A>_vs_<B>.csv          per-radius counts behind the figures
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from gripper_core import read_json
from rb5_eval import summarize
from eval_design_rb5 import cell_key, patch_aware_outline, read_rows

COLORS = ("#e08214", "#2166ac")  # A, B


def load_result(run, name):
    """dict(label, design, summary, rows, template) for one design of the run."""
    run_template = run/"source_design168.xml"
    if name in (None, "best"):
        meta = read_json(run/"best_design_fine.json")
        csv_path = run/"final"/f"{meta['label']}_workspace.csv"
        csv_path = csv_path if csv_path.is_file() else run/"final"/"best_workspace.csv"
        template = run_template
    elif (run/"references"/f"{name}.json").is_file():
        meta = read_json(run/"references"/f"{name}.json")
        csv_path = run/"references"/f"{name}_workspace.csv"
        template = run/"references"/meta.get("template_file", f"{name}_source.xml")
    elif (run/"final"/f"{name}.json").is_file():
        meta = read_json(run/"final"/f"{name}.json")
        csv_path = run/"final"/f"{name}_workspace.csv"
        template = run_template
    else:
        known = sorted({p.stem for p in (run/"final").glob("*.json") if p.stem != "comparison"}
                       | {p.stem for p in (run/"references").glob("*.json")})
        hint = (f"Evaluate it first: python eval_design_rb5.py --run {run} --xml {name}.xml"
                if name.startswith("design") else "")
        raise SystemExit(f"No result '{name}' in {run}. Available: best, {', '.join(known)}. {hint}")
    for path in (csv_path, template):
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
    return dict(label=meta["label"], design=meta["design"], summary=meta["summary"], patch=meta.get("patch"),
                rows=read_rows(csv_path), template=template.read_text(encoding="utf-8-sig"))


def common_cells(a, b, cfg):
    """Restrict both results to the cells they share; re-summarize if anything was dropped."""
    ka = {cell_key(r["r_mm"], r["x_mm"], r["y_mm"]) for r in a["rows"]}
    kb = {cell_key(r["r_mm"], r["x_mm"], r["y_mm"]) for r in b["rows"]}
    both = ka & kb
    if not both:
        raise SystemExit("The two results share no cells. Were they evaluated on different grids?")
    if ka != kb:
        print(f"WARNING: different grids ({len(ka)} vs {len(kb)} cells). Comparing the "
              f"{len(both)} shared cells only; J and counts are recomputed on those.")
        for e in (a, b):
            e["rows"] = [r for r in e["rows"] if cell_key(r["r_mm"], r["x_mm"], r["y_mm"]) in both]
            e["summary"] = summarize(e["rows"], cfg)


def counts(rows, r=None):
    sub = [x for x in rows if r is None or x["r_mm"] == r]
    env = sum(bool(x["success"]) and x["contact_type"] == "env" for x in sub)
    tip = sum(bool(x["success"]) and x["contact_type"] == "tip" for x in sub)
    return env, tip


def per_radius_table(a, b):
    base = {cell_key(r["r_mm"], r["x_mm"], r["y_mm"]): bool(r["success"]) for r in a["rows"]}
    table = []
    for rad in sorted({r["r_mm"] for r in a["rows"]}):
        ea, ta = counts(a["rows"], rad)
        eb, tb = counts(b["rows"], rad)
        sub = [r for r in b["rows"] if r["r_mm"] == rad]
        gained = sum(bool(r["success"]) and not base[cell_key(r["r_mm"], r["x_mm"], r["y_mm"])] for r in sub)
        lost = sum(not r["success"] and base[cell_key(r["r_mm"], r["x_mm"], r["y_mm"])] for r in sub)
        table.append(dict(r_mm=rad, cells=len(sub), **{f"{a['label']}_env": ea, f"{a['label']}_tip": ta,
                          f"{a['label']}_success": ea+ta, f"{b['label']}_env": eb, f"{b['label']}_tip": tb,
                          f"{b['label']}_success": eb+tb, f"gained_by_{b['label']}": gained,
                          f"lost_by_{b['label']}": lost}))
    return table


def plot_maps(path, entries, cfg):
    """One row per design, one column per radius, same categories as plot_workspace.py."""
    import matplotlib.pyplot as plt
    import plot_workspace as pw
    radii = sorted({row["r_mm"] for row in entries[0]["rows"]})
    shapes = [pw.hand_outline(e, cfg) for e in entries]
    lim = pw.limits(entries[0]["rows"], shapes[0]+shapes[1])
    aspect = (lim[1][1]-lim[1][0])/(lim[0][1]-lim[0][0])
    size = 2.7
    fig, axes = plt.subplots(len(entries), len(radii), squeeze=False,
                             figsize=(size*len(radii)+0.6, (size*aspect+0.45)*len(entries)+1.4))
    codes = {k: i for i, (k, _, _) in enumerate(pw.CATEGORIES)}
    for i, (e, sh) in enumerate(zip(entries, shapes)):
        for j, r in enumerate(radii):
            ax = axes[i, j]
            pw.category_map(ax, e["rows"], r, lambda row: codes[pw.category(row)], pw.CATEGORIES)
            pw.draw_hand(ax, sh)
            env, tip = counts(e["rows"], r)
            ax.set_title(f"{e['label']}  r={r:g}\nenv {env}, tip {tip}", fontsize=8.5)
            pw.finish_axes(ax, lim)
            ax.tick_params(labelsize=7)
            if j:
                ax.set_ylabel("")
            if i < len(entries)-1:
                ax.set_xlabel("")
    fig.suptitle("\n".join(pw.design_text(e["label"], e["design"], e["summary"]) for e in entries)
                 + "\nx<0 mirrored (symmetric hand); hand drawn open", fontsize=9.5)
    pw.legend(fig, pw.CATEGORIES)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.subplots_adjust(hspace=0.42, bottom=0.15)  # Room for two-line titles and the legend.
    fig.savefig(path, dpi=170)
    return fig


def plot_summary(path, a, b, table):
    import matplotlib.pyplot as plt
    entries = (a, b)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # (1) Totals and J.
    ax = axes[0, 0]
    keys = ("env", "tip", "success")
    pos = np.arange(len(keys))
    for k, e in enumerate(entries):
        vals = [e["summary"][key] for key in keys]
        bars = ax.bar(pos+(k-0.5)*0.38, vals, 0.38, color=COLORS[k],
                      label=f"{e['label']}  J={e['summary']['J']:.2f}")
        ax.bar_label(bars, fontsize=8, padding=1)
    ax.set_xticks(pos, ["env cells", "tip cells", "success (env+tip)"])
    ax.set_ylabel(f"cells (of {a['summary']['cells']} simulated, x >= 0)")
    ax.set_title("Workspace totals")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (2) Success per radius: solid env, hatched tip.
    ax = axes[0, 1]
    radii = [row["r_mm"] for row in table]
    pos = np.arange(len(radii))
    for k, e in enumerate(entries):
        env = [row[f"{e['label']}_env"] for row in table]
        tip = [row[f"{e['label']}_tip"] for row in table]
        x = pos+(k-0.5)*0.38
        ax.bar(x, env, 0.38, color=COLORS[k], label=e["label"])
        ax.bar(x, tip, 0.38, bottom=env, color=COLORS[k], alpha=0.4, hatch="///", edgecolor="white")
    ax.set_xticks(pos, [f"{r:g}" for r in radii])
    ax.set_xlabel("object radius r [mm]")
    ax.set_ylabel("successful cells (solid env, hatched tip)")
    ax.set_title("Success per object radius")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (3) Why cells fail.
    ax = axes[1, 0]
    fails = [Counter(r["reason"] for r in e["rows"] if not r["success"]) for e in entries]
    reasons = sorted(set(fails[0]) | set(fails[1]), key=lambda k: -(fails[0][k]+fails[1][k]))
    pos = np.arange(len(reasons))
    for k, e in enumerate(entries):
        bars = ax.barh(pos+(k-0.5)*0.38, [fails[k][r] for r in reasons], 0.38, color=COLORS[k], label=e["label"])
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_yticks(pos, reasons, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("failed cells")
    ax.set_title("Failure reasons (rb5_eval reason column)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # (4) Grip force on the cells each design holds.
    ax = axes[1, 1]
    data, labels = [], []
    for e in entries:
        ok = [r for r in e["rows"] if r["success"] and r.get("total_normal_N") is not None]
        data.append([float(r["total_normal_N"]) for r in ok])
        mf = np.mean([float(r["m_F"]) for r in ok]) if ok else float("nan")
        labels.append(f"{e['label']}\n(n={len(ok)}, mean m_F={mf:.2f})")
    if any(data):
        kw = {"tick_labels": labels} if _mpl_has_tick_labels() else {"labels": labels}
        parts = ax.boxplot([d if d else [np.nan] for d in data], patch_artist=True, widths=0.5,
                           medianprops=dict(color="black"), **kw)
        for patch, color in zip(parts["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for k, d in enumerate(data):
            if d:
                jitter = np.random.default_rng(k).uniform(-0.12, 0.12, len(d))
                ax.scatter(np.full(len(d), k+1)+jitter, d, s=6, color="black", alpha=0.35, zorder=3)
        ax.set_ylabel("total link normal force in lift hold [N]")
    else:
        ax.text(0.5, 0.5, "no successful cells", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Grip force on successful cells")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{a['label']} vs {b['label']} on the RB5 (same run config and grid)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    return fig


def _mpl_has_tick_labels():
    import matplotlib
    major, minor = (int(v) for v in matplotlib.__version__.split(".")[:2])
    return (major, minor) >= (3, 9)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Output folder of optimize_rb5_cmaes.py")
    p.add_argument("--a", default="design001", help="Reference design (default design001)")
    p.add_argument("--b", default="best", help="Design compared against it (default: the run's final best)")
    p.add_argument("--show", action="store_true", help="Also open the figures in windows")
    args = p.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_workspace as pw
    patch_aware_outline(pw)  # design001 (stage0 XML) is drawn with its own distal angle.

    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    a, b = load_result(run, args.a), load_result(run, args.b)
    if a["label"] == b["label"]:
        raise SystemExit("--a and --b are the same design.")
    common_cells(a, b, cfg)
    table = per_radius_table(a, b)

    out = run/"plots"
    out.mkdir(exist_ok=True)
    stem = f"compare_{a['label']}_vs_{b['label']}"
    written = [out/f"{stem}_maps.png", out/f"{stem}_diff.png", out/f"{stem}_summary.png", out/f"{stem}.csv"]
    plot_maps(written[0], [a, b], cfg)
    pw.plot_diff(written[1], b, a, cfg)  # Green: only b succeeds, red: only a succeeds.
    plot_summary(written[2], a, b, table)
    with written[3].open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    na, nb = a["label"], b["label"]
    print(f"{'':10}{na:>14}{nb:>14}")
    for key in ("J", "success", "env", "tip", "rb5_pass", "prefiltered", "cells"):
        va, vb = a["summary"][key], b["summary"][key]
        fmt = "{:>14.3f}" if isinstance(va, float) and not float(va).is_integer() else "{:>14g}"
        print(f"{key:10}{fmt.format(va)}{fmt.format(vb)}")
    wa, wb = max(len(na), 7)+2, max(len(nb), 7)+2
    print(f"\nenv/tip cells per radius; gained/lost = cells {nb} holds and {na} does not, and vice versa")
    print(f"{'r [mm]':>7}{na:>{wa}}{nb:>{wb}}{'gained':>8}{'lost':>6}")
    for row in table:
        ca = f"{row[na+'_env']}/{row[na+'_tip']}"
        cb = f"{row[nb+'_env']}/{row[nb+'_tip']}"
        print(f"{row['r_mm']:>7g}{ca:>{wa}}{cb:>{wb}}{row['gained_by_'+nb]:>8}{row['lost_by_'+nb]:>6}")
    reach = {e["label"]: max((r["r_mm"] for r in e["rows"] if r["success"]), default=math.nan) for e in (a, b)}
    print("\nlargest graspable r [mm]: " + ", ".join(f"{k} {v:g}" for k, v in reach.items()))
    for path in written:
        print("Saved:", path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
