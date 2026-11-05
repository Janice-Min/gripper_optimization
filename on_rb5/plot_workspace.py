"""Graspable workspace plots for an optimize_rb5_cmaes.py run.

Writes to <run>/plots/:
  workspace_<label>.png       one panel per object radius: env / tip / failed / not
                              attempted, with the open hand drawn in the hand frame
  diff_<label>_vs_<ref>.png   cells gained / lost against the reference (default
                              design168; --baseline design001 after add_reference.py)
  compare_designs.png         successful cells per radius for every final candidate

Each cell is drawn at the object center (x, y+r) of the hand frame (palm face at y=0,
fingers toward +y). Only x >= 0 is simulated; x < 0 is mirrored (symmetric hand).

  python plot_workspace.py --run runs/rb5_cmaes
  python plot_workspace.py --run runs/rb5_cmaes --label candidate_01
  python plot_workspace.py --run runs/rb5_cmaes --baseline design001
  python plot_workspace.py --run runs/rb5_cmaes --show
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import mujoco

from gripper_core import LINKS, build_xml, read_json
from rb5_eval import hand_config, run_result

BLOCKED = {"approach_overlap", "initial_palm_clearance", "initial_overlap", "approach_collision"}
CATEGORIES = [("env", "env: all 4 finger links", "#1a9850"),
              ("tip", "tip: partial contact, held", "#fdd835"),
              ("fail", "failed", "#bdbdbd"),
              ("blocked", "not attempted: open hand hits object", "#ffffff")]


def diff_palette(label, base_label):
    return [("both", "success in both", "#4575b4"),
            ("gained", f"success only with {label}", "#1a9850"),
            ("lost", f"success only with {base_label}", "#d73027"),
            ("neither", "neither", "#ffffff")]


def category(row):
    if row["success"]:
        return row["contact_type"] if row["contact_type"] in ("env", "tip") else "env"
    if row["prefiltered"] or row["reason"] in BLOCKED:
        return "blocked"
    return "fail"


def edges(centers):
    c = np.asarray(sorted(set(centers)), dtype=float)
    if len(c) == 1:
        return np.array([c[0]-5, c[0]+5])
    mid = (c[:-1]+c[1:])/2
    return np.r_[2*c[0]-mid[0], mid, 2*c[-1]-mid[-1]]


def mirrored_grid(rows, r, value):
    """(x_edges, y_edges, Z) for one radius; Z holds value(row), mirrored to x < 0."""
    sub = {(row["x_mm"], row["y_mm"]): row for row in rows if row["r_mm"] == r}
    xs = sorted({x for x, _ in sub})
    ys = sorted({y for _, y in sub})
    full_x = [-x for x in reversed(xs) if x > 0] + xs
    z = np.full((len(ys), len(full_x)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(full_x):
            row = sub.get((abs(x), y))
            if row is not None:
                z[j, i] = value(row)
    return edges(full_x), edges([y+r for y in ys]), z


def capsule_outline(p0, p1, radius, n=14):
    d = p1-p0
    a = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-12 else math.pi/2
    t1 = np.linspace(a-math.pi/2, a+math.pi/2, n)
    t0 = np.linspace(a+math.pi/2, a+3*math.pi/2, n)
    return np.vstack([p1+radius*np.c_[np.cos(t1), np.sin(t1)], p0+radius*np.c_[np.cos(t0), np.sin(t0)]])


def hand_outline(entry, cfg):
    """Open-hand link outlines (palm and four finger links) in hand-frame mm, built from
    the design's own template (design168 for optimizer results)."""
    xml = build_xml(entry["template"], entry["design"], hand_config(cfg), radius_mm=20)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shapes = []
    for name in LINKS:
        g = model.geom(name).id
        c, rot, size = 1000*data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), 1000*model.geom_size[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = rot[:, 2]*size[1]
            shapes.append(capsule_outline((c-axis)[:2], (c+axis)[:2], size[0]))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = [c+rot[:, 0]*sx*size[0]+rot[:, 1]*sy*size[1] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            shapes.append(np.array(corners)[:, :2])
    return shapes


def draw_hand(ax, shapes, dashed=False):
    from matplotlib.patches import Polygon
    for pts in shapes:
        ax.add_patch(Polygon(pts, closed=True, facecolor="none" if dashed else "#444444",
                             alpha=1.0 if dashed else 0.55, edgecolor="black",
                             linestyle="--" if dashed else "-", linewidth=0.8, zorder=3))


def limits(rows, shapes):
    """Common axis limits for every radius panel, so panels share one scale."""
    xe, ye = [], []
    for r in sorted({row["r_mm"] for row in rows}):
        x, y, _ = mirrored_grid(rows, r, lambda row: 0)
        xe += [x[0], x[-1]]
        ye += [y[0], y[-1]]
    pts = np.vstack(shapes)
    return ((min(min(xe), pts[:, 0].min())-5, max(max(xe), pts[:, 0].max())+5),
            (min(min(ye), pts[:, 1].min())-5, max(max(ye), pts[:, 1].max())+5))


def panel_layout(n, lim):
    import matplotlib.pyplot as plt
    cols = min(4, n)
    rows = math.ceil(n/cols)
    aspect = (lim[1][1]-lim[1][0])/(lim[0][1]-lim[0][0])
    fig, axes = plt.subplots(rows, cols, figsize=(3.3*cols, (3.3*aspect+0.7)*rows+1.0), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()[:n]


def category_map(ax, rows, r, codes, palette):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    xe, ye, z = mirrored_grid(rows, r, codes)
    cmap = ListedColormap([c for _, _, c in palette])
    norm = BoundaryNorm(np.arange(len(palette)+1)-0.5, cmap.N)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(z), cmap=cmap, norm=norm,
                  edgecolors="#e0e0e0", linewidth=0.3, zorder=1)


def finish_axes(ax, lim):
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5, zorder=2)
    ax.axvline(0, color="#999999", linewidth=0.4, linestyle=":", zorder=2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("object center y+r [mm]")


def legend(fig, palette):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#888888", label=t) for _, t, c in palette],
               loc="lower center", ncol=len(palette), frameon=False, fontsize=9)


def design_text(label, design, summary=None):
    text = (f"{label}: L={design['length_mm']:.1f} mm, Lp/L={design['lp_ratio']:.3f}, "
            f"G={design['gap_mm']:.1f} mm, $\\theta_p$={design['theta_p_deg']:.1f}°")
    if summary:
        text += f"   |   J={summary['J']:.2f} (env {summary['env']}, tip {summary['tip']})"
    return text


def plot_workspace(path, rows, label, design, summary, shapes):
    import matplotlib.pyplot as plt
    codes = {k: i for i, (k, _, _) in enumerate(CATEGORIES)}
    radii = sorted({row["r_mm"] for row in rows})
    lim = limits(rows, shapes)
    fig, axes = panel_layout(len(radii), lim)
    best = {}
    for row in rows:
        if row["success"] and row["contact_type"] == "env":
            if row["r_mm"] not in best or row["m_F"] > best[row["r_mm"]]["m_F"]:
                best[row["r_mm"]] = row
    for ax, r in zip(axes, radii):
        category_map(ax, rows, r, lambda row: codes[category(row)], CATEGORIES)
        draw_hand(ax, shapes)
        if r in best:  # Object size for scale, at the strongest env grasp.
            b = best[r]
            ax.add_patch(plt.Circle((b["x_mm"], b["y_mm"]+r), r, fill=False, color="black",
                                    linestyle="--", linewidth=0.9, zorder=4))
        sub = [row for row in rows if row["r_mm"] == r]
        env = sum(row["success"] and row["contact_type"] == "env" for row in sub)
        tip = sum(row["success"] and row["contact_type"] == "tip" for row in sub)
        ax.set_title(f"r = {r:g} mm   env {env}, tip {tip}", fontsize=10)
        finish_axes(ax, lim)
    fig.suptitle(design_text(label, design, summary)+"\nx<0 mirrored; dashed circle: object at the strongest env grasp",
                 fontsize=10)
    legend(fig, CATEGORIES)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=200)
    return fig


def plot_diff(path, cur, ref, cfg):
    """Cells gained / lost by `cur` against the reference design `ref` (both entries)."""
    palette = diff_palette(cur["label"], ref["label"])
    codes = {k: i for i, (k, _, _) in enumerate(palette)}
    base = {(r["r_mm"], r["x_mm"], r["y_mm"]): r["success"] for r in ref["rows"]}
    rows = cur["rows"]

    def value(row):
        b = base.get((row["r_mm"], row["x_mm"], row["y_mm"]), False)
        key = {(True, True): "both", (True, False): "gained", (False, True): "lost"}.get((row["success"], b), "neither")
        return codes[key]

    shapes, ref_shapes = hand_outline(cur, cfg), hand_outline(ref, cfg)
    radii = sorted({row["r_mm"] for row in rows})
    lim = limits(rows, shapes+ref_shapes)
    fig, axes = panel_layout(len(radii), lim)
    for ax, r in zip(axes, radii):
        category_map(ax, rows, r, value, palette)
        draw_hand(ax, shapes)
        draw_hand(ax, ref_shapes, dashed=True)
        sub = [row for row in rows if row["r_mm"] == r]
        gained = sum(value(row) == codes["gained"] for row in sub)
        lost = sum(value(row) == codes["lost"] for row in sub)
        ax.set_title(f"r = {r:g} mm   +{gained} / -{lost}", fontsize=10)
        finish_axes(ax, lim)
    sc, sr = cur["summary"], ref["summary"]
    fig.suptitle(f"{cur['label']} (filled hand, J={sc['J']:.2f}, {sc['success']} cells) vs "
                 f"{ref['label']} (dashed hand, J={sr['J']:.2f}, {sr['success']} cells)\n"
                 "env and tip both count as success", fontsize=10)
    legend(fig, palette)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=200)
    return fig


def plot_compare(path, entries):
    import matplotlib.pyplot as plt
    radii = sorted({row["r_mm"] for e in entries for row in e["rows"]})
    width = 0.8/len(entries)
    fig, ax = plt.subplots(figsize=(max(7, 1.1*len(radii)+3), 4.5))
    colors = plt.get_cmap("tab10").colors
    for k, e in enumerate(entries):
        env = [sum(r["success"] and r["contact_type"] == "env" and r["r_mm"] == rad for r in e["rows"]) for rad in radii]
        tip = [sum(r["success"] and r["contact_type"] == "tip" and r["r_mm"] == rad for r in e["rows"]) for rad in radii]
        pos = np.arange(len(radii))-0.4+width*(k+0.5)
        s = e["summary"]
        ax.bar(pos, env, width, color=colors[k % 10], label=f"{e['label']}  J={s['J']:.2f} (env {s['env']}, tip {s['tip']})")
        ax.bar(pos, tip, width, bottom=env, color=colors[k % 10], alpha=0.45, hatch="///", edgecolor="white")
    ax.set_xticks(np.arange(len(radii)), [f"{r:g}" for r in radii])
    ax.set_xlabel("object radius r [mm]")
    ax.set_ylabel("successful cells (solid: env, hatched: tip)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Output folder of optimize_rb5_cmaes.py")
    p.add_argument("--label", help="Design to plot (default: the final best)")
    p.add_argument("--baseline", help="Reference for the gain/loss plot, e.g. design001 added with "
                                      "add_reference.py (default: baseline = design168)")
    p.add_argument("--show", action="store_true", help="Also open the figures in windows")
    args = p.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    final = run/"final"
    entries = {}

    def load(name):
        label, design, summary, rows, template = run_result(run, name)
        entries[label] = dict(label=label, design=design, summary=summary, rows=rows, template=template)
        return entries[label]

    chosen = load(args.label)
    if (final/"comparison.json").is_file():
        names = [r["label"] for r in read_json(final/"comparison.json")["results"]]
    else:
        names = [chosen["label"]] + (["baseline"] if (run/"baseline_workspace.csv").is_file() else [])
    for name in names:
        if name not in entries:
            load(name)
    ref_name = args.baseline or "baseline"
    if args.baseline and ref_name not in entries:
        raise SystemExit(f"No result '{ref_name}' in {run}. Available: {names}. "
                         f"Add it first: python add_reference.py --run {args.run} --xml {ref_name}.xml")
    out = run/"plots"
    out.mkdir(exist_ok=True)
    label = chosen["label"]
    written = [out/f"workspace_{label}.png"]
    plot_workspace(written[-1], chosen["rows"], label, chosen["design"], chosen["summary"],
                   hand_outline(chosen, cfg))
    if ref_name in entries and ref_name != label:
        written.append(out/f"diff_{label}_vs_{ref_name}.png")
        plot_diff(written[-1], chosen, entries[ref_name], cfg)
    if len(names) > 1:
        written.append(out/"compare_designs.png")
        plot_compare(written[-1], [entries[n] for n in names])
    for path in written:
        print("Saved:", path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
