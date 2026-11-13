#!/usr/bin/env python3
"""Per-design picture of the largest F of every force direction, drawn as arrows around the cylinder.

Input is a peak_<group>_<value>.csv from compare_force.py --peak (any direction set: axes, edges, ...).
For every design it draws one figure:
  top row   - the cylinder seen from three sides (x-y, x-z, y-z) with one arrow per measured direction
              in that plane; arrow length is proportional to F, the label gives F in m g.
  bottom    - the same values as bars, with the grasp cell (r, x, y, env/tip) each maximum came from.
and one heat map of all designs x directions.

Hand frame: +x toward the R finger, +y out of the hand (pull-out), -y into the palm, z = cylinder axis.
Each direction's maximum may come from a different cell (object radius / position); the bars name it.

  python plot_force_arrows.py --peak-csv runs/rb5_cmaes_final/plots/disturbance/peak_all_ramp.csv
  python plot_force_arrows.py --peak-csv ... --labels design075 design078 --out figs
"""
import argparse
import math
from pathlib import Path

import numpy as np

from compare_force import peak_dirs, read_peak_csv
from disturb_force import direction_vector

# Views: (file tag, title, horizontal axis, vertical axis)
VIEWS = (("xy", "Top view  (x-y plane)", 0, 1), ("xz", "Front view  (x-z plane)", 0, 2),
         ("yz", "Side view  (y-z plane)", 1, 2))
AXIS_NAME = "xyz"
R_CYL = 1.0     # Cylinder radius in drawing units (schematic, not to scale).
R0 = 1.75       # Arrows start on this circle.
L_MAX = 2.9     # Arrow length of the largest value on the scale.


def scale_top(rows, dirs):
    """Value that maps to L_MAX: the ramp cap if any value reached it, else the largest value rounded up."""
    top = max(float(r[f"max_{d}_mg"]) for r in rows for d in dirs)
    return max(1.0, math.ceil(top))


def in_plane(d, h, v):
    """2-D unit vector of direction d in a view, or None if d has a component off that plane."""
    vec = direction_vector(d)
    off = [i for i in range(3) if i not in (h, v)][0]
    if abs(vec[off]) > 1e-9:
        return None
    u = np.array([vec[h], vec[v]])
    return u/np.linalg.norm(u)


def draw_body(ax, tag, P):
    """Schematic cylinder, fingers and palm for one view."""
    from matplotlib.patches import Circle, Ellipse, Rectangle
    hand = dict(facecolor=P["hand"], edgecolor="none", zorder=1)
    body = dict(facecolor=P["cyl"], edgecolor=P["cyl_edge"], lw=1.0, zorder=2)
    txt = dict(color=P["muted"], fontsize=7.5, ha="center", va="center", zorder=5)
    if tag == "xy":  # Looking down the cylinder axis; palm at -y, fingers at -x (L) and +x (R).
        ax.add_patch(Rectangle((-1.45, -1.42), 2.9, 0.24, **hand))
        for s in (-1, 1):
            ax.add_patch(Rectangle((s*1.18-0.09, -1.18), 0.18, 2.05, **hand))
        ax.add_patch(Circle((0, 0), R_CYL, **body))
        ax.text(0, -1.30, "palm", **txt)
        ax.text(-1.18, 1.05, "L", **txt)
        ax.text(1.18, 1.05, "R", **txt)
    else:
        h = 1.0
        if tag == "xz":  # Fingers seen end-on at -x / +x.
            for s in (-1, 1):
                ax.add_patch(Rectangle((s*1.18-0.09, -0.45), 0.18, 0.9, **hand))
            ax.text(-1.18, 0.62, "L", **txt)
            ax.text(1.18, 0.62, "R", **txt)
        else:  # Palm at -y, a finger behind the cylinder.
            ax.add_patch(Rectangle((-1.42, -0.62), 0.24, 1.24, **hand))
            ax.add_patch(Rectangle((-1.18, -0.45), 2.05, 0.9, facecolor=P["hand"], edgecolor="none", alpha=0.55,
                                   zorder=1))
            ax.text(-1.48, -0.86, "palm", **txt)
        ax.add_patch(Rectangle((-R_CYL, -h), 2*R_CYL, 2*h, facecolor=P["cyl"], edgecolor="none", zorder=2))
        for x in (-R_CYL, R_CYL):
            ax.plot([x, x], [-h, h], color=P["cyl_edge"], lw=1.0, zorder=3)
        ax.add_patch(Ellipse((0, -h), 2*R_CYL, 0.42, **body))
        ax.add_patch(Ellipse((0, h), 2*R_CYL, 0.42, facecolor=P["cyl_top"], edgecolor=P["cyl_edge"], lw=1.0, zorder=3))


def draw_view(ax, tag, title, h, v, row, dirs, top, weakest, P):
    from matplotlib.patches import Circle, FancyArrow
    ax.set_aspect("equal")
    lim = R0+L_MAX+1.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=9.5, color=P["ink"])
    # Scale rings with their value, on a bisector between the 45-degree arrows.
    steps = [s for s in (0.25, 0.5, 0.75, 1.0)]
    for s in steps:
        ax.add_patch(Circle((0, 0), R0+s*L_MAX, fill=False, edgecolor=P["grid"], lw=0.8, ls=(0, (3, 3)), zorder=0))
        a = math.radians(22.5+45*3)
        val = s*top
        ax.text((R0+s*L_MAX)*math.cos(a), (R0+s*L_MAX)*math.sin(a), f"{val:g}" + (" m g" if s == 1.0 else ""), color=P["muted"], fontsize=6.5,
                ha="center", va="center", zorder=0, bbox=dict(facecolor=P["surface"], edgecolor="none", pad=0.6))
    ax.add_patch(Circle((0, 0), R0, fill=False, edgecolor=P["grid"], lw=0.8, zorder=0))
    draw_body(ax, tag, P)
    for d in dirs:
        u = in_plane(d, h, v)
        if u is None:
            continue
        F = float(row[f"max_{d}_mg"])
        cens = bool(row[f"max_{d}_censored"])
        L = max(0.02, min(F, top)/top*L_MAX)
        col = P["weak"] if d == weakest else P["arrow"]
        head = min(0.32, 0.6*L)
        start = R0*u
        ax.add_patch(FancyArrow(start[0], start[1], L*u[0], L*u[1], width=0.13, head_width=0.38, head_length=head,
                                length_includes_head=True, facecolor=P["surface"] if cens else col, edgecolor=col,
                                lw=1.3 if cens else 0, zorder=4))
        tip = (R0+L+0.62)*u
        lab = f"{'≥' if cens else ''}{F:.2f}"
        ax.text(tip[0], tip[1]+0.13, lab, ha="center", va="bottom", fontsize=8.2,
                color=P["weak"] if d == weakest else P["ink"], fontweight="bold", zorder=6)
        ax.text(tip[0], tip[1]+0.08, d, ha="center", va="top", fontsize=7, color=P["ink2"], zorder=6)
    # Small axis key in the corner.
    k0 = np.array([-lim+0.35, -lim+0.35])
    for vec, name in (((0.9, 0), AXIS_NAME[h]), ((0, 0.9), AXIS_NAME[v])):
        ax.annotate("", xy=k0+np.array(vec), xytext=k0, arrowprops=dict(arrowstyle="-|>", color=P["ink2"], lw=1))
        ax.text(*(k0+np.array(vec)*1.28), f"+{name}", fontsize=7.5, color=P["ink2"], ha="center", va="center")


def draw_bars(ax, row, dirs, top, weakest, P, value):
    vals = [float(row[f"max_{d}_mg"]) for d in dirs]
    cens = [bool(row[f"max_{d}_censored"]) for d in dirs]
    cols = [P["weak"] if d == weakest else P["arrow"] for d in dirs]
    bars = ax.bar(range(len(dirs)), vals, width=0.66, color=cols, linewidth=0, zorder=3)
    for b, c, col in zip(bars, cens, cols):
        if c:
            b.set_hatch("////")
            b.set_edgecolor(P["surface"])
    ymax = top*1.2
    for j, (val, c, d) in enumerate(zip(vals, cens, dirs)):
        ax.text(j, val+0.015*ymax, f"{'≥' if c else ''}{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                color=P["weak"] if d == weakest else P["ink"])
    ax.axhline(1.0, color=P["ink2"], lw=0.8, ls="--", zorder=2)
    ax.text(len(dirs)-0.35, 1.0, "1 m g\n(object\nweight)", ha="left", va="center", fontsize=7, color=P["ink2"])
    ax.set_xlim(-0.6, len(dirs)-0.4)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(len(dirs)), dirs, fontsize=8.5)
    # Cell of every maximum below the direction names.
    for j, d in enumerate(dirs):
        cell = str(row.get(f"max_{d}_cell") or "")
        ax.annotate(cell.replace(" ", "\n"), xy=(j, 0), xycoords=("data", "axes fraction"), xytext=(0, -22),
                    textcoords="offset points", ha="center", va="top", fontsize=6.3, color=P["muted"],
                    linespacing=1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"max F over cells [m g] ({value})")
    n_axes = sum(1 for d in dirs if len(d) == 2)
    if 0 < n_axes < len(dirs):
        ax.axvline(n_axes-0.5, color=P["muted"], lw=0.8)
        ax.text(n_axes/2-0.5, ymax*0.97, "axes", ha="center", va="top", fontsize=7.5, color=P["ink2"])
        ax.text((n_axes+len(dirs))/2-0.5, ymax*0.97, "diagonals", ha="center", va="top", fontsize=7.5, color=P["ink2"])


def palette():
    from plot_disturbance import GRID, INK, INK_2, MUTED, SERIES, SURFACE
    return dict(arrow=SERIES[0], weak=SERIES[7], ink=INK, ink2=INK_2, muted=MUTED, grid="#d6d5d0", surface=SURFACE,
                hand="#dcdad5", cyl="#eceae5", cyl_top="#f6f5f2", cyl_edge="#8f8d88", grid_line=GRID)


def plot_design(out, row, dirs, top, value, group):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    P = palette()
    vals = {d: float(row[f"max_{d}_mg"]) for d in dirs}
    weakest = min(vals, key=vals.get)
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = GridSpec(2, 3, height_ratios=[1.0, 0.62], hspace=0.12, wspace=0.06, figure=fig)
    for k, (tag, title, h, v) in enumerate(VIEWS):
        draw_view(fig.add_subplot(gs[0, k]), tag, title, h, v, row, dirs, top, weakest, P)
    draw_bars(fig.add_subplot(gs[1, :]), row, dirs, top, weakest, P, value)
    kind = "env and tip" if group == "all" else group
    cens = sum(bool(row[f"max_{d}_censored"]) for d in dirs)
    fig.suptitle(f"{row['label']}: largest F of each direction over its {int(float(row['cells']))} {kind} grasp cells "
                 f"({value})", x=0.06, y=0.985, ha="left", fontsize=13, color=P["ink"])
    note = (f"Arrow = force direction on the object (at its COM), length ∝ F, label = F [m g] (m = 0.1 kg, 1 m g = 0.98 N). "
            f"Red = weakest direction ({weakest}, {vals[weakest]:.2f} m g).")
    if cens:
        note += f"  Hollow arrow / hatched bar / ≥ = held up to the {top:g} m g cap ({cens} direction{'s' if cens > 1 else ''})."
    note += "  Each direction's maximum can come from a different cell (shown under the bars). Gripper drawn schematically."
    fig.text(0.06, 0.953, note, ha="left", va="top", fontsize=8.3, color=P["ink2"], wrap=True)
    path = Path(out)/f"arrows_{row['label']}_{group}_{value}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(out, rows, dirs, top, value, group):
    import matplotlib.pyplot as plt
    from plot_disturbance import INK, INK_2, SEQUENTIAL, SURFACE
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    M = np.array([[float(r[f"max_{d}_mg"]) for d in dirs] for r in rows])
    C = np.array([[bool(r[f"max_{d}_censored"]) for d in dirs] for r in rows])
    fig, ax = plt.subplots(figsize=(0.62*len(dirs)+2.4, 0.5*len(rows)+1.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=top, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(dirs)):
            dark = M[i, j] > 0.55*top
            ax.text(j, i, f"{'≥' if C[i, j] else ''}{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color=SURFACE if dark else INK)
        j = int(np.argmin(M[i]))
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="#e34948", lw=1.6))
    ax.set_xticks(range(len(dirs)), dirs, rotation=90)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    n_axes = sum(1 for d in dirs if len(d) == 2)
    if 0 < n_axes < len(dirs):
        ax.axvline(n_axes-0.5, color=SURFACE, lw=3)
    ax.set_title(f"Largest F of each direction [m g] ({value}; red box = weakest direction; ≥ = held to the cap)",
                 loc="left", fontsize=9.5, color=INK_2)
    path = Path(out)/f"arrows_heatmap_{group}_{value}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--peak-csv", type=Path, required=True, help="peak_<group>_<value>.csv from compare_force.py --peak")
    p.add_argument("--labels", nargs="+", help="Only these designs (default: all rows)")
    p.add_argument("--out", type=Path, help="Output folder (default: next to the CSV, subfolder arrows)")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_disturbance import style
    style(plt)
    rows = read_peak_csv(args.peak_csv)
    if args.labels:
        rows = [r for l in args.labels for r in rows if r["label"] == l]
    if not rows:
        raise SystemExit("No designs to draw.")
    stem = args.peak_csv.stem.split("_")  # peak_<group>_<value>
    group, value = (stem[1], stem[2]) if len(stem) == 3 else ("all", "ramp")
    dirs = [d for d in peak_dirs(rows[0]) if all(r.get(f"max_{d}_mg") is not None for r in rows)]
    top = scale_top(rows, dirs)
    out = args.out or args.peak_csv.resolve().parent/"arrows"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        print(f"Saved: {plot_design(out, r, dirs, top, value, group)}")
    print(f"Saved: {plot_heatmap(out, rows, dirs, top, value, group)}")


if __name__ == "__main__":
    main()
