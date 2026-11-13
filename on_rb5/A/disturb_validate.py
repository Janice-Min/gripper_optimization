"""Disturbance experiment A, method check: is F_max a property of the grasp, not of the
simulation settings? Run on a sample of saved cells BEFORE trusting the 204-design batch.

Needs the states of disturb_collect.py (any designs, e.g. design001 and candidate_02). For a
stratified sample of cells (env / tip / hold_mixed, spread over the radii) and every
direction it simulates
  base       the ramp of disturb_force.py (--rate, default 1 N/s)
  rate_*     the same ramp at other rates (--rates, default 0.5 and 2 N/s): convergence of
             the 'slow' ramp. If halving the rate still moves F_max a lot, the ramp is
             not quasi-static and either a slower rate or the hold test should be used.
  dt_half    the base ramp with half the timestep: numerical convergence.
  hold       the hold-test bisection (disturb_force.py --refine): the rate-independent
             holding force. hold/ramp tells how much the ramp overestimates; the rank
             correlation tells whether ranking designs by the cheap ramp is safe.
  zero       the zero-force control: how much the grasp creeps with no load at all.
All F are judged with the same criteria as disturb_force.py (options are shared).

  python A/disturb_validate.py --labels candidate_02 design001 --workers 8      (from on_rb5/)
  python A/disturb_validate.py --cells-per-group 2 --dirs px nz   # quick

Writes A/results/<run name>/_validation/
  rows.csv        one row per cell, direction and variant: F [m*g], cause, censored
  zero.csv        one row per cell: zero-force control
  summary.json    per variant against base: median and 90th percentile of |dF|/F, rank
                  correlation; hold/ramp ratio per group; creep fractions
  validation.png  base ramp against every variant (needs matplotlib)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from gripper_core import read_json, save_rows, write_json
from rb5_eval import validate_rb5_config
from eval_design_rb5 import read_rows, to_json
from disturb_collect import label_dir, resolve_labels, run_tasks
import disturb_force as DF



def pick_cells(cells, per_group, seed):
    """Up to per_group cells of every group, spread evenly over the group's (r, x, y) order."""
    rng = np.random.default_rng(seed)
    out = []
    by = {}
    for c in cells:
        by.setdefault(DF.group_of(c), []).append(c)
    for g, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda c: (c["r_mm"], c["x_mm"], c["y_mm"]))
        if len(rows) > per_group:
            edges = np.linspace(0, len(rows), per_group+1)
            idx = [int(rng.integers(int(a), max(int(a)+1, int(b)))) for a, b in zip(edges[:-1], edges[1:])]
            rows = [rows[i] for i in idx]
        out += rows
    return out


def validate_task(task):
    """All variants for the directions of one cell."""
    label, out, cfg, s, crit, rates, timestep_check, c = task
    out = Path(out)
    model, data, st = DF.open_cell(out, c, cfg)
    mg = s["mg_N"]
    base = dict(label=label, r_mm=c["r_mm"], x_mm=c["x_mm"], y_mm=c["y_mm"], group=DF.group_of(c))
    rows, start = [], time.perf_counter()

    def add(direction, variant, F_N, cause, censored, extra=None):
        rows.append(dict(base, dir=direction, variant=variant, F_mg=F_N/mg, cause=cause, censored=censored,
                         **(extra or {})))

    zero = DF.zero_control(model, data, st["state"], s)
    zres = dict(base, **DF.zero_result(zero["trace"], zero["ended"], crit))
    for name in s["dirs"]:
        d = DF.DIRECTIONS[name]
        res = DF.ramp(model, data, st["state"], d, s)
        j = DF.judge_trace(res["trace"], res["ended"], crit, s["cap_N"])
        add(name, "base", j["F_max_N"], j["cause"], j["censored"], dict(t_Fmax_s=j["t_Fmax_s"]))
        for rate in rates:
            sr = dict(s, rate_N_s=rate)
            r2 = DF.ramp(model, data, st["state"], d, sr)
            j2 = DF.judge_trace(r2["trace"], r2["ended"], crit, s["cap_N"])
            add(name, f"rate_{rate:g}", j2["F_max_N"], j2["cause"], j2["censored"])
        if timestep_check:
            dt = float(model.opt.timestep)
            try:
                model.opt.timestep = dt/2
                r3 = DF.ramp(model, data, st["state"], d, s)
            finally:
                model.opt.timestep = dt  # The compiled model is cached and shared by later tasks.
            j3 = DF.judge_trace(r3["trace"], r3["ended"], crit, s["cap_N"])
            add(name, "dt_half", j3["F_max_N"], j3["cause"], j3["censored"])
        f_hold, cens, trials = DF.refine(model, data, st["state"], d, s, crit, j["F_max_N"], j["censored"])
        add(name, "hold", f_hold, (trials[-1][3] or "held") if trials else "", cens, dict(trials=len(trials)))
    return rows, zres, time.perf_counter()-start


def ranks(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for val in np.unique(v):  # Average ranks of ties.
        idx = np.flatnonzero(v == val)
        r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def compare(rows, zeros, rates, cap_mg):
    """summary.json: every variant against base, hold/ramp per group, creep."""
    base = {(r["label"], r["r_mm"], r["x_mm"], r["y_mm"], r["dir"]): r for r in rows if r["variant"] == "base"}
    variants = [v for v in ("dt_half", "hold", *(f"rate_{x:g}" for x in rates)) if any(r["variant"] == v for r in rows)]
    out = dict(cells=len(zeros), directions=len(base), cap_mg=cap_mg, variants={})
    for v in variants:
        pairs = []
        for r in rows:
            if r["variant"] != v:
                continue
            b = base[(r["label"], r["r_mm"], r["x_mm"], r["y_mm"], r["dir"])]
            if b["censored"] and r["censored"]:
                continue  # Both at the cap: no information.
            pairs.append((b, r))
        if not pairs:
            continue
        fb = np.array([b["F_mg"] for b, _ in pairs])
        fv = np.array([r["F_mg"] for _, r in pairs])
        rel = np.abs(fv-fb)/np.maximum(fb, 0.05)  # Below 0.05 mg the relative error means little.
        groups = {}
        for g in sorted({b["group"] for b, _ in pairs}):
            sel = np.array([b["group"] == g for b, _ in pairs])
            ratio = fv[sel]/np.maximum(fb[sel], 1e-9)
            groups[g] = dict(n=int(sel.sum()), median_ratio=float(np.median(ratio)),
                             median_abs_rel_diff=float(np.median(rel[sel])))
        # Cell level: weakest direction (what pass levels and robust_area use).
        cell_b, cell_v = {}, {}
        for b, r in pairs:
            k = (b["label"], b["r_mm"], b["x_mm"], b["y_mm"])
            cell_b[k] = min(cell_b.get(k, math.inf), b["F_mg"])
            cell_v[k] = min(cell_v.get(k, math.inf), r["F_mg"])
        keys = sorted(cell_b)
        out["variants"][v] = dict(
            n=len(pairs), median_ratio=float(np.median(fv/np.maximum(fb, 1e-9))),
            median_abs_rel_diff=float(np.median(rel)), p90_abs_rel_diff=float(np.percentile(rel, 90)),
            spearman_directions=spearman(fb, fv),
            spearman_cells_weakest=spearman([cell_b[k] for k in keys], [cell_v[k] for k in keys]),
            groups=groups)
    zg = {}
    for z in zeros:
        g = zg.setdefault(z["group"], dict(n=0, failed=0, causes=Counter(), max_disp_mm=[], max_rot_deg=[]))
        g["n"] += 1
        g["failed"] += bool(z["zero_fail"])
        g["causes"][z["zero_cause"] or "held"] += 1
        g["max_disp_mm"].append(z["zero_max_disp_mm"])
        g["max_rot_deg"].append(z["zero_max_rot_deg"])
    out["zero_control"] = {g: dict(n=v["n"], failed=v["failed"], causes=dict(v["causes"]),
                                   median_max_disp_mm=float(np.median(v["max_disp_mm"])),
                                   median_max_rot_deg=float(np.median(v["max_rot_deg"])),
                                   duration_s=float(zeros[0]["zero_T_s"]) if zeros else None)
                           for g, v in zg.items()}
    return out


def plot(path, rows, cap_mg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    base = {(r["label"], r["r_mm"], r["x_mm"], r["y_mm"], r["dir"]): r for r in rows if r["variant"] == "base"}
    variants = [v for v in dict.fromkeys(r["variant"] for r in rows) if v != "base"]
    colors = {"env": "#2a78d6", "tip": "#eb6834", "hold_mixed": "#9a9893"}  # plot_disturbance palette
    fig, axes = plt.subplots(1, len(variants), figsize=(3.6*len(variants), 3.8), squeeze=False)
    top = min(cap_mg, max(r["F_mg"] for r in rows)*1.05)
    for ax, v in zip(axes[0], variants):
        for g, col in colors.items():
            pts = [(base[(r["label"], r["r_mm"], r["x_mm"], r["y_mm"], r["dir"])]["F_mg"], r["F_mg"])
                   for r in rows if r["variant"] == v and r["group"] == g]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, s=12, alpha=0.7, color=col, label=g, edgecolors="none")
        ax.plot([0, top], [0, top], color="#444", lw=0.8, ls="--")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_xlabel("base ramp F_max [m g]")
        ax.set_ylabel(f"{v} [m g]")
        ax.set_title(v, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=layout.DEFAULT_RUN)
    p.add_argument("--labels", nargs="+", default=["all"], help="Designs collected by disturb_collect.py")
    p.add_argument("--cells-per-group", type=int, default=4, help="Cells per design and group (default 4)")
    p.add_argument("--rates", type=float, nargs="*", default=[0.5, 2.0], help="Other ramp rates [N/s]")
    p.add_argument("--no-timestep-check", action="store_true", help="Skip the half-timestep ramp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    DF.add_arguments(p)
    args = p.parse_args()
    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    crit = DF.criteria(args, cfg)
    s = DF.settings(args, cfg)
    s.update(refine=True, zero_control=True)
    DF.check_settings(s, crit)
    collected = sorted(d.name for d in layout.states_root(run).glob("*") if (d/"cells.csv").is_file())
    labels = resolve_labels(args.labels, collected)
    tasks = []
    for label in labels:
        out = label_dir(run, label)
        cells = [c for c in read_rows(out/"cells.csv") if c["state_file"]]
        if not cells:
            continue
        model, _ = DF.model_for(str(out/"hands"/str(np.load(out/cells[0]["state_file"])["hand_file"])), cfg)
        mg = float(model.body_mass[model.body("obj").id])*cfg["simulation"]["gravity_m_s2"]
        sl = dict(s, mg_N=mg, cap_N=s["cap_mg"]*mg, tol_N=s["tol_mg"]*mg)
        for c in pick_cells(cells, args.cells_per_group, args.seed):
            tasks.append((label, str(out), cfg, sl, crit, args.rates, not args.no_timestep_check, c))
    if not tasks:
        raise SystemExit("No saved cells. Run disturb_collect.py first.")
    print(f"{len(tasks)} cells from {labels} x {len(s['dirs'])} directions; variants: base {s['rate_N_s']} N/s, "
          f"rates {args.rates}, {'dt/2, ' if not args.no_timestep_check else ''}hold test, zero control", flush=True)
    rows, zeros, wall = [], [], time.perf_counter()

    def report(part, zero, sec):
        rows.extend(part)
        zeros.append(zero)
        print(f"  {zero['label']} r={zero['r_mm']:g} x={zero['x_mm']:g} y={zero['y_mm']:g} ({zero['group']}): "
              f"{sec:.0f} s ({len(zeros)}/{len(tasks)}, {time.perf_counter()-wall:.0f} s)", flush=True)

    run_tasks(validate_task, tasks, args.workers, report)
    out = layout.a_root(run)/"_validation"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    save_rows(out/"rows.csv", [{k: r.get(k) for k in keys} for r in rows])
    save_rows(out/"zero.csv", zeros)
    sm = compare(rows, zeros, args.rates, s["cap_mg"])
    sm.update(labels=labels, criteria=crit, settings={k: v for k, v in s.items() if k != "dirs"}, dirs=s["dirs"])
    write_json(out/"summary.json", json.loads(json.dumps(sm, default=to_json)))
    print(f"\nAgainst the base ramp ({s['rate_N_s']} N/s; ratio = variant/base, |dF|/F relative difference):")
    print(f"  {'variant':<10} {'n':>4} {'median ratio':>13} {'median |dF|/F':>14} {'p90 |dF|/F':>11} "
          f"{'rank corr dir':>14} {'rank corr cell':>15}")
    for v, r in sm["variants"].items():
        rc = "-" if r["spearman_directions"] is None else f"{r['spearman_directions']:.3f}"
        cc = "-" if r["spearman_cells_weakest"] is None else f"{r['spearman_cells_weakest']:.3f}"
        print(f"  {v:<10} {r['n']:>4} {r['median_ratio']:>13.3f} {r['median_abs_rel_diff']:>14.3f} "
              f"{r['p90_abs_rel_diff']:>11.3f} {rc:>14} {cc:>15}")
        for g, gr in r["groups"].items():
            print(f"      {g:<10} n={gr['n']:<4} median ratio {gr['median_ratio']:.3f}, "
                  f"median |dF|/F {gr['median_abs_rel_diff']:.3f}")
    print("Zero-force control (no load, full ramp duration):")
    for g, z in sm["zero_control"].items():
        print(f"  {g:<10} {z['failed']}/{z['n']} failed {z['causes']}, median max drift {z['median_max_disp_mm']:.2f} mm, "
              f"rotation {z['median_max_rot_deg']:.2f} deg over {z['duration_s']:.1f} s")
    print("Saved:", out/"summary.json")
    fig = plot(out/"validation.png", rows, s["cap_mg"])
    if fig:
        print("Saved:", fig)


if __name__ == "__main__":
    mp.freeze_support()
    main()
