"""Graspable workspace of one design XML mounted on the RB5, plus its plot.

The design is evaluated exactly like the final candidates of an optimize_rb5_cmaes.py
run: same config_used.json, same final grid (rb5_fine), same rb5_eval.Rb5Arm.run_cells
(mounting, pick-and-lift, env/tip judging, prefilter). The design XML (same format as
design168.xml) is used as the hand template, so its own fixed values (phi_d, k_p, k_d,
preload) are kept. Only tendon tension, object and protocol come from the config.

  python eval_design_rb5.py --run runs/rb5_cmaes_final --xml design001.xml --workers 8

Running the same command again continues an interrupted evaluation, or only redraws the
plot when it is finished. --fresh discards earlier results of this label.

Writes to <run>/references/:
  <label>.json              design, summary (J, env, tip, failures), grid
  <label>_workspace.csv     one row per cell, same columns as final/*_workspace.csv
  <label>_source.xml        copy of the design XML (template used for every cell)
  <label>_hand.xml          hand at its strongest grasp cell
  <label>_rb5_scene.xml     viewer scene: python run_pick_lift.py --xml <this file>
and <run>/plots/workspace_<label>.png
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from gripper_core import DesignSpace, VARIABLES, baseline_parameters, build_xml, read_json, save_rows
from rb5_eval import Rb5Arm, best_cell, group_tasks, hand_config, summarize, validate_rb5_config

WORKER = {}


def init_worker(template, cfg):
    WORKER.clear()
    WORKER.update(template=template, cfg=cfg, arm=None)


def run_task(task):
    """Same work as optimize_rb5_cmaes.worker_task: cells of one design at one radius."""
    _, design, r_mm, cells = task
    if WORKER["arm"] is None:
        WORKER["arm"] = Rb5Arm(WORKER["cfg"])
    start = time.perf_counter()
    rows = WORKER["arm"].run_cells(WORKER["template"], design, r_mm, cells)
    return rows, time.perf_counter()-start


def to_json(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=to_json), encoding="utf-8")


def parse_value(v):
    if v is None or v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def read_rows(path):
    """Rows of a *_workspace.csv (save_rows output) with bool/float/None restored."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return [{k: parse_value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def cell_key(r, x, y):
    return (round(float(r), 6), round(float(x), 6), round(float(y), 6))


def load_design(template, cfg):
    """Design dict of the XML, derived the way the optimizer derives its baseline."""
    base = baseline_parameters(template)
    space = DesignSpace(cfg, base)
    design = space.derived({v: float(base[v]) for v in VARIABLES})
    try:
        space.encode(design)
        note = ""
    except Exception as exc:  # Outside the CMA-ES bounds is fine for a plain evaluation.
        note = f"outside the CMA-ES search bounds ({exc})"
    return design, note


def hand_parameters(hand_xml):
    """Finger values of a generated hand XML, i.e. what the RB5 run actually simulates."""
    root = ET.fromstring(hand_xml)
    joint = {j.get("name"): j for j in root.iter("joint")}
    geom = {g.get("name"): g for g in root.iter("geom")}
    body = {b.get("name"): b for b in root.iter("body")}
    motor = root.find(".//motor[@name='pulley_pull']")

    def seg(name):
        v = [float(t) for t in geom[name].get("fromto").split()]
        return 1000*math.dist(v[:3], v[3:])

    out = {"Lp [mm]": seg("R_gp"), "Ld [mm]": seg("R_gd"),
           "G [mm]": 2000*float(body["R_prox"].get("pos").split()[0])}
    q_open = {}
    for tag, name in (("p", "R_jp"), ("d", "R_jd")):
        j = joint[name]
        k, spring = float(j.get("stiffness", 0)), float(j.get("springref", 0))
        q_open[tag] = float(j.get("range").split()[0])  # Open stop; closing is positive.
        out[f"k_{tag} [N m/rad]"] = k
        out[f"preload_{tag} [N m]"] = k*(q_open[tag]-spring)
        out[f"damping_{tag}"] = float(j.get("damping", 0))
    out["theta_p [deg]"] = -math.degrees(q_open["p"])
    out["theta_d [deg]"] = -math.degrees(q_open["d"])
    out["phi_d [deg]"] = out["theta_p [deg]"]+out["theta_d [deg]"]
    gear = float(motor.get("gear").split()[0])
    out["max tendon tension [N]"] = -gear*float(motor.get("ctrlrange").split()[1])
    return out


def print_parameters(columns):
    names = list(columns)
    keys = list(dict.fromkeys(k for c in columns.values() for k in c))
    print(f"{'':24}" + "".join(f"{n:>16}" for n in names))
    for k in keys:
        vals = [columns[n].get(k) for n in names]
        text = "".join(f"{v:>16.5g}" if v is not None else f"{'-':>16}" for v in vals)
        nums = [v for v in vals if v is not None]
        flag = "  <-- differs" if len(nums) > 1 and not np.allclose(nums, nums[0], rtol=1e-4, atol=1e-9) else ""
        print(f"{k:24}{text}{flag}")


def signature(template, cfg, grid):
    text = template+json.dumps(cfg, sort_keys=True)+json.dumps(grid, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def read_progress(path, sig):
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or json.loads(lines[0]).get("signature") != sig:
        raise SystemExit(f"{path} belongs to a different XML, config or grid. "
                         "Add --fresh to discard it, or use another --label.")
    done = {}
    for line in lines[1:]:
        try:
            part = json.loads(line)
        except json.JSONDecodeError:  # A line cut off by Ctrl+C.
            continue
        for row in part["rows"]:
            done[cell_key(row["r_mm"], row["x_mm"], row["y_mm"])] = row
    return done


def split_tasks(design, grid, done, workers):
    """(index, design, r, cells) tasks for the cells not done yet. Every cell is simulated
    independently, so splitting a radius into chunks only spreads the work over workers."""
    todo = []
    for _, _, r, cells in group_tasks([design], grid):
        left = [(x, y) for x, y in cells if cell_key(r, x, y) not in done]
        if left:
            todo.append((r, left))
    n_cells = sum(len(c) for _, c in todo)
    if not n_cells:
        return []
    per_task = max(4, math.ceil(n_cells/(3*workers)))
    tasks = []
    for r, cells in todo:
        size = math.ceil(len(cells)/math.ceil(len(cells)/per_task))  # Even chunks per radius.
        tasks += [(0, design, r, cells[i:i+size]) for i in range(0, len(cells), size)]
    return tasks


def evaluate(template, design, cfg, grid, progress, workers):
    sig = signature(template, cfg, grid)
    done = read_progress(progress, sig)
    total = sum(len(c) for _, _, _, c in group_tasks([design], grid))
    tasks = split_tasks(design, grid, done, workers)
    if done:
        print(f"Continuing: {len(done)}/{total} cells already done in {progress.name}", flush=True)
    if tasks:
        if not progress.is_file():
            progress.write_text(json.dumps({"signature": sig})+"\n", encoding="utf-8")
        print(f"Simulating {total-len(done)} cells in {len(tasks)} tasks with {workers} workers ...",
              flush=True)
        wall = time.perf_counter()
        n_new = 0
        pool = None
        interrupted = False
        try:
            if workers > 1:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                           initializer=init_worker, initargs=(template, cfg))
                results = (f.result() for f in as_completed([pool.submit(run_task, t) for t in tasks]))
            else:
                init_worker(template, cfg)
                results = (run_task(t) for t in tasks)
            for rows, sec in results:
                with progress.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"rows": rows, "seconds": sec}, default=to_json)+"\n")
                for row in rows:
                    done[cell_key(row["r_mm"], row["x_mm"], row["y_mm"])] = row
                n_new += len(rows)
                elapsed = time.perf_counter()-wall
                left = total-len(done)
                ok = sum(bool(r["success"]) for r in rows)
                print(f"  r={rows[0]['r_mm']:g} mm: {len(rows)} cells, {ok} success, {sec:.0f} s | "
                      f"{len(done)}/{total} done, ~{elapsed/n_new*left/60:.0f} min left", flush=True)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if pool:
                pool.shutdown(wait=not interrupted, cancel_futures=True)
        if interrupted:
            raise SystemExit("\nInterrupted. Finished cells are saved; run the same command again to continue.")
    # Same row order as the optimizer: radius, then the grid's (x, y) order.
    return [done[cell_key(r, x, y)] for _, _, r, cells in group_tasks([design], grid) for x, y in cells]


def plot(run, cfg, label, template, design, summary, rows):
    import matplotlib
    matplotlib.use("Agg")
    import plot_workspace as pw
    out = run/"plots"
    out.mkdir(exist_ok=True)
    path = out/f"workspace_{label}.png"
    shapes = pw.hand_outline(dict(template=template, design=design), cfg)
    pw.plot_workspace(path, rows, label, design, summary, shapes)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Output folder of optimize_rb5_cmaes.py")
    p.add_argument("--xml", type=Path, default=Path("design001.xml"),
                   help="Design XML in the design168.xml format (default design001.xml)")
    p.add_argument("--label", help="Name of the result (default: XML file name without .xml)")
    p.add_argument("--grid", help="Grid name in config_used.json (default: the run's final grid)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--fresh", action="store_true", help="Discard earlier results of this label")
    args = p.parse_args()
    run = args.run.resolve()
    label = args.label or args.xml.stem
    ref = run/"references"
    files = {k: ref/f"{label}{s}" for k, s in (("json", ".json"), ("csv", "_workspace.csv"),
             ("src", "_source.xml"), ("hand", "_hand.xml"), ("scene", "_rb5_scene.xml"),
             ("progress", "_progress.jsonl"))}
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not (run/"config_used.json").is_file():
        raise SystemExit(f"No config_used.json in {run}. --run must be an optimize_rb5_cmaes.py output folder.")
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)

    if args.fresh:
        for f in files.values():
            f.unlink(missing_ok=True)
    if files["json"].is_file() and files["csv"].is_file():  # Finished earlier: only redraw.
        result = read_json(files["json"])
        template = files["src"].read_text(encoding="utf-8-sig")
        rows = read_rows(files["csv"])
        print(f"{label} was already evaluated (use --fresh to redo). Redrawing the plot.")
        print("Saved:", plot(run, cfg, label, template, result["design"], result["summary"], rows))
        return

    if not args.xml.is_file():
        raise SystemExit(f"Design XML not found: {args.xml}")
    template = args.xml.read_text(encoding="utf-8-sig")
    design, note = load_design(template, cfg)
    if args.grid:
        if args.grid not in cfg["grids"]:
            raise SystemExit(f"Grid '{args.grid}' not in config_used.json: {sorted(cfg['grids'])}")
        grid, grid_name = cfg["grids"][args.grid], args.grid
    elif (run/"best_design_fine.json").is_file():
        grid, grid_name = read_json(run/"best_design_fine.json")["grid"], "final grid of the run"
    else:
        grid, grid_name = cfg["grids"]["rb5_fine"], "rb5_fine"

    # What the RB5 runs will simulate, next to the run's final best design.
    hcfg = hand_config(cfg)
    columns = {label: hand_parameters(build_xml(template, design, hcfg, radius_mm=40))}
    if (run/"best_design_fine.json").is_file() and (run/"source_design168.xml").is_file():
        best = read_json(run/"best_design_fine.json")
        run_template = (run/"source_design168.xml").read_text(encoding="utf-8-sig")
        columns[best["label"]] = hand_parameters(build_xml(run_template, best["design"], hcfg, radius_mm=40))
    print(f"Hand as mounted on the RB5 ({', '.join(columns)}):")
    print_parameters(columns)
    if note:
        print(f"Note: {label} is {note}.")
    tension = float(cfg["simulation"]["tendon_tension_N"])
    if columns[label]["max tendon tension [N]"] < tension-1e-6:
        raise SystemExit(f"The generated hand's actuator ctrlrange allows only "
                         f"{columns[label]['max tendon tension [N]']:.1f} N, but the run uses {tension:.1f} N.")

    ref.mkdir(parents=True, exist_ok=True)
    files["src"].write_text(template, encoding="utf-8")
    print(f"Grid: {grid_name} = {grid}")
    wall = time.perf_counter()
    rows = evaluate(template, design, cfg, grid, files["progress"], args.workers)
    summary = summarize(rows, cfg)
    save_rows(files["csv"], rows)
    cell = best_cell(rows)
    try:
        files["hand"].write_text(build_xml(template, design, cfg, radius_mm=cell["r_mm"], x_mm=cell["x_mm"],
                                           y_mm=cell["y_mm"]), encoding="utf-8")
        Rb5Arm(cfg).scene_file(files["scene"], template, design, cell["r_mm"], cell["x_mm"], cell["y_mm"])
    except Exception as exc:  # The workspace result is already saved.
        print(f"Warning: could not export the viewer scene: {exc}")
    dump(files["json"], dict(label=label, design=design, summary=summary, grid=grid,
                             source_xml=str(args.xml.resolve()), viewer_cell=cell))
    print(f"\n{label}: J={summary['J']:.4f}, success={summary['success']}/{summary['cells']} "
          f"(env={summary['env']}, tip={summary['tip']}), rb5_pass={summary['rb5_pass']}, "
          f"prefiltered={summary['prefiltered']}, wall={time.perf_counter()-wall:.0f}s")
    print(f"  failures: {summary['failures']}")
    print("Saved:", files["json"])
    print("Saved:", files["csv"])
    print("Saved:", plot(run, cfg, label, template, design, summary, rows))
    print(f"View the strongest grasp: python run_pick_lift.py --xml {files['scene']}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
