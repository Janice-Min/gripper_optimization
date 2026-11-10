"""Disturbance experiment, step 1: re-run the successful cells and save the grasp state.

For every design with a per-cell workspace CSV in the run (default --labels all:
final/ = baseline (design168), candidate_01..03, and references/ = design001) this
  * takes that design's OWN success cells from its workspace CSV (not the common cells),
    plus the cells that lifted the object but failed only the contact rule
    (reason hold_mixed by default, see --extra-reasons),
  * re-runs exactly the pick-and-lift of rb5_eval.Rb5Arm.run_cells (same mounting, IK
    path, CellMonitor and judge()), without the approach prefilter, which only decides
    whether a cell is simulated at all,
  * checks that the original verdict is reproduced (same reason and contact type), and
  * saves the full MuJoCo integration state (qpos, qvel, ctrl, warmstart, ...) at the
    end of 'lift hold' for every cell whose object was lifted and held (rb5_pass), so
    disturb_force.py continues from exactly that instant instead of re-lifting.

The workspace CSVs hold no contact state, which is why the cells are simulated again.
Run on the same machine and MuJoCo version as the original evaluation (3.3.7): the
pick-and-lift is deterministic there, but other platforms can flip borderline cells.

  python disturb_collect.py --run runs/rb5_cmaes_final --workers 8
  python disturb_collect.py --run runs/rb5_cmaes_final --labels candidate_02 --limit 5   # smoke test

Writes <run>/disturbance/<label>/
  cells.csv          one row per selected cell: original verdict (orig_*), re-run verdict
                     (rb5_eval row columns), reproduced, state_file
  states/*.npz       integration state at the end of 'lift hold' (cells with rb5_pass)
  hands/hand_r*.xml  hand XML of each radius, compiled again by disturb_force.py
  meta.json          design, stage0 patch, signatures, counts, non-reproduced cells
Earlier states and disturb_force.py results of the same design are deleted first.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import mujoco

from gripper_core import read_json, save_rows, write_json
from rb5_eval import CellMonitor, empty_row, run_result, validate_rb5_config
from eval_design_rb5 import load_reference, make_arm, read_rows, to_json
from run_pick_lift import ARM_JOINTS, PickLift

ROOT = Path(__file__).resolve().parent
OUT_DIR = "disturbance"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
DEFAULT_LABELS = ("all",)


# ---------------------------------------------------------------- shared helpers
def label_dir(run, label):
    return Path(run)/OUT_DIR/label


def cell_name(r, x, y):
    return f"r{float(r):g}_x{float(x):g}_y{float(y):g}"


def load_label(run, label):
    """(design for build_xml, stage0 patch or None, hand template, workspace rows).

    references/<label>.json (eval_design_rb5.py, e.g. design001) is read first, then
    final/<label> (baseline, candidate_01..03 or a design added with add_reference.py)."""
    run = Path(run)
    ref = run/"references"
    if (ref/f"{label}.json").is_file() and (ref/f"{label}_workspace.csv").is_file():
        design, patch, template, _, _ = load_reference(ref, label)
        return design, patch, template, read_rows(ref/f"{label}_workspace.csv")
    _, design, _, _, template = run_result(run, label)
    return design, None, template, read_rows(run/"final"/f"{label}_workspace.csv")


def available_labels(run):
    """Designs of the run that have a per-cell workspace CSV: final/ results, then references/.
    (history.csv holds only per-design summaries of the coarse search grid, no cells.)"""
    run = Path(run)
    labels = []
    if (run/"final"/"comparison.json").is_file():
        labels += [r["label"] for r in read_json(run/"final"/"comparison.json")["results"]
                   if (run/"final"/f"{r['label']}_workspace.csv").is_file()]
    labels += sorted(p.stem for p in (run/"references").glob("*.json")
                     if not p.stem.endswith("_meta") and (run/"references"/f"{p.stem}_workspace.csv").is_file())
    return list(dict.fromkeys(labels))


def resolve_labels(labels, available):
    """Expand 'all' and refuse unknown names."""
    out = list(dict.fromkeys(l for name in labels for l in (available if name == "all" else [name])))
    unknown = [l for l in out if l not in available]
    if unknown:
        raise SystemExit(f"Unknown design(s) {unknown}. Available: {available}")
    return out


def save_table(path, rows):
    """save_rows, but a design without any selected cell gets an empty file instead of a crash."""
    if rows:
        save_rows(path, rows)
    else:
        Path(path).write_text("", encoding="utf-8-sig")


def scene_signature(arm, hand_xml):
    """Everything the compiled RB5 scene depends on: hand XML, RB5 and task sources, MuJoCo.
    disturb_force.py refuses a saved state whose signature differs."""
    src = arm.root/arm.opts["sources_dir"]
    h = hashlib.sha256()
    for part in (hand_xml, (src/"rb5_850e.xml").read_text(encoding="utf-8-sig"),
                 (src/"task_scene.xml").read_text(encoding="utf-8-sig"), mujoco.__version__):
        h.update(part.encode())
    return h.hexdigest()


def get_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def set_state(model, data, state):
    if state.size != mujoco.mj_stateSize(model, STATE_SPEC):
        raise ValueError("Saved state does not fit this model (different scene).")
    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------- selection
def select_cells(rows, extra_reasons):
    """Own success cells of the design plus cells failed only for one of extra_reasons."""
    out = []
    for r in rows:
        if r["success"]:
            group = str(r["contact_type"])
        elif r["reason"] in extra_reasons:
            group = str(r["reason"])
        else:
            continue
        out.append(dict(r_mm=float(r["r_mm"]), x_mm=float(r["x_mm"]), y_mm=float(r["y_mm"]),
                        orig_group=group, margin_negative=bool(float(r["m_F"] or 0) < 0),
                        orig_success=bool(r["success"]), orig_contact_type=str(r["contact_type"]),
                        orig_reason=str(r["reason"]), orig_m_F=float(r["m_F"] or 0),
                        orig_total_normal_N=float(r["total_normal_N"] or 0)))
    return out


# ---------------------------------------------------------------- simulation
def collect_task(task):
    """Re-run the selected cells of one design at one radius; save states. Runs in a worker."""
    label, design, patch, template, cfg, r_mm, cells, out = task
    out = Path(out)
    arm = make_arm(cfg, patch)
    start = time.perf_counter()
    try:
        hand_xml, model, q_home = arm.prepare(template, design, r_mm)
    except RuntimeError as exc:  # Home IK: an arm problem, independent of the hand.
        failed = [dict(empty_row(r_mm, c["x_mm"], c["y_mm"]), reason="home_ik_failed",
                       error=str(exc)[:200]) for c in cells]
        return [merge(c, row, "") for c, row in zip(cells, failed)], time.perf_counter()-start
    sig = scene_signature(arm, hand_xml)
    hand_file = out/"hands"/f"hand_r{r_mm:g}.xml"
    hand_file.write_text(hand_xml, encoding="utf-8")
    data = mujoco.MjData(model)
    armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
    rows = []
    for c in cells:
        t0 = time.perf_counter()
        x_mm, y_mm = c["x_mm"], c["y_mm"]
        row = empty_row(r_mm, x_mm, y_mm)
        # Same steps as Rb5Arm.run_cells after its prefilter.
        arm.reset_cell(model, data, q_home, r_mm, x_mm, y_mm)
        sim = PickLift(model, data)
        sim.record_trace = True
        monitor = CellMonitor(model, cfg, arm.opts)
        sim.callbacks.append(monitor)
        result, error = None, None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = sim.run(arm.tension, arm.lift)
        except RuntimeError as exc:
            error = exc
        arm.finish_row(row, model, data, sim, monitor, result, error)
        row["seconds"] = time.perf_counter()-t0
        state_file = ""
        if result is not None and row["rb5_pass"]:
            # sim.run() returns right after the last 'lift hold' step: nothing has moved since.
            name = cell_name(r_mm, x_mm, y_mm)
            np.savez_compressed(out/"states"/f"{name}.npz", state=get_state(model, data),
                                spec=int(STATE_SPEC), r_mm=r_mm, x_mm=x_mm, y_mm=y_mm,
                                arm_ctrl=data.ctrl[armact].copy(), signature=sig,
                                hand_file=hand_file.name, mujoco=mujoco.__version__,
                                sim_time_s=float(data.time))
            state_file = f"states/{name}.npz"
        rows.append(merge(c, row, state_file))
    return rows, time.perf_counter()-start


def merge(cell, row, state_file):
    """cells.csv row: selection and original verdict, then the re-run row of rb5_eval."""
    reproduced = row["reason"] == cell["orig_reason"] and row["contact_type"] == cell["orig_contact_type"]
    out = dict(cell, reproduced=reproduced, d_m_F=float(row["m_F"])-cell["orig_m_F"])
    out.update({k: v for k, v in row.items()
                if k not in ("r_mm", "x_mm", "y_mm", "prefiltered", "approach_clearance_mm")})
    out["state_file"] = state_file
    return out


def chunk_tasks(label, design, patch, template, cfg, cells, out, workers):
    """One compiled model per radius; large radii are split so all workers stay busy."""
    by_r = {}
    for c in cells:
        by_r.setdefault(c["r_mm"], []).append(c)
    per = max(2, math.ceil(len(cells)/(3*max(1, workers))))
    tasks = []
    for r, group in sorted(by_r.items()):
        size = math.ceil(len(group)/math.ceil(len(group)/per))
        tasks += [(label, design, patch, template, cfg, r, group[i:i+size], str(out))
                  for i in range(0, len(group), size)]
    return tasks


def run_tasks(fn, tasks, workers, report):
    """Run tasks serially or on a spawn pool (Windows-safe); report(rows, seconds) per task."""
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            for f in as_completed([pool.submit(fn, t) for t in tasks]):
                report(*f.result())
    else:
        for t in tasks:
            report(*fn(t))


def collect(run, label, cfg, extra_reasons, workers, limit=None):
    design, patch, template, orig = load_label(run, label)
    cells = select_cells(orig, extra_reasons)
    if limit:
        cells = cells[:limit]
    out = label_dir(run, label)
    for sub in ("states", "hands"):
        (out/sub).mkdir(parents=True, exist_ok=True)
    # Never mix two collections: old states and the disturbance results built on them go.
    for old in [*(out/"states").glob("*.npz"), *(out/"traces").glob("*.npz"), *out.glob("force_*")]:
        old.unlink()
    groups = Counter(c["orig_group"] for c in cells)
    print(f"\n[{label}] {len(cells)} cells selected from {len(orig)}: {dict(groups)}, "
          f"m_F<0 successes: {sum(c['margin_negative'] and c['orig_success'] for c in cells)}", flush=True)
    rows, wall = [], time.perf_counter()

    def report(part, sec):
        rows.extend(part)
        ok = sum(r["reproduced"] for r in part)
        print(f"  r={part[0]['r_mm']:g} mm: {len(part)} cells, {ok} reproduced, {sec:.0f} s "
              f"({len(rows)}/{len(cells)})", flush=True)

    run_tasks(collect_task, chunk_tasks(label, design, patch, template, cfg, cells, out, workers),
              workers, report)
    key = {(c["r_mm"], c["x_mm"], c["y_mm"]): i for i, c in enumerate(cells)}
    rows.sort(key=lambda r: key[(r["r_mm"], r["x_mm"], r["y_mm"])])
    save_table(out/"cells.csv", rows)

    bad = [r for r in rows if not r["reproduced"]]
    orig_ok = [r for r in rows if r["orig_success"]]
    meta = dict(label=label, design=design, patch=patch, run=str(Path(run).resolve()),
                mujoco=mujoco.__version__, python=platform.python_version(), platform=platform.platform(),
                extra_reasons=list(extra_reasons), limit=limit, cells_in_map=len(orig),
                selected=dict(groups), orig_success=len(orig_ok),
                reproduced_success=sum(r["reproduced"] for r in orig_ok),
                rerun_success=sum(bool(r["success"]) for r in rows),
                states_saved=sum(bool(r["state_file"]) for r in rows),
                not_reproduced=[dict(r_mm=r["r_mm"], x_mm=r["x_mm"], y_mm=r["y_mm"],
                                     orig=r["orig_reason"]+"/"+r["orig_contact_type"],
                                     rerun=r["reason"]+"/"+r["contact_type"]) for r in bad],
                wall_s=time.perf_counter()-wall)
    write_json(out/"meta.json", json.loads(json.dumps(meta, default=to_json)))
    print(f"[{label}] original successes reproduced: {meta['reproduced_success']}/{len(orig_ok)}; "
          f"re-run successes {meta['rerun_success']}; states saved {meta['states_saved']} "
          f"(rb5_pass); {meta['wall_s']:.0f} s", flush=True)
    for b in meta["not_reproduced"]:
        print(f"    not reproduced: r={b['r_mm']:g} x={b['x_mm']:g} y={b['y_mm']:g}  "
              f"{b['orig']} -> {b['rerun']}")
    print(f"Saved: {out/'cells.csv'}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=ROOT/"runs"/"rb5_cmaes_final")
    p.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                   help="'all' (default) or names: final/ labels (baseline, candidate_02, ...), "
                        "references/ labels (design001)")
    p.add_argument("--extra-reasons", nargs="*", default=["hold_mixed"],
                   help="Failed cells to include besides the successes (default hold_mixed; none: no value)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--limit", type=int, help="Only the first N selected cells per design (smoke test)")
    args = p.parse_args()
    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    env = read_json(run/"environment.json") if (run/"environment.json").is_file() else {}
    if env.get("mujoco") and env["mujoco"] != mujoco.__version__:
        print(f"Warning: the run used MuJoCo {env['mujoco']}, this is {mujoco.__version__}. "
              "Expect more non-reproduced cells.")
    labels = resolve_labels(args.labels, available_labels(run))
    print(f"Designs: {labels}")
    for label in labels:
        collect(run, label, cfg, tuple(args.extra_reasons), args.workers, args.limit)


if __name__ == "__main__":
    mp.freeze_support()
    main()
