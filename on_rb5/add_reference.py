"""Evaluate another design XML (e.g. design001.xml) on the RB5 with the same config and
the same final grid as a finished optimize_rb5_cmaes.py run, and add it to that run.

The design is used exactly as written in its own XML, including the parameters the
optimizer keeps fixed (distal absolute angle, joint stiffness and preload, object mass).
Its template is copied into <run>/final, so plot_workspace.py and run_task.py draw and
run it with its own geometry.

  python add_reference.py --run runs/rb5_cmaes --xml design001.xml --workers 8
  python plot_workspace.py --run runs/rb5_cmaes --baseline design001
  python run_task.py --run runs/rb5_cmaes --label design001
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gripper_core import baseline_parameters, grid_cells, read_json, save_rows, write_json
from optimize_rb5_cmaes import evaluate_many, init_worker
from rb5_eval import reference_design, validate_rb5_config

FIXED = (("phi_d_deg", "distal absolute angle phi_d [deg]"), ("kp", "proximal stiffness kp"),
         ("kd", "distal stiffness kd"), ("taup", "proximal preload"), ("taud", "distal preload"),
         ("mass_kg", "object mass [kg]"), ("halfheight_m", "object half height [m]"))
OPTIMIZER_LABELS = ("baseline", "start")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Finished optimize_rb5_cmaes.py folder")
    p.add_argument("--xml", type=Path, required=True, help="Design XML to evaluate, e.g. design001.xml")
    p.add_argument("--label", help="Name in the plots (default: XML file name, e.g. design001)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    args = p.parse_args()
    run = args.run.resolve()
    final = run/"final"
    if not (final/"comparison.json").is_file():
        raise SystemExit(f"{run} has no final/comparison.json. Use a run that finished its final evaluation.")
    label = args.label or args.xml.stem
    comparison = read_json(final/"comparison.json")
    existing = {r["label"]: r for r in comparison["results"]}
    if label in OPTIMIZER_LABELS or label.startswith("candidate_") or (
            label in existing and not existing[label].get("source_xml")):
        raise SystemExit(f"'{label}' is an optimizer result name. Choose another --label.")
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    template = args.xml.read_text(encoding="utf-8-sig")
    design = reference_design(template)
    grid = comparison["grid"]

    ref, opt = baseline_parameters(template), baseline_parameters((run/"source_design168.xml").read_text(encoding="utf-8-sig"))
    print(f"{label}: L={design['length_mm']:.1f} mm, Lp/L={design['lp_ratio']:.3f}, "
          f"G={design['gap_mm']:.1f} mm, theta_p={design['theta_p_deg']:.1f} deg")
    print(f"Fixed parameters ({label} vs the optimizer's template design168):")
    for key, name in FIXED:
        flag = "" if abs(ref[key]-opt[key]) <= 1e-9*max(1.0, abs(opt[key])) else "   <-- differs"
        print(f"  {name:34s} {ref[key]:<12.6g} {opt[key]:<12.6g}{flag}")
    b = cfg["bounds"]
    outside = [k for k in ("length_mm", "lp_ratio", "gap_mm", "theta_p_deg")
               if not b[k][0]-1e-9 <= design[k] <= b[k][1]+1e-9]
    if outside:
        print(f"  Note: {outside} outside the optimizer bounds; evaluated as written anyway.")

    n = len(grid_cells(grid))
    print(f"Evaluating {label} on the final grid ({n} cells) with {args.workers} workers ...", flush=True)
    start = time.perf_counter()
    init_worker(template, cfg)
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                initializer=init_worker, initargs=(template, cfg))
            if args.workers > 1 else None)
    try:
        (summary, rows, _), = evaluate_many([design], grid, cfg, pool)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)

    source = f"{label}_source.xml"
    (final/source).write_text(template, encoding="utf-8")
    save_rows(final/f"{label}_workspace.csv", rows)
    result = dict(label=label, design=design, summary=summary, source_xml=source,
                  note="Reference design evaluated as written by add_reference.py; not an optimizer result.")
    write_json(final/f"{label}.json", result)
    comparison["results"] = [r for r in comparison["results"] if r["label"] != label]+[result]
    write_json(final/"comparison.json", comparison)
    print(f"{label}: J={summary['J']:.4f}, env={summary['env']}, tip={summary['tip']}, "
          f"rb5_pass={summary['rb5_pass']}/{summary['cells']}  ({time.perf_counter()-start:.0f}s)")
    print(f"  failures: {summary['failures']}")
    for r in comparison["results"]:
        s = r["summary"]
        print(f"  {r['label']:14s} J={s['J']:.4f}  env={s['env']}  tip={s['tip']}")
    print(f"Added to {final}. Plot with: python plot_workspace.py --run {args.run} --baseline {label}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
