"""Workspace map + disturbance experiment for many design XMLs (e.g. design001..design204).

For every XML in --xml-dir (default: the xml/ folder next to on_rb5/; stage0 format like
design001.xml, or the RB5 hand format of
design168.xml), with label = file name without .xml:
  1. workspace  eval_design_rb5.py   RB5 pick-and-lift map on the run's final grid
                                     -> <run>/references/<label>*, plots/workspace_<label>.png
  2. collect    disturb_collect.py   re-run success + hold_mixed cells, save post-lift states
  3. force      disturb_force.py     ramped force, F_max per direction (method A)
                                     -> <run>/disturbance/<label>/
Each design runs its three steps one after another, single-threaded; --workers designs
run at the same time. That keeps every core busy without starting a process pool for
every design, and one design that fails (log in disturbance/_logs/<label>.log) does not
stop the others.

Stopped or crashed? Run the same command again: finished steps are skipped and a
half-finished workspace map continues from its progress file. When a step runs again,
the steps after it run again too.

At the end (and with --only-table, without simulating) it writes
  <run>/disturbance/all_designs.csv   one row per design: design values, stiffness and
                                      preload, J and success of the map, re-run success,
                                      and per level: passed cells, conditional robustness,
                                      J_robust; F_min medians of env / tip / hold_mixed
  <run>/disturbance/summary.json/csv  as disturb_force.py writes them, for all designs
Designs already in final/ (baseline, candidate_01..03) are included once they have been
run through disturb_collect.py and disturb_force.py.

  python batch_disturbance.py --workers 8                        # designs in ..\\xml
  python batch_disturbance.py --xml-dir ..\\xml --pattern "design00[2-4].xml" --workers 3   # try a few
  python batch_disturbance.py --run runs/rb5_cmaes_final --only-table
  python batch_disturbance.py --force-args "--slip-mm 20 --rot-deg 0"
A path with spaces is safest in quotes ("C:\\...\\BEAR lab\\...\\xml") or relative (..\\xml);
an unquoted --xml-dir path is joined back together as well.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gripper_core import read_json
from disturb_collect import OUT_DIR, ROOT, available_labels, save_table
from disturb_force import write_summary

STEPS = ("workspace", "collect", "force")
DESIGN_KEYS = ("length_mm", "lp_ratio", "gap_mm", "theta_p_deg", "theta_d_deg", "phi_d_deg")
HAND_KEYS = {"k_p [N m/rad]": "k_p", "k_d [N m/rad]": "k_d", "preload_p [N m]": "preload_p",
             "preload_d [N m]": "preload_d"}


def step_done(run, label, step):
    ref, out = run/"references", run/OUT_DIR/label
    if step == "workspace":
        return (ref/f"{label}.json").is_file() and (ref/f"{label}_workspace.csv").is_file()
    if step == "collect":
        return (out/"meta.json").is_file() and (out/"cells.csv").is_file()
    return (out/"force_summary.json").is_file()


def commands(run, label, xml, args):
    py = [sys.executable, "-u"]
    grid = ["--grid", args.grid] if args.grid else []
    return {"workspace": [*py, str(ROOT/"eval_design_rb5.py"), "--run", str(run), "--xml", str(xml),
                          "--label", label, "--workers", "1", *grid],
            "collect": [*py, str(ROOT/"disturb_collect.py"), "--run", str(run), "--labels", label,
                        "--workers", "1"],
            "force": [*py, str(ROOT/"disturb_force.py"), "--run", str(run), "--labels", label,
                      "--workers", "1", "--no-summary", *shlex.split(args.force_args)]}


STOP = threading.Event()


def process(run, label, xml, args, log_dir):
    """The three steps of one design, in order; returns a status dict."""
    start, ran, redo = time.perf_counter(), [], False
    cmds = commands(run, label, xml, args)
    for step in STEPS:
        if not redo and step_done(run, label, step):
            continue
        if STOP.is_set():
            return dict(label=label, ok=False, step=step, ran=ran, seconds=time.perf_counter()-start)
        redo = True
        with (log_dir/f"{label}.log").open("a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {step}: {' '.join(cmds[step])}\n")
            log.flush()
            rc = subprocess.run(cmds[step], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
        if rc or not step_done(run, label, step):
            return dict(label=label, ok=False, step=step, ran=ran, seconds=time.perf_counter()-start)
        ran.append(step)
    return dict(label=label, ok=True, step="", ran=ran, seconds=time.perf_counter()-start)


def design_info(run, label):
    """Design values, hand stiffness/preload and map summary of a references/ or final/ design."""
    for path in (run/"references"/f"{label}.json", run/"final"/f"{label}.json"):
        if path.is_file():
            res = read_json(path)
            break
    else:
        return {}
    out = {k: res.get("design", {}).get(k) for k in DESIGN_KEYS}
    meta = run/"references"/f"{label}_meta.json"
    hand = read_json(meta).get("mounted_hand", {}) if meta.is_file() else {}
    out.update({short: hand.get(k) for k, short in HAND_KEYS.items()})
    sm = res.get("summary", {})
    out.update(J=sm.get("J"), map_success=sm.get("success"), map_env=sm.get("env"), map_tip=sm.get("tip"),
               map_hold_mixed=sm.get("failures", {}).get("hold_mixed", 0))
    return out


def write_table(run, rank_level):
    """all_designs.csv and summary.json/csv from every design's force_summary.json."""
    summaries = []
    for d in sorted((run/OUT_DIR).iterdir()) if (run/OUT_DIR).is_dir() else []:
        if (d/"force_summary.json").is_file():
            summaries.append(read_json(d/"force_summary.json"))
    if not summaries:
        print("No design has disturbance results yet.")
        return
    rows = []
    for sm in summaries:
        row = dict(label=sm["label"], **design_info(run, sm["label"]), reproduced_success=sm["reproduced_success"],
                   success=sm["success"], env=sm["env"], tip=sm["tip"], J_rerun=sm["J_rerun"])
        for L in sm["levels"]:
            lv = f"{L['level_mg']:g}mg"
            row.update({f"pass_{lv}": L["passed"], f"cond_{lv}": L["conditional_robustness"],
                        f"J_robust_{lv}": L["J_robust"]})
        for g in ("env", "tip", "hold_mixed"):
            info = sm["groups"].get(g, {})
            row[f"n_{g}"] = info.get("n", 0)
            row[f"F_min_median_{g}_mg"] = info.get("F_min_mg_median")
        row["criteria"] = " ".join(f"{k}={v}" for k, v in sm.get("criteria", {}).items())
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))  # Designs may differ in levels.
    rows = [{k: r.get(k) for k in keys} for r in rows]
    save_table(run/OUT_DIR/"all_designs.csv", rows)
    write_summary(run, summaries)
    print(f"\n{len(rows)} designs -> {run/OUT_DIR/'all_designs.csv'}")
    col = f"J_robust_{rank_level:g}mg"
    ranked = sorted((r for r in rows if r.get(col) is not None), key=lambda r: -r[col])
    if ranked:
        print(f"Top 10 by {col} (J = map objective, cond = conditional robustness at {rank_level:g} mg):")
        for r in ranked[:10]:
            cond = r.get(f"cond_{rank_level:g}mg")
            print(f"  {r['label']:<24} {col}={r[col]:7.2f}  pass={r[f'pass_{rank_level:g}mg']:>3}  "
                  f"cond={'-' if cond is None else f'{cond:.2f}'}  J={r['J'] if r['J'] is None else round(r['J'], 2)}"
                  f"  success={r['success']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=ROOT/"runs"/"rb5_cmaes_final")
    p.add_argument("--xml-dir", nargs="+", default=[str(ROOT.parent/"xml")],
                   help="Folder with the design XMLs (default: ../xml). Unquoted spaces are joined back")
    p.add_argument("--pattern", default="*.xml", help="File pattern in --xml-dir (default *.xml)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1),
                   help="Designs simulated at the same time (default: CPU count - 1)")
    p.add_argument("--grid", help="Grid for the workspace map (default: the run's final grid). "
                   "Only for tests: every design must use the same grid")
    p.add_argument("--force-args", default="", help='Extra disturb_force.py options, e.g. "--slip-mm 20"')
    p.add_argument("--rank-level", type=float, default=1.0, help="Level [m*g] for the printed ranking")
    p.add_argument("--only-table", action="store_true", help="Only rebuild all_designs.csv")
    args = p.parse_args()
    run = args.run.resolve()
    if not (run/"config_used.json").is_file():
        raise SystemExit(f"No config_used.json in {run}.")
    if not args.only_table:
        # PowerShell splits an unquoted path at its spaces ("BEAR lab"): join it back.
        xml_dir = Path(" ".join(args.xml_dir)).resolve()
        if not xml_dir.is_dir():
            raise SystemExit(f"XML folder not found: {xml_dir}\n"
                             'Put the path in quotes, e.g. --xml-dir "C:\\...\\BEAR lab\\...\\xml", or use ..\\xml')
        args.xml_dir = xml_dir
        xmls = sorted(xml_dir.glob(args.pattern))
        if not xmls:
            raise SystemExit(f"No {args.pattern} in {xml_dir}")
        stems = [x.stem for x in xmls]
        dup = sorted({s for s in stems if stems.count(s) > 1})
        if dup:
            raise SystemExit(f"Duplicate design names: {dup}")
        clash = [s for s in stems if s in available_labels(run) and not (run/"references"/f"{s}.json").is_file()]
        if clash:
            raise SystemExit(f"{clash} are already final/ results of the run. Rename those XML files.")
        todo = [(x.stem, x) for x in xmls if not all(step_done(run, x.stem, s) for s in STEPS)]
        print(f"{len(xmls)} designs in {args.xml_dir}, {len(xmls)-len(todo)} already done, "
              f"{len(todo)} to run with {args.workers} at a time. Logs: {run/OUT_DIR/'_logs'}", flush=True)
        log_dir = run/OUT_DIR/"_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        wall, finished, failed = time.perf_counter(), 0, []
        pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
        try:
            futures = [pool.submit(process, run, label, xml, args, log_dir) for label, xml in todo]
            for f in as_completed(futures):
                r = f.result()
                finished += 1
                elapsed = time.perf_counter()-wall
                eta = elapsed/finished*(len(todo)-finished)
                if r["ok"]:
                    print(f"  [{finished}/{len(todo)}] {r['label']}: done ({'+'.join(r['ran']) or 'nothing'}, "
                          f"{r['seconds']/60:.1f} min) | elapsed {elapsed/60:.0f} min, ~{eta/60:.0f} min left",
                          flush=True)
                else:
                    failed.append(r)
                    print(f"  [{finished}/{len(todo)}] {r['label']}: FAILED in {r['step']} "
                          f"(see _logs/{r['label']}.log)", flush=True)
        except KeyboardInterrupt:
            STOP.set()
            pool.shutdown(wait=False, cancel_futures=True)
            raise SystemExit("\nStopped. Run the same command again to continue.")
        pool.shutdown()
        if failed:
            print(f"\n{len(failed)} design(s) failed: {[r['label'] for r in failed]}")
    write_table(run, args.rank_level)


if __name__ == "__main__":
    main()
