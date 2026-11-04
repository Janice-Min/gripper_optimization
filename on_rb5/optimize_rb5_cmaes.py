"""CMA-ES over (length, Lp ratio, G, theta_p) with the design168 hand mounted on the RB5.

Every design is scored by full RB5 pick-and-lift rollouts (rb5_eval.py) on a fixed
cell grid, J = env + w_tip*tip + lambda_force*sum(m_F). Work is split into
(design, radius) tasks, so a generation, the baseline and the final check all use
every worker. Checkpoints are written after each generation; --resume continues.

  python optimize_rb5_cmaes.py --grid rb5_smoke --generations 2 --popsize 4 --restarts 0 --no-final --out runs/rb5_smoke
  python optimize_rb5_cmaes.py --baseline-only --out runs/rb5_timing
  python optimize_rb5_cmaes.py --workers 8 --out runs/rb5_cmaes
  python optimize_rb5_cmaes.py --resume --workers 8 --out runs/rb5_cmaes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from gripper_core import (DesignSpace, VARIABLES, baseline_parameters, build_xml, grid_cells,
                          read_json, save_rows, write_json, np, mujoco)
from rb5_eval import (Rb5Arm, arm_options, best_cell, group_tasks, load_design, summarize,
                      validate_rb5_config)
try:
    import cma
except ImportError as exc:
    raise SystemExit("Install dependencies: python -m pip install mujoco cma numpy") from exc

HERE = Path(__file__).resolve().parent
WORKER = {}


def init_worker(template, cfg):
    WORKER.clear()
    WORKER.update(template=template, cfg=cfg, arm=None)


def worker_task(task):
    index, design, r_mm, cells = task
    if WORKER["arm"] is None:
        WORKER["arm"] = Rb5Arm(WORKER["cfg"])
    start = time.perf_counter()
    rows = WORKER["arm"].run_cells(WORKER["template"], design, r_mm, cells)
    return index, rows, time.perf_counter()-start


def evaluate_many(designs, grid, cfg, pool):
    """[(summary, rows, cpu_seconds)] in the order of designs."""
    tasks = group_tasks(designs, grid)
    done = pool.map(worker_task, tasks) if pool else map(worker_task, tasks)
    rows, seconds = [[] for _ in designs], [0.0]*len(designs)
    for index, part, sec in done:
        rows[index].extend(part)
        seconds[index] += sec
    return [(summarize(rows[i], cfg), rows[i], seconds[i]) for i in range(len(designs))]


def sources_digest(cfg):
    src = HERE/arm_options(cfg)["sources_dir"]
    return "".join(hashlib.sha256((src/n).read_bytes()).hexdigest()
                   for n in ("rb5_850e.xml", "task_scene.xml"))


def config_signature(template, cfg, grid_name, start_design):
    source = (template+json.dumps(cfg, sort_keys=True)+grid_name+sources_digest(cfg)
              + json.dumps(start_design, sort_keys=True))
    return hashlib.sha256(source.encode()).hexdigest()


def key_of(design):
    return tuple(round(design[v], 8) for v in VARIABLES)


def save_state(out, state, es):
    stored = dict(state)
    stored["es_bytes"] = es.pickle_dumps() if es is not None else None
    stored["numpy_random_state"] = np.random.get_state()
    temp = out/"checkpoint.pkl.tmp"
    temp.write_bytes(pickle.dumps(stored, protocol=4))
    temp.replace(out/"checkpoint.pkl")
    rows = []
    for rec in state["records"]:
        s = rec["summary"]
        rows.append(dict(restart=rec["restart"], generation=rec["generation"], eval_id=rec["eval_id"],
                         label=rec.get("label", ""), **{k: rec["design"][k] for k in VARIABLES},
                         lp_mm=rec["design"]["lp_mm"], ld_mm=rec["design"]["ld_mm"],
                         theta_d_deg=rec["design"]["theta_d_deg"], J=s["J"], env=s["env"], tip=s["tip"],
                         success=s["success"], rb5_pass=s["rb5_pass"], prefiltered=s["prefiltered"],
                         cells=s["cells"], workspace_term=s["workspace_term"], force_term=s["force_term"],
                         cpu_seconds=rec["seconds"], failures=json.dumps(s["failures"], sort_keys=True)))
    with (out/"history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_record(state, design, summary, rows, seconds, restart, generation, label=""):
    rec = dict(eval_id=len(state["records"]), restart=restart, generation=generation,
               design=design, summary=summary, seconds=seconds, label=label)
    state["records"].append(rec)
    # Keep cell rows only for the best design, so exports need no re-run.
    if state.get("best_rows") is None or summary["J"] > state["best_J"]+1e-12:
        state.update(best_rows=rows, best_J=summary["J"], best_eval_id=rec["eval_id"])
    return rec


def save_best(out, state, template, cfg):
    best = state["records"][state["best_eval_id"]]
    write_json(out/"best_design.json", dict(design=best["design"], summary=best["summary"],
               grid=state["grid"], evaluation_id=best["eval_id"],
               note="Best observed on the RB5 search grid, not a guaranteed global maximum."))
    return best


def export_search_best(out, state, template, cfg):
    best = state["records"][state["best_eval_id"]]
    rows = state["best_rows"]
    save_rows(out/"best_workspace.csv", rows)
    cell = best_cell(rows)
    write_json(out/"viewer_cell.json", cell)
    (out/"best_design_hand.xml").write_text(build_xml(template, best["design"], cfg,
        radius_mm=cell["r_mm"], x_mm=cell["x_mm"], y_mm=cell["y_mm"]), encoding="utf-8")
    Rb5Arm(cfg).scene_file(out/"best_rb5_scene.xml", template, best["design"],
                           cell["r_mm"], cell["x_mm"], cell["y_mm"])


def start_strategy(space, cfg, restart, start_design):
    opt = cfg["optimizer"]
    if restart == 0:
        # The start design is evaluated exactly; the CMA mean is nudged inward only for sampling.
        u0 = np.clip(space.encode(start_design), 1e-6, 1-1e-6)
    else:
        u0 = np.random.default_rng(opt["seed"]+restart*1009).uniform(.1, .9, 4)
    return cma.CMAEvolutionStrategy(u0, opt["sigma0"], {
        "bounds": [0.0, 1.0], "popsize": opt["population"], "seed": opt["seed"]+restart,
        "maxiter": opt["generations"], "verbose": -9, "verb_log": 0,
        # The objective has genuine plateaus. Do not mistake equal scores for convergence.
        "tolfun": 0, "tolfunhist": 0, "tolflatfitness": opt["generations"]+1, "tolx": 1e-7,
    })


def final_evaluation(out, state, template, cfg, pool, final_grid_name, references):
    target = out/"final"
    target.mkdir(exist_ok=True)
    grid = cfg["grids"][final_grid_name]
    candidates, seen = [], set()
    for label, design in references:
        if key_of(design) not in seen:
            candidates.append((label, design))
            seen.add(key_of(design))
    ranked = sorted(state["records"], key=lambda r: r["summary"]["J"], reverse=True)
    n = 0
    for rec in ranked:
        if n >= cfg["optimizer"]["final_top_k"]:
            break
        if key_of(rec["design"]) in seen:
            continue
        n += 1
        candidates.append((f"candidate_{n:02d}", rec["design"]))
        seen.add(key_of(rec["design"]))
    print(f"Final evaluation on '{final_grid_name}': {len(candidates)} designs x "
          f"{len(grid_cells(grid))} cells", flush=True)
    results = []
    for (label, design), (summary, rows, sec) in zip(candidates,
            evaluate_many([d for _, d in candidates], grid, cfg, pool)):
        save_rows(target/f"{label}_workspace.csv", rows)
        result = dict(label=label, design=design, summary=summary)
        write_json(target/f"{label}.json", result)
        results.append((result, rows))
        print(f"  {label}: J={summary['J']:.4f}, env={summary['env']}, tip={summary['tip']}, "
              f"rb5_pass={summary['rb5_pass']}/{summary['cells']}", flush=True)
    best, rows = max(results, key=lambda t: t[0]["summary"]["J"])
    write_json(target/"comparison.json", dict(grid=grid, results=[r for r, _ in results],
                                               best_label=best["label"]))
    save_rows(target/"best_workspace.csv", rows)
    write_json(out/"best_design_fine.json", dict(**best, grid=grid))
    cell = best_cell(rows)
    write_json(out/"viewer_cell_fine.json", cell)
    (out/"best_design_fine_hand.xml").write_text(build_xml(template, best["design"], cfg,
        radius_mm=cell["r_mm"], x_mm=cell["x_mm"], y_mm=cell["y_mm"]), encoding="utf-8")
    Rb5Arm(cfg).scene_file(out/"best_rb5_scene_fine.xml", template, best["design"],
                           cell["r_mm"], cell["x_mm"], cell["y_mm"])
    return best


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, default=HERE/"design168.xml")
    p.add_argument("--config", type=Path, default=HERE/"config_rb5_arm.json")
    p.add_argument("--out", type=Path, default=HERE/"runs"/"rb5_cmaes_design168")
    p.add_argument("--grid", default="rb5_coarse", help="Search grid name in the config")
    p.add_argument("--final-grid", default="rb5_fine", help="Grid for the final re-evaluation")
    p.add_argument("--start-from", type=Path,
                   help="JSON with a 'design' entry (e.g. the hand-only best_design_fine.json). "
                        "Evaluated exactly and used as the first CMA mean instead of design168.")
    p.add_argument("--generations", type=int)
    p.add_argument("--popsize", type=int)
    p.add_argument("--restarts", type=int, help="Additional independent random starts")
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--baseline-only", action="store_true", help="Evaluate references, report time, stop")
    p.add_argument("--no-final", action="store_true", help="Skip final-grid re-evaluation")
    p.add_argument("--resume", action="store_true", help="Resume your own checkpoint in --out")
    return p


def main():
    args = parser().parse_args()
    out = args.out.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.resume:
        if not (out/"checkpoint.pkl").is_file():
            raise FileNotFoundError(f"No checkpoint in {out}")
        cfg = read_json(out/"config_used.json")
        template = (out/"source_design168.xml").read_text(encoding="utf-8-sig")
        # Only unpickle files you generated yourself. Never load an untrusted checkpoint.
        state = pickle.loads((out/"checkpoint.pkl").read_bytes())
        args.grid = state["grid_name"]
        if any(x is not None for x in (args.generations, args.popsize, args.seed, args.restarts, args.start_from)):
            raise ValueError("Resume uses the saved search settings; omit optimizer overrides.")
        if state["signature"] != config_signature(template, cfg, args.grid, state["start_design"]):
            raise ValueError("Template, config or sources/ changed. Start a new output directory.")
        versions = {"mujoco": mujoco.__version__, "cma": cma.__version__, "numpy": np.__version__}
        if state["versions"] != versions:
            raise ValueError("Resume requires the same MuJoCo, cma and numpy versions as the saved run.")
        es = pickle.loads(state.pop("es_bytes")) if state["es_bytes"] else None
        np.random.set_state(state.pop("numpy_random_state"))
        if args.baseline_only:
            # Never let "--resume --baseline-only" start the long search by accident.
            for rec in state["records"]:
                if rec["restart"] == -1:
                    s = rec["summary"]
                    print(f"{rec['label']}: J={s['J']:.4f}, success={s['success']}/{s['cells']} "
                          f"(env={s['env']}, tip={s['tip']}), cpu={rec['seconds']:.0f}s\n"
                          f"  failures: {s['failures']}")
            return
    else:
        if out.exists() and any(out.iterdir()):
            if (out/"checkpoint.pkl").is_file():
                hint = ("It already holds a run (results: baseline_summary.json, history.csv). "
                        "Continue it with --resume, or pick a new --out.")
            else:
                hint = ("It holds files from a run that stopped before its first checkpoint "
                        "(see the error printed back then). Delete the folder or pick a new --out.")
            raise SystemExit(f"Output folder is not empty: {out}\n{hint}")
        cfg = read_json(args.config)
        for arg, key in ((args.generations, "generations"), (args.popsize, "population"),
                         (args.restarts, "restarts"), (args.seed, "seed")):
            if arg is not None:
                cfg["optimizer"][key] = arg
        template = args.xml.read_text(encoding="utf-8-sig")
        state, es = None, None
    validate_rb5_config(cfg)
    for name in (args.grid, args.final_grid):
        if name not in cfg["grids"]:
            raise ValueError(f"Grid '{name}' is not in the config. Available: {sorted(cfg['grids'])}")
    opt = cfg["optimizer"]
    if opt["population"] < 4 or opt["generations"] < 1 or opt["restarts"] < 0 or opt["seed"] <= 0:
        raise ValueError("Require population>=4, generations>=1, restarts>=0 and seed>0")
    base = baseline_parameters(template)
    space = DesignSpace(cfg, base)
    baseline = space.derived({v: float(base[v]) for v in VARIABLES})
    space.encode(baseline)
    if state is None:
        start_design = load_design(args.start_from, template, cfg) if args.start_from else None
    else:
        start_design = state["start_design"]
    grid = cfg["grids"][args.grid]
    out.mkdir(parents=True, exist_ok=True)
    Rb5Arm(cfg)  # Fail early if sources/ is missing.
    n_cells = len(grid_cells(grid))
    print(f"RB5 search grid '{args.grid}': {n_cells} cells/design, "
          f"{len(group_tasks([baseline], grid))} tasks/design, {args.workers} workers", flush=True)
    print(f"Objective: env + {cfg['objective']['w_tip']}*tip + {cfg['objective']['lambda_force']}*sum(m_F); "
          f"rb5_arm={arm_options(cfg)}", flush=True)

    init_worker(template, cfg)
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                initializer=init_worker, initargs=(template, cfg))
            if args.workers > 1 else None)
    references = [("baseline", baseline)] + ([("start", start_design)] if start_design else [])
    interrupted = False
    try:
        if state is None:
            write_json(out/"config_used.json", cfg)
            (out/"source_design168.xml").write_text(template, encoding="utf-8")
            metadata = dict(python=platform.python_version(), mujoco=mujoco.__version__, cma=cma.__version__,
                            numpy=np.__version__, platform=platform.platform(),
                            input_sha256=hashlib.sha256(template.encode()).hexdigest())
            write_json(out/"environment.json", metadata)
            print("Evaluating reference designs on the RB5 ...", flush=True)
            wall = time.perf_counter()
            evaluated = evaluate_many([d for _, d in references], grid, cfg, pool)
            wall = time.perf_counter()-wall
            state = dict(signature=config_signature(template, cfg, args.grid, start_design), grid=grid,
                         versions={k: metadata[k] for k in ("mujoco", "cma", "numpy")},
                         grid_name=args.grid, start_design=start_design, restart=0, generation=0,
                         stall=0, restart_best=-1.0, finished=False, records=[], best_rows=None, best_J=None)
            for (label, design), (summary, rows, sec) in zip(references, evaluated):
                add_record(state, design, summary, rows, sec, -1, 0, label)
                save_rows(out/f"{label}_workspace.csv", rows)
                write_json(out/f"{label}_summary.json", summary)
                print(f"{label}: J={summary['J']:.4f}, success={summary['success']}/{summary['cells']} "
                      f"(env={summary['env']}, tip={summary['tip']}), rb5_pass={summary['rb5_pass']}, "
                      f"prefiltered={summary['prefiltered']}, cpu={sec:.0f}s", flush=True)
                print(f"  failures: {summary['failures']}", flush=True)
            per_design = max(sec for _, _, sec in evaluated)
            budget = (opt["restarts"]+1)*opt["generations"]*opt["population"]
            print(f"Wall time for references: {wall:.0f}s. Search budget <= {budget} designs; "
                  f"estimate ~{budget*per_design/args.workers/3600:.1f} h with {args.workers} workers "
                  "(+ final evaluation; candidate run times vary).", flush=True)
            save_state(out, state, None)
            save_best(out, state, template, cfg)
            if args.baseline_only:
                return
        while state["restart"] <= opt["restarts"] and not state["finished"]:
            if es is None:
                es = start_strategy(space, cfg, state["restart"], start_design or baseline)
                state.update(generation=0, stall=0, restart_best=-1.0)
                save_state(out, state, es)
            if state["generation"] >= opt["generations"] or es.stop() or state["stall"] >= opt["stall_generations"]:
                print(f"Restart {state['restart']} ended: {dict(es.stop()) or 'score plateau'}", flush=True)
                state["restart"] += 1
                es = None
                save_state(out, state, es)
                continue
            # Checkpoint before ask and after tell: a partial generation is rerun exactly.
            population = es.ask()
            designs = [space.decode(u) for u in population]
            wall = time.perf_counter()
            evaluated = evaluate_many(designs, grid, cfg, pool)
            scores = [s["J"] for s, _, _ in evaluated]
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite objective score")
            es.tell(population, [-s for s in scores])  # CMA minimizes; we maximize J.
            state["generation"] += 1
            for design, (summary, rows, sec) in zip(designs, evaluated):
                add_record(state, design, summary, rows, sec, state["restart"], state["generation"])
            if max(scores) > state["restart_best"]+1e-12:
                state["restart_best"], state["stall"] = max(scores), 0
            else:
                state["stall"] += 1
            save_state(out, state, es)
            b = save_best(out, state, template, cfg)
            d = b["design"]
            print(f"R{state['restart']} G{state['generation']:03d} | gen_best={max(scores):.4f} | "
                  f"best J={b['summary']['J']:.4f} (env={b['summary']['env']}, tip={b['summary']['tip']}) "
                  f"L={d['length_mm']:.1f} ratio={d['lp_ratio']:.3f} G={d['gap_mm']:.1f} "
                  f"th_p={d['theta_p_deg']:.1f} | {time.perf_counter()-wall:.0f}s", flush=True)
        state["finished"] = True
        save_state(out, state, es)
        best = save_best(out, state, template, cfg)
        if best["summary"]["success"] == 0:
            print("WARNING: no successful cells found. Inspect failure counts.", flush=True)
        export_search_best(out, state, template, cfg)
        if not args.no_final:
            final = final_evaluation(out, state, template, cfg, pool, args.final_grid, references)
            print(f"Final-grid best: {final['label']}, J={final['summary']['J']:.4f}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Last completed generation is saved; restart with --resume.", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    if not interrupted:
        print(f"Saved results: {out}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()