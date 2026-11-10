"""Disturbance experiment, step 2 (method A): slowly increasing force -> holding limit F_max.

Needs the states of disturb_collect.py. For every saved cell and every direction the
grasp state at the end of 'lift hold' is restored (bit-identical to continuing the
pick-and-lift) and a force on the cylinder's center of mass (xfrc_applied) is ramped
    F(t) = rate * t,  0 -> cap (default 1 N/s up to 5 mg)
while the arm holds its last command and the tendon keeps its tension. Gravity stays on,
so -z (down) adds to the object's weight and +z lifts it.

Directions are in the hand frame (same frame as the workspace map):
  +x / -x   jaw-width direction, toward the R / L finger
  +y / -y   out of the hand (pull-out) / into the palm
  +z / -z   up / down along the cylinder axis (world z)

A direction fails at the first sample where (criteria are options, defaults from config):
  slip      object moved >= --slip-mm relative to the hand        (hold_displacement_mm)
  rotation  object turned >= --rot-deg relative to the hand       (hold_rotation_deg; 0 = off)
  lost      R or L finger normal force < 0.10 N for >= --grace-s   (lost_contact_grace_s)
F_max = applied force at that sample. If nothing fails up to the cap, F_max = cap and the
row is 'censored' (the true limit is higher). A run stopped by an unintended collision or
instability is censored at the force reached. The ramp of a direction stops early once the
object is clearly gone (moved >= stop_mm or no bilateral contact for stop_lost_s), so the
saved traces let --rescore re-judge with any slip <= stop_mm and grace <= stop_lost_s.

Pass at level k: F_max >= k*m*g in EVERY direction. Per design and level:
  conditional robustness = passing cells / re-run success cells   (grasp quality)
  robust workspace       = passing success cells out of all map cells (e.g. 792)
  J_robust               = env_pass + w_tip*tip_pass + lambda*sum(m_F), same scale as J
Groups env / tip / tip with m_F<0 / hold_mixed are summarized separately; the tip/env
pass ratio is an empirical check of w_tip, the hold_mixed group of the "mixed fails" rule.

  python disturb_force.py --run runs/rb5_cmaes_final --workers 8
  python disturb_force.py --run runs/rb5_cmaes_final --rescore --slip-mm 10 --rot-deg 0
  python disturb_force.py --run runs/rb5_cmaes_final --labels design001 --limit 2 --dirs nz py
(--dirs takes px nx py ny pz nz; +x/+y/+z also work, but argparse reads -x as an option)

Writes <run>/disturbance/<label>/
  traces/*.npz      per cell and direction ('px_trace', ...; column names in 'columns'):
                    t, F, hand-frame displacement, rotation, link normal forces, finger
                    joint change, plus how the ramp ended
  force_dirs.csv    one row per cell and direction
  force_cells.csv   one row per cell: F_max of each direction, weakest direction, pass_*
  force_meta.json   ramp settings and the criteria of the last analysis
and <run>/disturbance/summary.json, summary.csv (all designs analyzed so far).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import mujoco

from gripper_core import JOINTS, LINKS, read_json, save_rows, write_json
from rb5_eval import Rb5Arm, validate_rb5_config
from eval_design_rb5 import read_rows, to_json
from disturb_collect import (DEFAULT_LABELS, OUT_DIR, ROOT, label_dir, resolve_labels, run_tasks,
                             save_table, scene_signature, set_state)
from run_pick_lift import ARM_JOINTS, PickLift

DIRECTIONS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
              "+z": (0, 0, 1), "-z": (0, 0, -1)}
BILATERAL_N = 0.10  # run_pick_lift.PickLift: each side's finger force sum must exceed this.
DEFAULT_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
# Trace columns: t, F, object displacement in the hand frame (3), rotation, link normal
# forces R_gp R_gd L_gp L_gd palm (5), finger joint change R_jp R_jd L_jp L_jd [deg] (4).
TRACE_COLUMNS = ("t_s", "F_N", "dx_mm", "dy_mm", "dz_mm", "rot_deg", *(f"{n}_N" for n in LINKS),
                 *(f"d{n}_deg" for n in JOINTS))
WORKER = {}


def key(direction):
    return direction.replace("+", "p").replace("-", "n")


# ---------------------------------------------------------------- simulation
def link_forces(model, data, og, links):
    """Normal force of the object on R_gp, R_gd, L_gp, L_gd, palm (as CellMonitor.forces)."""
    f, buf = np.zeros(len(LINKS)), np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if og not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == og else c.geom1
        if other in links:
            mujoco.mj_contactForce(model, data, i, buf)
            f[links[other]] += max(0.0, float(buf[0]))
    return f


def ramp(model, data, state, d_hand, s):
    """Restore the post-lift state and ramp the force along hand direction d_hand."""
    set_state(model, data, state)
    ob, og, site = model.body("obj").id, model.geom("obj_g").id, model.site("grasp").id
    links = {model.geom(n).id: i for i, n in enumerate(LINKS)}
    armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
    qa = np.array([model.joint(n).qposadr[0] for n in JOINTS])
    q0 = data.qpos[qa].copy()
    dt = model.opt.timestep
    every = max(1, round(s["sample_s"]/dt))
    rs0 = data.site_xmat[site].reshape(3, 3).copy()
    p0 = rs0.T @ (data.xpos[ob]-data.site_xpos[site])
    r0 = rs0.T @ data.xmat[ob].reshape(3, 3)
    direction = rs0 @ np.asarray(d_hand, float)
    sim = PickLift(model, data)
    sim.stage = "disturb"
    sim.qcmd = data.ctrl[armact].copy()  # The arm keeps holding its last command.
    rows = []

    def sample(force):
        rs = data.site_xmat[site].reshape(3, 3)
        p = rs.T @ (data.xpos[ob]-data.site_xpos[site])
        rel = r0.T @ (rs.T @ data.xmat[ob].reshape(3, 3))
        rot = math.degrees(math.acos(np.clip((np.trace(rel)-1)/2, -1, 1)))
        f = link_forces(model, data, og, links)
        rows.append([data.time, force, *(1000*(p-p0)), rot, *f, *np.degrees(data.qpos[qa]-q0)])
        return 1000*float(np.linalg.norm(p-p0)), min(f[0]+f[1], f[2]+f[3]) > BILATERAL_N

    sample(0.0)
    n = math.ceil(s["cap_N"]/s["rate_N_s"]/dt-1e-9)
    ended, lost_since, t_start = "cap", None, data.time
    try:
        for i in range(n):
            force = min(s["rate_N_s"]*(i+1)*dt, s["cap_N"])
            data.xfrc_applied[ob, :3] = force*direction
            sim.step()
            if (i+1) % every and i != n-1:
                continue
            disp, bilateral = sample(force)
            lost_since = None if bilateral else (data.time if lost_since is None else lost_since)
            if disp >= s["stop_mm"] or (lost_since is not None and data.time-lost_since >= s["stop_lost_s"]-1e-9):
                ended = "drop"
                break
    except RuntimeError as exc:
        ended = "unintended_collision" if str(exc).startswith("Unintended collision") else "unstable"
        if rows[-1][1] != force:
            sample(force)
    data.xfrc_applied[:] = 0
    trace = np.asarray(rows, dtype=np.float64)
    trace[:, 0] -= t_start
    return dict(trace=trace, ended=ended, direction_world=direction,
                penetration_mm=1000*sim.max_unintended_penetration)


def model_for(hand_file, cfg):
    """Compiled RB5 scene of a saved hand XML, cached per worker process."""
    if hand_file not in WORKER:
        arm = WORKER.setdefault("_arm", Rb5Arm(cfg))
        hand_xml = Path(hand_file).read_text(encoding="utf-8")
        WORKER[hand_file] = (arm.compile(hand_xml), scene_signature(arm, hand_xml))
    return WORKER[hand_file]


def force_task(task):
    """All directions of the given cells (one radius). Writes one traces/*.npz per cell."""
    out, cfg, s, cells = task
    out = Path(out)
    done, start = [], time.perf_counter()
    for c in cells:
        st = np.load(out/c["state_file"])
        model, sig = model_for(str(out/"hands"/str(st["hand_file"])), cfg)
        if str(st["signature"]) != sig:
            raise SystemExit(f"{c['state_file']}: the scene (hand XML, sources/ or MuJoCo version) "
                             "differs from the one the state was saved with. Run disturb_collect.py again.")
        r, x, y = float(st["r_mm"]), float(st["x_mm"]), float(st["y_mm"])
        model.site_pos[model.site("grasp").id] = [x/1000, (y+r)/1000, 0.0]
        data = mujoco.MjData(model)
        saved = {}
        for name in s["dirs"]:
            t0 = time.perf_counter()
            res = ramp(model, data, st["state"], DIRECTIONS[name], s)
            k = key(name)
            saved.update({f"{k}_trace": res["trace"].astype(np.float32), f"{k}_ended": res["ended"],
                          f"{k}_world": res["direction_world"], f"{k}_pen_mm": res["penetration_mm"],
                          f"{k}_sec": time.perf_counter()-t0})
        np.savez_compressed(out/"traces"/Path(c["state_file"]).name, r_mm=r, x_mm=x, y_mm=y,
                            dirs=np.array(s["dirs"]), columns=np.array(TRACE_COLUMNS), **saved)
        done.append(dict(r_mm=r, x_mm=x, y_mm=y, ended=[str(saved[f"{key(d)}_ended"]) for d in s["dirs"]]))
    return done, time.perf_counter()-start


# ---------------------------------------------------------------- analysis
def first_index(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def judge_trace(trace, ended, crit, cap_N):
    """F_max and its cause from one direction's trace."""
    t, force = trace[:, 0], trace[:, 1]
    disp = np.linalg.norm(trace[:, 2:5], axis=1)
    rot, f, dq = trace[:, 5], trace[:, 6:11], np.abs(trace[:, 11:15])
    events = {"slip": first_index(disp >= crit["slip_mm"])}
    if crit["rot_deg"] > 0:
        events["rotation"] = first_index(rot >= crit["rot_deg"])
    lost = np.minimum(f[:, 0]+f[:, 1], f[:, 2]+f[:, 3]) <= BILATERAL_N
    onset, events["lost"] = None, None
    for i, bad in enumerate(lost):  # Onset of the first lost streak lasting >= grace.
        if not bad:
            onset = None
            continue
        onset = i if onset is None else onset
        if t[i]-t[onset] >= crit["grace_s"]-1e-9:
            events["lost"] = onset
            break
    out = {f"F_{k}_N": (float(force[i]) if i is not None else None) for k, i in events.items()}
    fired = {k: i for k, i in events.items() if i is not None}
    if fired:
        cause = min(fired, key=lambda k: (fired[k], k))
        i = fired[cause]
        censored = False
    elif ended == "drop":  # Only when re-judged with criteria looser than the stop rule.
        cause, i, censored = "drop", len(force)-1, False
    else:
        cause, i, censored = ended, len(force)-1, True  # cap: survived; error: lower bound.
    out.update(F_max_N=float(cap_N if cause == "cap" else force[i]), cause=cause, censored=censored,
               ended_by=ended, F_end_N=float(force[-1]), max_disp_mm=float(disp.max()),
               max_rot_deg=float(rot.max()), disp_at_Fmax_mm=float(disp[i]),
               finger_dq_at_Fmax_deg=float(dq[i].max()))
    return out


def group_of(cell):
    return cell["contact_type"] if cell["success"] else cell["reason"]


def analyze(run, label, cfg, crit, levels):
    """Rebuild force_dirs.csv, force_cells.csv and the design's summary from the traces."""
    out = label_dir(run, label)
    meta = read_json(out/"force_meta.json")
    s = meta["settings"]
    mg, cap_N = s["mg_N"], s["cap_N"]
    if crit["slip_mm"] > s["stop_mm"] or crit["grace_s"] > s["stop_lost_s"]:
        print(f"Warning: criteria looser than the stop rule (stop_mm={s['stop_mm']}, "
              f"stop_lost_s={s['stop_lost_s']}); stopped ramps count as 'drop' failures.")
    cells = [c for c in read_rows(out/"cells.csv") if c["state_file"]]
    dir_rows, cell_rows = [], []
    for c in cells:
        path = out/"traces"/Path(c["state_file"]).name
        if not path.is_file():
            continue
        tr = np.load(path)
        info = dict(r_mm=c["r_mm"], x_mm=c["x_mm"], y_mm=c["y_mm"], group=group_of(c),
                    orig_group=c["orig_group"], margin_negative=bool(c["success"]) and c["m_F"] < 0,
                    reproduced=c["reproduced"], m_F=c["m_F"])
        per = {}
        for name in [str(d) for d in tr["dirs"]]:
            k = key(name)
            res = judge_trace(tr[f"{k}_trace"].astype(float), str(tr[f"{k}_ended"]), crit, cap_N)
            res["F_max_mg"] = res["F_max_N"]/mg
            per[name] = res
            dir_rows.append(dict(info, dir=name, **res, max_unintended_penetration_mm=float(tr[f"{k}_pen_mm"]),
                                 wall_s=float(tr[f"{k}_sec"])))
        weakest = min(per, key=lambda d: per[d]["F_max_N"])
        row = dict(info, contact_type=c["contact_type"], F_min_mg=per[weakest]["F_max_mg"], weakest_dir=weakest,
                   weakest_cause=per[weakest]["cause"],
                   any_error=any(v["cause"] in ("unintended_collision", "unstable") for v in per.values()))
        row.update({f"F_{d}_mg": per[d]["F_max_mg"] for d in per})
        row.update({f"pass_{lv:g}mg": per[weakest]["F_max_N"] >= lv*mg-1e-9 for lv in levels})
        cell_rows.append(row)
    if cells and not cell_rows:
        raise SystemExit(f"No traces in {out/'traces'}. Run without --rescore first.")
    save_table(out/"force_dirs.csv", dir_rows)
    save_table(out/"force_cells.csv", cell_rows)
    meta["criteria"], meta["levels_mg"] = crit, list(levels)
    write_json(out/"force_meta.json", meta)
    sm = summarize(label, read_json(out/"meta.json"), cell_rows, dir_rows, cfg, levels, s)
    sm["criteria"] = crit
    write_json(out/"force_summary.json", json.loads(json.dumps(sm, default=to_json)))
    return sm


def pass_rate(rows, lv):
    return (sum(r[f"pass_{lv:g}mg"] for r in rows)/len(rows)) if rows else None


def summarize(label, collect_meta, cells, dirs, cfg, levels, s):
    o = cfg["objective"]
    ok = [c for c in cells if c["group"] in ("env", "tip")]
    env, tip = [c for c in ok if c["group"] == "env"], [c for c in ok if c["group"] == "tip"]
    n_map = collect_meta["cells_in_map"]
    out = dict(label=label, cells_in_map=n_map, orig_success=collect_meta["orig_success"],
               reproduced_success=collect_meta["reproduced_success"], success=len(ok), env=len(env),
               tip=len(tip), J_rerun=len(env)+o["w_tip"]*len(tip)+o["lambda_force"]*sum(c["m_F"] for c in ok),
               directions=s["dirs"], cap_mg=s["cap_mg"],
               causes=dict(Counter(d["cause"] for d in dirs if d["group"] in ("env", "tip"))),
               levels=[], groups={})
    for lv in levels:
        p = [c for c in ok if c[f"pass_{lv:g}mg"]]
        pe, pt = sum(c["group"] == "env" for c in p), sum(c["group"] == "tip" for c in p)
        per_dir = {d: sum(r["F_max_N"] >= lv*s["mg_N"]-1e-9 for r in dirs
                          if r["dir"] == d and r["group"] in ("env", "tip")) for d in s["dirs"]}
        out["levels"].append(dict(level_mg=lv, passed=len(p), env=pe, tip=pt,
                                  conditional_robustness=len(p)/len(ok) if ok else None,
                                  robust_workspace=len(p), robust_workspace_fraction=len(p)/n_map,
                                  J_robust=pe+o["w_tip"]*pt+o["lambda_force"]*sum(c["m_F"] for c in p),
                                  per_direction=per_dir))
    groups = {"env": env, "tip": tip, "tip_margin_negative": [c for c in tip if c["margin_negative"]]}
    for c in cells:
        if c["group"] not in ("env", "tip"):
            groups.setdefault(c["group"], []).append(c)
    for name, rows in groups.items():
        f = np.array([r["F_min_mg"] for r in rows])
        out["groups"][name] = dict(
            n=len(rows), F_min_mg_median=float(np.median(f)) if len(f) else None,
            F_min_mg_min=float(f.min()) if len(f) else None, F_min_mg_max=float(f.max()) if len(f) else None,
            weakest_dirs=dict(Counter(r["weakest_dir"] for r in rows)),
            pass_rate={f"{lv:g}mg": pass_rate(rows, lv) for lv in levels},
            cells=[[r["r_mm"], r["x_mm"], r["y_mm"], round(r["F_min_mg"], 3)] for r in rows] if len(rows) <= 12 else None)
    pe, pt = out["groups"]["env"]["pass_rate"], out["groups"]["tip"]["pass_rate"]
    out["tip_over_env_pass_ratio"] = {k: (pt[k]/pe[k] if pe[k] and pt[k] is not None else None) for k in pe}
    return out


def print_summary(sm):
    print(f"\n[{sm['label']}] re-run success {sm['success']} (env {sm['env']}, tip {sm['tip']}) of "
          f"{sm['orig_success']} original, map {sm['cells_in_map']} cells, J_rerun={sm['J_rerun']:.2f}")
    print(f"  failure causes (success cells x directions): {sm['causes']}")
    print(f"  {'level':>6} {'pass':>5} {'env':>4} {'tip':>4} {'cond.robust':>11} {'robust WS':>10} {'J_robust':>9}  "
          + " ".join(f"{d:>4}" for d in sm["directions"]))
    for L in sm["levels"]:
        cr = "-" if L["conditional_robustness"] is None else f"{L['conditional_robustness']:.2f}"
        print(f"  {L['level_mg']:>4g}mg {L['passed']:>5} {L['env']:>4} {L['tip']:>4} {cr:>11} "
              f"{L['robust_workspace']:>4}/{sm['cells_in_map']:<5} {L['J_robust']:>9.2f}  "
              + " ".join(f"{L['per_direction'][d]:>4}" for d in sm["directions"]))
    print("  groups (F_min = weakest direction, in m*g):")
    for name, g in sm["groups"].items():
        if not g["n"]:
            continue
        rates = " ".join(f"{k}:{v:.2f}" for k, v in g["pass_rate"].items() if v is not None)
        print(f"    {name:<20} n={g['n']:<3} median {g['F_min_mg_median']:.2f}  min {g['F_min_mg_min']:.2f}  "
              f"weakest {g['weakest_dirs']}  pass {rates}")
    ratio = " ".join(f"{k}:{v:.2f}" for k, v in sm["tip_over_env_pass_ratio"].items() if v is not None)
    print(f"  tip/env pass-rate ratio (compare with w_tip): {ratio}")


def write_summary(run, summaries):
    base = Path(run)/OUT_DIR
    path = base/"summary.json"
    allsm = read_json(path) if path.is_file() else {}
    allsm.update({sm["label"]: sm for sm in summaries})
    write_json(path, json.loads(json.dumps(allsm, default=to_json)))
    flat = []
    for sm in allsm.values():
        for L in sm["levels"]:
            flat.append(dict(label=sm["label"], level_mg=L["level_mg"], success=sm["success"],
                             passed=L["passed"], env=L["env"], tip=L["tip"],
                             conditional_robustness=L["conditional_robustness"],
                             robust_workspace=L["robust_workspace"], cells_in_map=sm["cells_in_map"],
                             J_rerun=sm["J_rerun"], J_robust=L["J_robust"],
                             **{f"pass_{d}": L["per_direction"].get(d) for d in DIRECTIONS}))
    save_rows(base/"summary.csv", flat)
    return path


# ---------------------------------------------------------------- main
def simulate(run, label, cfg, s, workers, limit):
    out = label_dir(run, label)
    if not (out/"cells.csv").is_file():
        raise SystemExit(f"No {out/'cells.csv'}. Run disturb_collect.py first.")
    cells = [c for c in read_rows(out/"cells.csv") if c["state_file"]]
    if limit:
        cells = cells[:limit]
    (out/"traces").mkdir(exist_ok=True)
    for old in [*(out/"traces").glob("*.npz"), *out.glob("force_*")]:
        old.unlink()
    if not cells:  # Nothing was lifted: the design's summary is all zeros.
        write_json(out/"force_meta.json", dict(label=label, settings=dict(s, mg_N=None, cap_N=None),
                                               mujoco=mujoco.__version__))
        print(f"\n[{label}] no lifted cell, nothing to disturb", flush=True)
        return
    # mg from the scene: object mass x the config's gravity.
    st = np.load(out/cells[0]["state_file"])
    model, _ = model_for(str(out/"hands"/str(st["hand_file"])), cfg)
    s = dict(s, mg_N=float(model.body_mass[model.body("obj").id])*cfg["simulation"]["gravity_m_s2"])
    s["cap_N"] = s["cap_mg"]*s["mg_N"]
    write_json(out/"force_meta.json", dict(label=label, settings=s, mujoco=mujoco.__version__))
    by_r = {}
    for c in cells:
        by_r.setdefault(c["r_mm"], []).append(c)
    per = max(1, math.ceil(len(cells)/(3*max(1, workers))))
    tasks = [(str(out), cfg, s, g[i:i+per]) for _, g in sorted(by_r.items()) for i in range(0, len(g), per)]
    print(f"\n[{label}] {len(cells)} cells x {len(s['dirs'])} directions, ramp {s['rate_N_s']} N/s to "
          f"{s['cap_mg']:g} mg = {s['cap_N']:.3f} N ({s['cap_N']/s['rate_N_s']:.1f} s per direction)", flush=True)
    done, wall = [], time.perf_counter()

    def report(part, sec):
        done.extend(part)
        ends = Counter(e for p in part for e in p["ended"])
        print(f"  r={part[0]['r_mm']:g} mm: {len(part)} cells, {sec:.0f} s, ended {dict(ends)} "
              f"({len(done)}/{len(cells)}, {time.perf_counter()-wall:.0f} s)", flush=True)

    run_tasks(force_task, tasks, workers, report)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=ROOT/"runs"/"rb5_cmaes_final")
    p.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                   help="'all' (default: every design collected by disturb_collect.py) or names")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--rescore", action="store_true", help="Only re-judge saved traces (no simulation)")
    p.add_argument("--limit", type=int, help="Only the first N saved cells per design (smoke test)")
    g = p.add_argument_group("ramp (simulation)")
    g.add_argument("--rate", type=float, default=1.0, help="Force rate [N/s] (default 1)")
    g.add_argument("--max-mg", type=float, default=5.0, help="Ramp end in units of m*g (default 5)")
    g.add_argument("--dirs", nargs="+", default=list(DIRECTIONS), metavar="DIR",
                   choices=[*DIRECTIONS, *map(key, DIRECTIONS)],
                   help="px nx py ny pz nz (= +x -x +y -y +z -z); default all six")
    g.add_argument("--stop-mm", type=float, help="Stop a direction once the object moved this far "
                   "(default simulation.lift_slip_mm)")
    g.add_argument("--stop-lost-s", type=float, default=0.1, help="... or lost bilateral contact this long")
    c = p.add_argument_group("failure criteria (also for --rescore)")
    c.add_argument("--slip-mm", type=float, help="default simulation.hold_displacement_mm")
    c.add_argument("--rot-deg", type=float, help="default simulation.hold_rotation_deg; 0 disables")
    c.add_argument("--grace-s", type=float, help="default simulation.lost_contact_grace_s")
    c.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS), help="Pass levels [m*g]")
    p.add_argument("--no-summary", action="store_true",
                   help="Do not update the shared summary.json/csv (batch_disturbance.py writes it at the end)")
    args = p.parse_args()
    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    sc = cfg["simulation"]
    crit = dict(slip_mm=args.slip_mm if args.slip_mm is not None else sc["hold_displacement_mm"],
                rot_deg=args.rot_deg if args.rot_deg is not None else sc["hold_rotation_deg"],
                grace_s=args.grace_s if args.grace_s is not None else sc.get("lost_contact_grace_s", 0.02))
    collected = sorted(d.name for d in (run/OUT_DIR).glob("*") if (d/"cells.csv").is_file())
    labels = resolve_labels(args.labels, collected)
    print(f"Designs: {labels}")
    if not args.rescore:
        names = {key(d): d for d in DIRECTIONS}
        s = dict(rate_N_s=args.rate, cap_mg=args.max_mg, dirs=list(dict.fromkeys(names.get(d, d) for d in args.dirs)),
                 stop_mm=args.stop_mm if args.stop_mm is not None else sc["lift_slip_mm"],
                 stop_lost_s=args.stop_lost_s, sample_s=sc["sample_interval_s"])
        if args.rate <= 0 or args.max_mg <= 0 or s["stop_mm"] < crit["slip_mm"] or s["stop_lost_s"] < crit["grace_s"]:
            raise SystemExit("Need rate>0, max-mg>0, stop-mm >= slip-mm and stop-lost-s >= grace-s")
        for label in labels:
            simulate(run, label, cfg, s, args.workers, args.limit)
    summaries = [analyze(run, label, cfg, crit, sorted(args.levels)) for label in labels]
    print(f"\nCriteria: slip >= {crit['slip_mm']} mm, rotation >= {crit['rot_deg']} deg (0 = off), "
          f"lost contact >= {crit['grace_s']} s")
    for sm in summaries:
        print_summary(sm)
    if not args.no_summary:
        print("\nSaved:", write_summary(run, summaries))
    for label in labels:
        print("Saved:", label_dir(run, label)/"force_cells.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
