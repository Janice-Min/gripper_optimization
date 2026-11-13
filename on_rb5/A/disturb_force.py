"""Disturbance experiment A, step 2: slowly increasing force -> holding limit F_max.

Needs the states of disturb_collect.py. For every saved cell the grasp state at the end of
'lift hold' is restored (bit-identical to continuing the pick-and-lift) and, per direction,
a force on the cylinder's center of mass (xfrc_applied) is ramped
    F(t) = rate * t,  0 -> cap (default 1 N/s up to 10 mg)
while the arm holds its last command and the tendon keeps its tension (motor = force
control, 20 N). Gravity stays on, so -z (down) adds to the object's weight and +z lifts it;
F_max is the EXTERNAL force on top of gravity.

Directions are in the hand frame (same frame as the workspace map):
  +x / -x   jaw-width direction, toward the R / L finger
  +y / -y   out of the hand (pull-out) / into the palm
  +z / -z   up / down along the cylinder axis = out of the finger plane (world z)

A direction fails at the first sample where (criteria are options, defaults from config):
  slip      object moved >= --slip-mm relative to the hand        (hold_displacement_mm)
  rotation  object turned >= --rot-deg relative to the hand       (hold_rotation_deg; 0 = off;
            --rot-mode tilt counts only the tilt of the cylinder axis, not its spin)
  lost      R or L finger normal force < 0.10 N for >= --grace-s   (lost_contact_grace_s)
F_max = applied force at that sample. If nothing fails up to the cap, F_max = cap and the
row is 'censored' (the true limit is higher). A run stopped by an unintended collision or
instability is censored at the force reached. The ramp of a direction stops early once the
object is clearly gone (moved >= stop_mm or no bilateral contact for stop_lost_s), so the
saved traces let --rescore re-judge with any slip <= stop_mm and grace <= stop_lost_s
(e.g. --slip-mm 30: "a tip grasp that slid into an env grasp but did not drop" holds).

Also per cell (both on by default or with --refine):
  zero-force control  the same state held with F = 0 for the full ramp duration. Soft
              contacts creep; a success cell whose control already fails the criteria
              cannot hold the object even without load and counts as FAILED (group
              'zero_fail': out of the success cells, robust_area, pass levels and J_stable;
              --zero-fail keep only flags it). A direction is 'creep_limited' when the
              control failed no later than the ramp did.
  contact type along the ramp  env / tip / mixed / none from the link forces (moving
              average over force_average_window_s, rules of the config): type at F_max and
              the first force at which the type changed (e.g. tip -> env), so tip grasps
              that turn into env grasps under load can be told apart.
  --refine    hold test by bisection: largest CONSTANT force (smooth rise --rise-s, then
              --hold-s) the grasp holds without meeting a criterion, to --refine-tol-mg.
              Rate-independent definition of the holding force; costs ~5x the ramp. The
              trials are judged during the simulation, so --rescore with other criteria
              cannot re-judge them (a warning says so). --metric hold ranks by it.

Success cells below = re-run success cells (env, tip) minus the zero_fail ones.
J_stable = env + w_tip*tip + lambda*sum(m_F) of those (J_rerun: before the zero-force rule).
Pass at level k: F_max >= k*m*g in EVERY direction. Per design and level:
  conditional robustness = passing cells / re-run success cells   (grasp quality)
  robust workspace       = passing success cells out of all map cells (e.g. 792)
  J_robust               = env_pass + w_tip*tip_pass + lambda*sum(m_F), same scale as J
and, level-free,
  robust_area            = sum over success cells of min(F_min, cap) [mg]: the area under
                           the survival curve "cells holding >= F"; cells the design cannot
                           grasp count 0, so designs are compared on the same 792 cells.
                           Grows with the workspace; F_min_median_mg (median over the
                           success cells) is the size-independent grasp quality.
Groups env / tip / tip with m_F<0 / hold_mixed are summarized separately, including the
F_max distribution of every direction per group (direction_stats).

  python A/disturb_force.py --workers 8                                         (from on_rb5/)
  python A/disturb_force.py --rescore --slip-mm 10 --rot-deg 0
  python A/disturb_force.py --labels design001 --limit 2 --dirs nz py
  python A/disturb_force.py --labels candidate_02 --refine
(--dirs takes px nx py ny pz nz; +x/+y/+z also work, but argparse reads -x as an option)

Reads the grasp states in <run>/grasp_states/<label>/ (<run> = grasp/results/rb5_cmaes_final)
and writes A/results/<run name>/<label>/
  traces/*.npz      per cell: '<dir>_trace' per direction ('px_trace', ...; column names
                    in 'columns'), 'z0_trace' of the zero-force control, '<dir>_hold_*'
                    with --refine; t, F, hand-frame displacement, rotation, link normal
                    forces, finger joint change, hand (grasp site) motion in the world
  force_dirs.csv    one row per cell and direction
  force_cells.csv   one row per cell: F_max of each direction, weakest direction, pass_*,
                    zero-force control
  force_meta.json   ramp settings and the criteria of the last analysis
  force_summary.json
and A/results/<run name>/summary.json, summary.csv (all designs analyzed so far).
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
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from gripper_core import JOINTS, LINKS, contact_rules, read_json, save_rows, write_json
from rb5_eval import Rb5Arm, validate_rb5_config
from eval_design_rb5 import read_rows, to_json
from disturb_collect import (DEFAULT_LABELS, label_dir, resolve_labels, run_tasks,
                             save_table, scene_signature, set_state)
from run_pick_lift import ARM_JOINTS, PickLift, smooth

DIRECTIONS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
              "+z": (0, 0, 1), "-z": (0, 0, -1)}
ZERO = "0"  # Zero-force control; saved under the key 'z0'.
BILATERAL_N = 0.10  # run_pick_lift.PickLift: each side's finger force sum must exceed this.
DEFAULT_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
GROUPS = ("env", "tip")
# Trace columns: t, F, object displacement in the hand frame (3), rotation, link normal
# forces R_gp R_gd L_gp L_gd palm (5), finger joint change R_jp R_jd L_jp L_jd [deg] (4),
# displacement of the hand (grasp site) in the world [mm] (arm compliance), tilt of the
# cylinder axis relative to the hand [deg] (rotation without the spin about its own axis).
# The last two exist in newer traces only.
TRACE_COLUMNS = ("t_s", "F_N", "dx_mm", "dy_mm", "dz_mm", "rot_deg", *(f"{n}_N" for n in LINKS),
                 *(f"d{n}_deg" for n in JOINTS), "hand_mm", "tilt_deg")
HAND_COL = TRACE_COLUMNS.index("hand_mm")
TILT_COL = TRACE_COLUMNS.index("tilt_deg")
WORKER = {}


def force_dir(run, label):
    """Experiment A results of one design: A/results/<run name>/<label>/."""
    return layout.a_root(run)/label


def key(direction):
    return "z0" if direction == ZERO else direction.replace("+", "p").replace("-", "n")


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


def load(model, data, state, d_hand, force_at, t_end, s, stop):
    """Restore the post-lift state and push the object along hand direction d_hand with
    force_at(t) [N] (t = 0 at the restored state) for t_end seconds.

    stop = dict(slip_mm, rot_deg (None/0 = off), lost_s): the run ends ('drop') at the
    first sample where one of them is reached. ended is 'end' when t_end was reached."""
    set_state(model, data, state)
    ob, og, site = model.body("obj").id, model.geom("obj_g").id, model.site("grasp").id
    links = {model.geom(n).id: i for i, n in enumerate(LINKS)}
    armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
    qa = np.array([model.joint(n).qposadr[0] for n in JOINTS])
    q0 = data.qpos[qa].copy()
    dt = model.opt.timestep
    every = max(1, round(s["sample_s"]/dt))
    rs0 = data.site_xmat[site].reshape(3, 3).copy()
    h0 = data.site_xpos[site].copy()
    p0 = rs0.T @ (data.xpos[ob]-h0)
    r0 = rs0.T @ data.xmat[ob].reshape(3, 3)
    direction = rs0 @ np.asarray(d_hand, float)
    sim = PickLift(model, data)
    sim.stage = "disturb"
    sim.qcmd = data.ctrl[armact].copy()  # The arm keeps holding its last command.
    rows = []

    def sample(force):
        rs = data.site_xmat[site].reshape(3, 3)
        p = rs.T @ (data.xpos[ob]-data.site_xpos[site])
        r = rs.T @ data.xmat[ob].reshape(3, 3)
        rel = r0.T @ r
        rot = math.degrees(math.acos(np.clip((np.trace(rel)-1)/2, -1, 1)))
        tilt = math.degrees(math.acos(np.clip(float(r0[:, 2] @ r[:, 2]), -1, 1)))
        f = link_forces(model, data, og, links)
        hand = 1000*float(np.linalg.norm(data.site_xpos[site]-h0))
        rows.append([data.time, force, *(1000*(p-p0)), rot, *f, *np.degrees(data.qpos[qa]-q0), hand, tilt])
        return 1000*float(np.linalg.norm(p-p0)), (tilt if stop.get("rot_mode") == "tilt" else rot), \
            min(f[0]+f[1], f[2]+f[3]) > BILATERAL_N

    sample(0.0)
    n = max(1, math.ceil(t_end/dt-1e-9))
    ended, lost_since, t_start, force = "end", None, data.time, 0.0
    rot_stop = stop.get("rot_deg") or 0.0
    try:
        for i in range(n):
            force = float(force_at((i+1)*dt))
            data.xfrc_applied[ob, :3] = force*direction
            sim.step()
            if (i+1) % every and i != n-1:
                continue
            disp, rot, bilateral = sample(force)
            lost_since = None if bilateral else (data.time if lost_since is None else lost_since)
            if (disp >= stop["slip_mm"] or (rot_stop > 0 and rot >= rot_stop)
                    or (lost_since is not None and data.time-lost_since >= stop["lost_s"]-1e-9)):
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


def ramp_stop(s):
    return dict(slip_mm=s["stop_mm"], rot_deg=None, lost_s=s["stop_lost_s"])


def ramp(model, data, state, d_hand, s):
    """F(t) = rate*t up to the cap. ended: cap / drop / unintended_collision / unstable."""
    res = load(model, data, state, d_hand, lambda t: min(s["rate_N_s"]*t, s["cap_N"]),
               s["cap_N"]/s["rate_N_s"], s, ramp_stop(s))
    res["ended"] = "cap" if res["ended"] == "end" else res["ended"]
    return res


def zero_control(model, data, state, s):
    """F = 0 for the duration of a full ramp. ended: held / drop / error."""
    res = load(model, data, state, (1.0, 0.0, 0.0), lambda t: 0.0, s["cap_N"]/s["rate_N_s"], s,
               ramp_stop(s))
    res["ended"] = "held" if res["ended"] == "end" else res["ended"]
    return res


def hold_trial(model, data, state, d_hand, force, s, crit):
    """Constant force (smooth rise over rise_s, then hold_s). (held, cause, seconds)."""
    rise = s["rise_s"]
    res = load(model, data, state, d_hand,
               (lambda t: force*float(smooth(t/rise))) if rise > 0 else (lambda t: force),
               rise+s["hold_s"], s, dict(slip_mm=crit["slip_mm"], rot_deg=crit["rot_deg"],
                                          rot_mode=crit.get("rot_mode", "total"), lost_s=crit["grace_s"]))
    t_end = float(res["trace"][-1, 0])
    if res["ended"] == "end":
        return True, "", t_end
    if res["ended"] == "drop":
        fired = failure_events(res["trace"], crit)
        cause = min((k for k, i in fired.items() if i is not None), key=lambda k: fired[k], default="drop")
        return False, cause, t_end
    return False, res["ended"], t_end  # Collision or instability: not a hold.


def refine(model, data, state, d_hand, s, crit, f_ramp, ramp_censored):
    """Bisection on the constant force. Returns (F_hold, censored, trials)."""
    tol, cap = s["tol_N"], s["cap_N"]
    trials = []

    def held(force):
        ok, cause, sec = hold_trial(model, data, state, d_hand, force, s, crit)
        trials.append((force, ok, sec, cause))
        return ok

    hi = cap if ramp_censored else min(cap, max(1.25*f_ramp, f_ramp+2*tol))
    if held(hi):
        return hi, True, trials
    lo = 0.0  # The cell held its own weight through 'lift hold'.
    while hi-lo > tol:
        mid = 0.5*(lo+hi)
        if held(mid):
            lo = mid
        else:
            hi = mid
    return lo, False, trials


def model_for(hand_file, cfg):
    """Compiled RB5 scene of a saved hand XML, cached per worker process."""
    if hand_file not in WORKER:
        arm = WORKER.setdefault("_arm", Rb5Arm(cfg))
        hand_xml = Path(hand_file).read_text(encoding="utf-8")
        WORKER[hand_file] = (arm.compile(hand_xml), scene_signature(arm, hand_xml))
    return WORKER[hand_file]


def open_cell(out, c, cfg):
    """(model, data, state npz) of one saved cell, with the grasp site moved to the cell."""
    st = np.load(out/c["state_file"])
    model, sig = model_for(str(out/"hands"/str(st["hand_file"])), cfg)
    if str(st["signature"]) != sig:
        raise SystemExit(f"{c['state_file']}: the scene (hand XML, sources/ or MuJoCo version) "
                         "differs from the one the state was saved with. Run disturb_collect.py again.")
    r, x, y = float(st["r_mm"]), float(st["x_mm"]), float(st["y_mm"])
    model.site_pos[model.site("grasp").id] = [x/1000, (y+r)/1000, 0.0]
    return model, mujoco.MjData(model), st


def force_task(task):
    """Zero control and all directions of the given cells (one radius). One traces/*.npz per cell.
    out: grasp states of the design, dst: its A results."""
    out, dst, cfg, s, crit, cells = task
    out, dst = Path(out), Path(dst)
    done, start = [], time.perf_counter()
    for c in cells:
        model, data, st = open_cell(out, c, cfg)
        saved = {}
        names = ([ZERO] if s["zero_control"] else [])+list(s["dirs"])
        for name in names:
            t0 = time.perf_counter()
            k = key(name)
            if name == ZERO:
                res = zero_control(model, data, st["state"], s)
            else:
                res = ramp(model, data, st["state"], DIRECTIONS[name], s)
            saved.update({f"{k}_trace": res["trace"].astype(np.float32), f"{k}_ended": res["ended"],
                          f"{k}_world": res["direction_world"], f"{k}_pen_mm": res["penetration_mm"]})
            if s["refine"] and name != ZERO:
                j = judge_trace(res["trace"], res["ended"], crit, s["cap_N"])
                f_hold, cens, trials = refine(model, data, st["state"], DIRECTIONS[name], s, crit,
                                              j["F_max_N"], j["censored"])
                saved.update({f"{k}_hold_N": f_hold, f"{k}_hold_censored": cens,
                              f"{k}_hold_trials": np.array([t[:3] for t in trials], dtype=np.float64),
                              f"{k}_hold_causes": np.array([t[3] for t in trials])})
            saved[f"{k}_sec"] = time.perf_counter()-t0
        np.savez_compressed(dst/"traces"/Path(c["state_file"]).name, r_mm=float(st["r_mm"]),
                            x_mm=float(st["x_mm"]), y_mm=float(st["y_mm"]), dirs=np.array(s["dirs"]),
                            columns=np.array(TRACE_COLUMNS), zero_control=s["zero_control"],
                            refine=s["refine"], refine_criteria=json.dumps(crit), **saved)
        done.append(dict(r_mm=float(st["r_mm"]), x_mm=float(st["x_mm"]), y_mm=float(st["y_mm"]),
                         ended=[str(saved[f"{key(d)}_ended"]) for d in s["dirs"]]))
    return done, time.perf_counter()-start


# ---------------------------------------------------------------- analysis
def first_index(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def failure_events(trace, crit):
    """First sample index of every criterion (None = never met)."""
    t = trace[:, 0]
    disp = np.linalg.norm(trace[:, 2:5], axis=1)
    rot, f = rotation(trace, crit), trace[:, 6:11]
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
    return events


def rotation(trace, crit):
    """Rotation the criterion looks at: total (default, as the 'lift hold' check of the
    workspace map) or tilt of the cylinder axis (spin about the axis does not count)."""
    if crit.get("rot_mode", "total") == "tilt":
        if trace.shape[1] <= TILT_COL:
            raise SystemExit("--rot-mode tilt needs traces with a tilt column; simulate again.")
        return trace[:, TILT_COL]
    return trace[:, 5]


def classify_many(forces, threshold, rule="links_or_palm", partial="fail"):
    """gripper_core.classify_contact for every row of forces (R_gp, R_gd, L_gp, L_gd, palm)."""
    b = np.asarray(forces) >= threshold
    rp, rd, lp, ld, palm = b.T
    env = (rp & rd & lp & ld) | (palm & (rule == "links_or_palm"))
    return np.select([~((rp | rd) & (lp | ld)), (rp & rd) != (lp & ld), env, rd & ld & ~rp & ~lp],
                     ["none", "mixed", "env", "tip"], "tip" if partial == "tip" else "unsupported")


def contact_types(forces, window, threshold, rules):
    """Contact type per sample from the causal moving average of the link forces."""
    c = np.vstack([np.zeros((1, forces.shape[1])), np.cumsum(forces, axis=0)])
    n = np.arange(1, len(forces)+1)
    lo = np.maximum(0, n-window)
    return classify_many((c[n]-c[lo])/(n-lo)[:, None], threshold, *rules)


def judge_trace(trace, ended, crit, cap_N, types=None, start_type=None):
    """F_max and its cause from one direction's trace (+ contact type changes with types)."""
    trace = np.asarray(trace, dtype=np.float64)
    force = trace[:, 1]
    disp = np.linalg.norm(trace[:, 2:5], axis=1)
    rot, dq = rotation(trace, crit), np.abs(trace[:, 11:15])
    events = failure_events(trace, crit)
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
               ended_by=ended, F_end_N=float(force[-1]), t_Fmax_s=float(trace[i, 0]),
               max_disp_mm=float(disp.max()), max_rot_deg=float(rot.max()),
               disp_at_Fmax_mm=float(disp[i]), finger_dq_at_Fmax_deg=float(dq[i].max()),
               hand_at_Fmax_mm=float(trace[i, HAND_COL]) if trace.shape[1] > HAND_COL else None)
    if types is not None:
        ref = start_type or str(types[0])
        change = first_index(types != ref)
        out.update(type_start=ref, type_at_Fmax=str(types[i]),
                   type_change_to=str(types[change]) if change is not None else "",
                   F_type_change_N=float(force[change]) if change is not None else None,
                   type_changed_before_Fmax=change is not None and change <= i)
    return out


def zero_result(trace, ended, crit):
    """Zero-force control of one cell: does the grasp meet a criterion with no load at all?"""
    trace = np.asarray(trace, dtype=np.float64)
    fired = {k: i for k, i in failure_events(trace, crit).items() if i is not None}
    cause = min(fired, key=lambda k: (fired[k], k)) if fired else ("" if ended == "held" else ended)
    t_fail = float(trace[fired[cause], 0]) if fired else None
    disp = np.linalg.norm(trace[:, 2:5], axis=1)
    return dict(zero_fail=bool(fired) or ended not in ("held", "drop"), zero_cause=cause,
                zero_t_fail_s=t_fail, zero_T_s=float(trace[-1, 0]), zero_max_disp_mm=float(disp.max()),
                zero_max_rot_deg=float(rotation(trace, crit).max()))


def group_of(cell):
    return cell["contact_type"] if cell["success"] else cell["reason"]


def analyze(run, label, cfg, crit, levels, metric="ramp", zero_rule="fail"):
    """Rebuild force_dirs.csv, force_cells.csv and the design's summary from the traces.
    zero_rule 'fail': a success cell whose zero-force control fails counts as failed."""
    out, dst = label_dir(run, label), force_dir(run, label)
    meta = read_json(dst/"force_meta.json")
    s = meta["settings"]
    mg, cap_N = s["mg_N"], s["cap_N"]
    sc = cfg["simulation"]
    rules = contact_rules(cfg)
    window = max(1, round(sc["force_average_window_s"]/s["sample_s"]))
    if crit["slip_mm"] > s["stop_mm"] or crit["grace_s"] > s["stop_lost_s"]:
        print(f"Warning: criteria looser than the stop rule (stop_mm={s['stop_mm']}, "
              f"stop_lost_s={s['stop_lost_s']}); stopped ramps count as 'drop' failures.")
    cells = [c for c in read_rows(out/"cells.csv") if c["state_file"]]
    dir_rows, cell_rows, warned, no_zero = [], [], False, 0
    for c in cells:
        path = dst/"traces"/Path(c["state_file"]).name
        if not path.is_file():
            continue
        tr = np.load(path)
        zero = zero_result(tr["z0_trace"], str(tr["z0_ended"]), crit) if "z0_trace" in tr.files else {}
        no_zero += not zero
        base = group_of(c)
        removed = zero_rule == "fail" and base in GROUPS and bool(zero.get("zero_fail"))
        info = dict(r_mm=c["r_mm"], x_mm=c["x_mm"], y_mm=c["y_mm"], group="zero_fail" if removed else base,
                    base_group=base, orig_group=c["orig_group"],
                    margin_negative=bool(c["success"]) and c["m_F"] < 0, reproduced=c["reproduced"], m_F=c["m_F"])
        refined = "refine" in tr.files and bool(tr["refine"])
        if refined and not warned and json.loads(str(tr["refine_criteria"])) != crit:
            print(f"Warning [{label}]: hold tests were judged with {tr['refine_criteria']} during the "
                  "simulation; only the ramp is re-judged with the current criteria.")
            warned = True
        if metric == "hold" and not refined:
            raise SystemExit(f"--metric hold needs traces made with --refine ({path.name} has none).")
        per = {}
        for name in [str(d) for d in tr["dirs"]]:
            k = key(name)
            trace = tr[f"{k}_trace"].astype(float)
            types = contact_types(trace[:, 6:11], window, sc["contact_threshold_N"], rules)
            res = judge_trace(trace, str(tr[f"{k}_ended"]), crit, cap_N, types, str(c["contact_type"]))
            res["F_max_mg"] = res["F_max_N"]/mg
            res["creep_limited"] = bool(zero.get("zero_t_fail_s") is not None and not res["censored"]
                                        and zero["zero_t_fail_s"] <= res["t_Fmax_s"]+1e-9)
            if refined:
                res.update(F_hold_N=float(tr[f"{k}_hold_N"]), F_hold_mg=float(tr[f"{k}_hold_N"])/mg,
                           hold_censored=bool(tr[f"{k}_hold_censored"]),
                           hold_trials=int(len(tr[f"{k}_hold_trials"])))
            res["F_mg"] = res["F_hold_mg"] if metric == "hold" else res["F_max_mg"]
            per[name] = res
            dir_rows.append(dict(info, dir=name, **res, max_unintended_penetration_mm=float(tr[f"{k}_pen_mm"]),
                                 wall_s=float(tr[f"{k}_sec"])))
        weakest = min(per, key=lambda d: per[d]["F_mg"])
        row = dict(info, contact_type=c["contact_type"], F_min_mg=per[weakest]["F_mg"], weakest_dir=weakest,
                   weakest_cause=per[weakest]["cause"], weakest_creep_limited=per[weakest]["creep_limited"],
                   any_error=any(v["cause"] in ("unintended_collision", "unstable") for v in per.values()),
                   any_type_change=any(v.get("type_changed_before_Fmax") for v in per.values()), **zero)
        row.update({f"F_{d}_mg": per[d]["F_mg"] for d in per})
        if refined:
            row.update({f"F_ramp_{d}_mg": per[d]["F_max_mg"] for d in per})
            row["F_min_hold_mg"] = min(per[d]["F_hold_mg"] for d in per)
        row.update({f"pass_{lv:g}mg": per[weakest]["F_mg"] >= lv-1e-9 for lv in levels})
        cell_rows.append(row)
    if cells and not cell_rows:
        raise SystemExit(f"No traces in {dst/'traces'}. Run without --rescore first.")
    if no_zero and zero_rule == "fail":
        print(f"Warning [{label}]: {no_zero} cell(s) have no zero-force control (made with --no-zero-control "
              "or an older disturb_force.py); the zero-force rule cannot remove them.")
    save_table(dst/"force_dirs.csv", dir_rows)
    save_table(dst/"force_cells.csv", cell_rows)
    meta["criteria"], meta["levels_mg"], meta["metric"], meta["zero_rule"] = crit, list(levels), metric, zero_rule
    write_json(dst/"force_meta.json", meta)
    sm = summarize(label, read_json(out/"meta.json"), cell_rows, dir_rows, cfg, levels, s)
    sm.update(criteria=crit, metric=metric, zero_rule=zero_rule)
    write_json(dst/"force_summary.json", json.loads(json.dumps(sm, default=to_json)))
    return sm


def pass_rate(rows, lv):
    return (sum(r[f"pass_{lv:g}mg"] for r in rows)/len(rows)) if rows else None


def stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not v.size:
        return dict(n=0)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return dict(n=int(v.size), median=float(med), q25=float(q25), q75=float(q75), min=float(v.min()),
                max=float(v.max()))


def direction_stats(dirs, groups, directions):
    """F (in m*g) of every direction, per group: the per-direction answer for env and tip."""
    out = {}
    for g in groups:
        out[g] = {}
        for d in directions:
            rows = [r for r in dirs if r["group"] == g and r["dir"] == d]
            if not rows:
                continue
            st = stats(r["F_mg"] for r in rows)
            st.update(censored=sum(r["censored"] for r in rows),
                      creep_limited=sum(r["creep_limited"] for r in rows),
                      type_changed=sum(bool(r.get("type_changed_before_Fmax")) for r in rows),
                      causes=dict(Counter(r["cause"] for r in rows)))
            if "F_hold_mg" in rows[0]:
                st["ramp_median"] = stats(r["F_max_mg"] for r in rows).get("median")
                st["hold_median"] = stats(r["F_hold_mg"] for r in rows).get("median")
            out[g][d] = st
    return out


def summarize(label, collect_meta, cells, dirs, cfg, levels, s):
    o = cfg["objective"]
    ok = [c for c in cells if c["group"] in GROUPS]  # Success cells after the zero-force rule.
    ok_all = [c for c in cells if c.get("base_group", c["group"]) in GROUPS]  # Re-run success cells.
    env, tip = [c for c in ok if c["group"] == "env"], [c for c in ok if c["group"] == "tip"]
    n_map = collect_meta["cells_in_map"]
    cap = s["cap_mg"]

    def area(rows):
        return float(sum(min(r["F_min_mg"], cap) for r in rows))

    def J(rows):
        return (sum(c.get("base_group", c["group"]) == "env" for c in rows)
                + o["w_tip"]*sum(c.get("base_group", c["group"]) == "tip" for c in rows)
                + o["lambda_force"]*sum(c["m_F"] for c in rows))

    out = dict(label=label, cells_in_map=n_map, orig_success=collect_meta["orig_success"],
               reproduced_success=collect_meta["reproduced_success"], success_rerun=len(ok_all),
               zero_fail_removed=dict(Counter(c["base_group"] for c in cells if c["group"] == "zero_fail")),
               success=len(ok), env=len(env), tip=len(tip), J_rerun=J(ok_all), J_stable=J(ok),
               directions=s["dirs"], cap_mg=cap, robust_area=area(ok), robust_area_env=area(env),
               robust_area_tip=area(tip),
               F_min_median_mg=float(np.median([c["F_min_mg"] for c in ok])) if ok else 0.0,
               causes=dict(Counter(d["cause"] for d in dirs if d["group"] in GROUPS)),
               zero_control_fail=dict(Counter(c.get("base_group", c["group"]) for c in ok_all if c.get("zero_fail"))),
               creep_limited_weakest=sum(bool(c["weakest_creep_limited"]) for c in ok),
               type_changed_cells=dict(Counter(c["group"] for c in ok if c["any_type_change"])),
               levels=[], groups={})
    for lv in levels:
        p = [c for c in ok if c[f"pass_{lv:g}mg"]]
        pe, pt = sum(c["group"] == "env" for c in p), sum(c["group"] == "tip" for c in p)
        per_dir = {d: sum(r["F_mg"] >= lv-1e-9 for r in dirs if r["dir"] == d and r["group"] in GROUPS)
                   for d in s["dirs"]}
        out["levels"].append(dict(level_mg=lv, passed=len(p), env=pe, tip=pt,
                                  conditional_robustness=len(p)/len(ok) if ok else None,
                                  robust_workspace=len(p), robust_workspace_fraction=len(p)/n_map,
                                  J_robust=pe+o["w_tip"]*pt+o["lambda_force"]*sum(c["m_F"] for c in p),
                                  per_direction=per_dir))
    groups = {"env": env, "tip": tip, "tip_margin_negative": [c for c in tip if c["margin_negative"]]}
    for c in cells:
        if c["group"] not in GROUPS:
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
    out["direction_stats"] = direction_stats(dirs, [g for g in groups if g != "tip_margin_negative"], s["dirs"])
    held = [c for c in ok if c.get("F_min_hold_mg") is not None]
    if held:  # --refine: the same numbers from the hold test (absolute holding force).
        out["hold"] = dict(n=len(held), F_min_median_mg=float(np.median([c["F_min_hold_mg"] for c in held])),
                           passed={f"{lv:g}mg": sum(c["F_min_hold_mg"] >= lv-1e-9 for c in held) for lv in levels},
                           passed_env={f"{lv:g}mg": sum(c["F_min_hold_mg"] >= lv-1e-9 for c in held
                                                        if c["group"] == "env") for lv in levels})
    return out


def print_summary(sm):
    removed = sm.get("zero_fail_removed") or {}
    print(f"\n[{sm['label']}] success {sm['success']} (env {sm['env']}, tip {sm['tip']}) = re-run success "
          f"{sm.get('success_rerun', sm['success'])} of {sm['orig_success']} original minus "
          f"{sum(removed.values())} failing the zero-force control {removed or ''}; map {sm['cells_in_map']} cells, "
          f"J_stable={sm.get('J_stable', sm['J_rerun']):.2f} (J_rerun {sm['J_rerun']:.2f}), F = {sm.get('metric', 'ramp')}")
    print(f"  robust_area {sm['robust_area']:.2f} mg (env {sm['robust_area_env']:.2f}, "
          f"tip {sm['robust_area_tip']:.2f}), median F_min {sm['F_min_median_mg']:.2f} mg; "
          f"failure causes (success cells x directions): {sm['causes']}")
    print(f"  zero-force control failed (re-run success cells): {sm['zero_control_fail'] or 0}, "
          f"weakest direction creep-limited: "
          f"{sm['creep_limited_weakest']}, contact type changed before F_max: {sm['type_changed_cells'] or 0}")
    if sm.get("hold"):
        h = sm["hold"]
        print(f"  hold test (--refine): median F_min {h['F_min_median_mg']:.2f} mg, cells holding >= level in every "
              f"direction: " + " ".join(f"{k}:{v}" for k, v in h["passed"].items()))
    print(f"  {'level':>6} {'pass':>5} {'env':>4} {'tip':>4} {'cond.robust':>11} {'robust WS':>10} {'J_robust':>9}  "
          + " ".join(f"{d:>4}" for d in sm["directions"]))
    for L in sm["levels"]:
        cr = "-" if L["conditional_robustness"] is None else f"{L['conditional_robustness']:.2f}"
        print(f"  {L['level_mg']:>4g}mg {L['passed']:>5} {L['env']:>4} {L['tip']:>4} {cr:>11} "
              f"{L['robust_workspace']:>4}/{sm['cells_in_map']:<5} {L['J_robust']:>9.2f}  "
              + " ".join(f"{L['per_direction'][d]:>4}" for d in sm["directions"]))
    print("  median F per direction [m*g] (q25-q75), * = some censored at the cap:")
    for g, per in sm["direction_stats"].items():
        if not per:
            continue
        cells = "  ".join(f"{d} {v['median']:.2f} ({v['q25']:.2f}-{v['q75']:.2f}){'*' if v['censored'] else ''}"
                          for d, v in per.items() if v["n"])
        n = max(v["n"] for v in per.values())
        print(f"    {g:<11} n={n:<3} {cells}")
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
    base = layout.a_root(run)
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
                             J_rerun=sm["J_rerun"], J_robust=L["J_robust"], robust_area=sm.get("robust_area"),
                             **{f"pass_{d}": L["per_direction"].get(d) for d in DIRECTIONS}))
    if flat:
        save_rows(base/"summary.csv", flat)
    return path


# ---------------------------------------------------------------- main
def simulate(run, label, cfg, s, crit, workers, limit):
    out, dst = label_dir(run, label), force_dir(run, label)
    if not (out/"cells.csv").is_file():
        raise SystemExit(f"No {out/'cells.csv'}. Run grasp/disturb_collect.py first.")
    cells = [c for c in read_rows(out/"cells.csv") if c["state_file"]]
    if limit:
        cells = cells[:limit]
    (dst/"traces").mkdir(parents=True, exist_ok=True)
    for old in [*(dst/"traces").glob("*.npz"), *dst.glob("force_*")]:
        old.unlink()
    if not cells:  # Nothing was lifted: the design's summary is all zeros.
        write_json(dst/"force_meta.json", dict(label=label, settings=dict(s, mg_N=None, cap_N=None),
                                               mujoco=mujoco.__version__))
        print(f"\n[{label}] no lifted cell, nothing to disturb", flush=True)
        return
    # mg from the scene: object mass x the config's gravity.
    model, _ = model_for(str(out/"hands"/str(np.load(out/cells[0]["state_file"])["hand_file"])), cfg)
    s = dict(s, mg_N=float(model.body_mass[model.body("obj").id])*cfg["simulation"]["gravity_m_s2"])
    s.update(cap_N=s["cap_mg"]*s["mg_N"], tol_N=s["tol_mg"]*s["mg_N"])
    write_json(dst/"force_meta.json", dict(label=label, settings=s, mujoco=mujoco.__version__))
    by_r = {}
    for c in cells:
        by_r.setdefault(c["r_mm"], []).append(c)
    per = max(1, math.ceil(len(cells)/(3*max(1, workers))))
    tasks = [(str(out), str(dst), cfg, s, crit, g[i:i+per]) for _, g in sorted(by_r.items())
             for i in range(0, len(g), per)]
    extra = [x for x, on in (("zero-force control", s["zero_control"]),
                             (f"hold-test bisection (rise {s['rise_s']:g} s, hold {s['hold_s']:g} s, "
                              f"tol {s['tol_mg']:g} mg)", s["refine"])) if on]
    print(f"\n[{label}] {len(cells)} cells x {len(s['dirs'])} directions, ramp {s['rate_N_s']} N/s to "
          f"{s['cap_mg']:g} mg = {s['cap_N']:.3f} N ({s['cap_N']/s['rate_N_s']:.1f} s per direction)"
          + (f" + {', '.join(extra)}" if extra else ""), flush=True)
    done, wall = [], time.perf_counter()

    def report(part, sec):
        done.extend(part)
        ends = Counter(e for p in part for e in p["ended"])
        print(f"  r={part[0]['r_mm']:g} mm: {len(part)} cells, {sec:.0f} s, ended {dict(ends)} "
              f"({len(done)}/{len(cells)}, {time.perf_counter()-wall:.0f} s)", flush=True)

    run_tasks(force_task, tasks, workers, report)


def criteria(args, cfg):
    sc = cfg["simulation"]
    return dict(slip_mm=args.slip_mm if args.slip_mm is not None else sc["hold_displacement_mm"],
                rot_deg=args.rot_deg if args.rot_deg is not None else sc["hold_rotation_deg"],
                rot_mode=args.rot_mode,
                grace_s=args.grace_s if args.grace_s is not None else sc.get("lost_contact_grace_s", 0.02))


def settings(args, cfg):
    sc = cfg["simulation"]
    names = {key(d): d for d in DIRECTIONS}
    return dict(rate_N_s=args.rate, cap_mg=args.max_mg,
                dirs=list(dict.fromkeys(names.get(d, d) for d in args.dirs)),
                stop_mm=args.stop_mm if args.stop_mm is not None else sc["lift_slip_mm"],
                stop_lost_s=args.stop_lost_s, sample_s=sc["sample_interval_s"],
                zero_control=not args.no_zero_control, refine=args.refine,
                rise_s=args.rise_s, hold_s=args.hold_s if args.hold_s is not None else sc["hold_s"],
                tol_mg=args.refine_tol_mg)


def add_arguments(p):
    """Ramp, refine and criteria options (shared with disturb_validate.py)."""
    g = p.add_argument_group("ramp (simulation)")
    g.add_argument("--rate", type=float, default=1.0, help="Force rate [N/s] (default 1)")
    g.add_argument("--max-mg", type=float, default=10.0, help="Ramp end in units of m*g (default 10)")
    g.add_argument("--dirs", nargs="+", default=list(DIRECTIONS), metavar="DIR",
                   choices=[*DIRECTIONS, *map(key, DIRECTIONS)],
                   help="px nx py ny pz nz (= +x -x +y -y +z -z); default all six")
    g.add_argument("--stop-mm", type=float, help="Stop a direction once the object moved this far "
                   "(default simulation.lift_slip_mm)")
    g.add_argument("--stop-lost-s", type=float, default=0.1, help="... or lost bilateral contact this long")
    g.add_argument("--no-zero-control", action="store_true", help="Skip the zero-force control run")
    h = p.add_argument_group("hold test (--refine)")
    h.add_argument("--refine", action="store_true", help="Bisection on a constant force per direction")
    h.add_argument("--rise-s", type=float, default=0.5, help="Smooth rise to the test force [s] (default 0.5)")
    h.add_argument("--hold-s", type=float, help="Hold time at the test force [s] (default simulation.hold_s)")
    h.add_argument("--refine-tol-mg", type=float, default=0.02, help="Bisection tolerance [m*g] (default 0.02)")
    c = p.add_argument_group("failure criteria (also for --rescore)")
    c.add_argument("--slip-mm", type=float, help="default simulation.hold_displacement_mm")
    c.add_argument("--rot-deg", type=float, help="default simulation.hold_rotation_deg; 0 disables")
    c.add_argument("--rot-mode", choices=("total", "tilt"), default="total",
                   help="Rotation measured as total angle (default, like the lift-hold check) or as "
                        "tilt of the cylinder axis (spin about the axis allowed)")
    c.add_argument("--grace-s", type=float, help="default simulation.lost_contact_grace_s")


def check_settings(s, crit):
    if (s["rate_N_s"] <= 0 or s["cap_mg"] <= 0 or s["stop_mm"] < crit["slip_mm"]
            or s["stop_lost_s"] < crit["grace_s"]):
        raise SystemExit("Need rate>0, max-mg>0, stop-mm >= slip-mm and stop-lost-s >= grace-s")
    if s["refine"] and (s["hold_s"] <= 0 or s["rise_s"] < 0 or s["tol_mg"] <= 0):
        raise SystemExit("Need hold-s>0, rise-s>=0 and refine-tol-mg>0")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=layout.DEFAULT_RUN)
    p.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                   help="'all' (default: every design collected by disturb_collect.py) or names")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--rescore", action="store_true", help="Only re-judge saved traces (no simulation)")
    p.add_argument("--limit", type=int, help="Only the first N saved cells per design (smoke test)")
    add_arguments(p)
    p.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS), help="Pass levels [m*g]")
    p.add_argument("--metric", choices=("ramp", "hold"), default="ramp",
                   help="F used for pass levels, F_min and robust_area: ramp F_max (default) or the "
                        "--refine hold force")
    p.add_argument("--zero-fail", choices=("fail", "keep"), default="fail",
                   help="Success cells whose zero-force control fails: count as failed (default) or "
                        "keep them as success cells (only flagged)")
    p.add_argument("--no-summary", action="store_true",
                   help="Do not update the shared summary.json/csv (batch_disturbance.py writes it at the end)")
    args = p.parse_args()
    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    crit = criteria(args, cfg)
    collected = sorted(d.name for d in layout.states_root(run).glob("*") if (d/"cells.csv").is_file())
    labels = resolve_labels(args.labels, collected)
    print(f"Designs: {labels}")
    if not args.rescore:
        s = settings(args, cfg)
        check_settings(s, crit)
        for label in labels:
            simulate(run, label, cfg, s, crit, args.workers, args.limit)
    summaries = [analyze(run, label, cfg, crit, sorted(args.levels), args.metric, args.zero_fail)
                 for label in labels]
    print(f"\nCriteria: slip >= {crit['slip_mm']} mm, {crit['rot_mode']} rotation >= {crit['rot_deg']} deg (0 = off), "
          f"lost contact >= {crit['grace_s']} s")
    for sm in summaries:
        print_summary(sm)
    if not args.no_summary:
        print("\nSaved:", write_summary(run, summaries))
    for label in labels:
        print("Saved:", force_dir(run, label)/"force_cells.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
