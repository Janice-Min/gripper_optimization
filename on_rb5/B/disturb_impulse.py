"""Disturbance experiment B: a short force pulse (impact) on the grasped object.

Uses the grasp states of experiment A (disturb_collect.py) of the designs A found graspable
(disturbance/graspable_designs.csv). For every success cell the state at the end of
'lift hold' is restored and, per direction and pulse size,
    F(t) = k*m*g  for 0 <= t < T      (default T = 50 ms, k = 0.5, 1, 2, 3; --shape halfsine:
                                        k*m*g*sin(pi*t/T), the same peak, 2/pi of the impulse)
is applied to the object's center of mass (xfrc_applied), then the object must stay in the
hand for --after-s (default 10 s) after the pulse. The arm holds its last command and the
tendon keeps its tension (20 N), gravity stays on: exactly the conditions of experiment A.

Initial values. Pulse sizes span A's ramp F_max of the success cells (weakest directions
about 0.2-2.6 m g, strong env directions 3-10 m g); 0.5 m g separates the many tip-heavy
designs that all fail 1 m g. With m = 0.1 kg a pulse moves a FREE object
d = k*g*T^2/2 = 12 mm per m g at T = 50 ms, so a grasp that cannot resist slips past 5 mm
within the pulse. Pilot (9 designs, 219 cells): the object slides only while the pulse lasts
(84 % of failures start during it), and a 50 ms pulse is held about up to A's ramp F_max
+ 0.2 m g (50 % point); at T = 20 ms the margin grows to about +1 m g. So 50 ms is close to
quasi-static; use --pulse-ms 10-20 for a more impact-like (impulse-limited) test.

A trial HOLDS when, from the start of the pulse to the end of the after-time, none of the
criteria of experiment A is met (defaults from the config, same as A):
  slip      object moved >= --slip-mm relative to the hand (5 mm)
            --slip-mode residual: only displacement from settle-s after the pulse on counts
            (an elastic deflection during the impact that springs back is allowed)
  rotation  object turned >= --rot-deg relative to the hand (5 deg; --rot-mode tilt)
  lost      R or L finger normal force < 0.10 N for >= --grace-s (20 ms)
A failure is labelled by when it starts: 'pulse' (during the pulse), 'transient' (up to
--settle-s after it) or 'delayed' (later: the impact left a weaker grasp that creeps out).
A trial that hits an unintended collision or becomes unstable counts as not held (cause
'unintended_collision' / 'unstable').

Zero-force control: every cell is also held with F = 0 for the same T + after-s. A cell
that fails it cannot hold the object even without an impact (soft-contact creep) and is
dropped (group 'zero_fail', as in A; --zero-fail keep only flags it). Experiment A's
control lasted 9.8 s, this one T + after-s, so a few more cells can drop out.

Per cell and direction: level = largest k such that the pulse of every size <= k was held
(0 if the smallest failed). Per cell: level_min = weakest direction. Per design and k:
  passed       success cells holding the k pulse in EVERY direction (level_min >= k)
  cond         passed / success cells                      (grasp quality)
  J_impulse    env_pass + w_tip*tip_pass + lambda*sum(m_F), the scale of J and J_robust
  impact_area  sum over success cells of level_min [mg]     (like robust_area of A)
and, per trial, A's ramp F_max of the same cell and direction, so the tables show where a
short pulse is held above the slowly-ramped limit (held_above_A) or fails below it.

Traces: every sample (sample_interval_s, 5 ms) up to --detail-s after the pulse start,
then one row per --coarse-s that carries the maxima of displacement, rotation, tilt and the
longest lost-contact streak since the previous row. Held / not held is therefore exact for
any criteria in --rescore; the failure time is exact inside the detail window and good to
coarse-s after it. As in A, a trial stops early once the object is clearly gone (moved
>= --stop-mm, default 30 mm, or no bilateral contact for --stop-lost-s, 0.1 s), so
--rescore works for any slip <= stop-mm and grace <= stop-lost-s.

  python B/disturb_impulse.py --workers 8                         # all graspable designs (from on_rb5/)
  python B/disturb_impulse.py --labels design075 candidate_02 --workers 8
  python B/disturb_impulse.py --labels design075 --limit 3 --dirs py nx       # quick test
  python B/disturb_impulse.py --pulse-ms 20 --mags 1 3 5 --tag p20            # another setting, own folder
  python B/disturb_impulse.py --rescore --slip-mode residual                  # re-judge only
Stopped? Run the same command again: finished cells are kept (one file per cell).

Reads the grasp success cells (grasp_cells/<label>/) and A's results (A/results/:
graspable_designs.csv, <label>/force_*). Writes
B/results/impulse[_<tag>]/<label>/
  traces/<cell>.npz   per trial '<dir>_<k>mg_trace' ('py_2mg_trace', ...; columns in
                      'columns'), '_ended', zero control 'z0_*'
  impulse_trials.csv  one row per cell x direction x pulse size
  impulse_cells.csv   one row per cell: level per direction, level_min, pass_<k>mg, zero control
  impulse_meta.json   settings and criteria;   impulse_summary.json   design summary
and B/results/impulse[_<tag>]/all_designs.csv (one row per design, with A's numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from gripper_core import JOINTS, LINKS, contact_rules, read_json, write_json
from rb5_eval import validate_rb5_config
from eval_design_rb5 import read_rows, to_json
from disturb_collect import cell_name, label_dir, save_table, set_state
import disturb_force as df
from disturb_force import BILATERAL_N, GROUPS, classify_many, force_dir, link_forces
from run_pick_lift import ARM_JOINTS, PickLift

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
EDGES = {f"{sa}{a}{sb}{b}": tuple((1 if sa == "+" else -1)*(c == a) + (1 if sb == "+" else -1)*(c == b)
                                  for c in "xyz")
         for a, b in (("x", "y"), ("x", "z"), ("y", "z")) for sa in "+-" for sb in "+-"}
DIRECTIONS = {**AXES, **EDGES}
DIR_SETS = {"axes": list(AXES), "edges": list(EDGES), "all": list(DIRECTIONS)}
ZERO = "0"
DEFAULT_MAGS = (0.5, 1.0, 2.0, 3.0)
# A's trace columns, then the maxima since the previous row (equal to the sample itself
# inside the detail window) and the longest lost-contact streak [s] in that interval.
TRACE_COLUMNS = (*df.TRACE_COLUMNS, "disp_max_mm", "rot_max_deg", "tilt_max_deg", "lost_s")
C = {n: i for i, n in enumerate(TRACE_COLUMNS)}
LINK_COLS = slice(C["R_gp_N"], C["palm_N"]+1)
SIM_KEYS = ("pulse_s", "shape", "mags", "dirs", "after_s", "sample_s", "detail_s", "coarse_s", "stop_mm",
            "stop_lost_s", "zero_control", "monotonic", "mg_N")
MAX_MODELS = 6  # Compiled scenes kept per worker (one per design and radius).
EPS = 1e-6  # Time tolerance [s]; traces are float32 and the sample interval is 5 ms.


def dkey(direction):
    return "z0" if direction == ZERO else direction.replace("+", "p").replace("-", "n")


def tkey(direction, k):
    return "z0" if direction == ZERO else f"{dkey(direction)}_{k:g}mg".replace(".", "p")


def parse_dirs(names):
    """'axes' / 'edges' / 'all' and single directions (px or +x, pxpy or +x+y)."""
    back = {dkey(d): d for d in DIRECTIONS}
    out = []
    for n in names:
        out += DIR_SETS.get(n) or [back.get(n, n)]
    bad = [d for d in out if d not in DIRECTIONS]
    if bad:
        raise SystemExit(f"Unknown direction(s) {bad}. Use axes, edges, all or e.g. px nx py ny pz nz pxpy.")
    return list(dict.fromkeys(out))


def sub_dir(tag):
    return "impulse" + (f"_{tag}" if tag else "")


def impulse_dir(run, sub, label=None):
    """B results of one setting: B/results/<sub>/, or of one design in it."""
    d = layout.b_root(run)/sub
    return d/label if label else d


# ---------------------------------------------------------------- simulation
def pulse(shape, peak_N, T, dt):
    """Force of every step during the pulse (step i covers t = i*dt .. (i+1)*dt)."""
    n = max(1, round(T/dt))
    if shape == "rect":
        return np.full(n, float(peak_N))
    return float(peak_N)*np.sin(np.pi*(np.arange(n)+0.5)/n)


def impact(model, data, state, d_hand, forces, s):
    """Restore the post-lift state, push the object along the hand direction d_hand with
    forces[i] [N] in step i (nothing after the pulse) and run pulse_s + after_s.

    ended: 'end' (time over), 'drop' (object clearly gone: moved >= stop_mm or no bilateral
    contact for >= stop_lost_s), 'unintended_collision' or 'unstable'."""
    set_state(model, data, state)
    ob, og, site = model.body("obj").id, model.geom("obj_g").id, model.site("grasp").id
    links = {model.geom(n).id: i for i, n in enumerate(LINKS)}
    armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
    qa = np.array([model.joint(n).qposadr[0] for n in JOINTS])
    q0 = data.qpos[qa].copy()
    dt = model.opt.timestep
    every = max(1, round(s["sample_s"]/dt))
    coarse = max(1, round(s["coarse_s"]/s["sample_s"]))
    rs0 = data.site_xmat[site].reshape(3, 3).copy()
    h0 = data.site_xpos[site].copy()
    p0 = rs0.T @ (data.xpos[ob]-h0)
    r0 = rs0.T @ data.xmat[ob].reshape(3, 3)
    d = np.asarray(d_hand, float)
    direction = rs0 @ (d/np.linalg.norm(d))
    sim = PickLift(model, data)
    sim.stage = "impulse"
    sim.qcmd = data.ctrl[armact].copy()  # The arm keeps holding its last command.
    t_start = data.time
    rows, acc = [], np.zeros(4)  # acc: maxima since the last stored row (disp, rot, tilt, lost streak).
    state_ = dict(onset=None, since_store=0)

    def measure(force):
        rs = data.site_xmat[site].reshape(3, 3)
        p = rs.T @ (data.xpos[ob]-data.site_xpos[site])
        r = rs.T @ data.xmat[ob].reshape(3, 3)
        rot = math.degrees(math.acos(np.clip((np.trace(r0.T @ r)-1)/2, -1, 1)))
        tilt = math.degrees(math.acos(np.clip(float(r0[:, 2] @ r[:, 2]), -1, 1)))
        f = link_forces(model, data, og, links)
        t = data.time-t_start
        disp = 1000*float(np.linalg.norm(p-p0))
        bilateral = min(f[0]+f[1], f[2]+f[3]) > BILATERAL_N
        state_["onset"] = None if bilateral else (t if state_["onset"] is None else state_["onset"])
        streak = 0.0 if state_["onset"] is None else t-state_["onset"]
        np.maximum(acc, (disp, rot, tilt, streak), out=acc)
        row = [t, force, *(1000*(p-p0)), rot, *f, *np.degrees(data.qpos[qa]-q0),
               1000*float(np.linalg.norm(data.site_xpos[site]-h0)), tilt]
        return row, disp, streak

    def store(row):
        rows.append(row+list(acc))
        acc[:] = 0
        state_["since_store"] = 0

    store(measure(0.0)[0])
    n = max(1, math.ceil((s["pulse_s"]+s["after_s"])/dt-1e-9))
    ended, force, row = "end", 0.0, None
    try:
        for i in range(n):
            force = float(forces[i]) if i < len(forces) else 0.0
            data.xfrc_applied[ob, :3] = force*direction
            sim.step()
            if (i+1) % every and i != n-1:
                continue
            row, disp, streak = measure(force)
            state_["since_store"] += 1
            gone = disp >= s["stop_mm"] or streak >= s["stop_lost_s"]-EPS
            if gone or i == n-1 or row[0] <= s["detail_s"]+EPS or state_["since_store"] >= coarse:
                store(row)
                row = None
            if gone:
                ended = "drop"
                break
    except RuntimeError as exc:
        ended = "unintended_collision" if str(exc).startswith("Unintended collision") else "unstable"
        if row is not None:
            store(row)
    data.xfrc_applied[:] = 0
    return dict(trace=np.asarray(rows, dtype=np.float32), ended=ended, direction_world=direction,
                penetration_mm=1000*sim.max_unintended_penetration, impulse_Ns=float(np.sum(forces))*dt)


def open_cell(out, c, cfg):
    """disturb_force.open_cell with a bounded per-worker model cache (many designs per worker)."""
    if len(df.WORKER) > MAX_MODELS:
        arm = df.WORKER.get("_arm")
        df.WORKER.clear()
        if arm is not None:
            df.WORKER["_arm"] = arm
    return df.open_cell(out, c, cfg)


def impulse_task(task):
    """Zero control and every direction x pulse size of the given cells (one design, one radius)."""
    out, dst, cfg, s, crit, h, cells = task  # grasp states of the design, its B results
    out = Path(out)
    dest = Path(dst)/"traces"
    done, start = [], time.perf_counter()
    for c in cells:
        model, data, st = open_cell(out, c, cfg)
        dt = model.opt.timestep
        saved, failed = {}, set()
        trials = ([(ZERO, 0.0)] if s["zero_control"] else []) + [(d, k) for d in s["dirs"] for k in s["mags"]]
        for d, k in trials:
            name = tkey(d, k)
            if s["monotonic"] and d in failed:  # A larger pulse after a failure: assumed to fail.
                saved[f"{name}_ended"] = "skipped"
                continue
            t0 = time.perf_counter()
            forces = np.zeros(0) if d == ZERO else pulse(s["shape"], k*s["mg_N"], s["pulse_s"], dt)
            res = impact(model, data, st["state"], DIRECTIONS.get(d, (1, 0, 0)), forces, s)
            saved.update({f"{name}_trace": res["trace"], f"{name}_ended": res["ended"],
                          f"{name}_world": res["direction_world"], f"{name}_pen_mm": res["penetration_mm"],
                          f"{name}_impulse_Ns": res["impulse_Ns"], f"{name}_sec": time.perf_counter()-t0})
            if s["monotonic"] and d != ZERO and not judge(res["trace"], res["ended"], crit, s)["held"]:
                failed.add(d)
        name = cell_name(st["r_mm"], st["x_mm"], st["y_mm"])
        tmp = dest/f"{name}.tmp.npz"
        np.savez_compressed(tmp, r_mm=float(st["r_mm"]), x_mm=float(st["x_mm"]), y_mm=float(st["y_mm"]),
                            settings_hash=h, columns=np.array(TRACE_COLUMNS), dirs=np.array(s["dirs"]),
                            mags=np.array(s["mags"], dtype=float), state_file=c["state_file"],
                            monotonic_criteria=json.dumps(crit), **saved)
        os.replace(tmp, dest/f"{name}.npz")
        done.append(name)
    return out.name, done, time.perf_counter()-start


# ---------------------------------------------------------------- judging
def first_index(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def events(trace, crit, s):
    """First row index of every criterion (None = never met), from the interval maxima."""
    t = trace[:, 0]
    disp = trace[:, C["disp_max_mm"]].astype(float)
    if crit["slip_mode"] == "residual":
        disp = np.where(t >= s["pulse_s"]+s["settle_s"]-EPS, disp, 0.0)
    out = {"slip": first_index(disp >= crit["slip_mm"])}
    if crit["rot_deg"] > 0:
        rot = trace[:, C["tilt_max_deg" if crit["rot_mode"] == "tilt" else "rot_max_deg"]]
        out["rotation"] = first_index(rot >= crit["rot_deg"])
    out["lost"] = first_index(trace[:, C["lost_s"]] >= crit["grace_s"]-EPS)
    return out


def judge(trace, ended, crit, s):
    """Held or not, cause, onset time and phase of the failure."""
    trace = np.asarray(trace, dtype=np.float64)
    fired = {k: i for k, i in events(trace, crit, s).items() if i is not None}
    t = trace[:, 0]
    if fired:
        cause = min(fired, key=lambda k: (fired[k], k))
        i = fired[cause]
        onset = float(t[i]-(trace[i, C["lost_s"]] if cause == "lost" else 0.0))
    elif ended == "end":
        cause, onset = "", None
    else:  # 'drop' with criteria looser than the stop rule, collision or instability.
        cause, onset = ended, float(t[-1])
    T, settle = s["pulse_s"], s["settle_s"]
    phase = "" if onset is None else ("pulse" if onset <= T+EPS else "transient" if onset <= T+settle+EPS
                                      else "delayed")
    return dict(held=onset is None, cause=cause, phase=phase, t_fail_s=onset)


def describe(trace, s, window, threshold, rules):
    """Size of the response: displacement, rotation, hand motion, contact type at the end."""
    trace = np.asarray(trace, dtype=np.float64)
    t = trace[:, 0]
    after = t >= s["pulse_s"]+s["settle_s"]-EPS
    disp_now = np.linalg.norm(trace[:, 2:5], axis=1)
    end = trace[t >= t[-1]-window-EPS, LINK_COLS].mean(axis=0, keepdims=True)
    return dict(max_disp_mm=float(trace[:, C["disp_max_mm"]].max()),
                max_disp_after_settle_mm=float(trace[after, C["disp_max_mm"]].max()) if after.any() else None,
                residual_disp_mm=float(disp_now[-1]), max_rot_deg=float(trace[:, C["rot_max_deg"]].max()),
                max_tilt_deg=float(trace[:, C["tilt_max_deg"]].max()), max_lost_s=float(trace[:, C["lost_s"]].max()),
                max_hand_mm=float(trace[:, C["hand_mm"]].max()),
                max_finger_dq_deg=float(np.abs(trace[:, C["dR_jp_deg"]:C["dL_jd_deg"]+1]).max()),
                type_end=str(classify_many(end, threshold, *rules)[0]), T_s=float(t[-1]))


# ---------------------------------------------------------------- analysis
def selected_cells(out, fdir, which):
    """States to test: A's success cells after A's zero-force rule (default) or every
    re-run success cell. Each with its A group, contact type and m_F."""
    cells = [c for c in read_rows(out/"cells.csv") if c["state_file"] and c["success"]]
    groups = {}
    if (fdir/"force_cells.csv").is_file() and (fdir/"force_cells.csv").stat().st_size:
        groups = {cell_name(r["r_mm"], r["x_mm"], r["y_mm"]): r["group"] for r in read_rows(fdir/"force_cells.csv")}
    for c in cells:
        c["A_group"] = groups.get(cell_name(c["r_mm"], c["x_mm"], c["y_mm"]), "")
    if which == "success" and groups:
        cells = [c for c in cells if c["A_group"] in GROUPS]
    return cells


def a_forces(fdir):
    """A's ramp F_max [mg] (and hold value, if measured) per (cell, direction)."""
    path = fdir/"force_dirs.csv"
    if not path.is_file() or not path.stat().st_size:
        return {}
    return {(cell_name(r["r_mm"], r["x_mm"], r["y_mm"]), r["dir"]): (r["F_max_mg"], r.get("F_hold_mg"), r["censored"])
            for r in read_rows(path)}


def analyze(run, label, cfg, crit, sub, zero_rule="fail", quiet=False):
    """impulse_trials.csv, impulse_cells.csv and impulse_summary.json from the saved traces."""
    out, fdir, base = label_dir(run, label), force_dir(run, label), impulse_dir(run, sub, label)
    meta = read_json(base/"impulse_meta.json")
    s = meta["settings"]
    if crit["slip_mm"] > s["stop_mm"] or crit["grace_s"] > s["stop_lost_s"]:
        print(f"Warning: criteria looser than the stop rule (stop_mm={s['stop_mm']}, stop_lost_s={s['stop_lost_s']}); "
              "trials stopped by it count as failures ('drop').")
    if s["monotonic"] and not quiet:
        print(f"Note [{label}]: --monotonic run, larger pulses after a failure were not simulated "
              f"(decided with {meta.get('criteria_at_simulation')}).")
    sc = cfg["simulation"]
    rules, thr = contact_rules(cfg), sc["contact_threshold_N"]
    window = sc["force_average_window_s"]
    info = {cell_name(c["r_mm"], c["x_mm"], c["y_mm"]): c for c in selected_cells(out, fdir, "rerun")}
    A = a_forces(fdir)
    trial_rows, cell_rows = [], []
    for path in sorted((base/"traces").glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        tr = np.load(path)
        name = path.stem
        c = info[name]
        a_group = c["A_group"]
        zero = {}
        if "z0_trace" in tr.files:
            z = judge(tr["z0_trace"], str(tr["z0_ended"]), crit, s)
            zero = dict(zero_fail=not z["held"], zero_cause=z["cause"], zero_t_fail_s=z["t_fail_s"],
                        zero_max_disp_mm=float(tr["z0_trace"][:, C["disp_max_mm"]].max()),
                        zero_max_rot_deg=float(tr["z0_trace"][:, C["rot_max_deg"]].max()))
        group0 = str(c["contact_type"])
        removed = zero_rule == "fail" and bool(zero.get("zero_fail"))
        cinfo = dict(r_mm=c["r_mm"], x_mm=c["x_mm"], y_mm=c["y_mm"], group="zero_fail" if removed else group0,
                     base_group=group0, m_F=c["m_F"])
        dirs, mags = [str(d) for d in tr["dirs"]], [float(k) for k in tr["mags"]]
        held, levels, non_mono = {}, {}, []
        for d in dirs:
            for k in mags:
                n = tkey(d, k)
                ended = str(tr[f"{n}_ended"])
                Fa, Fh, Fcens = A.get((name, d), (None, None, None))
                row = dict(cinfo, dir=d, mag_mg=k, A_F_ramp_mg=Fa, A_F_hold_mg=Fh, A_censored=Fcens)
                if ended == "skipped":
                    row.update(held=False, cause="skipped", phase="", ended_by=ended)
                else:
                    trace = tr[f"{n}_trace"]
                    row.update(judge(trace, ended, crit, s), ended_by=ended,
                               impulse_Ns=float(tr[f"{n}_impulse_Ns"]), **describe(trace, s, window, thr, rules),
                               max_unintended_penetration_mm=float(tr[f"{n}_pen_mm"]), wall_s=float(tr[f"{n}_sec"]))
                    row["type_changed"] = bool(row["held"] and row["type_end"] != group0)
                held[d, k] = row["held"]
                trial_rows.append(row)
            lv = 0.0
            for k in mags:
                if not held[d, k]:
                    break
                lv = k
            levels[d] = lv
            if any(not held[d, a] and held[d, b] for a in mags for b in mags if a < b):
                non_mono.append(d)
        weakest = min(dirs, key=lambda d: levels[d])
        row = dict(cinfo, contact_type=group0, A_group=a_group or None, level_min_mg=levels[weakest],
                   weakest_dir=weakest, non_monotonic=" ".join(non_mono), **zero)
        row.update({f"level_{d}_mg": levels[d] for d in dirs})
        row.update({f"pass_{k:g}mg": levels[weakest] >= k-1e-9 for k in mags})
        cell_rows.append(row)
    save_table(base/"impulse_trials.csv", trial_rows)
    save_table(base/"impulse_cells.csv", cell_rows)
    meta.update(criteria=crit, zero_rule=zero_rule)
    write_json(base/"impulse_meta.json", meta)
    sm = summarize(label, cell_rows, trial_rows, cfg, s, out, fdir)
    sm.update(criteria=crit, zero_rule=zero_rule)
    write_json(base/"impulse_summary.json", json.loads(json.dumps(sm, default=to_json)))
    return sm


def summarize(label, cells, trials, cfg, s, out, fdir):
    o = cfg["objective"]
    ok = [c for c in cells if c["group"] in GROUPS]
    okset = {(c["r_mm"], c["x_mm"], c["y_mm"]) for c in ok}
    tr_ok = [t for t in trials if (t["r_mm"], t["x_mm"], t["y_mm"]) in okset]
    env, tip = [c for c in ok if c["group"] == "env"], [c for c in ok if c["group"] == "tip"]
    a_sm = read_json(fdir/"force_summary.json") if (fdir/"force_summary.json").is_file() else {}
    meta = read_json(out/"meta.json") if (out/"meta.json").is_file() else {}
    mags, dirs = s["mags"], s["dirs"]
    out_ = dict(label=label, cells_in_map=meta.get("cells_in_map"), tested=len(cells),
                zero_fail_removed=dict(Counter(c["base_group"] for c in cells if c["group"] == "zero_fail")),
                success=len(ok), env=len(env), tip=len(tip), A_success=a_sm.get("success"),
                A_J_stable=a_sm.get("J_stable"), A_robust_area=a_sm.get("robust_area"),
                A_F_min_median_mg=a_sm.get("F_min_median_mg"),
                pulse_ms=1000*s["pulse_s"], shape=s["shape"], after_s=s["after_s"], mags=mags, dirs=dirs,
                impact_area=float(sum(c["level_min_mg"] for c in ok)),
                level_min_median_mg=float(np.median([c["level_min_mg"] for c in ok])) if ok else 0.0,
                weakest_dirs=dict(Counter(c["weakest_dir"] for c in ok if c["level_min_mg"] < mags[-1])),
                causes=dict(Counter(t["cause"] for t in tr_ok if not t["held"])),
                phases=dict(Counter(t["phase"] for t in tr_ok if not t["held"] and t["phase"])),
                non_monotonic_cells=sum(bool(c["non_monotonic"]) for c in ok),
                type_changed=dict(Counter(f"{t['base_group']}->{t['type_end']}" for t in tr_ok if t.get("type_changed"))),
                levels=[], held_fraction={})
    for k in mags:
        p = [c for c in ok if c[f"pass_{k:g}mg"]]
        pe, pt = sum(c["group"] == "env" for c in p), sum(c["group"] == "tip" for c in p)
        at_k = [t for t in tr_ok if abs(t["mag_mg"]-k) < 1e-9]
        withA = [t for t in at_k if t["A_F_ramp_mg"] is not None]
        out_["levels"].append(dict(
            mag_mg=k, passed=len(p), env=pe, tip=pt, cond=len(p)/len(ok) if ok else None,
            J_impulse=pe+o["w_tip"]*pt+o["lambda_force"]*sum(c["m_F"] for c in p),
            per_direction={d: sum(t["held"] for t in at_k if t["dir"] == d) for d in dirs},
            held_above_A=sum(t["held"] and t["A_F_ramp_mg"] < k for t in withA),
            below_A=sum(t["A_F_ramp_mg"] < k for t in withA),
            failed_within_A=sum((not t["held"]) and t["A_F_ramp_mg"] >= k for t in withA),
            within_A=sum(t["A_F_ramp_mg"] >= k for t in withA)))
    for g, rows in (("env", env), ("tip", tip)):
        names = {(c["r_mm"], c["x_mm"], c["y_mm"]) for c in rows}
        sel = [t for t in tr_ok if (t["r_mm"], t["x_mm"], t["y_mm"]) in names]
        out_["held_fraction"][g] = {d: {f"{k:g}": (float(np.mean([t["held"] for t in sel if t["dir"] == d
                                                                 and abs(t["mag_mg"]-k) < 1e-9])) if sel else None)
                                        for k in mags} for d in dirs}
    return out_


def print_summary(sm):
    rm = sm["zero_fail_removed"]
    print(f"\n[{sm['label']}] {sm['tested']} cells tested, {sum(rm.values())} fail the zero-force control {rm or ''}"
          f" -> success {sm['success']} (env {sm['env']}, tip {sm['tip']}); pulse {sm['pulse_ms']:g} ms {sm['shape']}, "
          f"then {sm['after_s']:g} s")
    print(f"  impact_area {sm['impact_area']:.1f} mg, median level_min {sm['level_min_median_mg']:.2f} mg, "
          f"weakest {sm['weakest_dirs']}, failures {sm['causes']} by phase {sm['phases']}"
          + (f", non-monotonic cells {sm['non_monotonic_cells']}" if sm["non_monotonic_cells"] else ""))
    print(f"  {'pulse':>6} {'pass':>5} {'env':>4} {'tip':>4} {'cond':>5} {'J_imp':>7}  "
          + " ".join(f"{d:>5}" for d in sm["dirs"]) + "   held>A  fail<=A")
    for L in sm["levels"]:
        cond = "-" if L["cond"] is None else f"{L['cond']:.2f}"
        print(f"  {L['mag_mg']:>4g}mg {L['passed']:>5} {L['env']:>4} {L['tip']:>4} {cond:>5} {L['J_impulse']:>7.2f}  "
              + " ".join(f"{L['per_direction'][d]:>5}" for d in sm["dirs"])
              + f"   {L['held_above_A']:>3}/{L['below_A']:<4} {L['failed_within_A']:>3}/{L['within_A']}")
    if sm["type_changed"]:
        print(f"  held trials whose contact type changed: {sm['type_changed']}")


def all_designs_table(run, sub, sms, rank_mg):
    """One row per design: design values, A's numbers, B's numbers."""
    try:
        from batch_disturbance import design_info
    except (ImportError, SystemExit):
        def design_info(run, label):
            return {}
    rows = []
    for sm in sms:
        row = dict(label=sm["label"], **design_info(Path(run), sm["label"]))
        row.update({k: sm.get(k) for k in ("A_success", "A_J_stable", "A_robust_area", "A_F_min_median_mg",
                                            "tested", "success", "env", "tip", "impact_area", "level_min_median_mg",
                                            "non_monotonic_cells")})
        row["zero_fail_removed"] = sum(sm["zero_fail_removed"].values())
        for L in sm["levels"]:
            k = f"{L['mag_mg']:g}mg"
            row.update({f"pass_{k}": L["passed"], f"pass_env_{k}": L["env"], f"pass_tip_{k}": L["tip"],
                        f"cond_{k}": L["cond"], f"J_impulse_{k}": L["J_impulse"]})
        for g in ("env", "tip"):
            for d, per in sm["held_fraction"].get(g, {}).items():
                row.update({f"held_{d}_{k}mg_{g}": v for k, v in per.items()})
        row.update({f"fail_{p}": sm["phases"].get(p, 0) for p in ("pulse", "transient", "delayed")})
        row["criteria"] = " ".join(f"{k}={v}" for k, v in sm.get("criteria", {}).items())
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    rows = [{k: r.get(k) for k in keys} for r in rows]
    path = impulse_dir(run, sub)/"all_designs.csv"
    save_table(path, rows)
    col = f"J_impulse_{rank_mg:g}mg"
    for key, text in (("impact_area", "impact_area [mg] (sum over success cells of the weakest direction's level)"),
                      (col, f"J_impulse at {rank_mg:g} mg")):
        ranked = sorted((r for r in rows if r.get(key) is not None), key=lambda r: -r[key])
        if ranked:
            print(f"Top 10 by {text}:")
            for r in ranked[:10]:
                print(f"  {r['label']:<14} {key}={r[key]:7.2f}  pass@{rank_mg:g}mg={r.get(f'pass_{rank_mg:g}mg')!s:>3}"
                      f"  success={r['success']} (env {r['env']}, tip {r['tip']})  A: robust_area="
                      f"{r['A_robust_area'] if r['A_robust_area'] is None else round(r['A_robust_area'], 1)}")
    return path


# ---------------------------------------------------------------- main
def graspable_labels(run):
    path = layout.a_root(run)/"graspable_designs.csv"
    if path.is_file():
        return [str(r["label"]) for r in read_rows(path)]
    return sorted(d.name for d in layout.a_root(run).iterdir() if (d/"force_cells.csv").is_file())


def settings_hash(s):
    return hashlib.sha256(json.dumps({k: s[k] for k in SIM_KEYS}, sort_keys=True).encode()).hexdigest()[:16]


def mg_of(out, fdir, cfg, cells):
    meta = fdir/"force_meta.json"
    if meta.is_file() and read_json(meta).get("settings", {}).get("mg_N"):
        return float(read_json(meta)["settings"]["mg_N"])
    st = np.load(out/cells[0]["state_file"])
    model, _ = df.model_for(str(out/"hands"/str(st["hand_file"])), cfg)
    return float(model.body_mass[model.body("obj").id])*cfg["simulation"]["gravity_m_s2"]


def prepare(run, label, cfg, s, crit, sub, which, limit, overwrite):
    """Settings of one design, cells still to simulate. Refuses to mix settings in one folder."""
    out, fdir = label_dir(run, label), force_dir(run, label)
    if not (out/"cells.csv").is_file():
        raise SystemExit(f"No {out/'cells.csv'}. Run grasp/disturb_collect.py (and A/disturb_force.py) first.")
    cells = selected_cells(out, fdir, which)
    if limit:
        cells = cells[:limit]
    base = impulse_dir(run, sub, label)
    (base/"traces").mkdir(parents=True, exist_ok=True)
    s = dict(s, mg_N=mg_of(out, fdir, cfg, cells) if cells else None)
    h = settings_hash(s)
    old = read_json(base/"impulse_meta.json") if (base/"impulse_meta.json").is_file() else None
    if old and old.get("settings_hash") != h:
        if not overwrite:
            raise SystemExit(f"{base} holds results made with other settings ({old['settings']}).\n"
                             "Use --overwrite to replace them or --tag NAME to keep both.")
        for p in (base/"traces").glob("*.npz"):
            p.unlink()
    for p in (base/"traces").glob("*.tmp.npz"):
        p.unlink()
    write_json(base/"impulse_meta.json", dict(label=label, settings=s, settings_hash=h, cells=which, limit=limit,
                                              criteria_at_simulation=crit, mujoco=mujoco.__version__))
    todo = [c for c in cells if not (base/"traces"/f"{cell_name(c['r_mm'], c['x_mm'], c['y_mm'])}.npz").is_file()]
    return s, h, cells, todo


def simulate(run, labels, cfg, s, crit, args):
    sub = sub_dir(args.tag)
    tasks, total, per_label = [], 0, {}
    for label in labels:
        sl, h, cells, todo = prepare(run, label, cfg, s, crit, sub, args.cells, args.limit, args.overwrite)
        per_label[label] = len(todo)
        total += len(todo)
        by_r = {}
        for c in todo:
            by_r.setdefault(c["r_mm"], []).append(c)
        for _, g in sorted(by_r.items()):
            for i in range(0, len(g), args.chunk):
                tasks.append((str(label_dir(run, label)), str(impulse_dir(run, sub, label)), cfg, sl, crit, h,
                              g[i:i+args.chunk]))
    n_trials = (len(s["dirs"])*len(s["mags"])+s["zero_control"])
    print(f"{len(labels)} designs, {total} cells to simulate ({sum(1 for v in per_label.values() if not v)} designs "
          f"already done) x {n_trials} trials of {s['pulse_s']+s['after_s']:g} s; pulse {1000*s['pulse_s']:g} ms "
          f"{s['shape']} of {', '.join(f'{k:g}' for k in s['mags'])} m*g in {len(s['dirs'])} directions; "
          f"{args.workers} workers -> {impulse_dir(run, sub)}", flush=True)
    finished = [l for l, v in per_label.items() if v == 0]
    done, wall = 0, time.perf_counter()

    def report(label, names, sec):
        nonlocal done
        done += len(names)
        per_label[label] -= len(names)
        if per_label[label] == 0:
            finished.append(label)
            sm = analyze(run, label, cfg, crit, sub, args.zero_fail, quiet=True)
            el = time.perf_counter()-wall
            print(f"  [{len(finished)}/{len(labels)}] {label}: success {sm['success']} (env {sm['env']}, tip {sm['tip']}), "
                  "pass " + " ".join(f"{L['mag_mg']:g}mg:{L['passed']}" for L in sm["levels"])
                  + f", impact_area {sm['impact_area']:.1f} | {done}/{total} cells, {el/60:.1f} min, "
                  f"~{el/done*(total-done)/60:.0f} min left", flush=True)

    if not tasks:
        return
    if args.workers > 1 and len(tasks) > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
        try:
            for f in as_completed([pool.submit(impulse_task, t) for t in tasks]):
                report(*f.result())
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise SystemExit("\nStopped. Run the same command again to continue (finished cells are kept).")
        pool.shutdown()
    else:
        for t in tasks:
            report(*impulse_task(t))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=layout.DEFAULT_RUN)
    p.add_argument("--labels", nargs="+", default=["graspable"],
                   help="'graspable' (default: disturbance/graspable_designs.csv of experiment A) or names")
    p.add_argument("--cells", choices=("success", "rerun"), default="success",
                   help="success: A's success cells after A's zero-force rule (default); rerun: every re-run "
                        "success cell with a state")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--chunk", type=int, default=4, help="Cells per worker task (default 4)")
    p.add_argument("--limit", type=int, help="Only the first N cells per design (smoke test)")
    p.add_argument("--tag", default="", help="Results go to impulse_<tag>/ (keeps runs with other settings apart)")
    p.add_argument("--overwrite", action="store_true", help="Replace results made with other settings")
    p.add_argument("--rescore", action="store_true", help="Only re-judge the saved traces (no simulation)")
    g = p.add_argument_group("pulse (simulation)")
    g.add_argument("--pulse-ms", type=float, default=50.0, help="Pulse duration T [ms] (default 50)")
    g.add_argument("--shape", choices=("rect", "halfsine"), default="rect",
                   help="rect: constant k*m*g for T (default); halfsine: peak k*m*g")
    g.add_argument("--mags", type=float, nargs="+", default=list(DEFAULT_MAGS),
                   help="Pulse sizes k [m*g] (default 0.5 1 2 3)")
    g.add_argument("--dirs", nargs="+", default=["axes"], metavar="DIR",
                   help="axes (default: px nx py ny pz nz), edges (12 diagonals), all, or single directions")
    g.add_argument("--after-s", type=float, default=10.0, help="Hold time after the pulse [s] (default 10)")
    g.add_argument("--settle-s", type=float, default=0.5,
                   help="End of the 'transient' phase after the pulse [s]; also the start of --slip-mode residual")
    g.add_argument("--detail-s", type=float, help="Store every sample up to this time [s] (default T+settle-s)")
    g.add_argument("--coarse-s", type=float, default=0.1, help="Row interval after the detail window [s]")
    g.add_argument("--stop-mm", type=float, help="Stop a trial once the object moved this far (default lift_slip_mm)")
    g.add_argument("--stop-lost-s", type=float, default=0.1, help="... or lost bilateral contact this long [s]")
    g.add_argument("--no-zero-control", action="store_true", help="Skip the zero-force control")
    g.add_argument("--monotonic", action="store_true",
                   help="Skip larger pulses in a direction after one failed (fewer trials; assumes monotonic)")
    c = p.add_argument_group("criteria (also for --rescore)")
    c.add_argument("--slip-mm", type=float, help="default simulation.hold_displacement_mm")
    c.add_argument("--slip-mode", choices=("any", "residual"), default="any",
                   help="any: displacement at any time (default, as A); residual: only after settle-s")
    c.add_argument("--rot-deg", type=float, help="default simulation.hold_rotation_deg; 0 disables")
    c.add_argument("--rot-mode", choices=("total", "tilt"), default="total")
    c.add_argument("--grace-s", type=float, help="default simulation.lost_contact_grace_s")
    c.add_argument("--zero-fail", choices=("fail", "keep"), default="fail",
                   help="Cells failing the zero-force control: drop (default) or keep and only flag")
    p.add_argument("--rank-mg", type=float, help="Pulse size for the printed J_impulse ranking (default: middle size)")
    args = p.parse_args()

    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    sc = cfg["simulation"]
    crit = dict(slip_mm=args.slip_mm if args.slip_mm is not None else sc["hold_displacement_mm"],
                slip_mode=args.slip_mode,
                rot_deg=args.rot_deg if args.rot_deg is not None else sc["hold_rotation_deg"], rot_mode=args.rot_mode,
                grace_s=args.grace_s if args.grace_s is not None else sc.get("lost_contact_grace_s", 0.02))
    sub = sub_dir(args.tag)
    if args.rescore:
        labels = [l for l in (graspable_labels(run) if args.labels == ["graspable"] else args.labels)
                  if (impulse_dir(run, sub, l)/"impulse_meta.json").is_file()]
        if not labels:
            raise SystemExit(f"No {sub}/ results to re-judge.")
    else:
        labels = graspable_labels(run) if args.labels == ["graspable"] else args.labels
        T = args.pulse_ms/1000
        s = dict(pulse_s=T, shape=args.shape, mags=sorted(set(args.mags)), dirs=parse_dirs(args.dirs),
                 after_s=args.after_s, settle_s=args.settle_s, sample_s=sc["sample_interval_s"],
                 detail_s=args.detail_s if args.detail_s is not None else T+args.settle_s,
                 coarse_s=args.coarse_s, stop_mm=args.stop_mm if args.stop_mm is not None else sc["lift_slip_mm"],
                 stop_lost_s=args.stop_lost_s, zero_control=not args.no_zero_control, monotonic=args.monotonic)
        if (T <= 0 or args.after_s < 0 or min(s["mags"]) <= 0 or s["detail_s"] < T+args.settle_s
                or s["stop_mm"] < crit["slip_mm"] or s["stop_lost_s"] < crit["grace_s"]):
            raise SystemExit("Need pulse-ms>0, after-s>=0, mags>0, detail-s >= T+settle-s, stop-mm >= slip-mm "
                             "and stop-lost-s >= grace-s")
        simulate(run, labels, cfg, s, crit, args)
    sms = []
    for label in labels:
        if not (impulse_dir(run, sub, label)/"impulse_meta.json").is_file():
            continue
        sm = analyze(run, label, cfg, crit, sub, args.zero_fail, quiet=len(labels) > 1)
        sms.append(sm)
    print(f"\nCriteria: slip >= {crit['slip_mm']} mm ({crit['slip_mode']}), {crit['rot_mode']} rotation >= "
          f"{crit['rot_deg']} deg (0 = off), lost contact >= {crit['grace_s']} s; zero-force control: {args.zero_fail}")
    if len(sms) <= 8:
        for sm in sms:
            print_summary(sm)
    if sms:
        mags = sms[0]["mags"]
        rank = args.rank_mg if args.rank_mg is not None else mags[len(mags)//2]
        print("\nSaved:", all_designs_table(run, sub, sms, rank))


if __name__ == "__main__":
    mp.freeze_support()
    main()
