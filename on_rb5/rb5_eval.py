"""Graspable-workspace evaluation with the design168 hand mounted on the RB5 arm.

Every cell (r, x, y) of a design is a full run_pick_lift.PickLift rollout: the hand is
regenerated with gripper_core.build_xml, mounted with build_scene.compose (the same
mounting as rb5_best_design_scene.xml), and the arm lowers behind the cylinder, enters
horizontally, closes the tendon, lifts and holds. Map convention as in gripper_core:
object radius r, object center at hand (x, y+r).

One compiled model is shared by all (x, y) cells of one (design, r); only the 'grasp'
site moves between cells, exactly like rb5_cell.py.

Cell success = run_pick_lift success, plus the workspace rules of the hand-only map
(all switchable in config "rb5_arm"):
  * contact type at the end of 'lift hold' must be env or tip (mixed = one-sided
    enveloping fails); env counts 1, tip counts w_tip in J
  * the object may not be pushed more than max_pre_grasp_shift_mm before closing
  * hold stability in the hand frame during 'lift hold' (hold_displacement_mm,
    hold_rotation_deg of "simulation"), if check_hold_stability
  * mu*sum(N) >= safety_factor*m*g, only if require_force_margin
J = env + w_tip*tip + lambda_force*sum(m_F), the same objective as gripper_core.
The commanded arm path depends only on the cell, not on the hand, so arm-side failures
(reason ik_failed) hit the same cells for every design and do not change the ranking.

Quick checks (design168 by default, or --design a JSON with a "design" entry):
  python rb5_eval.py --cell 40 0 25
  python rb5_eval.py --cell 40 0 25 --save-scene check_scene.xml
  python rb5_eval.py --grid rb5_smoke
  python rb5_eval.py --rescore runs/rb5_cmaes/best_workspace.csv   # new rules, old runs
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

import build_scene
from gripper_core import (CellEvaluator, LINKS, VARIABLES, DesignSpace, axis_values,
                          baseline_parameters, build_xml, classify_contact, contact_rules, read_json,
                          save_rows, validate_config, vector)
from run_pick_lift import ARM_JOINTS, PickLift

ROOT = Path(__file__).resolve().parent
PRE_GRASP_STAGES = ("lower behind cylinder", "approach cylinder", "settle at grasp")
RB5_DEFAULTS = dict(sources_dir="sources", meshdir=build_scene.MESHDIR,
                    max_pre_grasp_shift_mm=2.0, prefilter_approach=True,
                    check_hold_stability=True, require_force_margin=False)
FORCE_KEYS = ("R_prox_N", "R_dist_N", "L_prox_N", "L_dist_N", "palm_N")


class CellFail(RuntimeError):
    def __init__(self, reason, message=""):
        super().__init__(message or reason)
        self.reason = reason


def arm_options(cfg):
    opts = dict(RB5_DEFAULTS)
    opts.update(cfg.get("rb5_arm", {}))
    return opts


def hand_config(cfg):
    """Hand XML always uses the rb5 protocol: object on a table 60 mm below the finger
    plane, as in the RB5 scene. Only the approach prefilter reads that table."""
    c = copy.deepcopy(cfg)
    c["simulation"]["protocol"] = "rb5"
    return c


def validate_rb5_config(cfg):
    validate_config(hand_config(cfg))
    o = arm_options(cfg)
    if o["max_pre_grasp_shift_mm"] is not None and o["max_pre_grasp_shift_mm"] <= 0:
        raise ValueError("rb5_arm.max_pre_grasp_shift_mm must be positive or null")
    s = cfg["simulation"]
    if not 0 < s["tendon_tension_N"] <= 20:
        raise ValueError("run_pick_lift.py supports tendon tension 0..20 N")
    if not 30 <= s["lift_mm"] <= 250:
        raise ValueError("run_pick_lift.py supports lift 30..250 mm")


class CellMonitor:
    """PickLift callback: pre-grasp push, link forces and hold stability."""

    def __init__(self, model, cfg, opts):
        s = cfg["simulation"]
        self.ob = model.body("obj").id
        self.og = model.geom("obj_g").id
        self.site = model.site("grasp").id
        self.links = {model.geom(n).id: i for i, n in enumerate(LINKS)}
        self.every = max(1, round(s["sample_interval_s"]/model.opt.timestep))
        self.shift_tol = opts["max_pre_grasp_shift_mm"]
        self.buf = np.zeros(6)
        self.p0 = None
        self.max_shift_mm = 0.0
        self.samples = {"grasp hold": [], "lift hold": []}
        self.hold_ref = None
        self.hold_drift_mm = 0.0
        self.hold_rotation_deg = 0.0

    def forces(self, m, d):
        f = np.zeros(len(LINKS))
        for i in range(d.ncon):
            c = d.contact[i]
            if self.og not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.og else c.geom1
            if other in self.links:
                mujoco.mj_contactForce(m, d, i, self.buf)
                f[self.links[other]] += max(0.0, float(self.buf[0]))
        return f

    def __call__(self, sim):
        if sim.frame % self.every:
            return
        m, d, stage = sim.m, sim.d, sim.stage
        if stage in PRE_GRASP_STAGES:
            p = d.xpos[self.ob]
            if self.p0 is None:
                self.p0 = p.copy()
            shift = 1000*float(np.linalg.norm(p-self.p0))
            self.max_shift_mm = max(self.max_shift_mm, shift)
            if self.shift_tol is not None and shift > self.shift_tol:
                raise CellFail("approach_push", f"object pushed {shift:.2f} mm before closing")
        elif stage in self.samples:
            self.samples[stage].append((float(d.time), self.forces(m, d)))
            if stage == "lift hold":
                rs = d.site_xmat[self.site].reshape(3, 3)
                p = rs.T @ (d.xpos[self.ob]-d.site_xpos[self.site])
                r = rs.T @ d.xmat[self.ob].reshape(3, 3)
                if self.hold_ref is None:
                    self.hold_ref = (p.copy(), r.copy())
                    return
                p0, r0 = self.hold_ref
                cos = np.clip((np.trace(r0.T @ r)-1)/2, -1, 1)
                self.hold_drift_mm = max(self.hold_drift_mm, 1000*float(np.linalg.norm(p-p0)))
                self.hold_rotation_deg = max(self.hold_rotation_deg, math.degrees(math.acos(cos)))

    def window_mean(self, stage, window_s):
        rows = self.samples[stage]
        if not rows:
            return None
        end = rows[-1][0]
        return np.mean([f for t, f in rows if t >= end-window_s-1e-9], axis=0)


def failed_run_reason(result, lift):
    if result["last_second_min_lift_m"] <= 0.8*lift:
        return "lift_height"
    if not result["last_second_off_table"]:
        return "lift_table_contact"
    if not result["last_second_bilateral_contact"]:
        return "lift_lost_bilateral"
    if result["max_unintended_penetration_m"] >= 0.0001:
        return "unintended_penetration"
    if result["mujoco_warning_count"]:
        return "mujoco_warning"
    return "rb5_fail"


def exception_reason(exc):
    if isinstance(exc, CellFail):
        return exc.reason
    text = str(exc)
    for prefix, reason in (("Unintended collision", "unintended_collision"),
                           ("No bilateral finger contact", "close_no_bilateral"),
                           ("IK failed", "ik_failed"),
                           ("Simulation became unstable", "unstable")):
        if text.startswith(prefix):
            return reason
    return "error"


def empty_row(r_mm, x_mm, y_mm):
    """Every result row has the same columns, in this order."""
    return dict(r_mm=float(r_mm), x_mm=float(x_mm), y_mm=float(y_mm), success=False,
                contact_type="none", reason="unclassified", m_F=0.0, rb5_pass=False,
                prefiltered=False, approach_clearance_mm=None, pre_grasp_shift_mm=None,
                close_type="", total_normal_N=0.0, **{k: 0.0 for k in FORCE_KEYS},
                hold_drift_mm=None, hold_rotation_deg=None, min_lift_mm=None,
                end_tilt_deg=None, max_tracking_error_mm=None,
                max_unintended_penetration_mm=None, unintended_pair="", error="", seconds=0.0)


def judge(row, cfg, opts):
    """Verdict for a cell whose RB5 run passed (object lifted, both sides touching, off
    the table). Shared by run_cells() and rescore(), so both apply identical rules."""
    s = cfg["simulation"]
    if row["contact_type"] not in ("env", "tip"):
        return "hold_"+row["contact_type"]
    if opts["check_hold_stability"] and row["hold_drift_mm"] >= s["hold_displacement_mm"]:
        return "hold_displacement"
    if opts["check_hold_stability"] and row["hold_rotation_deg"] >= s["hold_rotation_deg"]:
        return "hold_rotation"
    if opts["require_force_margin"] and row["m_F"] < 0:
        return "hold_force_margin"
    return "success"


class Rb5Arm:
    """Composes, compiles and runs RB5 scenes. One instance per process."""

    def __init__(self, cfg, root=ROOT):
        self.cfg = cfg
        self.hcfg = hand_config(cfg)
        self.opts = arm_options(cfg)
        self.root = Path(root).resolve()
        src = self.root/self.opts["sources_dir"]
        missing = [n for n in ("rb5_850e.xml", "task_scene.xml") if not (src/n).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {src}. Run from the RB5 folder that "
                                    "contains build_scene.py, sources/ and assets/.")
        self.robot = ET.parse(src/"rb5_850e.xml").getroot()
        self.task = ET.parse(src/"task_scene.xml").getroot()
        self.meshdir = (self.root/self.opts["meshdir"]).resolve()
        s = cfg["simulation"]
        self.tension = float(s["tendon_tension_N"])
        self.lift = s["lift_mm"]/1000

    def hand_xml(self, template, design, r_mm, x_mm=0.0, y_mm=40.0):
        return build_xml(template, design, self.hcfg, radius_mm=r_mm, x_mm=x_mm, y_mm=y_mm)

    def compose(self, hand_xml, meshdir=None):
        scene = build_scene.compose(self.robot, ET.fromstring(hand_xml), copy.deepcopy(self.task),
                                    meshdir=meshdir or self.meshdir)
        # Same idea as rb5_cell.py: relative asset folders must survive compiling from a string.
        if meshdir is None:
            comp = scene.find("compiler")
            for attr in ("texturedir", "assetdir"):
                value = comp.get(attr)
                if value and not Path(value).is_absolute():
                    comp.set(attr, str((self.root/value).resolve()))
        return scene

    def compile(self, hand_xml):
        scene = self.compose(hand_xml)
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
        # The hand XML is written in radians. A degree compiler would silently bend it.
        jp = ET.fromstring(hand_xml).find(".//joint[@name='R_jp']")
        if not np.allclose(model.jnt_range[model.joint("R_jp").id], vector(jp.get("range")), atol=1e-6):
            raise ValueError("Hand joint ranges changed when mounted. task_scene.xml must use "
                             "<compiler angle='radian'>.")
        return model

    def run_cells(self, template, design, r_mm, cells):
        """All (x, y) cells for one design and one radius. Returns result rows."""
        s, o = self.cfg["simulation"], self.cfg["objective"]
        hand_xml = self.hand_xml(template, design, r_mm)
        model = self.compile(hand_xml)
        data = mujoco.MjData(model)
        try:
            q_home = build_scene.home_arm_q(model)
        except RuntimeError as exc:  # Arm-only problem; the hand design cannot cause it.
            return [dict(empty_row(r_mm, x, y), reason="home_ik_failed", error=str(exc)[:200])
                    for x, y in cells]
        qadr = np.array([model.joint(n).qposadr[0] for n in ARM_JOINTS])
        armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
        site = model.site("grasp").id
        weight = float(model.body_mass[model.body("obj").id])*s["gravity_m_s2"]
        prefilter = (self.opts["prefilter_approach"] and self.opts["max_pre_grasp_shift_mm"] is not None)
        hand = CellEvaluator(hand_xml, self.hcfg) if prefilter or s.get("approach_mm", 0) > 0 else None
        rows = []
        for x_mm, y_mm in cells:
            start = time.perf_counter()
            row = empty_row(r_mm, x_mm, y_mm)
            rows.append(row)
            if y_mm <= 0:
                row.update(reason="initial_palm_clearance", prefiltered=True)
                continue
            if hand is not None:
                hand.reset(x_mm, y_mm)
                clearance = hand.approach_clearance(x_mm, y_mm)
                row["approach_clearance_mm"] = 1000*float(clearance)
                # The open hand sweeps through the cylinder: the arm run could only push it.
                if prefilter and 1000*clearance < -self.opts["max_pre_grasp_shift_mm"]:
                    row.update(reason="approach_overlap", prefiltered=True,
                               seconds=time.perf_counter()-start)
                    continue
            model.site_pos[site] = [x_mm/1000, (y_mm+r_mm)/1000, 0.0]
            mujoco.mj_resetData(model, data)
            data.qpos[qadr] = q_home
            data.ctrl[armact] = q_home
            mujoco.mj_forward(model, data)
            sim = PickLift(model, data)
            monitor = CellMonitor(model, self.cfg, self.opts)
            sim.callbacks.append(monitor)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = sim.run(self.tension, self.lift)
            except RuntimeError as exc:
                result = None
                row.update(reason=exception_reason(exc), error=str(exc)[:200])
            row["pre_grasp_shift_mm"] = monitor.max_shift_mm
            row["max_tracking_error_mm"] = 1000*sim.max_tracking_error
            row["max_unintended_penetration_mm"] = 1000*sim.max_unintended_penetration
            row["unintended_pair"] = "-".join(sim.worst_collision or [])
            rot = data.xmat[model.body("obj").id].reshape(3, 3)
            row["end_tilt_deg"] = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
            close_f = monitor.window_mean("grasp hold", s["force_average_window_s"])
            if close_f is not None:
                row["close_type"] = classify_contact(close_f, s["contact_threshold_N"], *contact_rules(self.cfg))
            if result is not None:
                row["rb5_pass"] = bool(result["success"])
                row["min_lift_mm"] = 1000*result["last_second_min_lift_m"]
                hold_f = monitor.window_mean("lift hold", s["force_average_window_s"])
                if hold_f is not None:
                    kind = classify_contact(hold_f, s["contact_threshold_N"], *contact_rules(self.cfg))
                    margin = o["friction_mu"]*float(hold_f.sum())/weight-o["safety_factor"]
                    row.update(contact_type=kind, m_F=float(margin), total_normal_N=float(hold_f.sum()),
                               hold_drift_mm=monitor.hold_drift_mm,
                               hold_rotation_deg=monitor.hold_rotation_deg,
                               **dict(zip(FORCE_KEYS, map(float, hold_f))))
                if not result["success"]:
                    row["reason"] = failed_run_reason(result, self.lift)
                else:
                    row["reason"] = judge(row, self.cfg, self.opts)
                    row["success"] = row["reason"] == "success"
            row["seconds"] = time.perf_counter()-start
        return rows

    def scene_file(self, path, template, design, r_mm, x_mm, y_mm):
        """Write a viewer scene (with 'home' keyframe) for run_pick_lift.py / rb5_cell.py."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        hand_xml = self.hand_xml(template, design, r_mm, x_mm, y_mm)
        try:
            meshdir = Path(os.path.relpath(self.meshdir, path.parent)).as_posix()
        except ValueError:  # Different drive on Windows.
            meshdir = str(self.meshdir)
        scene = self.compose(hand_xml, meshdir=meshdir)
        build_scene.add_home_keyframe(scene, self.compile(hand_xml))
        ET.indent(scene, space="  ")
        ET.ElementTree(scene).write(path, encoding="utf-8", xml_declaration=True)
        mujoco.MjModel.from_xml_path(str(path))
        return path


def summarize(rows, cfg):
    o = cfg["objective"]
    env = sum(r["success"] and r["contact_type"] == "env" for r in rows)
    tip = sum(r["success"] and r["contact_type"] == "tip" for r in rows)
    margin = sum(r["m_F"] for r in rows if r["success"])
    main = env+o["w_tip"]*tip
    bonus = o["lambda_force"]*margin
    return dict(J=float(main+bonus), J_per_cell=float((main+bonus)/len(rows)), cells=len(rows),
                success=env+tip, env=env, tip=tip, workspace_term=float(main),
                force_margin_sum=float(margin), force_term=float(bonus),
                rb5_pass=sum(bool(r["rb5_pass"]) for r in rows),
                prefiltered=sum(bool(r["prefiltered"]) for r in rows),
                cell_seconds=float(sum(r["seconds"] for r in rows)),
                failures=dict(Counter(r["reason"] for r in rows if not r["success"])))


def group_tasks(designs, grid):
    """One task per (design, radius): the model is compiled once per task."""
    xy = [(x, y) for x in axis_values(grid["x_mm"]) for y in axis_values(grid["y_mm"])]
    return [(i, d, r, xy) for i, d in enumerate(designs) for r in axis_values(grid["r_mm"])]


def evaluate_design(template, design, cfg, grid, arm=None, with_rows=False):
    """Serial evaluation of one design. The optimizer parallelizes over group_tasks."""
    arm = arm or Rb5Arm(cfg)
    rows = []
    for _, d, r, xy in group_tasks([design], grid):
        rows.extend(arm.run_cells(template, d, r, xy))
    summary = summarize(rows, cfg)
    return (summary, rows) if with_rows else summary


def rescore(path, cfg):
    """Re-judge a saved RB5 workspace CSV with this config's rules, without simulating.

    Cells whose RB5 run failed stay failed. For every cell whose run passed, the stored
    lift-hold link forces, hold drift/rotation and m_F are judged again by judge(), so the
    result equals a re-run under the new contact rules and rb5_arm options."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw or "rb5_pass" not in raw[0]:
        raise ValueError("Not an RB5 workspace CSV (no rb5_pass column). Hand-only CSVs need a re-run.")
    thr, opts = cfg["simulation"]["contact_threshold_N"], arm_options(cfg)

    def num(v):
        return None if v in ("", None) else float(v)

    old, new, changed = [], [], []
    for r in raw:
        row = dict(r, success=r["success"] == "True", rb5_pass=r["rb5_pass"] == "True",
                   prefiltered=r["prefiltered"] == "True", m_F=float(r["m_F"]),
                   seconds=float(r["seconds"] or 0), hold_drift_mm=num(r["hold_drift_mm"]),
                   hold_rotation_deg=num(r["hold_rotation_deg"]))
        old.append(row)
        row = dict(row)
        if row["rb5_pass"]:
            row["contact_type"] = classify_contact([float(r[k]) for k in FORCE_KEYS], thr,
                                                   *contact_rules(cfg))
            row["reason"] = judge(row, cfg, opts)
            row["success"] = row["reason"] == "success"
            before = old[-1]["contact_type"] if old[-1]["success"] else "fail:"+old[-1]["reason"]
            after = row["contact_type"] if row["success"] else "fail:"+row["reason"]
            if before != after:
                changed.append((row["r_mm"], row["x_mm"], row["y_mm"], before, after,
                                {k: round(float(r[k]), 3) for k in FORCE_KEYS}))
        new.append(row)
    out = Path(path).with_name(Path(path).stem+"_rescored.csv")
    save_rows(out, new)
    return summarize(old, cfg), summarize(new, cfg), changed, out


def best_cell(rows):
    ok = [r for r in rows if r["success"]]
    if not ok:
        return dict(r_mm=40.0, x_mm=0.0, y_mm=40.0)
    row = max(ok, key=lambda r: (r["contact_type"] == "env", float(r["m_F"])))
    return {k: float(row[k]) for k in ("r_mm", "x_mm", "y_mm")}


def load_design(path, template, cfg):
    """Read {"design": {...}} (best_design*.json) or a bare design dict, re-derive, check bounds."""
    raw = read_json(path)
    raw = raw.get("design", raw)
    space = DesignSpace(cfg, baseline_parameters(template))
    design = space.derived({v: float(raw[v]) for v in VARIABLES})
    space.encode(design)
    return design


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, default=ROOT/"design168.xml")
    p.add_argument("--config", type=Path, default=ROOT/"config_rb5_arm.json")
    p.add_argument("--design", type=Path, help="JSON with a 'design' entry; default = design168")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell", type=float, nargs=3, metavar=("R_MM", "X_MM", "Y_MM"))
    g.add_argument("--grid", help="Grid name in the config, e.g. rb5_smoke")
    g.add_argument("--rescore", type=Path, help="Re-judge a saved RB5 workspace CSV with this config")
    p.add_argument("--save-scene", type=Path, help="With --cell: write a run_pick_lift.py scene")
    args = p.parse_args()
    cfg = read_json(args.config)
    validate_rb5_config(cfg)
    if args.rescore:
        before, after, changed, out = rescore(args.rescore, cfg)
        print(f"rules={contact_rules(cfg)}  J {before['J']:.4f} -> {after['J']:.4f}, "
              f"env {before['env']} -> {after['env']}, tip {before['tip']} -> {after['tip']}")
        for r, x, y, a, b, forces in changed:
            print(f"  r={r} x={x} y={y}: {a} -> {b}  {forces}")
        print("Saved:", out)
        return
    template = args.xml.read_text(encoding="utf-8-sig")
    if args.design:
        design = load_design(args.design, template, cfg)
    else:
        base = baseline_parameters(template)
        design = DesignSpace(cfg, base).derived({v: float(base[v]) for v in VARIABLES})
    arm = Rb5Arm(cfg)
    if args.cell:
        r, x, y = args.cell
        row = arm.run_cells(template, design, r, [(x, y)])[0]
        print(json.dumps(row, indent=2, default=float))
        print("PASS" if row["success"] else f"FAIL ({row['reason']})")
        if args.save_scene:
            print("Scene:", arm.scene_file(args.save_scene, template, design, r, x, y))
    else:
        summary = evaluate_design(template, design, cfg, cfg["grids"][args.grid], arm)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
