"""Graspable workspace of one design XML mounted on the RB5, plus its plot.

The design is evaluated exactly like the final candidates of an optimize_rb5_cmaes.py
run: same config_used.json, same final grid (rb5_fine), same rb5_eval.Rb5Arm.run_cells
(mounting, pick-and-lift, env/tip judging, prefilter).

Two XML formats are accepted:
  * the RB5 hand format of design168.xml (R_gp, R_jp, ...): used directly as the template.
  * the stage0 format of design001.xml (upper_green_link, joint1, ...): the hand is built
    with gripper_core.build_xml from the run's own template, then the XML's distal open
    angle, stiffness and preload are put back (damping follows the run's damping ratio).
    Before anything runs on the RB5, the result is checked against the original model:
    every finger value, the open-pose site positions and the tendon moment arms over the
    finger range must match.
Tendon tension, object and protocol always come from the run's config.

  python eval_design_rb5.py --run runs/rb5_cmaes_final --xml design001.xml --workers 8

Running the same command again continues an interrupted evaluation, or only redraws the
plot when it is finished. --fresh discards earlier results of this label.

Writes to <run>/references/:
  <label>.json              {label, design, summary}, same format as final/candidate_XX.json;
                            design holds the values actually simulated (stage0: the XML's
                            own theta_d and phi_d)
  <label>_meta.json         grid, build_xml input, stage0 patch, mounted hand values
  <label>_workspace.csv     one row per cell, same columns as final/*_workspace.csv
  <label>_source.xml        copy of the design XML
  <label>_template.xml      hand template given to build_xml
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

import mujoco
from gripper_core import DesignSpace, VARIABLES, baseline_parameters, build_xml, read_json, save_rows
from rb5_eval import Rb5Arm, best_cell, group_tasks, hand_config, summarize, validate_rb5_config

WORKER = {}
STAGE0_SITES = {  # stage0 site -> RB5 hand site. Finger sites must coincide exactly.
    "upper_tip": "R_tip", "upper_prox_top": "R_prox_top", "upper_prox_bot": "R_prox_bot",
    "joint2_side": "R_jd_side", "lower_tip": "L_tip", "lower_prox_top": "L_prox_top",
    "lower_prox_bot": "L_prox_bot", "joint4_side": "L_jd_side"}
PALM_SITES = {"joint1_side": "R_jp_side", "joint3_side": "L_jp_side", "palm_up": "R_palm",
              "palm_lo": "L_palm", "pulley": "pulley"}


# ---------------------------------------------------------------- stage0 conversion
def vec(text):
    return np.array([float(t) for t in text.split()])


def num(v):
    """Full-precision plain number for XML (repr of a numpy float is 'np.float64(..)')."""
    return repr(float(v))


def stage0_values(text):
    """Finger values of a stage0-format XML (design001.xml), in the RB5 hand's joint
    convention: open stop = lower range limit, closing positive."""
    root = ET.fromstring(text)
    body = {b.get("name"): b for b in root.iter("body")}
    geom = {g.get("name"): g for g in root.iter("geom")}
    joint = {j.get("name"): j for j in root.iter("joint")}

    def seg(name):
        v = vec(geom[name].get("fromto"))
        return float(np.linalg.norm(v[3:]-v[:3]))

    j1, j2, j3, j4 = (joint[f"joint{i}"] for i in range(1, 5))
    lo1, lo2 = float(vec(j1.get("range"))[0]), float(vec(j2.get("range"))[0])
    if not (np.isclose(seg("upper_green_link"), seg("lower_green_link"))
            and np.isclose(seg("upper_red_link"), seg("lower_red_link"))
            and np.isclose(-vec(j3.get("range"))[1], lo1) and np.isclose(-vec(j4.get("range"))[1], lo2)
            and j1.get("stiffness") == j3.get("stiffness") and j2.get("stiffness") == j4.get("stiffness")):
        raise SystemExit("The stage0 XML is not left/right symmetric. The RB5 hand model is symmetric.")
    k_p, k_d = float(j1.get("stiffness", 0)), float(j2.get("stiffness", 0))
    return dict(Lp=seg("upper_green_link"), Ld=seg("upper_red_link"),
                G=float(vec(body["upper_link1"].get("pos"))[1]-vec(body["lower_link1"].get("pos"))[1]),
                theta_p=-lo1, q_open_d=lo2, k_p=k_p, k_d=k_d,
                off_p=lo1-float(j1.get("springref", 0)), off_d=lo2-float(j2.get("springref", 0)))


def joint_order(root):
    """qpos address of every named joint, from document order (= MuJoCo's tree order)."""
    adr, out = 0, {}
    for el in root.find("worldbody").iter():
        if el.tag == "freejoint" or (el.tag == "joint" and el.get("type") == "free"):
            out[el.get("name")] = adr
            adr += 7
        elif el.tag == "joint":
            if el.get("type", "hinge") not in ("hinge", "slide"):
                raise ValueError("Unexpected joint type in the hand XML")
            out[el.get("name")] = adr
            adr += 1
    return out


def apply_patch(hand_xml, patch):
    """Put the stage0 design's distal open angle, stiffness, preload and damping into a
    hand XML made by build_xml. Geometry and everything else stay as build_xml made them."""
    if not patch:
        return hand_xml
    root = ET.fromstring(hand_xml)
    joint = {j.get("name"): j for j in root.iter("joint")}
    body = {b.get("name"): b for b in root.iter("body")}
    qd = float(patch["q_open_d"])
    for side in ("R", "L"):
        jp, jd = joint[f"{side}_jp"], joint[f"{side}_jd"]
        q_open_p = float(vec(jp.get("range"))[0])
        jp.set("stiffness", num(patch["k_p"]))
        jp.set("springref", num(q_open_p-patch["off_p"]))
        axis = vec(body[f"{side}_dist"].get("axisangle"))[:3]
        body[f"{side}_dist"].set("axisangle", " ".join(f"{v:g}" for v in axis)+f" {num(qd)}")
        jd.set("ref", num(qd))
        jd.set("range", f"{num(qd)} {num(vec(jd.get('range'))[1])}")
        jd.set("stiffness", num(patch["k_d"]))
        jd.set("springref", num(qd-patch["off_d"]))
        for j, key in ((jp, "damping_p"), (jd, "damping_d")):
            if key in patch:
                j.set("damping", num(patch[key]))
    key = root.find(".//key[@name='open']")
    if key is not None:
        qpos = key.get("qpos").split()
        order = joint_order(root)
        for side in ("R", "L"):
            qpos[order[f"{side}_jd"]] = num(qd)
        key.set("qpos", " ".join(qpos))
    return ET.tostring(root, encoding="unicode")


def full_inertia(m, d):
    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)  # Newer MuJoCo.
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)  # Older MuJoCo.
    return M


def open_pose(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if m.nkey and "open" in [m.key(i).name for i in range(m.nkey)]:
        mujoco.mj_resetDataKeyframe(m, d, m.key("open").id)
    mujoco.mj_forward(m, d)
    return m, d


def joint_inertia(xml):
    m, d = open_pose(xml)
    M = full_inertia(m, d)
    return {n: float(M[m.joint(n).dofadr[0], m.joint(n).dofadr[0]]) for n in ("R_jp", "R_jd")}


def stage0_conversion(src_text, run_template, cfg):
    """(design, patch): build_xml(run_template, design) + apply_patch(.., patch) is the
    stage0 design as an RB5 hand."""
    s = stage0_values(src_text)
    base = baseline_parameters(run_template)
    space = DesignSpace(cfg, base)
    L = s["Lp"]+s["Ld"]
    design = space.derived(dict(length_mm=1000*L, lp_ratio=s["Lp"]/L, gap_mm=1000*s["G"],
                                theta_p_deg=math.degrees(s["theta_p"])))
    patch = {k: s[k] for k in ("q_open_d", "k_p", "k_d", "off_p", "off_d")}
    raw = build_xml(run_template, design, hand_config(cfg), radius_mm=40)
    # Keep the run's damping ratio: damping = 2*zeta*sqrt(k*I) at the open pose.
    m, _ = open_pose(raw)
    inertia_raw = joint_inertia(raw)
    inertia_new = joint_inertia(apply_patch(raw, patch))
    for tag, name in (("p", "R_jp"), ("d", "R_jd")):
        j = m.joint(name)
        k, c = float(m.jnt_stiffness[j.id]), float(m.dof_damping[j.dofadr[0]])
        if k > 0 and c > 0:
            zeta = c/(2*math.sqrt(k*inertia_raw[name]))
            patch[f"damping_{tag}"] = float(2*zeta*math.sqrt(patch[f"k_{tag}"]*inertia_new[name]))
            patch[f"zeta_{tag}"] = float(zeta)
    return design, patch


def hand_frame_sites(m, d, palm_body=None):
    """Site positions in the RB5 hand frame (x = jaw width, y = approach) [m]."""
    if palm_body is None:
        return {m.site(i).name: d.site_xpos[i][:2].copy() for i in range(m.nsite)}
    b = m.body(palm_body).id
    rot, origin = d.xmat[b].reshape(3, 3), d.xpos[b]
    out = {}
    for i in range(m.nsite):
        local = rot.T @ (d.site_xpos[i]-origin)
        out[m.site(i).name] = np.array([local[1], -local[0]])  # stage0: fingers point along -x.
    return out


def check_conversion(src_text, hand_xml):
    """Raise unless the converted hand reproduces the stage0 fingers."""
    s = stage0_values(src_text)
    h = hand_parameters(hand_xml)
    want = {"Lp [mm]": 1000*s["Lp"], "Ld [mm]": 1000*s["Ld"], "G [mm]": 1000*s["G"],
            "theta_p [deg]": math.degrees(s["theta_p"]), "theta_d [deg]": -math.degrees(s["q_open_d"]),
            "k_p [N m/rad]": s["k_p"], "k_d [N m/rad]": s["k_d"],
            "preload_p [N m]": s["k_p"]*s["off_p"], "preload_d [N m]": s["k_d"]*s["off_d"]}
    bad = [f"{k}: XML {v:.6g}, RB5 hand {h[k]:.6g}" for k, v in want.items()
           if not np.isclose(h[k], v, rtol=1e-4, atol=1e-6)]
    m0, d0 = open_pose(src_text)
    m1, d1 = open_pose(hand_xml)
    a, b = hand_frame_sites(m0, d0, "palm"), hand_frame_sites(m1, d1)
    worst = max(1000*np.linalg.norm(a[k]-b[v]) for k, v in STAGE0_SITES.items())
    if worst > 0.5:
        bad.append(f"open-pose finger sites differ by up to {worst:.2f} mm")
    palm = {k: 1000*np.linalg.norm(a[k]-b[v]) for k, v in PALM_SITES.items() if k in a and v in b}
    arm_mm = moment_arm_difference(src_text, hand_xml)
    if arm_mm > MAX_ARM_DIFF_MM:
        bad.append(f"tendon moment arms over the finger range differ by up to {arm_mm:.3f} mm")
    if bad:
        raise SystemExit("The converted hand does not match the stage0 XML:\n  "+"\n  ".join(bad))
    return worst, max(palm.values(), default=0.0), arm_mm


STAGE0_JOINTS = ("joint1", "joint2", "joint3", "joint4")
RB5_JOINTS = ("R_jp", "R_jd", "L_jp", "L_jd")
MAX_ARM_DIFF_MM = 0.05


def closing_coordinates(m, d, joints):
    """(qpos address, q_open, closing sign, closing span) per joint; d at the open pose.
    The open stop is the range limit nearer q_open, closing runs towards the other one."""
    out = []
    for n in joints:
        j = m.joint(n)
        lo, hi = (float(v) for v in m.jnt_range[j.id])
        q0 = float(d.qpos[j.qposadr[0]])
        sign = 1.0 if abs(q0-lo) <= abs(q0-hi) else -1.0
        out.append((j.qposadr[0], q0, sign, (hi-q0) if sign > 0 else (q0-lo)))
    return out


def tendon_moment_arms(xml, joints, poses, eps=1e-6):
    """Moment arm of the tendon at each finger joint [m] (= -dL/d(closing angle)) for each
    pose (proximal closing, distal closing) [rad], both fingers in the same pose."""
    m, d = open_pose(xml)
    coords = closing_coordinates(m, d, joints)
    t = m.tendon("tendon").id
    out = []
    for dp, dd in poses:
        closing = [dp, dd, dp, dd]

        def length(k=None):
            for i, (adr, q0, sign, _) in enumerate(coords):
                d.qpos[adr] = q0+sign*(closing[i]+(eps if i == k else 0.0))
            mujoco.mj_fwdPosition(m, d)
            return float(d.ten_length[t])
        base = length()
        out.append([-(length(k)-base)/eps for k in range(len(coords))])
    return np.array(out)


def moment_arm_difference(src_text, hand_xml, n=5):
    """Largest |moment arm difference| [mm] between the stage0 XML and the RB5 hand on an
    n x n grid of finger poses up to 90 % of the closing range both models share.

    This replaced a contact-free tendon pull compared in joint angles, which is ill-posed
    for these designs: a preloaded joint resting on its open stop sinks into the soft limit
    by an amount set by solreflimit (stage0 0.02, RB5 template 0.002), soft fingers run
    into closing limits that differ between the models (stage0 1.4/1.5 rad, RB5 1.92 rad),
    and a preloaded joint near its break-away tension is bistable, so the end pose depends
    on the inertial kick of the other joint (finger density 1100 vs 620). 168 of the 204
    designs in ../xml failed that test with matching geometry. With stiffness and preload
    already compared above, equal moment arms mean equal static finger behaviour."""
    m0, d0 = open_pose(src_text)
    m1, d1 = open_pose(hand_xml)
    c0 = closing_coordinates(m0, d0, STAGE0_JOINTS)
    c1 = closing_coordinates(m1, d1, RB5_JOINTS)
    span_p, span_d = 0.9*min(c0[0][3], c1[0][3]), 0.9*min(c0[1][3], c1[1][3])
    poses = [(a*span_p, b*span_d) for a in np.linspace(0, 1, n) for b in np.linspace(0, 1, n)]
    a0 = tendon_moment_arms(src_text, STAGE0_JOINTS, poses)
    a1 = tendon_moment_arms(hand_xml, RB5_JOINTS, poses)
    return 1000*float(np.abs(a0-a1).max())


# ---------------------------------------------------------------- evaluation
class PatchedArm(Rb5Arm):
    """Rb5Arm whose hands carry the stage0 design's own distal angle, stiffness, preload."""

    def __init__(self, cfg, patch):
        super().__init__(cfg)
        self.patch = patch
        self.calls = 0

    def hand_xml(self, *args, **kwargs):
        self.calls += 1
        return apply_patch(super().hand_xml(*args, **kwargs), self.patch)


def make_arm(cfg, patch):
    return PatchedArm(cfg, patch) if patch else Rb5Arm(cfg)


def init_worker(template, cfg, patch=None):
    WORKER.clear()
    WORKER.update(template=template, cfg=cfg, patch=patch, arm=None)


def run_task(task):
    """Same work as optimize_rb5_cmaes.worker_task: cells of one design at one radius."""
    _, design, r_mm, cells = task
    if WORKER["arm"] is None:
        WORKER["arm"] = make_arm(WORKER["cfg"], WORKER["patch"])
    arm = WORKER["arm"]
    calls = getattr(arm, "calls", 0)
    start = time.perf_counter()
    rows = arm.run_cells(WORKER["template"], design, r_mm, cells)
    if WORKER["patch"] and arm.calls == calls:
        raise RuntimeError("rb5_eval.Rb5Arm.run_cells no longer builds the hand through hand_xml(); "
                           "the stage0 values would be lost. Send rb5_eval.py.")
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


def signature(template, cfg, grid, patch):
    text = (template+json.dumps(cfg, sort_keys=True)+json.dumps(grid, sort_keys=True)
            + json.dumps(patch, sort_keys=True))
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


def evaluate(template, design, patch, cfg, grid, progress, workers):
    sig = signature(template, cfg, grid, patch)
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
                                           initializer=init_worker, initargs=(template, cfg, patch))
                results = (f.result() for f in as_completed([pool.submit(run_task, t) for t in tasks]))
            else:
                init_worker(template, cfg, patch)
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


DESIGN_KEYS = ("length_mm", "lp_ratio", "gap_mm", "theta_p_deg", "lp_mm", "ld_mm",
               "theta_d_deg", "phi_d_deg")


def design_record(build_design, mounted):
    """design block of final/candidate_XX.json, describing the hand actually simulated.
    Geometry is the build_xml input; theta_d/phi_d come from the mounted hand, because a
    stage0 patch replaces the run template's distal angle."""
    out = {k: float(build_design[k]) for k in DESIGN_KEYS[:6]}
    out["theta_d_deg"] = float(mounted["theta_d [deg]"])+0.0  # +0.0 turns -0.0 into 0.0.
    out["phi_d_deg"] = float(mounted["phi_d [deg]"])+0.0
    return out


def write_result(files, label, build_design, patch, summary, mounted, meta):
    dump(files["json"], dict(label=label, design=design_record(build_design, mounted), summary=summary))
    dump(files["meta"], dict(build_design=build_design, patch=patch, mounted_hand=mounted, **meta))


def load_reference(ref, label):
    """(design for build_xml, patch, template text, json) of a references/ result.
    Also reads the older single-file format of this script."""
    result = read_json(ref/f"{label}.json")
    meta_path = ref/f"{label}_meta.json"
    meta = read_json(meta_path) if meta_path.is_file() else result
    build_design = meta.get("build_design", result["design"])
    template = (ref/meta.get("template_file", f"{label}_source.xml")).read_text(encoding="utf-8-sig")
    return build_design, meta.get("patch"), template, result, meta


def patch_aware_outline(pw):
    """Make plot_workspace.hand_outline draw entries that carry a stage0 patch correctly."""
    original = pw.hand_outline
    if getattr(original, "patch_aware", False):
        return

    def hand_outline(entry, cfg):
        if not entry.get("patch"):
            return original(entry, cfg)
        build = pw.build_xml
        pw.build_xml = lambda *a, **k: apply_patch(build(*a, **k), entry["patch"])
        try:
            return original(entry, cfg)
        finally:
            pw.build_xml = build
    hand_outline.patch_aware = True
    pw.hand_outline = hand_outline


def plot(run, cfg, label, template, design, patch, summary, rows):
    import matplotlib
    matplotlib.use("Agg")
    import plot_workspace as pw
    patch_aware_outline(pw)
    out = run/"plots"
    out.mkdir(exist_ok=True)
    path = out/f"workspace_{label}.png"
    shapes = pw.hand_outline(dict(template=template, design=design, patch=patch), cfg)
    pw.plot_workspace(path, rows, label, design, summary, shapes)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Output folder of optimize_rb5_cmaes.py")
    p.add_argument("--xml", type=Path, default=Path("design001.xml"),
                   help="Design XML, stage0 or RB5 hand format (default design001.xml)")
    p.add_argument("--label", help="Name of the result (default: XML file name without .xml)")
    p.add_argument("--grid", help="Grid name in config_used.json (default: the run's final grid)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument("--fresh", action="store_true", help="Discard earlier results of this label")
    args = p.parse_args()
    run = args.run.resolve()
    label = args.label or args.xml.stem
    ref = run/"references"
    files = {k: ref/f"{label}{s}" for k, s in (("json", ".json"), ("csv", "_workspace.csv"),
             ("meta", "_meta.json"), ("src", "_source.xml"), ("template", "_template.xml"),
             ("hand", "_hand.xml"),
             ("scene", "_rb5_scene.xml"), ("progress", "_progress.jsonl"))}
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
        build_design, patch, template, result, meta = load_reference(ref, label)
        print(f"{label} was already evaluated (use --fresh to redo). Redrawing the plot.")
        if not files["meta"].is_file():  # Older single-file result: rewrite, no simulation.
            mounted = hand_parameters(apply_patch(build_xml(template, build_design, hand_config(cfg),
                                                            radius_mm=40), patch))
            keep = {k: v for k, v in meta.items()
                    if k not in ("label", "design", "summary", "patch", "mounted_hand", "note")}
            write_result(files, label, build_design, patch, result["summary"], mounted, keep)
            print(f"Rewrote {files['json'].name} in the final/candidate_XX.json format "
                  f"(details in {files['meta'].name}).")
        print("Saved:", plot(run, cfg, label, template, build_design, patch, result["summary"],
                             read_rows(files["csv"])))
        return

    if not args.xml.is_file():
        raise SystemExit(f"Design XML not found: {args.xml}")
    source = args.xml.read_text(encoding="utf-8-sig")
    hcfg = hand_config(cfg)
    best = read_json(run/"best_design_fine.json") if (run/"best_design_fine.json").is_file() else None
    run_template = (run/"source_design168.xml").read_text(encoding="utf-8-sig")
    tags = {e.get("name") for e in ET.fromstring(source).iter() if e.get("name")}
    if "R_gp" in tags:
        template, patch = source, None
        design, note = load_design(template, cfg)
    elif "upper_green_link" in tags:
        print(f"{args.xml.name} is in the stage0 format: building it on the run's hand template ...")
        template = run_template
        design, patch = stage0_conversion(source, template, cfg)
        note = ""
        sites, palm_mm, arm_mm = check_conversion(source, apply_patch(build_xml(template, design, hcfg,
                                                                                    radius_mm=40), patch))
        print(f"Check passed: finger values equal, open-pose finger sites within {sites:.3f} mm, "
              f"tendon moment arms within {arm_mm:.2g} mm"
              + (f" (palm tendon sites moved up to {palm_mm:.1f} mm)" if palm_mm > 0.5 else "") + ".")
    else:
        raise SystemExit(f"{args.xml} is neither in the RB5 hand format (R_gp, ...) nor the stage0 "
                         "format (upper_green_link, ...).")
    if args.grid:
        if args.grid not in cfg["grids"]:
            raise SystemExit(f"Grid '{args.grid}' not in config_used.json: {sorted(cfg['grids'])}")
        grid, grid_name = cfg["grids"][args.grid], args.grid
    elif best is not None:
        grid, grid_name = best["grid"], "final grid of the run"
    else:
        grid, grid_name = cfg["grids"]["rb5_fine"], "rb5_fine"

    # What the RB5 runs will simulate, next to the run's final best design.
    mounted = apply_patch(build_xml(template, design, hcfg, radius_mm=40), patch)
    columns = {label: hand_parameters(mounted)}
    if best is not None:
        columns[best["label"]] = hand_parameters(build_xml(run_template, best["design"], hcfg, radius_mm=40))
    print(f"Hand as mounted on the RB5 ({', '.join(columns)}):")
    print_parameters(columns)
    if patch and "zeta_p" in patch:
        print(f"Damping follows the run's damping ratio (zeta_p={patch['zeta_p']:.3g}, "
              f"zeta_d={patch['zeta_d']:.3g}); tendon tension comes from config_used.json.")
    if note:
        print(f"Note: {label} is {note}.")
    tension = float(cfg["simulation"]["tendon_tension_N"])
    if columns[label]["max tendon tension [N]"] < tension-1e-6:
        raise SystemExit(f"The generated hand's actuator ctrlrange allows only "
                         f"{columns[label]['max tendon tension [N]']:.1f} N, but the run uses {tension:.1f} N.")

    ref.mkdir(parents=True, exist_ok=True)
    files["src"].write_text(source, encoding="utf-8")
    files["template"].write_text(template, encoding="utf-8")
    print(f"Grid: {grid_name} = {grid}")
    wall = time.perf_counter()
    rows = evaluate(template, design, patch, cfg, grid, files["progress"], args.workers)
    summary = summarize(rows, cfg)
    save_rows(files["csv"], rows)
    cell = best_cell(rows)
    try:
        files["hand"].write_text(apply_patch(build_xml(template, design, cfg, radius_mm=cell["r_mm"],
                                 x_mm=cell["x_mm"], y_mm=cell["y_mm"]), patch), encoding="utf-8")
        make_arm(cfg, patch).scene_file(files["scene"], template, design, cell["r_mm"], cell["x_mm"],
                                        cell["y_mm"])
    except Exception as exc:  # The workspace result is already saved.
        print(f"Warning: could not export the viewer scene: {exc}")
    write_result(files, label, design, patch, summary, columns[label],
                 dict(grid=grid, template_file=files["template"].name, source_xml=str(args.xml.resolve()),
                      viewer_cell=cell))
    print(f"\n{label}: J={summary['J']:.4f}, success={summary['success']}/{summary['cells']} "
          f"(env={summary['env']}, tip={summary['tip']}), rb5_pass={summary['rb5_pass']}, "
          f"prefiltered={summary['prefiltered']}, wall={time.perf_counter()-wall:.0f}s")
    print(f"  failures: {summary['failures']}")
    print("Saved:", files["json"])
    print("Saved:", files["csv"])
    print("Saved:", plot(run, cfg, label, template, design, patch, summary, rows))
    print(f"View the strongest grasp: python run_pick_lift.py --xml {files['scene']}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
