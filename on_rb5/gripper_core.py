"""MJCF generation and deterministic workspace evaluation for design168.

Units in configuration/results: mm, degrees, N. MuJoCo uses SI units.
The object remains a free body. No position reset is used after closing starts.

simulation.protocol selects the cell procedure:
  "free" (default): original procedure. Object floats in zero gravity while closing,
          gravity is switched on only for the hold test.
  "rb5":  mirrors run_pick_lift.py in the hand frame. The object rests on a table
          under gravity, the open hand must reach it along +y without touching it,
          tension follows the same quintic ramp, and the table is lowered along the
          RB5 lift profile before the hold test. Arm and IK are not simulated.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

try:
    import numpy as np
    import mujoco
except ImportError as exc:
    raise SystemExit("Missing package. Run: python -m pip install -r requirements.txt") from exc

JOINTS = ("R_jp", "R_jd", "L_jp", "L_jd")
LINKS = ("R_gp", "R_gd", "L_gp", "L_gd", "palm")
VARIABLES = ("length_mm", "lp_ratio", "gap_mm", "theta_p_deg")


def smooth(u):
    """Quintic 0..1 profile used by run_pick_lift.py for both tension and motion."""
    u = np.clip(u, 0.0, 1.0)
    return u**3*(10.0-15.0*u+6.0*u*u)


def protocol(cfg):
    mode = cfg["simulation"].get("protocol", "free")
    if mode not in ("free", "rb5"):
        raise ValueError("simulation.protocol must be 'free' or 'rb5'")
    return mode


def object_center_z(cfg, halfheight_m):
    """Object center height in the hand frame (finger plane is z=0)."""
    if protocol(cfg) != "rb5":
        return 0.0
    return -cfg["simulation"]["table_height_mm"]/1000+halfheight_m


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def element(root, tag, name):
    result = root.find(f".//{tag}[@name='{name}']")
    if result is None:
        raise ValueError(f"Required {tag} is missing: {name}")
    return result


def vector(text):
    return np.fromstring(text, sep=" ")


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def baseline_parameters(xml_text):
    root = ET.fromstring(xml_text)
    if root.find("compiler").get("angle") != "radian":
        raise ValueError("This generator expects compiler angle='radian'.")
    lp = np.linalg.norm(vector(element(root, "geom", "R_gp").get("fromto"))[3:])
    ld = np.linalg.norm(vector(element(root, "geom", "R_gd").get("fromto"))[3:])
    gap = 2 * vector(element(root, "body", "R_prox").get("pos"))[0]
    jp, jd = [element(root, "joint", n) for n in JOINTS[:2]]
    theta_p = -math.degrees(float(jp.get("ref")))
    theta_d = -math.degrees(float(jd.get("ref")))
    kp, kd = float(jp.get("stiffness")), float(jd.get("stiffness"))
    taup = kp * (float(jp.get("ref")) - float(jp.get("springref")))
    taud = kd * (float(jd.get("ref")) - float(jd.get("springref")))
    # Suppress only XML decimal-rounding noise, not an actual preload.
    taup, taud = (0.0 if abs(x) < 1e-9 else x for x in (taup, taud))
    return dict(length_mm=1000*(lp+ld), lp_ratio=lp/(lp+ld), gap_mm=1000*gap,
                theta_p_deg=theta_p, theta_d_deg=theta_d, phi_d_deg=theta_p+theta_d,
                kp=kp, kd=kd, taup=taup, taud=taud,
                mass_kg=float(element(root, "geom", "obj_g").get("mass")),
                halfheight_m=vector(element(root, "geom", "obj_g").get("size"))[1])


class DesignSpace:
    """Four bounded [0,1] coordinates; conditional bounds enforce size constraints."""
    def __init__(self, cfg, base):
        self.cfg, self.base = cfg, base
        b = cfg["bounds"]
        self.length = b["length_mm"]
        self.ratio = b["lp_ratio"]
        self.gap = b["gap_mm"]
        self.theta = list(b["theta_p_deg"])
        self.link_min = b["min_link_mm"]
        if not 0 < self.ratio[0] < self.ratio[1] < 1 or self.link_min <= 0 or self.gap[0] <= 0:
            raise ValueError("Require 0<lp_ratio<1, positive link minimum and positive gap")
        self.mode = cfg["fixed"]["distal_angle_mode"]
        if self.mode not in ("absolute", "relative"):
            raise ValueError("distal_angle_mode must be 'absolute' or 'relative'.")
        # Both angle restrictions in the user table are enforced. The stricter
        # theta_d >= 10 - theta_p implies phi_d >= 10, rather than merely >= 0.
        if self.mode == "relative":
            td = base["theta_d_deg"]
            self.theta[0] = max(self.theta[0], 10-td)
            self.theta[1] = min(self.theta[1], 75-td)
        else:
            if not 10-1e-7 <= base["phi_d_deg"] <= 75+1e-7:
                raise ValueError("Fixed distal absolute angle must satisfy 10<=phi_d<=75 degrees")
            self.theta[0] = max(self.theta[0], base["phi_d_deg"]-30)
        if self.theta[0] >= self.theta[1]:
            raise ValueError("No nonzero proximal-angle interval satisfies the distal constraints.")
        for bounds in (self.length, self.ratio, self.gap, self.theta):
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"Invalid bounds: {bounds}")
        if self.length[0] <= 2*self.link_min or self.gap[0] >= self.length[0]:
            raise ValueError("Length bounds cannot satisfy the link/gap constraints.")

    def local_bounds(self, length):
        return (max(self.ratio[0], self.link_min/length),
                min(self.ratio[1], 1-self.link_min/length),
                min(self.gap[1], length))

    def decode(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape != (4,) or not np.isfinite(u).all() or np.any(u < -1e-10) or np.any(u > 1+1e-10):
            raise ValueError("Expected four normalized coordinates in [0,1].")
        u = np.clip(u, 0, 1)
        length = self.length[0]+u[0]*(self.length[1]-self.length[0])
        rlo, rhi, ghi = self.local_bounds(length)
        d = dict(length_mm=float(length), lp_ratio=float(rlo+u[1]*(rhi-rlo)),
                 gap_mm=float(self.gap[0]+u[2]*(ghi-self.gap[0])),
                 theta_p_deg=float(self.theta[0]+u[3]*(self.theta[1]-self.theta[0])))
        return self.derived(d)

    def derived(self, d):
        d = dict(d)
        d["lp_mm"] = d["length_mm"]*d["lp_ratio"]
        d["ld_mm"] = d["length_mm"]-d["lp_mm"]
        d["theta_d_deg"] = (self.base["phi_d_deg"]-d["theta_p_deg"]
                              if self.mode == "absolute" else self.base["theta_d_deg"])
        d["phi_d_deg"] = d["theta_p_deg"]+d["theta_d_deg"]
        return d

    def encode(self, d):
        length = d["length_mm"]
        rlo, rhi, ghi = self.local_bounds(length)
        values = [(length-self.length[0])/(self.length[1]-self.length[0]),
                  (d["lp_ratio"]-rlo)/(rhi-rlo),
                  (d["gap_mm"]-self.gap[0])/(ghi-self.gap[0]),
                  (d["theta_p_deg"]-self.theta[0])/(self.theta[1]-self.theta[0])]
        if np.any(np.asarray(values) < -1e-7) or np.any(np.asarray(values) > 1+1e-7):
            raise ValueError("Baseline/start design lies outside the configured design space.")
        return np.clip(values, 0, 1).tolist()


def joint_inertias(model, data, joint_ids):
    """Mass-matrix diagonal from body COM Jacobians; works across MuJoCo 3.x."""
    dofs = model.jnt_dofadr[joint_ids]
    inertia = np.zeros(len(joint_ids))
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for b in range(1, model.nbody):
        mujoco.mj_jacBodyCom(model, data, jp, jr, b)
        rot = data.ximat[b].reshape(3, 3)
        world_I = rot @ np.diag(model.body_inertia[b]) @ rot.T
        for i, dof in enumerate(dofs):
            inertia[i] += model.body_mass[b]*np.dot(jp[:, dof], jp[:, dof])
            inertia[i] += jr[:, dof] @ world_I @ jr[:, dof]
    return inertia


def build_xml(template, design, cfg, radius_mm=40, x_mm=0, y_mm=40):
    """Regenerate geometry, routing sites, reference angles, preload, and damping."""
    root = ET.fromstring(template)  # Removes stale template comments intentionally.
    base = baseline_parameters(template)
    lp, ld, gap = (design[k]/1000 for k in ("lp_mm", "ld_mm", "gap_mm"))
    qp, qd = -math.radians(design["theta_p_deg"]), -math.radians(design["theta_d_deg"])
    origin = vector(element(root, "body", "obj").get("pos"))
    origin[:] = [x_mm/1000, (y_mm+radius_mm)/1000, object_center_z(cfg, base["halfheight_m"])]
    obj = element(root, "geom", "obj_g")
    obj.set("size", numbers([radius_mm/1000, base["halfheight_m"]]))
    obj.set("mass", str(base["mass_kg"]))
    element(root, "body", "obj").set("pos", numbers(origin))
    root.find("option").set("timestep", str(cfg["simulation"]["timestep_s"]))
    root.find("option").set("gravity", "0 0 0")
    worldbody = root.find("worldbody")
    for old in worldbody.findall("body[@name='table']"):
        worldbody.remove(old)  # Never stack tables when a generated XML is reused.
    if protocol(cfg) == "rb5":
        s = cfg["simulation"]
        # Same contact parameters as RB5 bench_top, which has MuJoCo's built-in geom
        # defaults. Explicit attributes keep the template's finger defaults off it.
        table = ET.SubElement(worldbody, "body", {
            "name": "table", "mocap": "true",
            "pos": numbers([0, 0.1, -s["table_height_mm"]/1000])})
        ET.SubElement(table, "geom", {
            "name": "table_g", "type": "box", "size": "0.4 0.4 0.01", "pos": "0 0 -0.01",
            "condim": "3", "friction": numbers(s.get("table_friction", [1.0, 0.005, 0.0001])),
            "solref": numbers(s.get("table_solref", [0.02, 1.0])),
            "solimp": "0.9 0.95 0.001 0.5 2", "rgba": "0.36 0.39 0.42 0.5"})
    # A configurable mu must affect both simulated contacts and the force criterion.
    for g in [root.find("default/geom")] + [element(root, "geom", n) for n in (*LINKS, "obj_g")]:
        if g is not None and g.get("friction"):
            fr = vector(g.get("friction")); fr[0] = cfg["objective"]["friction_mu"]
            g.set("friction", numbers(fr))
    # G is the hinge-to-hinge spacing, not the gap between capsule surfaces.
    palm = element(root, "geom", "palm")
    size = vector(palm.get("size"))
    oldgap = base["gap_mm"]/1000
    size[0] += (gap-oldgap)/2
    palm.set("size", numbers(size))
    for side, sign in (("R", 1), ("L", -1)):
        for tag, suffix in (("geom", "jp_wrap"), ("site", "jp_side"), ("site", "palm")):
            e = element(root, tag, f"{side}_{suffix}")
            pos = vector(e.get("pos"))
            pos[0] += sign*(gap-oldgap)/2
            e.set("pos", numbers(pos))
        p, d = element(root, "body", f"{side}_prox"), element(root, "body", f"{side}_dist")
        p.set("pos", numbers([sign*gap/2, 0, 0]))
        p.set("axisangle", numbers([0, 0, sign, qp]))
        d.set("pos", numbers([0, lp, 0]))
        d.set("axisangle", numbers([0, 0, sign, qd]))
        for suffix, length in (("gp", lp), ("gd", ld)):
            element(root, "geom", f"{side}_{suffix}").set("fromto", numbers([0, 0, 0, 0, length, 0]))
        for suffix, delta in (("prox_top", lp-base["length_mm"]*base["lp_ratio"]/1000),
                              ("tip", ld-base["length_mm"]*(1-base["lp_ratio"])/1000)):
            e = element(root, "site", f"{side}_{suffix}")
            pos = vector(e.get("pos")); pos[1] += delta
            e.set("pos", numbers(pos))
        for suffix, q, k, tau in (("jp", qp, base["kp"], base["taup"]),
                                  ("jd", qd, base["kd"], base["taud"])):
            j = element(root, "joint", f"{side}_{suffix}")
            upper = vector(j.get("range"))[1]
            j.set("range", numbers([q, upper]))
            j.set("ref", f"{q:.12g}")
            j.set("springref", f"{q-tau/k:.12g}")
    key = element(root, "key", "open")
    key.set("qpos", numbers([qp, qd, qp, qd, *origin, 1, 0, 0, 0]))
    key.set("ctrl", "0")
    # Keep the model's gear=-0.5 convention: tendon tension T requires ctrl=2T.
    act = element(root, "motor", "pulley_pull")
    gear = vector(act.get("gear"))[0]
    if gear >= 0:
        raise ValueError("Expected negative tendon gear for a pulling motor.")
    control = cfg["simulation"]["tendon_tension_N"]/abs(gear)
    act.set("ctrlrange", numbers([0, max(control, vector(act.get("ctrlrange"))[1])]))
    stat = root.find("statistic")
    if stat is not None:
        stat.set("center", numbers([0, (lp+ld)/2, 0]))
        stat.set("extent", str(2*max(gap/2+lp+ld, .15)))
    if cfg["simulation"]["recompute_joint_damping"]:
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        ids = [model.joint(n).id for n in JOINTS]
        inertia = joint_inertias(model, data, ids)
        zeta = cfg["simulation"]["damping_ratio"]
        for n, j, I in zip(JOINTS, ids, inertia):
            element(root, "joint", n).set("damping", f"{2*zeta*math.sqrt(model.jnt_stiffness[j]*I):.12g}")
    comment = ("Generated from design168. Four-variable symmetric geometry. "
               f"Lp={lp*1000:.6f} mm; Ld={ld*1000:.6f} mm; G={gap*1000:.6f} mm; "
               f"theta_p={design['theta_p_deg']:.6f} deg; theta_d={design['theta_d_deg']:.6f} deg. "
               f"Object center=(x,y+r,{origin[2]:.6g}); mass={base['mass_kg']} kg. "
               f"Protocol={protocol(cfg)}. "
               f"Tendon tension={cfg['simulation']['tendon_tension_N']} N; motor ctrl={control:.6f} N.")
    root.insert(0, ET.Comment(comment))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def axis_values(spec):
    start, stop, step = map(float, spec)
    if step <= 0 or stop < start or not np.isfinite([start, stop, step]).all():
        raise ValueError(f"Invalid grid axis: {spec}")
    n = round((stop-start)/step)
    if not math.isclose(start+n*step, stop, abs_tol=1e-7):
        raise ValueError(f"Grid interval must be an integer multiple of step: {spec}")
    return np.linspace(start, stop, n+1).tolist()


def grid_cells(grid):
    return [(r, x, y) for r in axis_values(grid["r_mm"])
            for x in axis_values(grid["x_mm"]) for y in axis_values(grid["y_mm"])]


ENV_RULES = ("links_or_palm", "all_links")
PARTIAL_RULES = ("fail", "tip")


def contact_rules(cfg):
    """(env_rule, partial_contact) from "objective". Defaults reproduce the original rule.

    env_rule  'links_or_palm': palm contact upgrades any bilateral grasp to env.
              'all_links': env needs all four finger links; palm never changes the type.
    partial_contact  what a bilateral grasp with one link per side is, other than
              distal-distal (e.g. proximal-proximal, proximal-distal):
              'fail' -> "unsupported"; 'tip' -> "tip" (weight w_tip; the later lift and
              hold checks still decide whether the object stays in the hand).
    One side with both links and the other with one is always "mixed" (failure), and
    contact on one side only is always "none" (failure)."""
    o = cfg["objective"]
    rules = (o.get("env_rule", "links_or_palm"), o.get("partial_contact", "fail"))
    if rules[0] not in ENV_RULES:
        raise ValueError(f"objective.env_rule must be one of {ENV_RULES}")
    if rules[1] not in PARTIAL_RULES:
        raise ValueError(f"objective.partial_contact must be one of {PARTIAL_RULES}")
    return rules


def classify_contact(forces, threshold, rule="links_or_palm", partial="fail"):
    """Order: right proximal/distal, left proximal/distal, palm."""
    rp, rd, lp, ld, palm = np.asarray(forces) >= threshold
    if not ((rp or rd) and (lp or ld)):
        return "none"
    if bool(rp and rd) != bool(lp and ld):
        return "mixed"
    if rp and rd and lp and ld:
        return "env"
    if palm and rule == "links_or_palm":
        return "env"
    if rd and ld and not rp and not lp:
        return "tip"
    return "tip" if partial == "tip" else "unsupported"


class CellEvaluator:
    def __init__(self, xml, cfg):
        self.cfg, self.sim = cfg, cfg["simulation"]
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        self.data = d = mujoco.MjData(m)
        self.joints = np.array([m.joint(n).id for n in JOINTS])
        self.qadr, self.dofs = m.jnt_qposadr[self.joints], m.jnt_dofadr[self.joints]
        self.og = m.geom("obj_g").id
        self.ob = m.body("obj").id
        self.oj = m.joint("obj_free").id
        self.oq, self.od = int(m.jnt_qposadr[self.oj]), int(m.jnt_dofadr[self.oj])
        self.aid = m.actuator("pulley_pull").id
        self.gids = [m.geom(n).id for n in LINKS]
        self.gmap = {g:i for i,g in enumerate(self.gids)}
        self.default_damping = m.dof_damping.copy()
        self.collision = int(m.geom_contype[self.og]), int(m.geom_conaffinity[self.og])
        self.open_q = m.qpos0[self.qadr].copy()
        self.control = self.sim["tendon_tension_N"]/abs(m.actuator_gear[self.aid, 0])
        self.radius_m = m.geom_size[self.og, 0]
        self.weight_N = m.body_mass[self.ob]*self.sim["gravity_m_s2"]
        self.mu = float(cfg["objective"]["friction_mu"])
        self.sf = float(cfg["objective"]["safety_factor"])
        self.force_buffer = np.zeros(6)
        self.dt = m.opt.timestep
        self.check_every = max(1, round(self.sim["sample_interval_s"]/self.dt))
        self.tick = 0
        self.hook = None
        self.rb5 = protocol(cfg) == "rb5"
        self.contact_rules = contact_rules(cfg)
        self.obj_z0 = float(m.body_pos[self.ob][2])
        self.ramp = smooth if self.sim.get("ramp_shape", "linear") == "quintic" else (lambda u: min(1.0, u))
        self.min_side_N = float(self.sim.get("min_side_force_N", 0.0))
        limit = self.sim.get("max_unintended_penetration_mm")
        self.unintended_limit_m = None if limit is None else limit/1000
        self.last_unintended_m = 0.0
        # Both fingers move in one plane and can meet tip to tip. When allowed, only
        # finger-finger pairs are exempt; finger-palm contacts are still checked.
        self.exempt_gids = set(self.gids[:4]) if self.sim.get("allow_finger_finger_contact", False) else set()
        if self.rb5:
            self.table_mocap = int(m.body_mocapid[m.body("table").id])
            self.table_z0 = float(m.body_pos[m.body("table").id][2])

    def reset(self, x_mm, y_mm):
        m, d = self.model, self.data
        # RB5 protocol: the object sits on the table under gravity from the start.
        m.opt.gravity[:] = [0, 0, -self.sim["gravity_m_s2"]] if self.rb5 else 0
        m.dof_damping[:] = self.default_damping
        # Translational damping only, in N s/m; rotation is not artificially braked.
        m.dof_damping[self.od:self.od+3] = self.sim["closing_object_damping_Ns_m"]
        m.geom_contype[self.og], m.geom_conaffinity[self.og] = self.collision
        mujoco.mj_resetData(m, d)
        d.qpos[self.oq:self.oq+3] = [x_mm/1000, y_mm/1000+self.radius_m, self.obj_z0]
        d.ctrl[:] = 0
        mujoco.mj_forward(m, d)
        self.tick = 0

    def sample(self):
        m, d = self.model, self.data
        # mj_step leaves some derived arrays at the previous integration state.
        mujoco.mj_forward(m, d)
        forces = np.zeros(5)
        penetration = 0.0
        self.last_unintended_m = 0.0
        self.last_unintended_pair = None
        for i in range(d.ncon):
            c = d.contact[i]
            if self.og not in (c.geom1, c.geom2):
                # Finger-finger or finger-palm. run_pick_lift.py fails on these.
                if not (c.geom1 in self.exempt_gids and c.geom2 in self.exempt_gids):
                    if -float(c.dist) > self.last_unintended_m:
                        self.last_unintended_m = -float(c.dist)
                        self.last_unintended_pair = (int(c.geom1), int(c.geom2))
                continue
            other = c.geom2 if c.geom1 == self.og else c.geom1
            if other not in self.gmap:
                continue
            penetration = max(penetration, -float(c.dist))
            mujoco.mj_contactForce(m, d, i, self.force_buffer)
            # First component is normal force. Do NOT add friction or joint contacts.
            forces[self.gmap[other]] += max(0.0, float(self.force_buffer[0]))
        return forces, penetration

    def step(self, ctrl, stage):
        self.data.ctrl[self.aid] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.tick += 1
        if self.hook and self.tick % self.check_every == 0:
            self.hook(self.model, self.data, stage)

    def approach_clearance(self, x_mm, y_mm):
        """Smallest object-to-hand distance while the open hand advances along +y.

        Kinematic sweep of the RB5 approach (object starts approach_mm ahead).
        The fingers sit on their opening limits, so the open pose is used as is.
        """
        m, d, s = self.model, self.data, self.sim
        q = d.qpos.copy()
        y0 = y_mm/1000+self.radius_m
        cutoff = s["approach_clearance_mm"]/1000+0.05
        best = cutoff
        offsets = np.arange(s["approach_mm"], -1e-9, -s.get("approach_step_mm", 1.0))/1000
        for off in np.append(offsets, 0.0):
            d.qpos[self.oq+1] = y0+off
            mujoco.mj_kinematics(m, d)
            for gid in self.gids:
                best = min(best, mujoco.mj_geomDistance(m, d, self.og, gid, cutoff, None))
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return best

    def numerical_failure(self):
        d = self.data
        return (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                or any(w.number for w in d.warning))

    def run(self, r_mm, x_mm, y_mm, hook: Callable | None = None):
        self.hook = hook
        self.reset(x_mm, y_mm)
        m, d, s = self.model, self.data, self.sim
        result = dict(r_mm=float(r_mm), x_mm=float(x_mm), y_mm=float(y_mm),
                      success=False, contact_type="none", reason="unclassified", m_F=0.0,
                      total_normal_N=0.0, min_margin_hold=None, max_displacement_mm=0.0,
                      max_rotation_deg=0.0, max_penetration_mm=0.0, final_speed_mm_s=0.0,
                      return_error_deg=None, return_speed_rad_s=None, close_time_s=0.0,
                      R_prox_N=0.0, R_dist_N=0.0, L_prox_N=0.0, L_dist_N=0.0, palm_N=0.0,
                      approach_clearance_mm=None, max_unintended_penetration_mm=0.0,
                      lift_slip_mm=None, lift_rotation_deg=None, unintended_pair="")

        def fail(reason):
            result["reason"] = reason
            return result

        if y_mm <= 0:
            return fail("initial_palm_clearance")
        # Negative signed geometric distances are overlaps, independent of force.
        for gid in self.gids:
            distance = mujoco.mj_geomDistance(m, d, self.og, gid, 1.0, None)
            if distance < -s["initial_overlap_tolerance_mm"]/1000:
                return fail("initial_overlap")
        if s.get("approach_mm", 0) > 0:
            clearance = self.approach_clearance(x_mm, y_mm)
            result["approach_clearance_mm"] = float(1000*clearance)
            if clearance < s["approach_clearance_mm"]/1000:
                return fail("approach_collision")

        def unintended():
            depth = self.last_unintended_m
            if 1000*depth > result["max_unintended_penetration_mm"] and self.last_unintended_pair:
                result["unintended_pair"] = "-".join(sorted(m.geom(g).name or f"geom{g}" for g in self.last_unintended_pair))
            result["max_unintended_penetration_mm"] = max(result["max_unintended_penetration_mm"], 1000*depth)
            return self.unintended_limit_m is not None and depth > self.unintended_limit_m
        window = deque()
        settled = False
        # "fixed" mirrors run_pick_lift.py: ramp, hold grasp_hold_s, then lift, whether or
        # not the object still creeps. A table-supported object that tips against the
        # palm keeps creeping slowly, so the zero-gravity settle test would never pass.
        fixed = s.get("close_end", "settle") == "fixed"
        close_steps = math.ceil((s["ramp_up_s"]+s["grasp_hold_s"] if fixed else s["close_timeout_s"])/self.dt)
        for step in range(close_steps):
            t = (step+1)*self.dt
            self.step(self.control*self.ramp(t/s["ramp_up_s"]), "close")
            if (step+1) % self.check_every and not (fixed and step == close_steps-1):
                continue
            if self.numerical_failure():
                return fail("numerical_warning")
            forces, depth = self.sample()
            result["max_penetration_mm"] = max(result["max_penetration_mm"], depth*1000)
            if depth*1000 > s["max_penetration_mm"]:
                return fail("excess_penetration")
            if unintended():
                return fail("unintended_penetration")
            slow = (np.max(np.abs(d.qvel[self.dofs])) < s["joint_speed_rad_s"]
                    and np.linalg.norm(d.qvel[self.od:self.od+3]) < s["object_speed_mm_s"]/1000)
            if fixed:
                if t >= s["ramp_up_s"]+s["grasp_hold_s"]-s["settle_window_s"]-1e-9:
                    window.append((t, forces))
                if step == close_steps-1:
                    settled = True
                continue
            if t >= s["ramp_up_s"] and slow:
                window.append((t, forces))
                if window[-1][0]-window[0][0] >= s["settle_window_s"]-1e-9:
                    settled = True
                    break
            else:
                window.clear()
        result["close_time_s"] = float(d.time)
        if not settled:
            return fail("close_timeout")
        close_forces = np.mean([v for _,v in window], axis=0)
        kind = classify_contact(close_forces, s["contact_threshold_N"], *self.contact_rules)
        result["contact_type"] = kind
        if kind not in ("env", "tip"):
            return fail("close_"+kind)
        if min(close_forces[:2].sum(), close_forces[2:4].sum()) < self.min_side_N:
            return fail("close_side_force")
        if self.mu*sum(close_forces) < self.sf*self.weight_N:
            return fail("close_force_margin")
        hold_pos = d.qpos[self.oq:self.oq+3].copy()
        hold_quat = d.qpos[self.oq+3:self.oq+7].copy()
        m.opt.gravity[:] = [0, 0, -s["gravity_m_s2"]]
        m.dof_damping[self.od:self.od+6] = 0
        hold_forces = []
        min_margin = float("inf")
        n_hold = math.ceil(s["hold_s"]/self.dt)
        n_lift = 0
        if self.rb5:
            # Lift = lower the table along the RB5 lift profile (lift_mm over lift_s),
            # stopping table_drop_mm below its start. The arm's lift acceleration is ~1%
            # of g, so moving the table instead of the hand is equivalent. During this
            # load transfer a tipped object may re-seat, so only slip (as in RB5's
            # 0.8*lift rule) is limited. The strict hold test starts from the new pose.
            n_lift = math.ceil(s["lift_s"]/self.dt)
            t_lift = np.arange(1, n_lift+1)*self.dt
            drop = np.minimum(s["lift_mm"]/1000*smooth(t_lift/s["lift_s"]), s["table_drop_mm"]/1000)
            lift_forces = []
        n_total = n_lift+n_hold
        # Contact chatter: when the table drops away, single steps can report zero finger
        # force while the grasp holds. A lost-contact or negative-margin state must last
        # lost_contact_grace_s to fail (0 keeps the original fail-on-first-sample rule).
        grace = s.get("lost_contact_grace_s", 0.0)
        bad_since = None
        for step in range(n_total):
            lifting = step < n_lift
            if lifting:
                d.mocap_pos[self.table_mocap, 2] = self.table_z0-drop[step]
            elif n_lift and step == n_lift:
                if not lift_forces:
                    return fail("lift_lost_contact")
                kind = classify_contact(np.mean(lift_forces, axis=0), s["contact_threshold_N"], *self.contact_rules)
                result["contact_type"] = kind
                if kind not in ("env", "tip"):
                    return fail("lift_"+kind)
                hold_pos = d.qpos[self.oq:self.oq+3].copy()
                hold_quat = d.qpos[self.oq+3:self.oq+7].copy()
            self.step(self.control, "lift" if lifting else "hold")
            if (step+1) % self.check_every and step != n_total-1 and step != n_lift-1:
                continue
            if self.numerical_failure():
                return fail("numerical_warning")
            forces, depth = self.sample()
            drift = 1000*np.linalg.norm(d.qpos[self.oq:self.oq+3]-hold_pos)
            dot = np.clip(abs(np.dot(d.qpos[self.oq+3:self.oq+7], hold_quat)), 0, 1)
            angle = math.degrees(2*math.acos(dot))
            if lifting:
                result["lift_slip_mm"] = max(result["lift_slip_mm"] or 0.0, float(drift))
                result["lift_rotation_deg"] = max(result["lift_rotation_deg"] or 0.0, angle)
                if drift >= s["lift_slip_mm"]:
                    return fail("lift_slip")
            else:
                result["max_displacement_mm"] = max(result["max_displacement_mm"], float(drift))
                result["max_rotation_deg"] = max(result["max_rotation_deg"], angle)
                if drift >= s["hold_displacement_mm"]:
                    return fail("hold_displacement")
                if angle >= s["hold_rotation_deg"]:
                    return fail("hold_rotation")
            result["max_penetration_mm"] = max(result["max_penetration_mm"], depth*1000)
            if depth*1000 > s["max_penetration_mm"]:
                return fail("excess_penetration")
            if unintended():
                return fail("unintended_penetration")
            threshold = s["contact_threshold_N"]
            lost = (max(forces[:2]) < threshold or max(forces[2:4]) < threshold
                    or min(forces[:2].sum(), forces[2:4].sum()) < self.min_side_N)
            margin = self.mu*float(sum(forces))/self.weight_N-self.sf
            if not lost:
                min_margin = min(min_margin, margin)
                result["min_margin_hold"] = min_margin
            if lost or margin < 0:
                now = (step+1)*self.dt
                bad_since = now if bad_since is None else bad_since
                if now-bad_since >= grace-1e-12:
                    phase = "lift" if lifting else "hold"
                    return fail(f"{phase}_lost_contact" if lost else f"{phase}_force_margin")
                continue
            bad_since = None
            if lifting:
                if (step+1)*self.dt >= s["lift_s"]-s["settle_window_s"]-1e-9:
                    lift_forces.append(forces)
            elif (step+1-n_lift)*self.dt >= s["hold_s"]-s["force_average_window_s"]-1e-9:
                hold_forces.append(forces)
        if bad_since is not None:
            return fail("hold_lost_contact")
        speed = float(1000*np.linalg.norm(d.qvel[self.od:self.od+3]))
        result["final_speed_mm_s"] = speed
        if speed >= s["object_speed_mm_s"]:
            return fail("hold_speed")
        averaged = np.mean(hold_forces, axis=0)
        final_kind = classify_contact(averaged, s["contact_threshold_N"], *self.contact_rules)
        if final_kind != kind:
            return fail("hold_changed_type")
        result["total_normal_N"] = float(sum(averaged))
        result.update(dict(zip(("R_prox_N", "R_dist_N", "L_prox_N", "L_dist_N", "palm_N"), map(float, averaged))))
        margin = self.mu*result["total_normal_N"]/self.weight_N-self.sf
        # Remove object interaction, retaining the terminal finger state and velocity.
        m.geom_contype[self.og] = m.geom_conaffinity[self.og] = 0
        m.opt.gravity[:] = 0
        n_down = math.ceil(s["ramp_down_s"]/self.dt)
        for step in range(n_down):
            self.step(self.control*max(0, 1-(step+1)*self.dt/s["ramp_down_s"]), "release")
        returned = False
        for step in range(math.ceil(s["return_timeout_s"]/self.dt)+1):
            if step:
                self.step(0, "return")
            if step % self.check_every:
                continue
            if self.numerical_failure():
                return fail("numerical_warning")
            err = float(np.degrees(np.max(np.abs(d.qpos[self.qadr]-self.open_q))))
            vel = float(np.max(np.abs(d.qvel[self.dofs])))
            result["return_error_deg"], result["return_speed_rad_s"] = err, vel
            if err <= s["return_angle_deg"] and vel < s["joint_speed_rad_s"]:
                returned = True
                break
        if not returned:
            return fail("return_timeout")
        result.update(success=True, reason="success", contact_type=kind, m_F=float(margin))
        return result


def summarize(rows, cfg):
    n = len(rows)
    env = sum(r["success"] and r["contact_type"] == "env" for r in rows)
    tip = sum(r["success"] and r["contact_type"] == "tip" for r in rows)
    margin = sum(r["m_F"] for r in rows if r["success"])
    main = env+cfg["objective"]["w_tip"]*tip
    bonus = cfg["objective"]["lambda_force"]*margin
    return dict(J=float(main+bonus), J_per_cell=float((main+bonus)/n), cells=n,
                success=env+tip, env=env, tip=tip, workspace_term=float(main),
                force_margin_sum=float(margin), force_term=float(bonus),
                failures=dict(Counter(r["reason"] for r in rows if not r["success"])))


def evaluate_design(template, design, cfg, grid, with_rows=False):
    rows = []
    for r in axis_values(grid["r_mm"]):
        evaluator = CellEvaluator(build_xml(template, design, cfg, radius_mm=r), cfg)
        for x in axis_values(grid["x_mm"]):
            for y in axis_values(grid["y_mm"]):
                rows.append(evaluator.run(r, x, y))
    summary = summarize(rows, cfg)
    return (summary, rows) if with_rows else summary


def save_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_config(cfg):
    s, o = cfg["simulation"], cfg["objective"]
    for name in ("timestep_s", "ramp_up_s", "ramp_down_s", "hold_s", "close_timeout_s",
                 "return_timeout_s", "settle_window_s", "force_average_window_s", "sample_interval_s",
                 "contact_threshold_N", "gravity_m_s2", "joint_speed_rad_s", "object_speed_mm_s",
                 "hold_displacement_mm", "hold_rotation_deg", "return_angle_deg", "max_penetration_mm"):
        if s[name] <= 0:
            raise ValueError(f"simulation.{name} must be positive")
    if s["close_timeout_s"] <= s["ramp_up_s"]+s["settle_window_s"]:
        raise ValueError("close_timeout_s must include ramp and a settling window")
    if s["force_average_window_s"] > s["hold_s"]:
        raise ValueError("force_average_window_s cannot exceed hold_s")
    if o["lambda_force"] < 0 or not 0 <= o["w_tip"] <= 1 or o["safety_factor"] < 1 or o["friction_mu"] <= 0:
        raise ValueError("Invalid objective parameters")
    if s["tendon_tension_N"] <= 0 or s["initial_overlap_tolerance_mm"] < 0 or s["closing_object_damping_Ns_m"] < 0:
        raise ValueError("Invalid tension, overlap tolerance or closing damping")
    mode = protocol(cfg)
    contact_rules(cfg)
    if s.get("close_end", "settle") not in ("settle", "fixed"):
        raise ValueError("simulation.close_end must be 'settle' or 'fixed'")
    if s.get("close_end") == "fixed" and s.get("grasp_hold_s", 0) < s["settle_window_s"]:
        raise ValueError("close_end 'fixed' needs grasp_hold_s >= settle_window_s")
    if s.get("ramp_shape", "linear") not in ("linear", "quintic"):
        raise ValueError("simulation.ramp_shape must be 'linear' or 'quintic'")
    if s.get("lost_contact_grace_s", 0) < 0:
        raise ValueError("simulation.lost_contact_grace_s cannot be negative")
    if s.get("min_side_force_N", 0) < 0 or s.get("max_unintended_penetration_mm", 0) < 0:
        raise ValueError("Side-force and unintended-penetration limits cannot be negative")
    if s.get("approach_mm", 0) < 0 or (s.get("approach_mm", 0) > 0 and (
            s.get("approach_clearance_mm", -1) < 0 or s.get("approach_step_mm", 1.0) <= 0)):
        raise ValueError("Approach sweep needs approach_clearance_mm>=0 and approach_step_mm>0")
    if mode == "rb5":
        for name in ("table_height_mm", "lift_mm", "lift_s", "table_drop_mm"):
            if s.get(name, 0) <= 0:
                raise ValueError(f"protocol 'rb5' requires positive simulation.{name}")
        if s.get("lift_slip_mm", 0) <= 0:
            raise ValueError("protocol 'rb5' requires positive simulation.lift_slip_mm")
        if not max(s["hold_displacement_mm"], s["lift_slip_mm"]) < s["table_drop_mm"] <= s["lift_mm"]:
            raise ValueError("Require hold_displacement_mm, lift_slip_mm < table_drop_mm <= lift_mm, "
                             "so a slipping object cannot land back on the table unnoticed")
        if s["settle_window_s"] > s["lift_s"]:
            raise ValueError("settle_window_s cannot exceed lift_s")
    for grid in cfg["grids"].values():
        for axis in grid.values():
            axis_values(axis)
        if grid["r_mm"][0] <= 0 or grid["x_mm"][0] < 0 or grid["y_mm"][0] <= 0:
            raise ValueError("Require r>0, x>=0 and y>0")
