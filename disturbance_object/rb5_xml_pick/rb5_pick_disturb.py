#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RB5 pick -> 5 s hold -> A (ramp), B (pulse), or C (arm shake).

Standalone runner for exported RB5 cell XMLs containing task_home/lifted keys.
Only numpy and mujoco are required. See --help and README_KO.md.
Motion and contact helpers are adapted from the user's gripper_optimization
repository (disturbance_object/rb5_xml_pick, retrieved 2026-11-17).

Usage:
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0009_r30y15.xml" --input A --dir px --mag 1
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0009_r30y15.xml" --input B --dir px --mag 1
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0009_r30y15.xml" --input C --dir px --mag 0.5 --cycles 5

python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0111_r40y20.xml" --input A --dir px --mag 1
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0111_r40y20.xml" --input B --dir px --mag 1
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0111_r40y20.xml" --input C --dir px --mag 0.5 --cycles 5

python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0001_r50y20.xml" --input A --dir px --mag 1
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0001_r50y20.xml" --input B --dir px --mag 1
python rb5_pick_disturb.py --xml ".\grasp_cells_xml\design0001_r50y20.xml" --input C --dir px --mag 0.5 --cycles 5

"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
    import mujoco
except ImportError:
    raise SystemExit('Install dependencies first: python -m pip install mujoco==3.3.7 numpy')

ARM = ARM_JOINTS = ('base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3')
ALL_DIRS = {
    'px': (1., 0., 0.), 'nx': (-1., 0., 0.),
    'py': (0., 1., 0.), 'ny': (0., -1., 0.),
    'pz': (0., 0., 1.), 'nz': (0., 0., -1.),
}
PICK_DEFAULTS = {
    'approach_distance': .12, 'settle_time': .6, 'move_time': 3.,
    'approach_time': 2.5, 'preclose_time': .5, 'ramp_time': 1.5,
    'grasp_hold': 1., 'lift_time': 3., 'lift': .15, 'ik_interval': .01,
    'open_kp': 1., 'open_kd': .03, 'open_torque_limit': .3,
    'open_release_time': .3, 'contact_threshold': .1, 'success_ratio': .8,
}


def numeric_steps(seconds, dt):
    return max(0, int(math.ceil(seconds / dt - 1e-10)))


def resolve_asset(file, compiler_dir, xml_dir, mesh_roots):
    """Resolve Windows/relative asset paths without editing the original XML."""
    parts = PureWindowsPath(file).parts if '\\' in file or ':' in file else Path(file).parts
    candidates = [Path(file), xml_dir / file]
    if compiler_dir:
        candidates += [Path(compiler_dir) / file, xml_dir / compiler_dir / file]
    lows = [p.lower() for p in parts]
    suffix = parts[lows.index('meshes') + 1:] if 'meshes' in lows else parts[-1:]
    for root in mesh_roots:
        candidates += [root.joinpath(*suffix), root / parts[-1]]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def load_model(path, explicit_mesh_dir):
    path = path.expanduser().resolve()
    root = ET.parse(path).getroot()
    if root.tag != 'mujoco':
        raise ValueError('The input must be a MuJoCo MJCF XML.')
    if any(root.iter('include')):
        raise ValueError('Use the fully exported scene XML, without external <include> files.')
    roots = []
    if explicit_mesh_dir:
        roots.append(explicit_mesh_dir.expanduser().resolve())
    for parent in dict.fromkeys([path.parent, *path.parents, Path.cwd(), Path(__file__).resolve().parent]):
        roots.extend([parent / 'assets/rb5_850e/meshes', parent / 'meshes'])
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    dropped, missing_collision = [], []
    asset = root.find('asset')
    if asset is not None:
        for node in list(asset.findall('mesh')):
            file = node.get('file')
            if not file:
                continue
            resolved = resolve_asset(file, meshdir, path.parent, roots)
            if resolved:
                node.set('file', resolved)
                continue
            name = node.get('name')
            uses = [(p, g) for p in root.iter() for g in p
                    if g.tag == 'geom' and g.get('mesh') == name]
            is_visual = lambda g: ('visual' in g.get('class', '').lower()
                                   or (g.get('contype') == '0' and g.get('conaffinity') == '0'))
            if any(not is_visual(g) for _, g in uses):
                missing_collision.append(file)
            else:
                for parent, geom in uses:
                    parent.remove(geom)
                asset.remove(node)
                dropped.append(file)
        # Built-in textures need no path; resolve any file-backed texture/hfield.
        for tag, directory in [('texture', 'texturedir'), ('hfield', 'assetdir')]:
            compdir = compiler.get(directory, '') if compiler is not None else ''
            for node in asset.findall(tag):
                if node.get('file'):
                    resolved = resolve_asset(node.get('file'), compdir, path.parent, roots)
                    if resolved is None:
                        raise FileNotFoundError('Missing asset: ' + node.get('file'))
                    node.set('file', resolved)
    if missing_collision:
        short = ', '.join(PureWindowsPath(f).name for f in missing_collision)
        raise FileNotFoundError(
            'RB5 collision mesh files are missing: ' + short + '\n'
            'Add --mesh-dir ".../assets/rb5_850e/meshes". Collision geometry is required.')
    if compiler is not None:
        for name in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(name, None)
    summary = {}
    for text in root.iter('text'):
        if text.get('name') == 'cell_summary' and text.get('data'):
            summary = json.loads(text.get('data'))
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    if dropped:
        print(f'[MESH] {len(dropped)} missing visual meshes omitted; collision meshes retained.')
    return model, xml, summary, dropped


def _name(m, objtype, *cands):
    import mujoco
    for c in cands:
        if c is None:
            continue
        for n in (c, f'hand__{c}'):
            i = mujoco.mj_name2id(m, objtype, n)
            if i >= 0:
                return i, n
    raise KeyError(f'none of {cands} found')


class Ctx:
    """모델에서 쓰는 id와 손 좌표계."""

    def __init__(self, m, summary):
        import mujoco
        B, G, A = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_ACTUATOR
        self.palm, self.palm_name = _name(m, B, summary.get('detected_palm_body'), 'palm')
        self.obj, self.obj_name = _name(m, B, summary.get('detected_object_body'), 'object_body')
        self.obj_geom, _ = _name(m, G, summary.get('object_geom'), 'object')
        groups = summary.get('finger_geom_groups') or [
            [n for n in ('upper_green_link', 'upper_red_link')], [n for n in ('lower_green_link', 'lower_red_link')]]
        self.link_names, self.link_ids, sides = [], [], []
        for s, grp in enumerate(groups[:2]):
            for g in grp:
                gid, gname = _name(m, G, g)
                self.link_names.append(gname)
                self.link_ids.append(gid)
                sides.append(s)
        self.nl = len(self.link_ids)
        self.side = np.array(sides)
        self.side0 = np.flatnonzero(self.side == 0)
        self.side1 = np.flatnonzero(self.side == 1)
        self.link_idx = -np.ones(m.ngeom, dtype=int)
        for i, gid in enumerate(self.link_ids):
            self.link_idx[gid] = i
        self.is_obj = np.zeros(m.ngeom, dtype=bool)
        self.is_obj[m.geom_bodyid == self.obj] = True
        # palm subtree = 손
        in_hand = np.zeros(m.nbody, dtype=bool)
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == self.palm:
                    in_hand[b] = True
                    break
                p = m.body_parentid[p]
        self.is_hand = in_hand[m.geom_bodyid]
        self.motors = []
        for mn in summary.get('motors') or ['pulley_pull']:
            try:
                self.motors.append(_name(m, A, mn)[0])
            except KeyError:
                pass
        # 손 좌표계 (palm 좌표로 표현): x = jaw, y = 손가락이 뻗는 방향, z = palm z
        a = math.radians(float(summary.get('forward_angle_deg', 180.0)))
        fy = np.array([math.cos(a), math.sin(a), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        fx = np.cross(fy, z)
        self.H = np.column_stack([fx, fy, z])
        self.buf = np.zeros(6)


def measure(m, d, ctx):
    """물체의 palm 기준 위치·자세, 링크별 법선 접촉력, 외부 접촉력."""
    import mujoco
    Rp = d.xmat[ctx.palm].reshape(3, 3)
    pp = d.xpos[ctx.palm]
    p = Rp.T @ (d.xpos[ctx.obj] - pp)
    Rr = Rp.T @ d.xmat[ctx.obj].reshape(3, 3)
    axis = Rp.T @ d.geom_xmat[ctx.obj_geom].reshape(3, 3)[:, 2]
    lf = np.zeros(ctx.nl)
    ext = 0.0
    palmf = 0.0
    n = d.ncon
    if n:
        c = d.contact
        g1, g2 = c.geom1, c.geom2
        io1, io2 = ctx.is_obj[g1], ctx.is_obj[g2]
        idx = np.flatnonzero(io1 ^ io2)
        if idx.size:
            adr = c.efc_address
            for i in idx:
                if adr[i] < 0:
                    continue
                mujoco.mj_contactForce(m, d, int(i), ctx.buf)
                fn = ctx.buf[0]
                other = g2[i] if io1[i] else g1[i]
                li = ctx.link_idx[other]
                if li >= 0:
                    lf[li] += fn
                elif ctx.is_hand[other]:
                    palmf += fn
                else:
                    ext += fn
    return p, Rr, axis, pp.copy(), lf, ext, palmf


def contact_type(lf, ctx, thr):
    on = lf > thr
    s0 = lf[ctx.side0].sum() > thr
    s1 = lf[ctx.side1].sum() > thr
    if on.all():
        return 'env'
    if s0 and s1:
        return 'tip'
    if s0 or s1:
        return 'one_side'
    return 'none'


def _angle_deg(cosv):
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def hand_open_from_key(m, key):
    """key(task_home)의 손 관절 값 = 열린 각도. {joint1: q, ...}"""
    import mujoco
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid < 0:
        return {}
    out = {}
    for j in range(m.njnt):
        n = m.joint(j).name
        if n.startswith('hand__joint') and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            out[n[len('hand__'):]] = float(m.key_qpos[kid, m.jnt_qposadr[j]])
    return out


def grasp_site_local(m, palm):
    """palm에 붙은 task_grasp site 위치 (palm 좌표). 없으면 None."""
    import mujoco
    for n in ('task_grasp', 'hand__task_grasp'):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        if sid >= 0 and m.site_bodyid[sid] == palm:
            return m.site_pos[sid].tolist()
    return None


def min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def accel_segments(profile, cycles):
    """(부호, t_a 단위 길이) 목록. 같은 부호가 이어지면 합친다."""
    if profile == 'p2p':
        segs = [(1.0, 1.0), (-1.0, 1.0)]
    else:
        segs = []
        for _ in range(max(1, int(cycles))):
            segs += [(1.0, 1.0), (-1.0, 2.0), (1.0, 1.0)]
    out = []
    for s, w in segs:
        if out and out[-1][0] == s:
            out[-1] = (s, out[-1][1] + w)
        else:
            out.append((s, w))
    return out


def unit_profile(sim, dt, sub=20):
    """가속도 최댓값 1 m/s^2인 u 방향 궤적. 반환 (x, v, a) 길이 n+1 (t = i*dt), Tm = n*dt.

    가속도 단계가 바뀌는 시각마다 길이 ramp의 코사인 계단을 그 시각을 중심으로 둔다.
    점프의 합이 0이고 계단 모양이 같으므로 왕복이면 끝 속도·위치가 정확히 0으로 돌아온다.
    """
    ta = sim['accel_ms'] / 1000.0
    tj = max(sim['ramp_ms'], 1e-6) / 1000.0
    segs = accel_segments(sim['profile'], sim['cycles'])
    levels = [0.0] + [s for s, _ in segs] + [0.0]
    t_sw = [tj / 2.0]
    for _, w in segs:
        t_sw.append(t_sw[-1] + w * ta)
    n = int(round((t_sw[-1] + tj / 2.0) / dt))
    h = dt / sub
    tf = np.arange(n * sub + 1) * h
    af = np.zeros_like(tf)
    for i, ts in enumerate(t_sw):
        jump = levels[i + 1] - levels[i]
        u = np.clip((tf - ts) / tj + 0.5, 0.0, 1.0)
        af += jump * (0.5 - 0.5 * np.cos(np.pi * u))
    vf = np.concatenate([[0.0], np.cumsum(0.5 * (af[1:] + af[:-1]) * h)])
    xf = np.concatenate([[0.0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * h)])
    return xf[::sub].copy(), vf[::sub].copy(), af[::sub].copy(), n


class Arm:
    """RB5 6관절: IK, 역동역학, 계획 궤적 충돌 확인."""

    def __init__(self, m, xml, palm):
        import mujoco
        self.m = m
        self.palm = palm
        self.jnt = np.array([m.joint(n).id for n in ARM_JOINTS])
        self.qa = m.jnt_qposadr[self.jnt].copy()
        self.dofs = m.jnt_dofadr[self.jnt].copy()
        self.act = np.array([m.actuator(n).id for n in ARM_JOINTS])
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        self.kv = -m.actuator_biasprm[self.act, 2].copy()
        lim = m.actuator_forcerange[self.act].copy()
        flimited = m.actuator_forcelimited[self.act].astype(bool)
        self.flim = np.where(flimited, np.maximum(np.abs(lim[:, 0]), np.abs(lim[:, 1])), np.inf)
        self.qrange = m.jnt_range[self.jnt].copy()
        self.qlimited = m.jnt_limited[self.jnt].astype(bool)
        self.dk = mujoco.MjData(m)  # IK / 충돌 확인용
        # 역동역학용 모델 (접촉·구속 끔: 팔 행만 쓴다)
        self.mi = mujoco.MjModel.from_xml_string(xml)
        self.mi.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(
            mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.di = mujoco.MjData(self.mi)
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))

    def _fk(self, q):
        import mujoco
        d = self.dk
        d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.xpos[self.palm].copy(), d.xmat[self.palm].reshape(3, 3).copy()

    def _jac(self):
        import mujoco
        mujoco.mj_jacBody(self.m, self.dk, self.jp, self.jr, self.palm)
        return np.vstack([self.jp[:, self.dofs], self.jr[:, self.dofs]])

    def ik(self, p, R, seed, tol=1e-9, iters=30):
        """palm 위치 p, 자세 R (world). Newton. 반환 (q, J(q), 수렴 여부)."""
        q = np.array(seed, dtype=float)
        for _ in range(iters):
            pc, Rc = self._fk(q)
            ep = p - pc
            er = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
            J = self._jac()
            if math.sqrt(ep @ ep) < tol and math.sqrt(er @ er) < tol:
                return q, J, True
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-12 * np.eye(6), np.r_[ep, er])
            s = np.max(np.abs(dq))
            if s > 0.2:
                dq *= 0.2 / s
            q = q + dq
        pc, Rc = self._fk(q)
        ok = np.linalg.norm(p - pc) < 1e-6
        return q, self._jac(), bool(ok)

    def inverse_tau(self, qpos0, q, qd, qdd):
        """팔 관절에 필요한 토크 (구속 없음, gravcomp·damping·armature 포함, 손은 강체로)."""
        import mujoco
        d = self.di
        d.qpos[:] = qpos0
        d.qpos[self.qa] = q
        d.qvel[:] = 0.0
        d.qvel[self.dofs] = qd
        d.qacc[:] = 0.0
        d.qacc[self.dofs] = qdd
        mujoco.mj_inverse(self.mi, d)
        return d.qfrc_inverse[self.dofs].copy()

    def object_tau(self, point, force):
        """palm에 붙은 점 point에 force를 내는 데 필요한 관절 토크 (di의 현재 자세)."""
        import mujoco
        mujoco.mj_jac(self.mi, self.di, self.jp, self.jr, point, self.palm)
        return self.jp[:, self.dofs].T @ force


def arm_contact(m, d, cat, buf):
    """팔·손이 환경(책상·받침대) 또는 팔 자신과 닿는 법선력 최댓값 [N]. 손가락끼리, 손-물체는 제외."""
    import mujoco
    n = d.ncon
    if not n:
        return 0.0
    c = d.contact
    k1 = cat[c.geom1]
    k2 = cat[c.geom2]
    rob1 = (k1 == 1) | (k1 == 2)
    rob2 = (k2 == 1) | (k2 == 2)
    bad = (rob1 & (k2 == 0)) | (rob2 & (k1 == 0)) | ((k1 == 1) & rob2) | ((k2 == 1) & rob1)
    idx = np.flatnonzero(bad)
    fmax = 0.0
    adr = c.efc_address
    for i in idx:
        if adr[i] < 0:
            continue
        mujoco.mj_contactForce(m, d, int(i), buf)
        fmax = max(fmax, float(buf[0]))
    return fmax


class Plan:
    def __init__(self, direction, k, u, x, v, acc, ctrl, feasible=True, reason='', info=None):
        self.direction, self.k, self.u = direction, k, u
        self.x, self.v, self.acc, self.ctrl = x, v, acc, ctrl
        self.feasible, self.reason = feasible, reason
        self.info = info or {}


def plan_trial(cs, direction, k):
    """방향(손 좌표 이름, z0이면 None)·크기 k [g]의 팔 명령. IK·충돌을 미리 확인한다."""
    arm, n = cs.arm, cs.n_mot
    if direction is None or k == 0:
        z = np.zeros(n + 1)
        return Plan(None, 0.0, np.zeros(3), z, z, z, np.tile(cs.c0, (n + 1, 1)))
    u = cs.Hw @ np.array(ALL_DIRS[direction], dtype=float)
    u /= np.linalg.norm(u)
    a = k * cs.g
    x, v, acc = cs.ux * a, cs.uv * a, cs.ua * a
    dt = cs.dt
    arm.dk.qpos[:] = cs.qpos0
    q = np.zeros((n + 1, 6))
    qd = np.zeros((n + 1, 6))
    seed = cs.q0.copy()
    w6 = np.zeros(6)
    for i in range(n + 1):
        qi, J, ok = arm.ik(cs.palm0 + x[i] * u, cs.Rpalm0, seed)
        if not ok:
            return Plan(direction, k, u, x, v, acc, None, False, f'ik_fail@{i * dt:.3f}s')
        q[i] = qi
        w6[:3] = v[i] * u
        qd[i] = np.linalg.solve(J, w6)
        seed = qi
    lim = arm.qlimited
    if lim.any() and ((q[:, lim] < arm.qrange[lim, 0]) | (q[:, lim] > arm.qrange[lim, 1])).any():
        return Plan(direction, k, u, x, v, acc, None, False, 'joint_limit')
    qdd = np.gradient(qd, dt, axis=0, edge_order=2)
    tau = np.zeros((n + 1, 6))
    for i in range(n + 1):
        tau[i] = arm.inverse_tau(cs.qpos0, q[i], qd[i], qdd[i])
        # 물체(0.3 kg)는 접촉으로 들려 있으므로 trees에 없다: 손에 붙은 질점으로 보고 하중을 더한다
        tau[i] += arm.object_tau(cs.objcom0 + x[i] * u, cs.mass * (acc[i] * u - cs.gvec))
    dtau = tau - tau[0]
    base = cs.c0 + (q - q[0])
    if cs.sim['arm_ctrl'] == 'ff':
        ctrl = base + (arm.kv * qd + dtau) / arm.kp
    else:
        ctrl = base
    pred = cs.f0 + dtau
    info = {'max_qd': np.abs(qd).max(axis=0).tolist(), 'max_qdd': np.abs(qdd).max(axis=0).tolist(),
            'pred_force_ratio': float(np.max(np.abs(pred) / arm.flim)),
            'max_disp_m': float(np.abs(x).max()), 'max_speed_mps': float(np.abs(v).max())}
    reason = preflight(cs, q, x, u)
    return Plan(direction, k, u, x, v, acc, ctrl, not reason, reason, info)


def preflight(cs, q, x, u, every_ms=5.0):
    """계획 궤적(물체는 손에 붙어 같이 움직인다고 봄)에서 팔·손·물체가 책상·받침대·팔과 겹치는지."""
    import mujoco
    m, cat, arm = cs.m, cs.cat, cs.arm
    d = arm.dk
    tol = cs.sim['pen_mm'] / 1000.0
    n = len(x) - 1
    stride = max(1, int(round(every_ms / 1000.0 / cs.dt)))
    idxs = sorted(set(list(range(0, n + 1, stride)) + [n, int(np.argmax(np.abs(x)))]))
    base = None
    for i in idxs:
        d.qpos[:] = cs.qpos0
        d.qpos[arm.qa] = q[i]
        if cs.obj_qadr is not None:
            d.qpos[cs.obj_qadr:cs.obj_qadr + 3] = cs.qpos0[cs.obj_qadr:cs.obj_qadr + 3] + x[i] * u
        mujoco.mj_fwdPosition(m, d)
        pen = {}
        for j in range(d.ncon):
            c = d.contact[j]
            k1, k2 = int(cat[c.geom1]), int(cat[c.geom2])
            pair = {k1, k2}
            if pair in ({2}, {2, 3}):           # 손가락끼리, 손-물체
                continue
            if pair == {3} or pair == {0}:
                continue
            key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
            pen[key] = max(pen.get(key, 0.0), -float(c.dist))
        if base is None:
            base = pen
            continue
        for key, depth in pen.items():
            if depth > base.get(key, 0.0) + tol:
                names = [m.geom(int(g)).name or m.body(int(m.geom_bodyid[g])).name for g in key]
                return f'path_collision@{i * cs.dt:.3f}s:{"/".join(names)}'
    return ''


class Episode:
    def __init__(self, args):
        self.a = args
        self.mj = mujoco
        self.m, self.xml, self.summary, self.dropped = load_model(args.xml, args.mesh_dir)
        m = self.m
        self.d = d = mujoco.MjData(m)
        self.ctx = ctx = Ctx(m, self.summary)
        self.dt = float(m.opt.timestep)
        self.cfg = dict(PICK_DEFAULTS, lift=args.lift_mm / 1000., lift_time=args.lift_s)
        self.g = float(np.linalg.norm(m.opt.gravity))
        self.gvec = m.opt.gravity.copy()
        self.mass = float(m.body_subtreemass[ctx.obj])
        self.mg = self.mass * self.g
        if self.mg <= 0:
            raise ValueError('Object mass and gravity magnitude must be positive.')
        home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'task_home')
        saved = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'lifted')
        if args.start == 'pick' and home < 0:
            raise ValueError('Pick mode requires <key name="task_home">.')
        if args.start == 'lifted' and saved < 0:
            raise ValueError('Lifted mode requires <key name="lifted">.')
        if len(ctx.motors) != 1:
            raise ValueError('This cell format must contain one pulley_pull tendon actuator.')
        self.motor = ctx.motors[0]
        gear = float(m.actuator_gear[self.motor, 0])
        if abs(gear) < 1e-12:
            raise ValueError('Tendon actuator gear must be nonzero.')
        stored_tension = -float(m.key_ctrl[saved, self.motor]) * gear if saved >= 0 else 10.
        self.tension = float(args.tension if args.tension is not None else stored_tension)
        if self.tension < 0:
            raise ValueError('The inferred tendon tension is negative; specify --tension explicitly.')
        self.ctrl_T = -self.tension / gear
        if m.actuator_ctrllimited[self.motor]:
            lo, hi = m.actuator_ctrlrange[self.motor]
            if not lo - 1e-9 <= self.ctrl_T <= hi + 1e-9:
                raise ValueError(f'Tension {self.tension:g} N needs ctrl={self.ctrl_T:g}, '
                                 f'outside XML ctrlrange [{lo:g}, {hi:g}].')
        self.arm_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM])
        self.arm_v = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM])
        self.arm_u = np.array([m.actuator(n).id for n in ARM])
        self.hand = []
        for n, q in (self.summary.get('initial_hand_qpos') or hand_open_from_key(m, 'task_home')).items():
            j, _ = _name(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.hand.append((m.jnt_qposadr[j], m.jnt_dofadr[j], float(q)))
        if home >= 0:
            mujoco.mj_resetDataKeyframe(m, d, home)
            mujoco.mj_forward(m, d)
            self.obj_z0 = float(d.xpos[ctx.obj, 2])
        else:
            self.obj_z0 = float(self.summary.get('initial_object_z_m', m.body_pos[ctx.obj, 2]))
        key = home if args.start == 'pick' else saved
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.time = 0.  # Use elapsed episode time, rather than the saved key's absolute timestamp.
        d.qfrc_applied[:] = 0.
        d.xfrc_applied[:] = 0.
        mujoco.mj_forward(m, d)
        if args.start == 'pick':
            self._setup_pick()
            self.n_lift = numeric_steps(self.t_lift_end, self.dt)
        else:
            self.stages = []
            self.n_lift = 0
            d.ctrl[self.motor] = self.ctrl_T
        self.n_wait = numeric_steps(args.wait_s, self.dt)
        self.n_dist = self.n_lift + self.n_wait
        self.t_up = self.n_lift * self.dt
        self.t_dist = self.n_dist * self.dt
        self.sim = dict(profile='roundtrip', cycles=args.cycles, accel_ms=args.accel_ms,
                        ramp_ms=args.shake_ramp_ms, arm_ctrl='ff', pen_mm=.5)
        self.n_rise = numeric_steps(args.mag / args.ramp_rate, self.dt)
        if args.input == 'A':
            self.n_input = self.n_rise + numeric_steps(args.force_hold_s, self.dt)
        elif args.input == 'B':
            self.n_input = max(1, numeric_steps(args.pulse_ms / 1000., self.dt))
        else:
            self.ux, self.uv, self.ua, self.n_mot = unit_profile(self.sim, self.dt)
            if self.n_mot < 2:
                raise ValueError('C motion is shorter than two physics steps; increase --accel-ms.')
            self.n_input = self.n_mot
        self.input_duration = self.n_input * self.dt
        self.n_end = self.n_dist + self.n_input + numeric_steps(args.after_s, self.dt)
        self.t_end = self.n_end * self.dt
        self.i = 0
        self.sample_n = max(1, numeric_steps(args.sample_ms / 1000., self.dt))
        self.rows = []
        self.ref = None
        self.hold_ref = None
        self.zero_max_slip = self.zero_max_rot = 0.
        self.zero_failure = self.failure = None
        self.pre_lost = np.zeros(2, dtype=int)
        self.lost = np.zeros(2, dtype=int)
        self.max_slip = self.max_rot = self.max_external = 0.
        self.max_tracking = self.max_palm_rot = self.max_sat = self.max_acc_error = self.max_armhit = 0.
        self.plan = None
        self.fworld = np.zeros(3)
        self.force = np.zeros(3)
        self.a_cmd = np.zeros(3)
        self.a_meas = np.zeros(3)
        self.tracking_mm = self.sat = 0.
        self.filter_n = max(1, numeric_steps(.010, self.dt))
        self.velocities = deque(maxlen=self.filter_n + 1)
        self.input_started = False
        self.input_interval_start = None
        self.status = 'running'
        self.done = False
        self.saved_path = None
        self.last_phase = None
        self.meas = measure(m, d, ctx)
        if self.n_lift == 0:
            self.hold_ref = self.reference()
        print(f'[MODEL] {args.xml.name}: mass={self.mass:g} kg; tension={self.tension:g} N; dt={self.dt:g} s')
        print(f'[TIME] lift end={self.t_up:.3f} s; wait={self.n_wait*self.dt:.3f} s; '
              f'input {args.input} starts={self.t_dist:.3f} s')

    def _setup_pick(self):
        m, d, ctx, cfg = self.m, self.d, self.ctx, self.cfg
        a = math.radians(float(self.summary.get('forward_angle_deg', 180.0)))
        R_star = np.array(self.summary.get('hand_rotation_world') or d.xmat[ctx.palm].reshape(3, 3).tolist())
        self.R_star = R_star
        fwd = R_star @ np.array([math.cos(a), math.sin(a), 0.0])
        gl = self.summary.get('source_grasp_local_m')
        if not gl and (self.summary.get('cell') or {}).get('grasp_local_xy_m'):
            gl = list(self.summary['cell']['grasp_local_xy_m']) + [0.0]
        if not gl:  # summary 없는 XML: 장면의 task_grasp site (palm 좌표) = 집기 목표점
            gl = grasp_site_local(m, ctx.palm)
        gl = np.array(gl if gl is not None else [-0.076, 0.0, 0.0], dtype=float)
        obj0 = d.xpos[ctx.obj].copy()
        self.p_home = d.xpos[ctx.palm].copy()
        self.p_grasp = obj0 - R_star @ gl
        self.p_pre = self.p_grasp - cfg['approach_distance'] * fwd
        t = 0.0
        self.stages = []
        for name, dur in (('settle', cfg['settle_time']), ('lower', cfg['move_time']),
                          ('approach', cfg['approach_time']), ('settle before closing', cfg['preclose_time']),
                          (f'close: {self.tension:g} N', cfg['ramp_time']), ('grasp hold', cfg['grasp_hold']),
                          ('lift', cfg['lift_time'])):
            self.stages.append((name, t, t + dur))
            t += dur
        st = {n.split(':')[0]: (t0, t1) for n, t0, t1 in self.stages}
        self.t_close = st['close'][0]
        self.t_lift0, self.t_lift_end = st['lift']
        self.t_lower = st['lower']
        self.t_app = st['approach']
        self.q_cmd = d.qpos[self.arm_q].copy()
        self.d_ik = self.mj.MjData(m)
        self.n_ik = max(1, int(round(cfg['ik_interval'] / self.dt)))
        self._step_i = 0
        self.ik_err = 0.0

    def _palm_target(self, t):
        cfg = self.cfg
        if t < self.t_lower[0]:
            return self.p_home
        if t < self.t_lower[1]:
            return self.p_home + (self.p_pre - self.p_home) * min_jerk((t - self.t_lower[0]) / cfg['move_time'])
        if t < self.t_app[1]:
            return self.p_pre + (self.p_grasp - self.p_pre) * min_jerk((t - self.t_app[0]) / cfg['approach_time'])
        if t < self.t_lift0:
            return self.p_grasp
        return self.p_grasp + np.array([0.0, 0.0, cfg['lift']]) * min_jerk((t - self.t_lift0) / cfg['lift_time'])

    def _ik(self, target):
        mj, m, di, ctx = self.mj, self.m, self.d_ik, self.ctx
        di.qpos[:] = self.d.qpos
        di.qpos[self.arm_q] = self.q_cmd
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        err = 0.0
        for _ in range(30):
            mj.mj_kinematics(m, di)
            mj.mj_comPos(m, di)
            R = di.xmat[ctx.palm].reshape(3, 3)
            e = np.concatenate([target - di.xpos[ctx.palm],
                                0.5 * sum(np.cross(R[:, i], self.R_star[:, i]) for i in range(3))])
            err = float(np.linalg.norm(e))
            if err < 1e-8:
                break
            mj.mj_jacBody(m, di, jp, jr, ctx.palm)
            J = np.vstack([jp[:, self.arm_v], jr[:, self.arm_v]])
            di.qpos[self.arm_q] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        self.ik_err = max(self.ik_err, err)
        return di.qpos[self.arm_q].copy()

    def _pick_controls(self):
        d, cfg, t = self.d, self.cfg, self.d.time
        if t <= self.t_lift_end + 1e-9 and self._step_i % self.n_ik == 0:
            self.q_cmd = self._ik(self._palm_target(min(t + cfg['ik_interval'], self.t_lift_end)))
            d.ctrl[self.arm_u] = self.q_cmd
        self._step_i += 1
        if self.motor is not None:
            d.ctrl[self.motor] = 0.0 if t < self.t_close else self.ctrl_T * min_jerk((t - self.t_close) / cfg['ramp_time'])
        # 열림 보조: 장력을 주기 시작하면 open_release_time 동안 부드럽게 0으로
        if t < self.t_close:
            w = 1.0
        else:
            s = (t - self.t_close) / cfg['open_release_time']
            w = 0.0 if s >= 1 else 0.5 * (1 + math.cos(math.pi * s))
        lim = cfg['open_torque_limit']
        for qa, va, q_open in self.hand:
            tau = cfg['open_kp'] * (q_open - d.qpos[qa]) - cfg['open_kd'] * d.qvel[va]
            d.qfrc_applied[va] = w * min(max(tau, -lim), lim)

    def reference(self):
        p, rot, axis, *_ = self.meas
        return p.copy(), rot.copy(), axis.copy()

    def phase(self):
        t = self.i * self.dt
        if self.i < self.n_lift:
            for name, t0, t1 in self.stages:
                if t0 - 1e-10 <= t < t1 - 1e-10:
                    return name
            return 'lift'
        if self.i < self.n_dist:
            return 'wait after lift'
        if self.i < self.n_dist + self.n_input:
            return {'A': 'A force ramp/hold', 'B': 'B pulse', 'C': 'C arm shake'}[self.a.input]
        return 'observe after input' if self.i < self.n_end else 'done'

    def begin_input(self):
        a, m, d, ctx = self.a, self.m, self.d, self.ctx
        lf = self.meas[4]
        lift = 1000. * (d.xpos[ctx.obj, 2] - self.obj_z0)
        bilateral = bool(lf[ctx.side0].sum() >= a.contact_n and lf[ctx.side1].sum() >= a.contact_n)
        if (a.start == 'pick' and lift < .8 * a.lift_mm) or not bilateral or self.zero_failure:
            self.status = 'pre_disturbance_failure'
            self.done = True
            self.zero_failure = self.zero_failure or ('insufficient_lift' if lift < .8*a.lift_mm else 'lost_contact')
            print('[NO INPUT] Pick/hold failed before the disturbance: ' + self.zero_failure)
            return
        self.ref = self.reference()
        self.Hw = d.xmat[ctx.palm].reshape(3, 3).copy() @ ctx.H
        self.fworld = self.Hw @ np.array(ALL_DIRS[a.dir])
        if a.input == 'C':
            print('[PLAN C] Checking IK, collision, and feedforward commands...')
            self.palm0 = d.xpos[ctx.palm].copy()
            self.Rpalm0 = d.xmat[ctx.palm].reshape(3, 3).copy()
            self.qpos0 = d.qpos.copy()
            self.objcom0 = d.xipos[ctx.obj].copy()
            self.arm = Arm(m, self.xml, ctx.palm)
            self.q0 = d.qpos[self.arm.qa].copy()
            self.c0 = d.ctrl[self.arm.act].copy()
            self.f0 = d.actuator_force[self.arm.act].copy()
            cat_body = np.zeros(m.nbody, dtype=np.int8)
            cat_body[m.body_rootid == m.body_rootid[ctx.palm]] = 1
            cat_body[0] = 0
            self.cat = cat_body[m.geom_bodyid]
            self.cat[ctx.is_hand] = 2
            self.cat[ctx.is_obj] = 3
            self.obj_qadr = int(m.jnt_qposadr[m.body_jntadr[ctx.obj]]) if m.body_jntnum[ctx.obj] else None
            self.plan = plan_trial(self, a.dir, a.mag)
            if not self.plan.feasible:
                self.status, self.done = 'infeasible', True
                print('[NO INPUT] C trajectory infeasible: ' + self.plan.reason)
                return
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ctx.palm, vel, 0)
            for _ in range(self.filter_n + 1):
                self.velocities.append(vel[3:].copy())
        self.input_started = True
        self.input_interval_start = self.i * self.dt
        self.lost = np.zeros(2, dtype=int)
        print(f'[INPUT {a.input}] t={self.input_interval_start:.3f} s; dir={a.dir}; '
              f'magnitude={a.mag:g} {"g" if a.input == "C" else "mg"}')

    def step(self):
        if self.done:
            return
        a, m, d, ctx = self.a, self.m, self.d, self.ctx
        phase = self.phase()
        if phase != self.last_phase:
            print(f'[STAGE] {self.i*self.dt:7.3f} s  {phase}')
            self.last_phase = phase
        if a.start == 'pick':
            self._pick_controls()
        else:
            d.ctrl[self.motor] = self.ctrl_T
        if self.i == self.n_dist:
            self.begin_input()
            if self.done:
                self.record(self.i*self.dt, phase)
                return
        self.force[:] = 0.
        self.a_cmd[:] = 0.
        j = self.i - self.n_dist
        if self.input_started and 0 <= j < self.n_input:
            if a.input == 'A':
                mag = min(a.mag, a.ramp_rate * (j + .5) * self.dt)
                self.force = self.fworld * mag * self.mg
            elif a.input == 'B':
                amp = math.sin(math.pi * (j + .5) / self.n_input) if a.shape == 'halfsine' else 1.
                self.force = self.fworld * a.mag * self.mg * amp
            else:
                d.ctrl[self.arm.act] = self.plan.ctrl[j]
                self.a_cmd = self.plan.acc[j] * self.plan.u
        elif self.input_started and a.input == 'C':
            d.ctrl[self.arm.act] = self.plan.ctrl[-1]
        d.xfrc_applied[:] = 0.
        d.xfrc_applied[ctx.obj, :3] = self.force
        interval_start = self.i * self.dt
        bad0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        mujoco.mj_step(m, d)
        self.i += 1
        if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                or int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > bad0
                or abs(d.time - self.i*self.dt) > self.dt*.1):
            self.status, self.done = 'unstable', True
            print('[STOP] Nonfinite state, MuJoCo divergence warning, or simulation time reset.')
            return
        # mj_step leaves kinematic arrays at the start of the step. Refresh them
        # so logged measurements and d.time refer to the same state.
        mujoco.mj_forward(m, d)
        self.meas = measure(m, d, ctx)
        if self.i == self.n_lift:
            self.hold_ref = self.reference()
        self.monitor()
        if self.i % self.sample_n == 0 or self.i in (self.n_lift, self.n_dist, self.n_end,
                                                      self.n_dist+self.n_input):
            self.record(interval_start, phase)
        if self.i >= self.n_end:
            self.status = 'failed' if self.failure else 'held'
            if a.input == 'C' and self.arm_limited():
                self.status = 'arm_limited'
            self.done = True

    def monitor(self):
        a = self.a
        p, rot, axis, _, lf, ext, _ = self.meas
        ref = self.ref if self.input_started else self.hold_ref
        if ref is None:
            return
        p0, r0, ax0 = ref
        slip = 1000.*float(np.linalg.norm(p-p0))
        angle = _angle_deg((np.trace(r0.T@rot)-1.)/2.)
        self.current_slip, self.current_rot = slip, angle
        sides = np.array([lf[self.ctx.side0].sum(), lf[self.ctx.side1].sum()])
        lost = self.lost if self.input_started else self.pre_lost
        lost[:] = np.where(sides < a.contact_n, lost+1, 0)
        cause = ('slip' if slip >= a.slip_mm else 'rotation' if angle >= a.rot_deg else
                 'lost_contact' if lost.max()*self.dt >= a.lost_s-1e-10 else
                 'external_collision' if ext >= a.collision_n else None)
        if not self.input_started:
            self.zero_max_slip = max(self.zero_max_slip, slip)
            self.zero_max_rot = max(self.zero_max_rot, angle)
            if cause and self.zero_failure is None:
                self.zero_failure = cause
            return
        self.max_slip = max(self.max_slip, slip)
        self.max_rot = max(self.max_rot, angle)
        self.max_external = max(self.max_external, float(ext))
        if cause and self.failure is None:
            self.failure = dict(cause=cause, time_s=self.i*self.dt,
                                after_input_s=self.i*self.dt-self.t_dist)
            print(f'[GRASP FAIL] {cause} at +{self.failure["after_input_s"]:.3f} s')
        if a.input != 'C':
            return
        j = self.i-self.n_dist
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.ctx.palm, vel, 0)
        self.velocities.append(vel[3:].copy())
        self.a_meas = (self.velocities[-1]-self.velocities[0])/(self.filter_n*self.dt)
        ip = min(j, self.n_input)
        desired = self.palm0+self.plan.x[ip]*self.plan.u
        self.tracking_mm = 1000.*float(np.linalg.norm(self.d.xpos[self.ctx.palm]-desired))
        self.sat = float(np.max(np.abs(self.d.actuator_force[self.arm.act])/self.arm.flim))
        if j <= self.n_input+numeric_steps(.5, self.dt):
            self.max_tracking = max(self.max_tracking, self.tracking_mm)
            pr = self.d.xmat[self.ctx.palm].reshape(3,3)
            self.max_palm_rot = max(self.max_palm_rot, _angle_deg((np.trace(self.Rpalm0.T@pr)-1.)/2.))
            self.max_sat = max(self.max_sat, self.sat)
            cmd0 = self.plan.v[max(0, min(j-self.filter_n, self.n_input))]
            cmd1 = self.plan.v[ip]
            command_filtered = (cmd1-cmd0)/(self.filter_n*self.dt)*self.plan.u
            self.max_acc_error = max(self.max_acc_error, float(np.linalg.norm(self.a_meas-command_filtered))/self.g)
            self.max_armhit = max(self.max_armhit, arm_contact(self.m, self.d, self.cat, np.zeros(6)))

    def arm_limited(self):
        return (self.max_tracking > 10. or self.max_palm_rot > 3. or self.max_sat >= .999
                or self.max_acc_error > .03+.15*self.a.mag or self.max_armhit > .5)

    def record(self, interval_start, phase):
        lf = self.meas[4]
        row = dict(time_s=self.i*self.dt, phase=phase, applied_interval_start_s=interval_start,
                   after_input_s=self.i*self.dt-self.t_dist,
                   object_lift_mm=1000.*(self.d.xpos[self.ctx.obj,2]-self.obj_z0),
                   slip_mm=getattr(self, 'current_slip', 0.), rotation_deg=getattr(self, 'current_rot', 0.),
                   side0_contact_N=float(lf[self.ctx.side0].sum()),
                   side1_contact_N=float(lf[self.ctx.side1].sum()), external_contact_N=float(self.meas[5]),
                   tracking_mm=self.tracking_mm, actuator_force_ratio=self.sat)
        for label, vec in [('force_world_N', self.force), ('accel_command_mps2', self.a_cmd),
                           ('accel_measured_mps2', self.a_meas)]:
            for axis, value in zip('xyz', vec):
                row[f'{label}_{axis}'] = float(value)
        for name, value in zip(self.ctx.link_names, lf):
            row[f'{name}_contact_N'] = float(value)
        self.rows.append(row)

    def result(self):
        a = self.a
        return dict(xml=str(a.xml.resolve()), mujoco_version=mujoco.__version__, input=a.input,
                    direction=a.dir, magnitude=a.mag, magnitude_unit='g' if a.input=='C' else 'mg',
                    direction_meaning='initial hand motion' if a.input=='C' else 'force on object',
                    start=a.start, status=self.status, input_applied=self.input_started,
                    input_start_s=self.input_interval_start, planned_input_start_s=self.t_dist,
                    lift_end_s=self.t_up, wait_after_lift_s=self.n_wait*self.dt,
                    input_duration_s=self.input_duration, observation_after_input_s=a.after_s,
                    simulation_end_s=self.i*self.dt, mass_kg=self.mass, mg_N=self.mg,
                    tendon_tension_N=self.tension, timestep_s=self.dt,
                    requested_arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                    pre_disturbance_failure=self.zero_failure,
                    pre_disturbance_max_slip_mm=self.zero_max_slip,
                    pre_disturbance_max_rotation_deg=self.zero_max_rot,
                    grasp_failure=self.failure, max_slip_mm=self.max_slip, max_rotation_deg=self.max_rot,
                    max_external_contact_N=self.max_external,
                    C_metrics=None if a.input!='C' else dict(
                        arm_limited=self.arm_limited() if self.input_started else None,
                        max_tracking_mm=self.max_tracking, max_palm_rotation_deg=self.max_palm_rot,
                        max_actuator_force_ratio=self.max_sat, max_acceleration_error_g=self.max_acc_error,
                        max_arm_environment_contact_N=self.max_armhit,
                        plan_reason=self.plan.reason if self.plan else None,
                        plan_info=self.plan.info if self.plan else None),
                    missing_visual_meshes=self.dropped)

    def save(self):
        if self.saved_path is not None:
            return self.saved_path
        if self.status == 'running':
            self.status = 'interrupted'
        out = self.a.out_dir
        if out is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            out = Path('single_disturb_results') / f'{self.a.xml.stem}_{self.a.input}_{stamp}'
        out = out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any((out/name).exists() for name in ('trace.csv','summary.json')):
            raise FileExistsError('Output already contains results. Use a new --out-dir: '+str(out))
        if self.rows:
            with (out/'trace.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        (out/'summary.json').write_text(json.dumps(self.result(), ensure_ascii=False, indent=2),encoding='utf-8')
        self.saved_path = out
        print(f'[RESULT] {self.status}; input applied={self.input_started}; '
              f'max slip={self.max_slip:.2f} mm; max rotation={self.max_rot:.2f} deg')
        print('[OUTPUT] '+str(out))
        return out


def draw_viewer(viewer, ep):
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0
        if ep.input_started and not ep.done and ep.i < ep.n_dist+ep.n_input:
            vector = ep.a_cmd if ep.a.input == 'C' else ep.force
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                center = ep.d.xpos[ep.ctx.palm if ep.a.input=='C' else ep.ctx.obj].copy()
                start = center + np.array([0.,0.,.06])
                end = start + .10*vector/norm
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),np.zeros(3),
                                   np.eye(3).ravel(),np.array([1.,.3,.05,1.],dtype=np.float32))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.006,start,end)
                scn.ngeom += 1
    if hasattr(viewer, 'set_texts'):
        left = 'stage\ntime [s]\ninput\nwait after lift [s]\ntension [N]\nslip [mm]\nrotation [deg]\nstatus'
        right = (f'{ep.phase()}\n{ep.i*ep.dt:.3f}\n{ep.a.input} {ep.a.dir} {ep.a.mag:g}'
                 f'\n{ep.n_wait*ep.dt:g}\n{ep.tension:g}\n{getattr(ep,"current_slip",0.):.2f}'
                 f'\n{getattr(ep,"current_rot",0.):.2f}\n{ep.status}')
        viewer.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,left,right)])
    viewer.sync()


def run_gui(ep):
    import mujoco.viewer
    paused = [False]
    def keypress(key):
        if key == 32:
            paused[0] = not paused[0]
    with mujoco.viewer.launch_passive(ep.m,ep.d,key_callback=keypress,
                                     show_left_ui=False,show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [.45, 0., .25]
        viewer.cam.distance = 1.25
        viewer.cam.azimuth,viewer.cam.elevation = 145., -25.
        viewer.opt.geomgroup[3] = 0 if not ep.dropped else 1
        saved = False
        while viewer.is_running():
            tick = time.perf_counter()
            if not paused[0] and not ep.done:
                target = ep.i*ep.dt+ep.a.speed/60.
                while not ep.done and ep.i*ep.dt < target-1e-10:
                    ep.step()
            draw_viewer(viewer,ep)
            if ep.done and not saved:
                ep.save()
                saved = True
                if ep.a.close_at_end:
                    break
                print('[VIEW] Finished. Close the window to exit. Space pauses/resumes.')
            time.sleep(max(0.,1./60.-(time.perf_counter()-tick)))
        if not saved:
            ep.save()


def parser():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--xml',type=Path,required=True,help='Exported RB5 cell XML path')
    p.add_argument('--input','--disturbance',dest='input',type=str.upper,choices=['A','B','C'],required=True)
    p.add_argument('--dir',choices=ALL_DIRS,default='py',help='Hand axes: x=jaw, y=outward, z=cylinder axis')
    p.add_argument('--mag',type=float,default=1.,help='A/B force in multiples of m*g; C hand acceleration in g')
    p.add_argument('--wait-s','--hold-s',dest='wait_s',type=float,default=5.,help='Wait AFTER lift completion')
    p.add_argument('--after-s',type=float,default=10.,help='Observation time after disturbance ends')
    p.add_argument('--start',choices=['pick','lifted'],default='pick',help='Reproduce pick or restore lifted key')
    p.add_argument('--mesh-dir',type=Path,help='Existing assets/rb5_850e/meshes directory')
    p.add_argument('--tension',type=float,help='Tendon tension in N; default inferred from lifted key ctrl and gear')
    p.add_argument('--lift-mm',type=float,default=150.,help='Pick mode lift height')
    p.add_argument('--lift-s',type=float,default=3.,help='Pick mode lift duration')
    p.add_argument('--ramp-rate',type=float,default=1.,help='A: force ramp slope in mg/s')
    p.add_argument('--force-hold-s',type=float,default=1.,help='A: hold peak force after ramp')
    p.add_argument('--pulse-ms',type=float,default=50.,help='B: pulse duration')
    p.add_argument('--shape',choices=['rect','halfsine'],default='rect',help='B: pulse shape')
    p.add_argument('--accel-ms',type=float,default=100.,help='C: basic acceleration segment duration')
    p.add_argument('--shake-ramp-ms',type=float,default=20.,help='C: cosine smoothing duration')
    p.add_argument('--cycles',type=int,default=1,help='C: number of +a -> -a -> +a round trips')
    p.add_argument('--slip-mm',type=float,default=5.)
    p.add_argument('--rot-deg',type=float,default=5.)
    p.add_argument('--contact-n',type=float,default=.1,help='Minimum contact force per finger side')
    p.add_argument('--lost-s',type=float,default=.02,help='Contact-loss duration threshold')
    p.add_argument('--collision-n',type=float,default=.05,help='Object contact with anything outside the hand')
    p.add_argument('--sample-ms',type=float,default=5.,help='CSV sample period; physics still runs at XML timestep')
    p.add_argument('--headless',action='store_true',help='Run physics and save results without a window')
    p.add_argument('--speed',type=float,default=1.,help='GUI playback speed; physics timestep is unchanged')
    p.add_argument('--close-at-end',action='store_true',help='Close GUI when simulation finishes')
    p.add_argument('--out-dir',type=Path,help='New result directory; existing result files are never overwritten')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if not a.xml.is_file():
        p.error('XML file not found: '+str(a.xml))
    for key in ('mag','wait_s','after_s','force_hold_s'):
        if not math.isfinite(getattr(a,key)) or getattr(a,key)<0:
            p.error('--'+key.replace('_','-')+' must be finite and nonnegative')
    for key in ('ramp_rate','pulse_ms','accel_ms','shake_ramp_ms','lift_mm','lift_s',
                'slip_mm','rot_deg','contact_n','lost_s','collision_n','sample_ms','speed'):
        if not math.isfinite(getattr(a,key)) or getattr(a,key)<=0:
            p.error('--'+key.replace('_','-')+' must be finite and positive')
    if a.tension is not None and (not math.isfinite(a.tension) or a.tension<0):
        p.error('--tension must be finite and nonnegative')
    if a.cycles<1:
        p.error('--cycles must be at least 1')
    if a.shake_ramp_ms>a.accel_ms:
        p.error('--shake-ramp-ms must not exceed --accel-ms')
    if a.out_dir and any((a.out_dir/f).exists() for f in ('summary.json','trace.csv')):
        p.error('--out-dir already contains results; choose a new directory')
    ep = None
    try:
        ep = Episode(a)
        if a.headless:
            while not ep.done:
                ep.step()
            ep.save()
        else:
            run_gui(ep)
    except KeyboardInterrupt:
        if ep is not None:
            ep.save()
        return 130
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,np.linalg.LinAlgError) as err:
        print('[ERROR] '+str(err),file=sys.stderr)
        if ep is not None:
            ep.status='error'
            ep.save()
        return 2
    return 0  # A physical grasp failure is a valid completed experiment; see summary.json.


if __name__ == '__main__':
    raise SystemExit(main())
