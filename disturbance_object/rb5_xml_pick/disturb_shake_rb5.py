#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 C — 들어 올린 뒤 팔 흔들기(가속·감속)에 대한 파지 유지 (RB5 + grasp_cells)

실험 A/B와 같은 출발점 S0(state.npz 복원 → 외력 없이 --pre-s 안정화)에서, 물체에 힘을 주는 대신
RB5 팔을 움직여 손(palm)을 손 좌표계의 한 방향 u로 왕복시킨다. 손 자세는 고정(평행 이동)이다.

  손 가속도 a(t) (u 방향, 기본 --profile roundtrip):
      +a (t_a)  →  -a (2 t_a)  →  +a (t_a)          → 제자리로 돌아와 멈춤
  가속도가 바뀌는 곳은 --ramp-ms 동안 코사인으로 부드럽게 바꾼다(가속도 연속, jerk 제한).
  손 기준으로 물체가 받는 관성력은 -m a(t)이므로 한 번 왕복하면 -u, +u 하중이 모두 걸리고,
  가운데 2 t_a 동안은 +u 하중(= 처음 움직이는 방향)이 걸린다. 방향 이름은 손이 처음 움직이는 방향이다.
  z는 중력이 더해진다: pz는 처음·끝 t_a에, nz는 가운데 2 t_a에 아래로 m(g + a)가 걸린다.
  크기 k [g]: 손 가속도 최댓값 = k * |gravity| (기본 0.5, 1 g).

운동이 끝난 뒤 --after-s(기본 10 s) 동안 유지한다. 판정은 실험 B와 같다:
  slip 5 mm / 회전 5 deg / 한쪽 손가락 접촉력 < 0.1 N이 20 ms / 손 밖 물체와 접촉 0.05 N / 발산.
  외력 0 대조(z0: 팔을 움직이지 않고 같은 시간 유지)에서 실패한 cell은 zero_fail로 뺀다.
  실패 시점: motion(운동 중) / transient(운동 후 --settle-s 안) / delayed(그 뒤).

팔 제어 (--arm-ctrl ff, 기본):
  손 목표 궤적 → 매 step 6D IK(자세 고정) → q_des, q̇_des = J^-1 v, q̈_des
  → ctrl = c0 + (q_des - q_des(0)) + (kv * q̇_des + Δτ_ff) / kp
  c0는 S0의 명령(저장 상태 그대로), Δτ_ff는 역동역학 토크의 S0 대비 변화(관성·코리올리·
  gravcomp가 없는 몸체의 중력·물체 하중). position actuator의 forcerange 제한은 그대로 걸린다.
  --arm-ctrl pos이면 위치 명령만 준다(run_rb5_xml.py의 lift와 같은 방식, 추종 지연이 큼).
  실제 손 가속도(--filter-ms 이동 평균), 손 위치 추종 오차, 손 자세 변화, actuator 포화, 팔 충돌을 기록하고,
  목표 가속도를 제대로 내지 못한 시행은 파지 실패와 섞지 않고 arm_limited로 따로 둔다(--rescore로 기준 변경).
  시뮬레이션 전에 계획 궤적의 IK와 충돌(팔·손·물체 vs 책상·받침대·팔)을 확인해 안 되는 시행은 infeasible로 둔다.

  python disturb_shake_rb5.py --designs design0001 --limit 3          # 빠른 시험
  python disturb_shake_rb5.py --workers 6                             # 전체 (끊기면 같은 명령으로 이어서)
  python disturb_shake_rb5.py --rescore --acc-tol 0.2                 # 시뮬레이션 없이 재판정
  python view_shake_rb5.py design0001/r060.0_y015.0 --dir py --mag 1  # 한 시행을 viewer로 (같은 폴더)

셀 입력: 기본은 이 파일 옆 grasp_cells_xml/ (collect_cell_xmls.py가 만든 XML 하나 = cell 하나,
상태는 <key name="lifted">, summary는 <custom>). 없으면 grasp_cells/ (폴더, state.npz). --cells-dir로 지정.
--clean: 시작 전에 results_X 안의 기존 결과를 모두 지움.
장면 읽기, 손 좌표계, 접촉력 함수는 disturb_impulse_rb5.py(같은 폴더)의 것을 그대로 쓴다.
시뮬레이션 결과는 cell마다 traces/<cell>.npz 하나에 저장하고, 표(CSV)는 항상 trace에서 다시 계산한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from disturb_impulse_rb5 import (  # noqa: E402  (같은 폴더)
    ALL_DIRS, Ctx, measure, contact_type, _angle_deg, _LF, parse_dirs,
    load_scene_xml, read_json, load_trace, save_trace, write_csv, read_csv,
    find_design_index, load_design_vars, select_cells, hms,
    load_cell, cell_parts, is_xml_cells_dir, clean_results, warn_old_xml)
from disturb_impulse_rb5 import default_cells_dir as _default_cells_dir  # noqa: E402

ARM_JOINTS = ('base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3')
DEFAULT_MAGS = [0.5, 1.0]
# 이 값이 다르면 trace의 의미가 달라진다 (같은 출력 폴더에 섞지 않는다)
SIM_KEYS = ('profile', 'cycles', 'accel_ms', 'ramp_ms', 'arm_ctrl', 'pre_s', 'after_s', 'settle_s',
            'sample_ms', 'coarse_s', 'check_ms', 'filter_ms', 'stop_mm', 'stop_lost_s', 'lost_n',
            'tension', 'pen_mm')
BASE_COLS = ['t', 'disp', 'rot', 'tilt', 'lost', 'ext', 'side_min',
             'dx', 'dy', 'dz', 'hand', 'palm_f', 'side0', 'side1']
# acmd/acf_*: 명령 가속도(측정과 같은 이동 평균), au*: 측정 손 가속도의 u 성분, aperp: u에 수직인 성분 크기 [g]
# track: 손(palm) 위치 - 계획 위치 [mm], prot: 손 자세 변화 [deg], sat: max |actuator force| / forcerange,
# armhit: 팔·손이 책상·받침대·팔 자신과 닿는 법선력 [N]  (모두 row 구간의 최댓값/최솟값)
C_COLS = ['acmd', 'acf_max', 'acf_min', 'au', 'au_max', 'au_min', 'aperp', 'track', 'prot', 'sat', 'armhit']
GROUPS = ('env', 'tip', 'other')
PHASES = ('motion', 'transient', 'delayed')


def trial_key(direction, k):
    return f'{direction}_{k:g}g'


def split_key(key):
    if key == 'z0':
        return '', 0.0
    dn, ks = key.rsplit('_', 1)
    return dn, float(ks[:-1])


def sim_hash(sim):
    blob = json.dumps({k: sim[k] for k in SIM_KEYS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def dir_label(vec):
    """손 좌표 단위벡터 -> 가장 가까운 방향 이름 (px, ny, ... , pxpy ...)."""
    best, bn = None, -2.0
    for name, v in ALL_DIRS.items():
        c = float(np.dot(vec, v))
        if c > bn:
            best, bn = name, c
    return best


# ----------------------------------------------------------------------------- motion profile
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


# ----------------------------------------------------------------------------- arm
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


class CellSim:
    """한 cell의 모델, S0, 손 좌표계, 팔. disturb_shake_rb5와 view_shake_rb5가 같이 쓴다."""

    def __init__(self, cell_dir, sim, mesh_roots, allow_missing_collision=False):
        import mujoco
        self.sim = sim
        m, d, xml, self.summary, self.cinfo = load_cell(cell_dir, mesh_roots, allow_missing_collision)
        self.cell_dir = Path(self.cinfo['cell_path'])
        self.dropped, self.dropped_col = self.cinfo['dropped_meshes'], self.cinfo['dropped_collision_meshes']
        self.m, self.d = m, d
        ctx = Ctx(m, self.summary)
        self.ctx = ctx
        self.applied = float(np.abs(d.qfrc_applied).sum() + np.abs(d.xfrc_applied).sum())
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        tension = sim['tension']
        if tension is not None:
            for a in ctx.motors:
                c = -float(tension) / m.actuator_gear[a, 0]
                lo, hi = m.actuator_ctrlrange[a]
                m.actuator_ctrlrange[a] = [min(lo, c), max(hi, c)]
                d.ctrl[a] = c
        self.tension_used = [float(-d.ctrl[a] * m.actuator_gear[a, 0]) for a in ctx.motors]
        self.dt = dt = m.opt.timestep
        self.g = float(np.linalg.norm(m.opt.gravity))
        self.gvec = m.opt.gravity.copy()
        self.mass = float(m.body_subtreemass[ctx.obj])
        self.mg = self.mass * self.g
        # 안정화 (B와 같음: 외력 없음, 팔은 저장된 명령 유지). 마지막 0.1 s 링크 접촉력 평균으로 시작 접촉 형태
        n_pre = int(round(sim['pre_s'] / dt))
        n_avg = min(n_pre, int(round(0.1 / dt)))
        lsum = np.zeros(ctx.nl)
        for i in range(n_pre):
            mujoco.mj_step(m, d)
            if i >= n_pre - n_avg:
                lsum += measure(m, d, ctx)[4]
        self.start_lf = lsum / max(n_avg, 1) if n_avg else measure(m, d, ctx)[4]
        self.start_type = contact_type(self.start_lf, ctx, sim['lost_n'])
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.S0 = np.zeros(mujoco.mj_stateSize(m, self.spec))
        mujoco.mj_getState(m, d, self.S0, self.spec)
        p0, R0, ax0, palm0, _, _, _ = measure(m, d, ctx)
        self.ref = (p0, R0, ax0, palm0)
        self.palm0 = d.xpos[ctx.palm].copy()
        self.Rpalm0 = d.xmat[ctx.palm].reshape(3, 3).copy()
        self.Hw = self.Rpalm0 @ ctx.H          # 손 좌표축 (world), 열 = x, y, z
        self.qpos0 = d.qpos.copy()
        self.objcom0 = d.xipos[ctx.obj].copy()
        self.arm = Arm(m, xml, ctx.palm)
        self.q0 = d.qpos[self.arm.qa].copy()
        self.c0 = d.ctrl[self.arm.act].copy()
        self.f0 = d.actuator_force[self.arm.act].copy()
        # 접촉 분류: 0 환경(책상·받침대 등), 1 팔(손 제외), 2 손, 3 물체
        root = m.body_rootid[ctx.palm]
        cat_body = np.zeros(m.nbody, dtype=np.int8)
        cat_body[m.body_rootid == root] = 1
        cat_body[0] = 0
        cat = cat_body[m.geom_bodyid]
        cat[ctx.is_hand] = 2
        cat[ctx.is_obj] = 3
        self.cat = cat
        self.obj_qadr = int(m.jnt_qposadr[m.body_jntadr[ctx.obj]]) if m.body_jntnum[ctx.obj] else None
        self.arm_hit0 = arm_contact(m, d, cat, np.zeros(6))
        # 운동 궤적 (가속도 1 m/s^2 기준, 크기는 곱해서 쓴다)
        self.ux, self.uv, self.ua, self.n_mot = unit_profile(sim, dt)
        self.Tm = self.n_mot * dt
        self.n_settle = int(round(sim['settle_s'] / dt))
        self.n_after = int(round(sim['after_s'] / dt))
        self.n_total = self.n_mot + self.n_after
        self.n_win = min(self.n_total, self.n_mot + self.n_settle)
        self.n_filt = max(1, int(round(sim['filter_ms'] / 1000.0 / dt)))
        self.ends = row_schedule(sim, dt, self.n_mot, self.n_total)

    def meta(self):
        import mujoco
        cell = self.summary.get('cell') or {}
        return {
            'r_mm': cell.get('r_mm'), 'y_mm': cell.get('y_mm'), 'x_mm': cell.get('x_mm'),
            'workspace_grasp_type': cell.get('workspace_grasp_type'),
            'rb5_four_link_contact': self.summary.get('four_link_contact'),
            'mass_kg': self.mass, 'mg_N': self.mg, 'g': self.g, 'tension_N': self.tension_used,
            'timestep': self.dt, 'motion_s': self.Tm, 'n_motion': self.n_mot,
            'start_type': self.start_type, 'start_link_forces_N': [float(x) for x in self.start_lf],
            'link_names': self.ctx.link_names, 'link_side': [int(s) for s in self.ctx.side],
            'hand_frame_world': self.Hw.tolist(),
            'object_in_palm_mm': (1000.0 * self.ref[0]).tolist(),
            'palm_pos_world_m': self.palm0.tolist(),
            'arm_q0': self.q0.tolist(), 'arm_ctrl0': self.c0.tolist(), 'arm_force0': self.f0.tolist(),
            'arm_kp': self.arm.kp.tolist(), 'arm_kv': self.arm.kv.tolist(),
            'arm_force_limit': [float(x) for x in self.arm.flim],
            'arm_contact_at_S0_N': self.arm_hit0,
            'applied_force_in_state': self.applied,
            'dropped_meshes': self.dropped, 'dropped_collision_meshes': self.dropped_col,
            'cell_path': self.cinfo['cell_path'], 'state_source': self.cinfo['state_source'],
            'summary_source': self.cinfo['summary_source'],
            'mujoco_version': mujoco.__version__,
            'source_mujoco_version': self.summary.get('source_mujoco_version'),
        }


def row_schedule(sim, dt, n_mot, n_total):
    fine_end = min(n_total, n_mot + int(round(sim['settle_s'] / dt)))
    fs = max(1, int(round(sim['sample_ms'] / 1000.0 / dt)))
    cs = max(1, int(round(sim['coarse_s'] / dt)))
    ends = list(range(fs, fine_end + 1, fs))
    if not ends or ends[-1] != fine_end:
        ends.append(fine_end)
    s = fine_end + cs
    while s < n_total:
        ends.append(s)
        s += cs
    if ends[-1] != n_total:
        ends.append(n_total)
    return [e for e in ends if e > 0]


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


# ----------------------------------------------------------------------------- planning
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


# ----------------------------------------------------------------------------- simulation
def run_trial(cs, plan, callback=None):
    """S0에서 한 시행. 반환 (rows[n_rows, ncols] float32, ended).

    callback(k, t, info)는 매 step 뒤 호출된다(viewer·영상용). info: a_cmd, a_meas (world, m/s^2).
    """
    import mujoco
    m, d, ctx, sim = cs.m, cs.d, cs.ctx, cs.sim
    arm = cs.arm
    dt = cs.dt
    ce = max(1, int(round(sim['check_ms'] / 1000.0 / dt)))
    lost_n, stop_mm, stop_lost = sim['lost_n'], sim['stop_mm'], sim['stop_lost_s']
    p0, R0, ax0, palm0 = cs.ref
    H = ctx.H
    u = plan.u
    g = cs.g
    n_mot, n_win, n_total, nf = cs.n_mot, cs.n_win, cs.n_total, cs.n_filt
    mujoco.mj_setState(m, d, cs.S0, cs.spec)
    d.xfrc_applied[:] = 0.0
    d.qfrc_applied[:] = 0.0
    w0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    end_set = set(cs.ends)
    act = arm.act
    ctrl = plan.ctrl
    # 명령 속도 (측정과 같은 이동 평균을 위해 앞에 0을 붙인다)
    vcmd = np.concatenate([np.zeros(nf), plan.v, np.zeros(max(0, n_win - n_mot) + 1)])
    vbuf = np.zeros((nf + 1, 3))   # 손 속도 고리 버퍼 (nf step 전 ~ 지금)
    vi = 0
    vfill = False
    res6 = np.zeros(6)
    buf6 = np.zeros(6)
    flim = arm.flim
    Rp0 = cs.Rpalm0
    rows = []
    dmax = rmax = tmax = lostmax = extmax = 0.0
    smin = math.inf
    acf_max = au_max = -math.inf
    acf_min = au_min = math.inf
    aperp = track = prot = sat = armhit = 0.0
    acmd_last = au_last = 0.0
    lsum = np.zeros(ctx.nl)
    lcnt = 0
    streak = [0, 0]
    cdt = ce * dt
    ended = 'complete'
    a_meas = np.zeros(3)
    for step in range(n_total):
        if step <= n_mot:
            d.ctrl[act] = ctrl[step]
        mujoco.mj_step(m, d)
        k = step + 1
        # 손 측정 (mj_step 뒤의 xpos·cvel은 step 시작 시점 = 계획 index step)
        if k <= n_win:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ctx.palm, res6, 0)
            if not vfill:
                vbuf[:] = res6[3:]
                vfill = True
            vi = (vi + 1) % (nf + 1)
            vbuf[vi] = res6[3:]
            a_meas = (vbuf[vi] - vbuf[(vi + 1) % (nf + 1)]) / (nf * dt)
            j = step + nf
            ac = (vcmd[j] - vcmd[j - nf]) / (nf * dt) / g
            au = float(a_meas @ u) / g
            ap = a_meas - (a_meas @ u) * u
            acf_max = max(acf_max, ac)
            acf_min = min(acf_min, ac)
            au_max = max(au_max, au)
            au_min = min(au_min, au)
            aperp = max(aperp, float(math.sqrt(ap @ ap)) / g)
            ip = min(step, n_mot)
            e = d.xpos[ctx.palm] - (cs.palm0 + plan.x[ip] * u)
            track = max(track, 1000.0 * float(math.sqrt(e @ e)))
            Rp = d.xmat[ctx.palm].reshape(3, 3)
            prot = max(prot, _angle_deg((np.trace(Rp0.T @ Rp) - 1.0) / 2.0))
            sat = max(sat, float(np.max(np.abs(d.actuator_force[act]) / flim)))
            acmd_last, au_last = ac, au
        elif k == n_win + 1:
            acmd_last = au_last = 0.0
            a_meas = np.zeros(3)
        if callback is not None:
            ip = min(step, n_mot)
            callback(k, k * dt, {'a_cmd': plan.acc[ip] * u if step < n_mot else np.zeros(3),
                                 'a_meas': a_meas, 'plan_palm': cs.palm0 + plan.x[ip] * u})
        at_end = k in end_set
        if k % ce and not at_end:
            continue
        p, Rr, ax, pp, lf, ext, palmf = measure(m, d, ctx)
        if k <= n_win:
            armhit = max(armhit, arm_contact(m, d, cs.cat, buf6))
        dp = p - p0
        disp = 1000.0 * math.sqrt(dp @ dp)
        rot = _angle_deg((np.trace(R0.T @ Rr) - 1.0) / 2.0)
        tilt = _angle_deg(float(ax @ ax0))
        s0 = lf[ctx.side0].sum()
        s1 = lf[ctx.side1].sum()
        streak[0] = streak[0] + 1 if s0 < lost_n else 0
        streak[1] = streak[1] + 1 if s1 < lost_n else 0
        lost = max(streak) * cdt
        dmax = max(dmax, disp)
        rmax = max(rmax, rot)
        tmax = max(tmax, tilt)
        lostmax = max(lostmax, lost)
        extmax = max(extmax, ext)
        smin = min(smin, s0, s1)
        lsum += lf
        lcnt += 1
        stop = disp >= stop_mm or lost >= stop_lost - 1e-12
        if at_end or stop:
            bad = (int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > w0
                   or not np.isfinite(d.qpos).all())
            dh = H.T @ dp * 1000.0
            zf = lambda v: 0.0 if not math.isfinite(v) else v  # noqa: E731
            row = [k * dt, dmax, rmax, tmax, lostmax, extmax, smin,
                   dh[0], dh[1], dh[2], 1000.0 * float(np.linalg.norm(pp - palm0)), palmf, s0, s1,
                   acmd_last, zf(acf_max), zf(acf_min), au_last, zf(au_max), zf(au_min),
                   aperp, track, prot, sat, armhit]
            rows.append(row + list(lsum / max(lcnt, 1)))
            dmax = rmax = tmax = lostmax = extmax = 0.0
            smin = math.inf
            acf_max = au_max = -math.inf
            acf_min = au_min = math.inf
            aperp = track = prot = sat = armhit = 0.0
            lsum[:] = 0.0
            lcnt = 0
            if bad:
                ended = 'unstable'
                break
            if stop:
                ended = 'stop_disp' if disp >= stop_mm else 'stop_lost'
                break
    return np.asarray(rows, dtype=np.float32), ended


def trace_columns(ctx):
    return BASE_COLS + C_COLS + [f'f_{n.replace("hand__", "")}' for n in ctx.link_names]


def simulate_cell(job):
    """worker: 한 cell의 필요한 시행을 돌려 trace 파일에 저장(기존 시행에 추가)."""
    t_start = time.time()
    cell_id = job['cell_id']
    try:
        sim = job['sim']
        cs = CellSim(job['cell_dir'], sim, job['mesh_roots'], job['allow_missing_collision'])
        cols = trace_columns(cs.ctx)
        out = Path(job['trace_path'])
        trials, plans_old = {}, {}
        if out.is_file() and not job['overwrite']:
            try:
                old_meta, old_cols, trials = load_trace(out)
                if old_meta.get('sim_hash') != job['sim_hash'] or old_cols != cols:
                    trials = {}
                else:
                    plans_old = old_meta.get('plans', {})
            except Exception:
                trials = {}
        todo = [k for k in job['keys'] if k not in trials]
        mono = job.get('monotonic_crit')
        failed_dirs = set()
        if mono:
            for key in sorted(trials, key=split_key):
                if key == 'z0':
                    continue
                dn, kk = split_key(key)
                r = score_trial(*trials[key], cols, mono, cs.Tm, sim['settle_s'], kk)
                if r['valid'] and not r['held']:
                    failed_dirs.add(dn)
        plans = dict(plans_old)
        ran = 0
        for key in sorted(todo, key=split_key):
            direction, k = split_key(key)
            if mono and direction in failed_dirs:
                continue
            plan = plan_trial(cs, direction or None, k)
            if direction:
                plans[key] = {'feasible': plan.feasible, 'reason': plan.reason, **plan.info}
            if not plan.feasible:
                trials[key] = (np.zeros((0, len(cols)), dtype=np.float32), f'infeasible:{plan.reason}')
                continue
            rows, ended = run_trial(cs, plan)
            trials[key] = (rows, ended)
            ran += 1
            if mono and key != 'z0':
                r = score_trial(rows, ended, cols, mono, cs.Tm, sim['settle_s'], k)
                if r['valid'] and not r['held']:
                    failed_dirs.add(direction)
        meta = {'cell_id': cell_id, 'design': job['design'], 'cell': job['cell'],
                'sim_hash': job['sim_hash'], 'sim': {k: sim[k] for k in SIM_KEYS},
                **cs.meta(), 'plans': plans}
        save_trace(out, meta, cols, trials)
        return {'cell_id': cell_id, 'ran': ran, 'sec': time.time() - t_start, 'error': None}
    except Exception:
        return {'cell_id': cell_id, 'ran': 0, 'sec': time.time() - t_start, 'error': traceback.format_exc()}


# ----------------------------------------------------------------------------- scoring
def score_trial(rows, ended, cols, crit, Tm, settle, k):
    """파지 판정(B와 같음) + 팔이 목표 운동을 냈는지(valid)."""
    c = {n: i for i, n in enumerate(cols)}
    res = {'held': False, 'valid': False, 'cause': 'no_data', 'phase': '', 't_fail_s': math.nan,
           'ended': ended, 'arm_reason': ''}
    if ended.startswith('infeasible'):
        res.update(cause='infeasible', arm_reason=ended.split(':', 1)[1] if ':' in ended else '')
        return res
    n = len(rows)
    if n == 0:
        return res
    t = rows[:, c['t']].astype(float)
    tstart = np.concatenate([[0.0], t[:-1]])
    after = tstart >= Tm + settle - 1e-9
    slip = rows[:, c['disp']] >= crit['slip_mm']
    if crit['slip_mode'] == 'residual':
        slip &= after
    rotv = rows[:, c['tilt' if crit['rot_mode'] == 'tilt' else 'rot']]
    masks = [('slip', slip),
             ('rotation', rotv >= crit['rot_deg']),
             ('lost', rows[:, c['lost']] >= crit['lost_s'] - 1e-9),
             ('collision', rows[:, c['ext']] >= crit['collision_n'])]
    first, cause = n, None
    for name, mk in masks:
        idx = np.flatnonzero(mk)
        if idx.size and idx[0] < first:
            first, cause = int(idx[0]), name
    if cause is None and ended != 'complete':
        cause = {'stop_disp': 'slip', 'stop_lost': 'lost', 'unstable': 'unstable'}.get(ended, ended)
        first = n - 1
    held = cause is None
    res.update(held=held, cause='' if held else cause)
    if not held:
        tf = float(t[first])
        res['t_fail_s'] = tf
        res['phase'] = 'motion' if tf <= Tm + 1e-9 else ('transient' if tf <= Tm + settle + 1e-9 else 'delayed')
        if res['phase'] == 'motion':
            a = float(rows[first, c['acmd']])
            res['load_sign'] = 0 if abs(a) < 0.05 * max(k, 1e-9) else (-1 if a > 0 else 1)
    disp = rows[:, c['disp']]
    res['max_disp_mm'] = float(disp.max())
    res['resid_disp_mm'] = float(disp[after].max()) if after.any() else math.nan
    res['max_rot_deg'] = float(rows[:, c['rot']].max())
    res['max_tilt_deg'] = float(rows[:, c['tilt']].max())
    res['max_hand_mm'] = float(rows[:, c['hand']].max())
    res['end_dx_mm'], res['end_dy_mm'], res['end_dz_mm'] = (float(rows[-1, c[kk]]) for kk in ('dx', 'dy', 'dz'))
    res['min_side_N'] = float(rows[:, c['side_min']].min())
    res['max_ext_N'] = float(rows[:, c['ext']].max())
    lcols = [i for i, nm in enumerate(cols) if nm.startswith('f_')]
    res['_end_lf'] = rows[-1, lcols].astype(float)
    # 팔 운동 확인: 운동 + settle 구간 (실패했으면 실패 row까지)
    win = tstart < Tm + settle - 1e-9
    if not held:
        win &= np.arange(n) <= first
    if k <= 0:
        res['valid'] = True
        res['acc_ratio_pos'] = res['acc_ratio_neg'] = math.nan
        res['max_track_mm'] = float(rows[win, c['track']].max()) if win.any() else 0.0
        return res
    reasons = []
    rp = rn = math.nan
    if win.any():
        # 가속도 비는 명령이 최댓값(0.95 k)에 닿은 뒤 30 ms 이상 지나서 실패한(또는 버틴) 부호만 본다
        t_ok = math.inf if held else res['t_fail_s'] - 0.03
        hi = np.flatnonzero(win & (rows[:, c['acf_max']] >= 0.95 * k))
        lo = np.flatnonzero(win & (rows[:, c['acf_min']] <= -0.95 * k))
        cmax, cmin = float(rows[win, c['acf_max']].max()), float(rows[win, c['acf_min']].min())
        mmax, mmin = float(rows[win, c['au_max']].max()), float(rows[win, c['au_min']].min())
        if hi.size and t[hi[0]] <= t_ok:
            rp = mmax / cmax
        if lo.size and t[lo[0]] <= t_ok:
            rn = mmin / cmin
        tol = max(crit['acc_tol'], crit.get('acc_abs_g', 0.0) / k)   # 작은 가속도에서는 절대 오차 [g]로
        for r in (rp, rn):
            if math.isfinite(r) and abs(r - 1.0) > tol:
                reasons.append('accel')
                break
        res['max_aperp_g'] = float(rows[win, c['aperp']].max())
        res['max_track_mm'] = float(rows[win, c['track']].max())
        res['max_palm_rot_deg'] = float(rows[win, c['prot']].max())
        res['max_sat'] = float(rows[win, c['sat']].max())
        res['max_arm_contact_N'] = float(rows[win, c['armhit']].max())
        if res['max_track_mm'] > crit['track_mm']:
            reasons.append('track')
        if res['max_palm_rot_deg'] > crit['palm_rot_deg']:
            reasons.append('palm_rot')
        if res['max_sat'] >= crit['sat']:
            reasons.append('saturation')
        if res['max_arm_contact_N'] > crit['arm_hit_n']:
            reasons.append('arm_contact')
    res['acc_ratio_pos'], res['acc_ratio_neg'] = rp, rn
    res['valid'] = not reasons
    res['arm_reason'] = '|'.join(reasons)
    return res


def score_all(out_dir, cells_dir, meta, crit, mags, dirs, designs_filter, w_tip, design_index):
    settle = meta['sim']['settle_s']
    trace_files = sorted(p for p in out_dir.glob('*/traces/*.npz') if not p.name.endswith('.tmp.npz'))
    per_design = {}
    stale = 0
    for tp in trace_files:
        design = tp.parent.parent.name
        if designs_filter and design not in designs_filter:
            continue
        try:
            tmeta, cols, trials = load_trace(tp)
        except Exception as e:
            print(f'[WARN] cannot read {tp}: {e}')
            continue
        if tmeta.get('sim_hash') != meta['sim_hash']:
            stale += 1
            continue
        per_design.setdefault(design, []).append((tmeta, cols, trials))

    pick = read_csv(cells_dir / 'rb5_pick_results.csv') or []
    attempted, rb5_pass = {}, {}
    for r in pick:
        dsg = r.get('design_id')
        attempted[dsg] = attempted.get(dsg, 0) + 1
        if str(r.get('success')).strip() in ('1', 'True', 'true'):
            rb5_pass[dsg] = rb5_pass.get(dsg, 0) + 1
    nopass = {dsg for dsg in attempted if not rb5_pass.get(dsg)}
    if designs_filter:
        nopass &= set(designs_filter)
    designs = sorted(set(per_design) | nopass)
    dvars, dvar_cols = load_design_vars(design_index, designs)

    all_rows, dir_stats = [], []
    totals = {'cells': 0, 'success': 0, 'zero_fail': 0, 'incomplete': 0, 'trials': 0,
              'arm_limited': 0, 'infeasible': 0}
    for design in designs:
        cells = sorted(per_design.get(design, []), key=lambda x: (x[0].get('r_mm') or 0, x[0].get('y_mm') or 0))
        trial_rows, cell_rows = [], []
        for tmeta, cols, trials in cells:
            Tm = tmeta['motion_s']
            lfctx = _LF(tmeta['link_side'])
            thr = tmeta['sim']['lost_n']
            base = {'design': design, 'cell': tmeta['cell'], 'r_mm': tmeta.get('r_mm'),
                    'y_mm': tmeta.get('y_mm'), 'workspace_grasp_type': tmeta.get('workspace_grasp_type'),
                    'start_type': tmeta['start_type']}
            z = score_trial(*trials['z0'], cols, crit, Tm, settle, 0.0) if 'z0' in trials else None
            zero_ok = bool(z and z['held'])
            group = tmeta['start_type'] if tmeta['start_type'] in ('env', 'tip') else 'other'
            if z is None:
                status = 'incomplete'
            elif not zero_ok and crit['zero_fail'] == 'fail':
                status = 'zero_fail'
            else:
                status = 'success'
            levels, censored, complete = {}, {}, z is not None
            n_lim = n_inf = 0
            for dn in dirs:
                lvl, broke, cen = 0.0, False, ''
                for k in mags:
                    key = trial_key(dn, k)
                    if key not in trials:
                        if not broke and not crit['monotonic']:
                            complete = False
                        broke = True
                        continue
                    r = score_trial(*trials[key], cols, crit, Tm, settle, k)
                    end_type = contact_type(r.pop('_end_lf'), lfctx, thr) if '_end_lf' in r else ''
                    plan = (tmeta.get('plans') or {}).get(key, {})
                    load_dir = ''
                    if r.get('load_sign'):
                        load_dir = dir_label(np.array(ALL_DIRS[dn]) * r['load_sign'])
                    n_lim += int(not r['valid'] and r['cause'] != 'infeasible')
                    n_inf += int(r['cause'] == 'infeasible')
                    trial_rows.append({
                        **base, 'dir': dn, 'k_g': k, 'valid': int(r['valid']),
                        'held': int(r['held']) if r['valid'] else '',
                        'held_raw': int(r['held']), 'cause': r['cause'], 'phase': r['phase'],
                        't_fail_s': r['t_fail_s'], 'load_dir_at_fail': load_dir,
                        'ended': r['ended'], 'arm_reason': r['arm_reason'],
                        'acc_ratio_pos': r.get('acc_ratio_pos', math.nan),
                        'acc_ratio_neg': r.get('acc_ratio_neg', math.nan),
                        'max_aperp_g': r.get('max_aperp_g', math.nan),
                        'max_track_mm': r.get('max_track_mm', math.nan),
                        'max_palm_rot_deg': r.get('max_palm_rot_deg', math.nan),
                        'max_sat': r.get('max_sat', math.nan),
                        'max_arm_contact_N': r.get('max_arm_contact_N', math.nan),
                        'pred_force_ratio': plan.get('pred_force_ratio', math.nan),
                        'max_speed_mps': plan.get('max_speed_mps', math.nan),
                        'max_disp_mm': r.get('max_disp_mm', math.nan),
                        'resid_disp_mm': r.get('resid_disp_mm', math.nan),
                        'max_rot_deg': r.get('max_rot_deg', math.nan),
                        'max_tilt_deg': r.get('max_tilt_deg', math.nan),
                        'end_dx_mm': r.get('end_dx_mm', math.nan), 'end_dy_mm': r.get('end_dy_mm', math.nan),
                        'end_dz_mm': r.get('end_dz_mm', math.nan),
                        'min_side_N': r.get('min_side_N', math.nan), 'max_ext_N': r.get('max_ext_N', math.nan),
                        'end_type': end_type,
                        'type_changed': int(bool(r['held'] and end_type and end_type != tmeta['start_type'])),
                        'cell_status': status})
                    if broke:
                        continue
                    if not r['valid']:
                        cen = r['arm_reason'] or r['cause']
                        broke = True
                    elif r['held']:
                        lvl = k
                    else:
                        broke = True
                levels[dn] = lvl
                censored[dn] = cen
            if not complete and status == 'success':
                status = 'incomplete'
            lmin = min(levels.values()) if levels else math.nan
            weakest = '|'.join(dn for dn in dirs if levels.get(dn) == lmin)
            crow = {**base, 'group': group, 'status': status, 'zero_held': int(zero_ok) if z else '',
                    'zero_cause': (z['cause'] if z else ''),
                    'zero_max_disp_mm': z.get('max_disp_mm', math.nan) if z else math.nan,
                    'zero_max_rot_deg': z.get('max_rot_deg', math.nan) if z else math.nan,
                    'level_min': lmin, 'weakest_dir': weakest,
                    'censored_dirs': '|'.join(f'{dn}:{censored[dn]}' for dn in dirs if censored.get(dn)),
                    'arm_limited_trials': n_lim, 'infeasible_trials': n_inf,
                    'mass_kg': tmeta['mass_kg'], 'tension_N': tmeta['tension_N'][0] if tmeta['tension_N'] else '',
                    'motion_s': Tm, 'rb5_four_link_contact': tmeta.get('rb5_four_link_contact')}
            for dn in dirs:
                crow[f'level_{dn}'] = levels.get(dn, math.nan)
            for k in mags:
                crow[f'pass@{k:g}'] = int(status == 'success' and lmin >= k - 1e-12)
            cell_rows.append(crow)

        ddir = out_dir / design
        if cells:
            write_csv(ddir / 'shake_trials.csv', trial_rows)
            write_csv(ddir / 'shake_cells.csv', cell_rows)
        succ = [c for c in cell_rows if c['status'] == 'success']
        ok_cells = {c['cell'] for c in succ}
        row = {'design': design, 'ws_cells_attempted': attempted.get(design, ''),
               'rb5_pass_cells': rb5_pass.get(design, ''), 'traced_cells': len(cell_rows),
               'incomplete': sum(c['status'] == 'incomplete' for c in cell_rows),
               'zero_fail': sum(c['status'] == 'zero_fail' for c in cell_rows),
               'success': len(succ),
               'success_env': sum(c['group'] == 'env' for c in succ),
               'success_tip': sum(c['group'] == 'tip' for c in succ),
               'success_other': sum(c['group'] == 'other' for c in succ),
               'cells_with_censored_dir': sum(bool(c['censored_dirs']) for c in succ)}
        lm = [c['level_min'] for c in succ]
        row['shake_area'] = float(sum(lm))
        row['level_min_median'] = float(np.median(lm)) if lm else math.nan
        row['level_min_mean'] = float(np.mean(lm)) if lm else math.nan
        for k in mags:
            pe = sum(c['level_min'] >= k - 1e-12 and c['group'] == 'env' for c in succ)
            pt = sum(c['level_min'] >= k - 1e-12 and c['group'] != 'env' for c in succ)
            row[f'pass@{k:g}'] = pe + pt
            row[f'cond@{k:g}'] = (pe + pt) / len(succ) if succ else math.nan
            row[f'env_pass@{k:g}'] = pe
            row[f'tip_pass@{k:g}'] = pt
            row[f'J_shake@{k:g}'] = pe + w_tip * pt
        tr_ok = [t for t in trial_rows if t['cell'] in ok_cells]
        tr_valid = [t for t in tr_ok if t['valid']]
        row['trials'] = len(tr_ok)
        row['arm_limited'] = sum(not t['valid'] and t['cause'] != 'infeasible' for t in tr_ok)
        row['infeasible'] = sum(t['cause'] == 'infeasible' for t in tr_ok)
        for ph in PHASES:
            row[f'fail_{ph}'] = sum(t['phase'] == ph for t in tr_valid)
        for cs_ in ('slip', 'rotation', 'lost', 'collision', 'unstable'):
            row[f'cause_{cs_}'] = sum(t['cause'] == cs_ for t in tr_valid)
        row['type_changed'] = sum(t['type_changed'] for t in tr_valid)
        ratios = [t[f'acc_ratio_{s}'] for t in tr_ok for s in ('pos', 'neg')
                  if isinstance(t[f'acc_ratio_{s}'], float) and math.isfinite(t[f'acc_ratio_{s}'])]
        row['acc_ratio_median'] = float(np.median(ratios)) if ratios else math.nan
        row['acc_ratio_min'] = float(np.min(ratios)) if ratios else math.nan
        row['acc_ratio_max'] = float(np.max(ratios)) if ratios else math.nan
        for dn in dirs:
            for k in mags:
                sel = [t['held'] for t in tr_valid if t['dir'] == dn and t['k_g'] == k]
                row[f'held_{dn}@{k:g}'] = float(np.mean(sel)) if sel else math.nan
                for grp in GROUPS:
                    gsel = [t['held'] for t in tr_valid if t['dir'] == dn and t['k_g'] == k and
                            (t['start_type'] if t['start_type'] in ('env', 'tip') else 'other') == grp]
                    if gsel:
                        dir_stats.append({'design': design, 'group': grp, 'dir': dn, 'k_g': k,
                                          'n': len(gsel), 'held': int(sum(gsel)),
                                          'held_rate': float(np.mean(gsel))})
        for col in dvar_cols:
            row[col] = (dvars.get(design) or {}).get(col, '')
        all_rows.append(row)
        totals['cells'] += len(cell_rows)
        totals['success'] += len(succ)
        totals['zero_fail'] += row['zero_fail']
        totals['incomplete'] += row['incomplete']
        totals['trials'] += len(trial_rows)
        totals['arm_limited'] += row['arm_limited']
        totals['infeasible'] += row['infeasible']

    if all_rows:
        first = ['design'] + dvar_cols
        fields = first + [k for k in all_rows[0] if k not in first]
        write_csv(out_dir / 'all_designs.csv', all_rows, fields)
    write_csv(out_dir / 'direction_stats.csv', dir_stats, ['design', 'group', 'dir', 'k_g', 'n', 'held', 'held_rate'])
    ranked = sorted(all_rows, key=lambda r: -r['shake_area'])[:10]
    summary = {'criteria': crit, 'mags_g': mags, 'dirs': dirs, 'w_tip': w_tip, 'totals': totals,
               'stale_traces_skipped': stale, 'design_index': str(design_index) if design_index else None,
               'top10_shake_area': [{'design': r['design'], 'shake_area': r['shake_area'],
                                     'success': r['success'], 'level_min_median': r['level_min_median']}
                                    for r in ranked]}
    with open(out_dir / 'shake_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return totals, stale, ranked


# ----------------------------------------------------------------------------- CLI
def sim_from_args(args):
    return {'profile': args.profile, 'cycles': args.cycles, 'accel_ms': args.accel_ms, 'ramp_ms': args.ramp_ms,
            'arm_ctrl': args.arm_ctrl, 'pre_s': args.pre_s, 'after_s': args.after_s, 'settle_s': args.settle_s,
            'sample_ms': args.sample_ms, 'coarse_s': args.coarse_s, 'check_ms': args.check_ms,
            'filter_ms': args.filter_ms, 'stop_mm': args.stop_mm, 'stop_lost_s': args.stop_lost_s,
            'lost_n': args.lost_n, 'tension': args.tension, 'pen_mm': args.pen_mm}


def crit_from_args(args, monotonic=False):
    return {'slip_mm': args.slip_mm, 'slip_mode': args.slip_mode, 'rot_deg': args.rot_deg,
            'rot_mode': args.rot_mode, 'lost_s': args.lost_s, 'collision_n': args.collision_n,
            'zero_fail': args.zero_fail, 'monotonic': bool(monotonic),
            'acc_tol': args.acc_tol, 'acc_abs_g': args.acc_abs_g, 'track_mm': args.track_mm, 'palm_rot_deg': args.palm_rot_deg,
            'sat': args.sat, 'arm_hit_n': args.arm_hit_n}


def mesh_roots_for(args, cells_dir):
    return [str(Path(p)) for p in ([args.mesh_dir] if args.mesh_dir else [])] + [
        str(HERE / 'assets' / 'rb5_850e' / 'meshes'), str(cells_dir.parent / 'assets' / 'rb5_850e' / 'meshes')]


def default_cells_dir(args):
    return _default_cells_dir(args.cells_dir)


def add_sim_args(p):
    s = p.add_argument_group('흔들기 (바꾸면 다른 실험: --tag 사용)')
    s.add_argument('--profile', choices=('roundtrip', 'p2p'), default='roundtrip',
                   help='roundtrip: +a, -a(2배 길이), +a로 왕복 후 정지 / p2p: +a, -a로 한 번 이동 후 정지')
    s.add_argument('--cycles', type=int, default=1, help='roundtrip 반복 횟수')
    s.add_argument('--accel-ms', type=float, default=100.0, help='가속 단계 길이 t_a [ms]')
    s.add_argument('--ramp-ms', type=float, default=20.0, help='가속도가 바뀌는 데 걸리는 시간 [ms] (jerk 제한)')
    s.add_argument('--arm-ctrl', choices=('ff', 'pos'), default='ff',
                   help='ff: IK + 속도·역동역학 feedforward (기본) / pos: IK 위치 명령만')
    s.add_argument('--tension', type=float, default=None, help='텐던 장력 [N] (기본: 저장 상태 그대로, 10 N)')
    s.add_argument('--pre-s', type=float, default=0.5, help='운동 전 안정화 시간 [s] (B와 같음)')
    s.add_argument('--after-s', type=float, default=10.0, help='운동이 끝난 뒤 유지 시간 [s]')
    s.add_argument('--settle-s', type=float, default=0.5, help='transient 구간 / residual slip 시작 [s]')
    s.add_argument('--sample-ms', type=float, default=5.0, help='운동~settle 구간 기록 간격 [ms]')
    s.add_argument('--coarse-s', type=float, default=0.1, help='그 뒤 기록 간격 [s]')
    s.add_argument('--check-ms', type=float, default=1.0, help='판정값 계산 간격 [ms]')
    s.add_argument('--filter-ms', type=float, default=10.0, help='손 가속도 측정 이동 평균 [ms]')
    s.add_argument('--stop-mm', type=float, default=30.0, help='이만큼 미끄러지면 시행 중단 [mm]')
    s.add_argument('--stop-lost-s', type=float, default=0.1, help='접촉을 이만큼 잃으면 시행 중단 [s]')
    s.add_argument('--lost-n', type=float, default=0.1, help='한쪽 손가락 접촉력 합이 이 값 미만이면 접촉 상실 [N]')
    s.add_argument('--pen-mm', type=float, default=0.5, help='계획 궤적 충돌 확인: 이보다 더 겹치면 infeasible [mm]')
    return s


def add_crit_args(p):
    r = p.add_argument_group('판정 (--rescore로 시뮬레이션 없이 변경 가능)')
    r.add_argument('--slip-mm', type=float, default=5.0, help='손 기준 물체 변위 한계 [mm]')
    r.add_argument('--slip-mode', choices=('total', 'residual'), default='total',
                   help='residual: 운동 후 settle-s 이후의 변위만 셈')
    r.add_argument('--rot-deg', type=float, default=5.0, help='손 기준 물체 회전 한계 [deg]')
    r.add_argument('--rot-mode', choices=('total', 'tilt'), default='total', help='tilt: 원기둥 축 기울기만')
    r.add_argument('--lost-s', type=float, default=0.02, help='접촉 상실 지속 한계 [s]')
    r.add_argument('--collision-n', type=float, default=0.05, help='물체가 손 밖의 물체(팔·받침대)와 닿는 힘 한계 [N]')
    r.add_argument('--zero-fail', choices=('fail', 'keep'), default='fail', help='외력 0 대조 실패 cell 처리')
    r.add_argument('--acc-tol', type=float, default=0.15,
                   help='팔 운동 확인: 측정/명령 최대 가속도 비가 1 ± 이 값 밖이면 arm_limited')
    r.add_argument('--acc-abs-g', type=float, default=0.03,
                   help='팔 운동 확인: 최대 가속도 차이가 이 값 [g] 이하면 비와 상관없이 통과 (작은 k용)')
    r.add_argument('--track-mm', type=float, default=10.0, help='팔 운동 확인: 손 위치 추종 오차 한계 [mm]')
    r.add_argument('--palm-rot-deg', type=float, default=3.0, help='팔 운동 확인: 손 자세 변화 한계 [deg]')
    r.add_argument('--sat', type=float, default=0.999, help='팔 운동 확인: |actuator force|/forcerange 한계')
    r.add_argument('--arm-hit-n', type=float, default=0.5, help='팔 운동 확인: 팔·손이 책상·받침대·팔과 닿는 힘 한계 [N]')
    return r


def main(args):
    cells_dir = default_cells_dir(args)
    out_root = Path(args.out) if args.out else HERE / 'results_C'
    out_dir = (out_root / (f'shake_{args.tag}' if args.tag else 'shake')).resolve()
    if args.clean:
        if args.rescore or args.dry_run:
            raise SystemExit('--clean은 시뮬레이션을 새로 돌릴 때만 씁니다 (--rescore/--dry-run과 같이 못 씀)')
        clean_results(out_root, cells_dir, 'C')
    mesh_roots = mesh_roots_for(args, cells_dir)
    sim = sim_from_args(args)
    meta_path = out_dir / 'shake_meta.json'
    meta = read_json(meta_path)
    if args.rescore:
        if not meta:
            raise SystemExit(f'no results to rescore in {out_dir}')
    else:
        h = sim_hash(sim)
        if meta and meta['sim_hash'] != h and not args.overwrite:
            diff = {k: (meta['sim'].get(k), sim[k]) for k in SIM_KEYS if meta['sim'].get(k) != sim[k]}
            raise SystemExit(f'{out_dir} holds results with other settings {diff} (old, new).\n'
                             f'Use --tag to keep them apart, or --overwrite to replace them.')
        mags_run = sorted(set(args.mags or DEFAULT_MAGS))
        dirs_run = parse_dirs(args.dirs)
        if meta and meta['sim_hash'] == h:
            mags_all = sorted(set(meta['mags']) | set(mags_run))
            dirs_all = meta['dirs'] + [dn for dn in dirs_run if dn not in meta['dirs']]
        else:
            mags_all, dirs_all = mags_run, dirs_run
        meta = {'sim': sim, 'sim_hash': h, 'mags': mags_all, 'dirs': dirs_all,
                'cells_dir': str(cells_dir), 'created': (meta or {}).get('created') or time.strftime('%Y-%m-%d %H:%M:%S'),
                'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'monotonic_used': (meta or {}).get('monotonic_used')}
    if args.rescore:
        mags = sorted(set(args.mags)) if args.mags else meta['mags']
        dirs = parse_dirs(args.dirs) if args.dirs else meta['dirs']
    else:
        mags, dirs = meta['mags'], meta['dirs']
    crit = crit_from_args(args, args.monotonic or meta.get('monotonic_used'))
    if args.monotonic and not args.rescore:
        meta['monotonic_used'] = True
    if crit['slip_mm'] > meta['sim']['stop_mm']:
        print(f'[WARN] --slip-mm {crit["slip_mm"]} > stop {meta["sim"]["stop_mm"]} mm: trials stopped there count as slip.')

    if not args.rescore:
        cells = select_cells(cells_dir, args.designs, args.only, args.limit)
        warn_old_xml(cells, 'C')
        keys = ['z0'] + [trial_key(dn, k) for dn in dirs_run for k in mags_run]
        jobs = []
        for cdir in cells:
            design, cell = cell_parts(cdir)
            tp = out_dir / design / 'traces' / f'{cell}.npz'
            need = keys
            if tp.is_file() and not args.overwrite:
                try:
                    tm, _, tr = load_trace(tp)
                    if tm.get('sim_hash') == meta['sim_hash']:
                        need = [k for k in keys if k not in tr]
                except Exception:
                    pass
            if need:
                jobs.append({'cell_id': f'{design}/{cell}', 'design': design,
                             'cell': cell, 'cell_dir': str(cdir), 'trace_path': str(tp),
                             'keys': keys, 'sim': sim, 'sim_hash': meta['sim_hash'],
                             'mesh_roots': mesh_roots, 'overwrite': args.overwrite,
                             'monotonic_crit': crit if crit['monotonic'] else None,
                             'allow_missing_collision': args.allow_missing_collision})
        n_designs = len({cell_parts(c)[0] for c in cells})
        print(f'[C] cells: {cells_dir} ({"XML, <key lifted>" if is_xml_cells_dir(cells_dir) else "folders, state.npz"})')
        Tm = (sum(w for _, w in accel_segments(sim['profile'], sim['cycles'])) * sim['accel_ms'] + sim['ramp_ms']) / 1000
        print(f'[C] {len(cells)} cells in {n_designs} designs; {len(jobs)} to run '
              f'({len(cells) - len(jobs)} already done). Per cell: zero control + {len(dirs_run)} dirs x '
              f'{len(mags_run)} mags = {len(keys)} trials of {Tm:.3f} s motion + {sim["after_s"]:g} s hold.')
        print(f'[C] output: {out_dir}')
        if args.dry_run:
            est = len(jobs) * len(keys) * (0.06 * (Tm + sim['after_s']) + 0.3) / max(1, args.workers)
            print(f'[C] dry run. Rough estimate {hms(est)} with {args.workers} workers.')
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        errors = []
        t0 = time.time()
        done = 0

        def report(res):
            nonlocal done
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            if res['error']:
                errors.append(res)
                print(f'[{done}/{len(jobs)}] {res["cell_id"]}  ERROR (see _errors.log)', flush=True)
                with open(out_dir / '_errors.log', 'a', encoding='utf-8') as f:
                    f.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} {res["cell_id"]}\n{res["error"]}\n')
            else:
                print(f'[{done}/{len(jobs)}] {res["cell_id"]}  {res["ran"]} trials  {res["sec"]:.1f} s  '
                      f'elapsed {hms(el)}  ETA {hms(eta)}', flush=True)

        if args.workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                report(simulate_cell(j))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(simulate_cell, j) for j in jobs]
                try:
                    for fu in as_completed(futs):
                        report(fu.result())
                except KeyboardInterrupt:
                    print('[C] interrupted; finished cells are saved. Run the same command to continue.')
                    for fu in futs:
                        fu.cancel()
                    raise
        if errors:
            print(f'[C] {len(errors)} cells failed to run; details in {out_dir / "_errors.log"}')

    design_index = find_design_index(cells_dir, args.designs_index)
    totals, stale, ranked = score_all(out_dir, cells_dir, meta, crit, mags, dirs,
                                      args.designs if args.rescore else None, args.w_tip, design_index)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({**meta, 'last_score': {'criteria': crit, 'mags': mags, 'dirs': dirs,
                                          'time': time.strftime('%Y-%m-%d %H:%M:%S')}}, f, indent=2)
    print(f'[C] scored {totals["cells"]} cells: success {totals["success"]}, zero_fail {totals["zero_fail"]}, '
          f'incomplete {totals["incomplete"]}; arm_limited trials {totals["arm_limited"]}, '
          f'infeasible {totals["infeasible"]}' + (f'; skipped {stale} traces with other settings' if stale else ''))
    if ranked:
        print('[C] top designs by shake_area: ' + ', '.join(f'{r["design"]} {r["shake_area"]:g}' for r in ranked[:5]))
    print(f'[C] tables in {out_dir}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(description='실험 C: 들어 올린 뒤 팔을 흔들어(가속·감속) 파지 유지 판정 (grasp_cells 상태에서 시작)')
    g = p.add_argument_group('대상과 출력')
    g.add_argument('--cells-dir', default=None,
                   help='grasp_cells_xml(XML 파일들) 또는 grasp_cells(폴더) (기본: 이 파일 옆 grasp_cells_xml, 없으면 grasp_cells)')
    g.add_argument('--designs', nargs='+', default=None, help='설계 이름 (기본: 전부)')
    g.add_argument('--only', nargs='+', default=None,
                   help='특정 cell만, 예: design0001/r060.0_y015.0 또는 design0001_r60y15')
    g.add_argument('--limit', type=int, default=0, help='설계마다 (r, y) 순서로 고르게 N개만 (빠른 시험)')
    g.add_argument('--out', default=None, help='결과 상위 폴더 (기본: 이 파일 옆 results_C)')
    g.add_argument('--tag', default=None, help='결과 폴더 shake_<tag> (설정이 다른 실험을 따로 보관)')
    g.add_argument('--overwrite', action='store_true', help='설정이 다른 기존 결과를 덮어씀')
    g.add_argument('--clean', action='store_true', help='시작 전에 결과 상위 폴더(results_C) 안의 기존 파일을 모두 지움')
    g.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    g.add_argument('--dry-run', action='store_true', help='cell·시행 수만 출력')
    g.add_argument('--mesh-dir', default=None, help='scene.xml의 mesh 절대경로가 없을 때 찾을 …/assets/rb5_850e/meshes')
    g.add_argument('--allow-missing-collision', action='store_true', help='충돌 mesh를 못 찾아도 빼고 진행 (확인용)')
    g.add_argument('--designs-index', default=None, help='설계 변수 표 designs_index.csv (기본: 자동 탐색)')
    s = add_sim_args(p)
    s.add_argument('--mags', type=float, nargs='+', default=None, help='손 가속도 최댓값 k [g] (기본 0.5 1)')
    s.add_argument('--dirs', nargs='+', default=None,
                   help='손이 처음 움직이는 방향: axes(6, 기본) | edges(12) | all(18) | px nx py ny pz nz …')
    r = add_crit_args(p)
    r.add_argument('--rescore', action='store_true', help='시뮬레이션 없이 trace로 다시 판정')
    r.add_argument('--monotonic', action='store_true', help='한 방향에서 실패하면 더 큰 가속도를 실패로 봄(시행 생략)')
    r.add_argument('--w-tip', type=float, default=0.2, help='J_shake의 tip 가중치')
    return p


if __name__ == '__main__':
    sys.exit(main(build_parser().parse_args()))
