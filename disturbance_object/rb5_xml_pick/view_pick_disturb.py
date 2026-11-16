#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성공 cell 하나를 RB5로 집어 들어 올리고, hold-s(기본 5 s) 뒤에 물체에 펄스 외란을 주는 viewer.

  python view_pick_disturb.py design0001/r060.0_y015.0                       # 집기부터, py 방향 1 mg
  python view_pick_disturb.py design0001/r060.0_y015.0 --dir px --mag 2      # 방향·크기
  python view_pick_disturb.py design0001/r060.0_y015.0 --from saved          # 저장된 '들어 올린 직후' 상태에서 시작
  python view_pick_disturb.py design0001/r060.0_y015.0 --video b_px2.mp4     # 창 없이 영상으로 저장
  python view_pick_disturb.py design0001/r060.0_y015.0 --headless            # 판정만 출력

--from pick (기본): 장면의 task_home 자세에서 run_rb5_xml.py와 같은 단계와 시간
  (settle → lower → approach → settle → close → grasp hold → lift)으로 다시 집는다.
  팔은 palm 목표 경로를 IK로 따라가고, 텐던 장력 ramp와 열림 보조(PD)도 같은 설정을 쓴다.
  rb5_pick_config.json이 grasp_cells/에 있으면 그 값을 쓴다.
--from saved: state.npz를 복원해 바로 hold부터 시작한다. 실험 B(disturb_impulse_rb5.py)와 같은 출발점이다
  (--hold-s 0.5이면 B의 시행과 같다).

viewer: Space = 일시정지/재생. 펄스 전후는 --slowmo 배속으로 느리게 보여 준다(물리 시간 간격은 그대로).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import disturb_impulse_rb5 as B  # noqa: E402  (같은 폴더)

# run_rb5_workspace.py 기본 설정 (grasp_cells/rb5_pick_config.json이 있으면 그 값으로 덮어씀)
PICK_DEFAULTS = {
    'approach_distance': 0.12, 'settle_time': 0.6, 'move_time': 3.0, 'approach_time': 2.5,
    'preclose_time': 0.5, 'ramp_time': 1.5, 'grasp_hold': 1.0, 'lift_time': 3.0, 'lift': 0.15,
    'ik_interval': 0.01, 'open_kp': 1.0, 'open_kd': 0.03, 'open_torque_limit': 0.3,
    'open_release_time': 0.3, 'tension': 10.0, 'contact_threshold': 0.1, 'success_ratio': 0.8,
}
ARM = ('base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3')


def min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Episode:
    """한 cell: (집기 →) 들어 올림 → hold → 펄스 → 유지. step()으로 1 timestep씩 진행."""

    def __init__(self, a):
        import mujoco
        self.mj = mujoco
        self.a = a
        cells_dir = Path(a.cells_dir) if a.cells_dir else (
            HERE / 'grasp_cells' if (HERE / 'grasp_cells').is_dir() else HERE.parent / 'grasp_cells')
        cdir = Path(a.cell)
        if not (cdir / 'state.npz').is_file():
            cdir = cells_dir / a.cell
        if not (cdir / 'scene.xml').is_file():
            raise SystemExit(f'no scene.xml in {cdir}')
        self.cdir = cdir
        self.summary = B.read_json(cdir / 'summary.json', {}) or {}
        cfg = dict(PICK_DEFAULTS)
        for p in (cdir.parent.parent / 'rb5_pick_config.json', cells_dir / 'rb5_pick_config.json'):
            got = B.read_json(p)
            if got:
                cfg.update({k: got[k] for k in PICK_DEFAULTS if k in got})
                break
        self.cfg = cfg
        mesh_roots = ([a.mesh_dir] if a.mesh_dir else []) + [
            str(HERE / 'assets' / 'rb5_850e' / 'meshes'), str(cells_dir.parent / 'assets' / 'rb5_850e' / 'meshes')]
        xml, dropped, _ = B.load_scene_xml(cdir / 'scene.xml', mesh_roots, a.allow_missing_collision)
        if dropped:
            print(f'[WARN] mesh files not found, not drawn: {len(dropped)} (give --mesh-dir)')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        self.m, self.d = m, d
        self.ctx = ctx = B.Ctx(m, self.summary)
        self.dt = m.opt.timestep
        self.mg = float(m.body_subtreemass[ctx.obj] * np.linalg.norm(m.opt.gravity))
        self.motor = ctx.motors[0] if ctx.motors else None
        tension = a.tension if a.tension is not None else float(
            self.summary.get('tension_per_motor_N') or cfg['tension'])
        self.tension = tension
        if self.motor is not None:
            self.ctrl_T = -tension / m.actuator_gear[self.motor, 0]
            lo, hi = m.actuator_ctrlrange[self.motor]
            m.actuator_ctrlrange[self.motor] = [min(lo, 0.0, self.ctrl_T), max(hi, self.ctrl_T)]
        self.arm_q = [m.jnt_qposadr[m.joint(n).id] for n in ARM]
        self.arm_v = [m.jnt_dofadr[m.joint(n).id] for n in ARM]
        self.arm_u = [m.actuator(n).id for n in ARM]
        self.hand = []   # (qposadr, dofadr, open angle)
        for n, q in (self.summary.get('initial_hand_qpos') or self.summary.get('hand_joints') or {}).items():
            j = m.joint(f'hand__{n}').id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'hand__{n}') >= 0 \
                else m.joint(n).id
            self.hand.append((m.jnt_qposadr[j], m.jnt_dofadr[j], float(q)))
        T = a.pulse_ms / 1000.0
        self.T = T
        if a.start == 'pick':
            mujoco.mj_resetDataKeyframe(m, d, m.key('task_home').id)
            mujoco.mj_forward(m, d)
            self._setup_pick()
            t_up = self.t_lift_end
        else:
            with np.load(cdir / 'state.npz', allow_pickle=False) as z:
                mujoco.mj_setState(m, d, np.array(z['integration_state']), int(z['integration_spec']))
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            if self.motor is not None:
                d.ctrl[self.motor] = self.ctrl_T
            mujoco.mj_forward(m, d)
            self.stages = []
            t_up = d.time
        self.t_up = t_up
        self.t_dist = t_up + a.hold_s
        self.t_end = self.t_dist + T + a.after_s
        self.obj_z0 = float(self.summary.get('initial_object_z_m') or d.xpos[ctx.obj][2])
        if a.start == 'pick':
            self.obj_z0 = float(d.xpos[ctx.obj][2])
        # monitors
        self.lf_hist = []
        self.ref = None
        self.fworld = None
        self.fail = None
        self.max = {'disp': 0.0, 'rot': 0.0, 'tilt': 0.0}
        self.streak = [0, 0]
        self.now = {'disp': 0.0, 'rot': 0.0, 'lf': np.zeros(ctx.nl), 'ext': 0.0}
        self.lift_at_dist = None
        self.type_at_dist = None
        self.end_type = None

    # ------------------------------------------------------------------ pick (재현)
    def _setup_pick(self):
        m, d, ctx, cfg = self.m, self.d, self.ctx, self.cfg
        a = math.radians(float(self.summary.get('forward_angle_deg', 180.0)))
        R_star = np.array(self.summary.get('hand_rotation_world') or d.xmat[ctx.palm].reshape(3, 3).tolist())
        self.R_star = R_star
        fwd = R_star @ np.array([math.cos(a), math.sin(a), 0.0])
        gl = np.array(self.summary.get('source_grasp_local_m') or
                      list((self.summary.get('cell') or {}).get('grasp_local_xy_m', [-0.076, 0.0])) + [0.0])
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

    # ------------------------------------------------------------------ main step
    def stage(self, t=None):
        t = self.d.time if t is None else t
        for name, t0, t1 in self.stages:
            if t0 <= t < t1:
                return name
        if t < self.t_dist:
            return f'hold {self.a.hold_s:g} s'
        if t < self.t_dist + self.T:
            return f'PULSE {self.a.dir} {self.a.mag:g} mg'
        if t < self.t_end:
            return 'after pulse'
        return 'done'

    def step(self):
        mj, m, d, ctx, a = self.mj, self.m, self.d, self.ctx, self.a
        t = d.time
        if a.start == 'pick':
            self._pick_controls()
        # 외란 시작 시점: 기준 자세와 힘 방향(손 좌표계) 고정
        if self.ref is None and t >= self.t_dist - 1e-9:
            p, Rr, ax, pp, lf, ext, _ = B.measure(m, d, ctx)
            self.ref = (p, Rr, ax)
            Rp = d.xmat[ctx.palm].reshape(3, 3)
            self.fworld = Rp @ ctx.H @ np.array(B.ALL_DIRS[a.dir])
            self.lift_at_dist = 1000.0 * (d.xpos[ctx.obj][2] - self.obj_z0)
            lfm = np.mean(self.lf_hist, axis=0) if self.lf_hist else lf
            self.type_at_dist = B.contact_type(lfm, ctx, self.cfg['contact_threshold'])
        if self.ref is not None and t < self.t_dist + self.T - 1e-12:
            amp = math.sin(math.pi * (t - self.t_dist + 0.5 * self.dt) / self.T) if a.shape == 'halfsine' else 1.0
            d.xfrc_applied[ctx.obj, :3] = self.fworld * a.mag * self.mg * amp
        else:
            d.xfrc_applied[ctx.obj, :3] = 0.0
        mj.mj_step(m, d)
        self._monitor()

    def _monitor(self):
        m, d, ctx, a = self.m, self.d, self.ctx, self.a
        p, Rr, ax, pp, lf, ext, _ = B.measure(m, d, ctx)
        self.now['lf'] = lf
        self.now['ext'] = ext
        if self.ref is None:
            if d.time > self.t_dist - 0.1:
                self.lf_hist.append(lf)
            return
        p0, R0, ax0 = self.ref
        disp = 1000.0 * float(np.linalg.norm(p - p0))
        rot = B._angle_deg((np.trace(R0.T @ Rr) - 1.0) / 2.0)
        tilt = B._angle_deg(float(ax @ ax0))
        self.now['disp'], self.now['rot'] = disp, rot
        for k, v in (('disp', disp), ('rot', rot), ('tilt', tilt)):
            self.max[k] = max(self.max[k], v)
        s = (lf[ctx.side0].sum(), lf[ctx.side1].sum())
        for i in (0, 1):
            self.streak[i] = self.streak[i] + 1 if s[i] < self.cfg['contact_threshold'] else 0
        if self.fail is None and d.time <= self.t_end + 1e-9:
            tr = d.time - self.t_dist
            rv = tilt if a.rot_mode == 'tilt' else rot
            cause = ('slip' if disp >= a.slip_mm else 'rotation' if rv >= a.rot_deg else
                     'lost' if max(self.streak) * self.dt >= a.lost_s - 1e-12 else
                     'collision' if ext >= a.collision_n else None)
            if cause:
                ph = 'pulse' if tr <= self.T + 1e-9 else ('transient' if tr <= self.T + a.settle_s + 1e-9 else 'delayed')
                self.fail = (cause, tr, ph)
        if d.time >= self.t_end - 1e-9 and self.end_type is None:
            self.end_type = B.contact_type(lf, ctx, self.cfg['contact_threshold'])

    def arrow(self):
        """펄스 중 그릴 화살표 (꼬리, 머리): 물체 밖에서 물체 표면을 향해."""
        if self.ref is None or not (self.t_dist <= self.d.time < self.t_dist + self.T + 0.15):
            return None
        c = self.d.xpos[self.ctx.obj]
        r = float(self.m.geom_size[self.ctx.obj_geom][0]) if self.m.geom_type[self.ctx.obj_geom] in (3, 5) else 0.03
        L = min(0.15, 0.04 + 0.03 * self.a.mag)
        head = c - self.fworld * (r + 0.005)
        return head - self.fworld * L, head

    def texts(self):
        d, a = self.d, self.a
        lf = self.now['lf']
        left = ['stage', 'time', 'tension', 'link forces [N]']
        right = [self.stage(), f'{d.time:6.2f} s', f'{self.tension:g} N',
                 ' '.join(f'{x:.2f}' for x in lf)]
        if self.ref is not None:
            left += ['disturbance', 'slip in hand', 'rotation']
            right += [f'{a.dir} {a.mag:g} mg = {a.mag * self.mg:.2f} N x {a.pulse_ms:g} ms',
                      (f'{self.now["disp"]:.2f} mm (max {self.max["disp"]:.2f})' if self.max['disp'] <= 30 else 'dropped (> 30 mm)'),
                      f'{self.now["rot"]:.2f} deg (max {self.max["rot"]:.2f})']
            left.append('verdict')
            if self.fail:
                right.append(f'FAIL {self.fail[0]} at +{self.fail[1]:.3f} s ({self.fail[2]})')
            else:
                right.append('HELD' if d.time >= self.t_end - 1e-9 else 'holding')
        else:
            lift = 1000.0 * (d.xpos[self.ctx.obj][2] - self.obj_z0)
            left.append('object lift')
            right.append(f'{lift:.1f} mm')
        return '\n'.join(left), '\n'.join(right)

    def report(self):
        a = self.a
        lines = []
        if a.start == 'pick':
            ok = (self.lift_at_dist or 0) >= self.cfg['success_ratio'] * 1000 * self.cfg['lift']
            lines.append(f'[PICK] reproduced pick: lift {self.lift_at_dist:.1f} mm at disturbance time, '
                         f'contact {self.type_at_dist}' + ('' if ok else '  <- NOT LIFTED (pick did not reproduce)'))
            saved = (self.summary.get('lifted_state') or {}).get('lift_m')
            if saved is not None:
                lines.append(f'       saved cell: lift {1000 * saved:.1f} mm, four_link_contact '
                             f'{self.summary.get("four_link_contact")}; max IK error {self.ik_err:.1e}')
        else:
            lines.append(f'[SAVED] lift {self.lift_at_dist:.1f} mm, contact {self.type_at_dist} at disturbance time')
        imp = a.mag * self.mg * self.T * (2 / math.pi if a.shape == 'halfsine' else 1.0)
        lines.append(f'[DIST] {a.dir} {a.mag:g} mg = {a.mag * self.mg:.2f} N, {a.pulse_ms:g} ms {a.shape} '
                     f'(impulse {imp:.3f} N s) at t = {self.t_dist:.2f} s')
        mx = dict(self.max)
        if mx['disp'] > 30.0:
            mx['disp'] = float('inf')
        if self.fail:
            tail = ('object dropped' if math.isinf(mx['disp']) else
                    f'max slip {mx["disp"]:.2f} mm, max rot {mx["rot"]:.2f} deg')
            lines.append(f'[RESULT] FAIL: {self.fail[0]} at +{self.fail[1]:.3f} s ({self.fail[2]}); {tail}')
        else:
            lines.append(f'[RESULT] HELD for {a.after_s:g} s after the pulse: max slip {mx["disp"]:.2f} mm, '
                         f'max rot {mx["rot"]:.2f} deg, end contact {self.end_type}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------- drawing helpers
def add_arrow(mujoco, scn, tail, head, rgba=(0.92, 0.41, 0.2, 1.0), width=0.009):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array(rgba, dtype=np.float32))
    if hasattr(mujoco, 'mjv_connector'):
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(tail, float), np.asarray(head, float))
    else:  # MuJoCo < 3.2
        mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, *tail, *head)
    scn.ngeom += 1


def slow_factor(ep, t):
    a = ep.a
    if a.slowmo >= 1.0:
        return 1.0
    return a.slowmo if ep.t_dist - 0.05 <= t < ep.t_dist + ep.T + a.slowmo_after else 1.0


def set_camera(cam, ep):
    a = ep.a
    cam.lookat[:] = ep.d.xpos[ep.ctx.obj] if a.start == 'saved' else np.array(a.lookat)
    cam.distance = a.camera_distance
    cam.azimuth = a.azimuth
    cam.elevation = a.elevation


def run_viewer(ep):
    import mujoco
    import mujoco.viewer
    a = ep.a
    paused = {'v': False}

    def key_cb(key):
        if key == 32:  # Space
            paused['v'] = not paused['v']

    try:
        v = mujoco.viewer.launch_passive(ep.m, ep.d, key_callback=key_cb, show_left_ui=False, show_right_ui=False)
    except TypeError:
        v = mujoco.viewer.launch_passive(ep.m, ep.d, key_callback=key_cb)
    with v:
        set_camera(v.cam, ep)
        has_text = hasattr(v, 'set_texts')
        fps = a.fps
        last = time.time()
        printed = False
        while v.is_running():
            if not paused['v'] and ep.d.time < ep.t_end:
                target = ep.d.time + a.speed * slow_factor(ep, ep.d.time) / fps
                while ep.d.time < min(target, ep.t_end):
                    ep.step()
            with v.lock():
                v.user_scn.ngeom = 0
                arr = ep.arrow()
                if arr:
                    add_arrow(mujoco, v.user_scn, *arr)
            if has_text:
                l, r = ep.texts()
                v.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, l, r)])
            v.sync()
            if ep.d.time >= ep.t_end and not printed:
                print(ep.report())
                print('[VIEW] finished; the window stays open until you close it.')
                printed = True
            sleep = 1.0 / fps - (time.time() - last)
            if sleep > 0:
                time.sleep(sleep)
            last = time.time()
    if not printed:
        print(ep.report())


def run_video(ep):
    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit('video needs imageio + imageio-ffmpeg: python -m pip install -r requirements-video.txt')
    a = ep.a
    w = min(a.width, ep.m.vis.global_.offwidth)
    h = min(a.height, ep.m.vis.global_.offheight)
    renderer = mujoco.Renderer(ep.m, h, w)
    cam = mujoco.MjvCamera()
    set_camera(cam, ep)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
    writer = imageio.get_writer(a.video, fps=a.fps, **({'quality': 8} if a.video.endswith('.mp4') else {}))
    n = 0
    while ep.d.time < ep.t_end - 1e-9:
        target = ep.d.time + a.speed * slow_factor(ep, ep.d.time) / a.fps
        while ep.d.time < min(target, ep.t_end):
            ep.step()
        if a.start == 'saved':
            cam.lookat[:] = ep.d.xpos[ep.ctx.obj]
        renderer.update_scene(ep.d, cam)
        arr = ep.arrow()
        if arr:
            add_arrow(mujoco, renderer.scene, *arr)
        img = renderer.render()
        if Image is not None:
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            l, r = ep.texts()
            y = 8
            for ll, rr in zip(l.split('\n'), r.split('\n')):
                dr.text((10, y), f'{ll}: {rr}', fill=(20, 20, 20))
                y += 14
            img = np.asarray(im)
        writer.append_data(img)
        n += 1
    writer.close()
    print(ep.report())
    print(f'[VIDEO] {a.video}: {n} frames at {a.fps} fps')


def main():
    p = argparse.ArgumentParser(description='성공 cell: RB5로 집어 올리고 hold-s 뒤 펄스 외란을 주는 viewer')
    p.add_argument('cell', help='design0001/r060.0_y015.0 (grasp_cells 기준) 또는 cell 폴더 경로')
    p.add_argument('--cells-dir', default=None, help='grasp_cells 폴더 (기본: 이 파일 옆)')
    p.add_argument('--from', dest='start', choices=('pick', 'saved'), default='pick',
                   help='pick: 집기부터 다시 / saved: 저장된 들어 올린 직후 상태에서')
    p.add_argument('--hold-s', type=float, default=5.0, help='들어 올린 뒤 외란까지 시간 [s]')
    p.add_argument('--dir', default='py', help='외란 방향 (손 좌표): px nx py ny pz nz, 대각선 pxpy …')
    p.add_argument('--mag', type=float, default=1.0, help='외란 크기 k [mg]')
    p.add_argument('--pulse-ms', type=float, default=50.0, help='펄스 길이 [ms]')
    p.add_argument('--shape', choices=('rect', 'halfsine'), default='rect')
    p.add_argument('--after-s', type=float, default=10.0, help='펄스 후 관찰 시간 [s]')
    p.add_argument('--tension', type=float, default=None, help='텐던 장력 [N] (기본: cell 값, 10 N)')
    p.add_argument('--slip-mm', type=float, default=5.0)
    p.add_argument('--rot-deg', type=float, default=5.0)
    p.add_argument('--rot-mode', choices=('total', 'tilt'), default='total')
    p.add_argument('--lost-s', type=float, default=0.02)
    p.add_argument('--collision-n', type=float, default=0.05)
    p.add_argument('--settle-s', type=float, default=0.5, help='transient 구간 길이 [s]')
    p.add_argument('--mesh-dir', default=None, help='…/assets/rb5_850e/meshes (scene.xml의 mesh 경로가 없을 때)')
    p.add_argument('--allow-missing-collision', action='store_true')
    g = p.add_argument_group('화면')
    g.add_argument('--headless', action='store_true', help='창 없이 판정만 출력')
    g.add_argument('--video', default=None, help='창 대신 영상 파일로 (.mp4 / .gif)')
    g.add_argument('--speed', type=float, default=1.0, help='재생 속도')
    g.add_argument('--slowmo', type=float, default=0.1, help='펄스 전후 재생 배속 (1이면 끔)')
    g.add_argument('--slowmo-after', type=float, default=0.4, help='펄스 후 느리게 보여 줄 시간 [s]')
    g.add_argument('--fps', type=int, default=60)
    g.add_argument('--width', type=int, default=1280)
    g.add_argument('--height', type=int, default=720)
    g.add_argument('--camera-distance', type=float, default=0.85)
    g.add_argument('--azimuth', type=float, default=135.0)
    g.add_argument('--elevation', type=float, default=-20.0)
    g.add_argument('--lookat', type=float, nargs=3, default=[0.5, 0.0, 0.17], help='--from pick 카메라 중심 [m]')
    a = p.parse_args()
    if a.dir not in B.ALL_DIRS:
        raise SystemExit(f'--dir must be one of {", ".join(B.ALL_DIRS)}')
    ep = Episode(a)
    print(f'[CELL] {ep.cdir}  start={a.start}  hold {a.hold_s:g} s -> {a.dir} {a.mag:g} mg '
          f'({a.mag * ep.mg:.2f} N) for {a.pulse_ms:g} ms, then {a.after_s:g} s')
    if a.headless:
        while ep.d.time < ep.t_end - 1e-9:
            ep.step()
        print(ep.report())
    elif a.video:
        run_video(ep)
    else:
        run_viewer(ep)
    return 0 if ep.fail is None else 1


if __name__ == '__main__':
    sys.exit(main())
