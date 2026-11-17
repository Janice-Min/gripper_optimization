#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 C viewer — grasp_cells의 cell 하나에서 팔 흔들기 시행 하나를 MuJoCo viewer로 본다.

disturb_shake_rb5.py와 같은 코드(CellSim, plan_trial, run_trial, score_trial)로 돌리므로
출발점(S0), 팔 궤적, 판정이 실험 C와 같다. 끝나면 판정을 출력하고 창은 닫을 때까지 남는다.

  python view_shake_rb5.py design0001/r60y15 --dir py --mag 1
  python view_shake_rb5.py design0001/r60y15 --dir nz --mag 0.5 --speed 0.1 --after-s 3
  python view_shake_rb5.py design0001/r60y15 --zero                     # 외력 0 대조 (팔 정지)
  python view_shake_rb5.py design0001/r60y15 --dir px --mag 1 --headless --video px_1g.mp4 --plot px_1g.png

화면:
  주황 화살표 = 명령 손 가속도, 하늘색 화살표(1.5 cm 위) = 측정 손 가속도 (palm 위 8 cm의 회색 점에서, 1 g = --arrow-m),
  초록 점 = 계획한 palm 위치. 왼쪽 위 글자 = 시간, 단계, 물체 slip·회전, 손가락 접촉력.
  Space = 일시정지/계속. 운동과 settle 구간은 --speed(기본 0.25배), 그 뒤 유지 구간은 --hold-speed(기본 1배)로 재생.
  viewer는 실패한 뒤에도 계속 시뮬레이션해서 물체가 빠지는 모습을 보여 준다(판정은 처음 실패 시점이라 같음).
  실험과 똑같이 30 mm에서 멈추려면 --batch-stop.
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

import disturb_shake_rb5 as C  # noqa: E402

ORANGE = np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32)
CYAN = np.array([0.1, 0.8, 1.0, 1.0], dtype=np.float32)
GREEN = np.array([0.1, 0.9, 0.2, 0.9], dtype=np.float32)


def add_arrow(scn, p0, vec, rgba, width=0.006):
    import mujoco
    if scn.ngeom >= scn.maxgeom or float(np.linalg.norm(vec)) < 1e-4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float), np.asarray(p0, float) + vec)
    scn.ngeom += 1


def add_dot(scn, p, rgba, r=0.006):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float),
                        np.eye(3).ravel(), rgba)
    scn.ngeom += 1


def draw_markers(scn, cs, info, arrow_m, lift=0.08):
    """화살표는 palm 위 lift [m]에서 시작 (물체에 가리지 않게)."""
    p = cs.d.xpos[cs.ctx.palm] + np.array([0.0, 0.0, lift])
    add_arrow(scn, p, info['a_cmd'] / cs.g * arrow_m, ORANGE, 0.008)
    add_arrow(scn, p + np.array([0.0, 0.0, 0.015]), info['a_meas'] / cs.g * arrow_m, CYAN, 0.005)
    add_dot(scn, p, (0.2, 0.2, 0.2, 1.0), 0.005)
    add_dot(scn, info['plan_palm'], GREEN)


def status_text(cs, k, t, plan):
    from disturb_impulse_rb5 import measure
    p, Rr, ax, pp, lf, ext, palmf = measure(cs.m, cs.d, cs.ctx)
    p0, R0, _, _ = cs.ref
    disp = 1000.0 * float(np.linalg.norm(p - p0))
    rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ Rr) - 1.0) / 2.0))))
    s0, s1 = lf[cs.ctx.side0].sum(), lf[cs.ctx.side1].sum()
    if k <= cs.n_mot:
        stage = 'motion'
    elif k <= cs.n_win:
        stage = 'settle'
    else:
        stage = 'hold'
    name = 'zero control' if plan.direction is None else f'{plan.direction} {plan.k:g} g'
    left = '\n'.join(['trial', 'time', 'stage', 'slip', 'rotation', 'finger force', 'link forces'])
    right = '\n'.join([name, f'{t:6.3f} s', stage, f'{disp:5.2f} mm', f'{rot:5.2f} deg',
                       f'{s0:5.2f} / {s1:5.2f} N', ' '.join(f'{x:.2f}' for x in lf)])
    return left, right


def save_plot(path, rows, cols, cs, plan, r, crit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = {n: i for i, n in enumerate(cols)}
    t = rows[:, c['t']]
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    title = 'zero control' if plan.direction is None else f'{plan.direction} {plan.k:g} g'
    verdict = 'held' if r['held'] else f'FAIL {r["cause"]} ({r["phase"]}, {r["t_fail_s"]:.3f} s)'
    if not r['valid']:
        verdict += f'  [arm_limited: {r["arm_reason"] or r["cause"]}]'
    fig.suptitle(f'{cs.cell_dir.parent.name}/{cs.cell_dir.name}  {title}  start {cs.start_type}  ->  {verdict}')
    a = ax[0]
    a.plot(t, rows[:, c['acmd']], color='tab:orange', label='command (filtered)')
    a.plot(t, rows[:, c['au']], color='tab:cyan', label='measured')
    a.fill_between(t, rows[:, c['au_min']], rows[:, c['au_max']], color='tab:cyan', alpha=0.2, lw=0)
    a.set_ylabel('hand accel along u [g]')
    a.legend(loc='upper right', fontsize=8)
    a = ax[1]
    a.plot(t, rows[:, c['disp']], label='slip [mm]')
    a.plot(t, rows[:, c['rot']], label='rotation [deg]')
    a.axhline(crit['slip_mm'], color='tab:blue', ls=':', lw=1)
    a.axhline(crit['rot_deg'], color='tab:orange', ls=':', lw=1)
    top = max(rows[:, c['disp']].max(), rows[:, c['rot']].max()) if len(rows) else 1.0
    a.set_ylim(0, min(max(3 * max(crit['slip_mm'], crit['rot_deg']), 1.15 * top), 40.0))
    a.set_ylabel('object vs hand (clipped at 40)')
    a.legend(loc='upper right', fontsize=8)
    a = ax[2]
    a.plot(t, rows[:, c['side0']], label='side 0 (upper)')
    a.plot(t, rows[:, c['side1']], label='side 1 (lower)')
    for nm in cols:
        if nm.startswith('f_'):
            a.plot(t, rows[:, c[nm]], lw=0.8, ls='--', label=nm[2:])
    a.axhline(cs.sim['lost_n'], color='k', ls=':', lw=1)
    a.set_ylabel('contact force [N]')
    a.legend(loc='upper right', fontsize=7, ncol=2)
    a = ax[3]
    a.plot(t, rows[:, c['hand']], label='palm displacement [mm]')
    win = t <= cs.Tm + cs.sim['settle_s'] + 1e-9
    a.plot(t[win], rows[win, c['track']] * 10, label='palm tracking error x10 [mm]')
    a.set_ylabel('palm')
    a.set_xlabel('time from S0 [s]')
    a.legend(loc='upper right', fontsize=8)
    for a in ax:
        a.axvline(cs.Tm, color='gray', ls='--', lw=0.8)
        a.axvline(cs.Tm + cs.sim['settle_s'], color='gray', ls=':', lw=0.8)
        if not r['held'] and math.isfinite(r['t_fail_s']):
            a.axvline(r['t_fail_s'], color='red', lw=0.8)
        a.grid(alpha=0.3)
    xmax = min(float(t[-1]), cs.Tm + max(1.5, cs.sim['settle_s'] * 2)) if len(t) else 1.0
    if r['phase'] == 'delayed' or r['held']:
        xmax = float(t[-1])
    ax[-1].set_xlim(0, xmax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_result(cs, plan, r):
    name = 'zero control' if plan.direction is None else f'{plan.direction} {plan.k:g} g'
    print(f'[RESULT] {name}: ' + ('HELD' if r['held'] else
                                  f'FAIL {r["cause"]} in {r["phase"]} at {r["t_fail_s"]:.3f} s'))
    if r.get('load_sign'):
        uh = np.array(C.ALL_DIRS[plan.direction])
        print(f'         load on object at failure: {C.dir_label(uh * r["load_sign"])} '
              f'(hand accelerating toward {C.dir_label(-uh * r["load_sign"])})')
    print(f'         max slip {r.get("max_disp_mm", math.nan):.2f} mm, max rotation {r.get("max_rot_deg", math.nan):.2f} deg, '
          f'min finger side force {r.get("min_side_N", math.nan):.2f} N')
    if plan.direction is not None:
        print(f'         arm: peak accel measured/command  + side {r.get("acc_ratio_pos", math.nan):.3f}, '
              f'- side {r.get("acc_ratio_neg", math.nan):.3f}, '
              f'tracking {r.get("max_track_mm", math.nan):.2f} mm, palm rotation {r.get("max_palm_rot_deg", math.nan):.3f} deg, '
              f'actuator {100 * r.get("max_sat", math.nan):.0f} % of limit, arm contact {r.get("max_arm_contact_N", math.nan):.2f} N'
              + ('' if r['valid'] else f'  -> arm_limited ({r["arm_reason"]})'))


def main(args):
    import mujoco
    cells_dir = C.default_cells_dir(args)
    cdir = Path(args.cell)
    if not ((cdir / 'state.npz').is_file() or (cdir.suffix.lower() == '.xml' and cdir.is_file())):
        cdir = C.resolve_cell(cells_dir, args.cell)   # design0001/r60y15, 예전 이름, XML 이름 모두
    sim = C.sim_from_args(args)
    if not args.batch_stop:
        sim['stop_mm'] = 1e9
        sim['stop_lost_s'] = 1e9
    crit = C.crit_from_args(args)
    print(f'[VIEW] loading {args.cell} and settling {sim["pre_s"]:g} s ...', flush=True)
    cs = C.CellSim(cdir, sim, C.mesh_roots_for(args, cells_dir), args.allow_missing_collision)
    direction = None if args.zero else args.dir
    if direction is not None and direction not in C.ALL_DIRS:
        raise SystemExit(f'--dir must be one of {", ".join(C.ALL_DIRS)}')
    plan = C.plan_trial(cs, direction, 0.0 if args.zero else args.mag)
    print(f'[VIEW] start contact {cs.start_type}; link forces ' +
          ', '.join(f'{n.replace("hand__", "")} {f:.2f}' for n, f in zip(cs.ctx.link_names, cs.start_lf)) + ' N')
    if direction is not None:
        u_h = np.array(C.ALL_DIRS[direction])
        print(f'[VIEW] {direction} {args.mag:g} g: hand moves along hand {u_h.round(3).tolist()} '
              f'(world {plan.u.round(3).tolist()}), max displacement {1000 * plan.info.get("max_disp_m", 0):.1f} mm, '
              f'max speed {plan.info.get("max_speed_mps", 0):.2f} m/s, motion {cs.Tm:.3f} s, then hold {sim["after_s"]:g} s')
        if plan.info:
            print(f'[VIEW] planned joint speed max [rad/s] {np.round(plan.info["max_qd"], 2).tolist()}, '
                  f'predicted actuator load {100 * plan.info["pred_force_ratio"]:.0f} % of forcerange')
    if not plan.feasible:
        print(f'[VIEW] plan is infeasible: {plan.reason}')
        if plan.ctrl is None or not args.force:
            raise SystemExit('nothing to run' + ('' if plan.ctrl is None else ' (use --force to run it anyway)'))
    cols = C.trace_columns(cs.ctx)

    # ------------------------------------------------------------------ playback helpers
    rec = {'frames': [], 'next_t': 0.0}
    renderer = cam = None
    if args.video:
        renderer = mujoco.Renderer(cs.m, args.height, args.width) if (
            cs.m.vis.global_.offwidth >= args.width and cs.m.vis.global_.offheight >= args.height) else None
        if renderer is None:
            cs.m.vis.global_.offwidth = max(cs.m.vis.global_.offwidth, args.width)
            cs.m.vis.global_.offheight = max(cs.m.vis.global_.offheight, args.height)
            renderer = mujoco.Renderer(cs.m, args.height, args.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = cs.d.xpos[cs.ctx.obj]
        cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation

    def speed_at(k):
        return args.speed if k <= cs.n_win else args.hold_speed

    def capture(k, t, info):
        if renderer is None:
            return
        while t >= rec['next_t'] - 1e-12:
            renderer.update_scene(cs.d, camera=cam)
            if not args.no_markers:
                draw_markers(renderer.scene, cs, info, args.arrow_m)
            rec['frames'].append(renderer.render().copy())
            rec['next_t'] += speed_at(k) / args.fps

    if args.headless:
        def cb(k, t, info):
            capture(k, t, info)
        t0 = time.time()
        rows, ended = C.run_trial(cs, plan, callback=cb if renderer is not None else None)
        print(f'[VIEW] simulated {rows[-1, 0] if len(rows) else 0:.2f} s in {time.time() - t0:.1f} s ({ended})')
    else:
        import mujoco.viewer
        state = {'paused': True, 'start': time.time() + args.start_delay if args.start_delay > 0 else 0.0}

        def key_cb(key):
            if key == 32:  # space
                state['paused'] = not state['paused']
                state['start'] = 0.0

        print('[VIEW] Space = pause/continue. Starts in '
              f'{args.start_delay:g} s. Close the window to quit.', flush=True)
        with mujoco.viewer.launch_passive(cs.m, cs.d, key_callback=key_cb) as v:
            v.cam.lookat[:] = cs.d.xpos[cs.ctx.obj]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = args.distance, args.azimuth, args.elevation
            # 시작 전: S0를 보여 주며 대기
            mujoco.mj_setState(cs.m, cs.d, cs.S0, cs.spec)
            mujoco.mj_forward(cs.m, cs.d)
            while v.is_running():
                if state['start'] and time.time() >= state['start']:
                    state['paused'] = False
                    state['start'] = 0.0
                if not state['paused']:
                    break
                try:
                    v.set_texts((None, None, 'S0 (paused)', 'press Space to start'))
                except Exception:
                    pass
                v.sync()
                time.sleep(0.02)
            clock = {'wall0': time.perf_counter(), 'wall': 0.0, 'sim': 0.0}

            class _Closed(Exception):
                pass

            def cb(k, t, info):
                capture(k, t, info)
                sp = speed_at(k)
                if t - clock['sim'] < sp / args.viewer_fps:
                    return
                # 재생 속도: 구간마다 속도가 달라서 wall 시간을 누적한다
                clock['wall'] += (t - clock['sim']) / sp
                clock['sim'] = t
                with v.lock():
                    v.user_scn.ngeom = 0
                    if not args.no_markers:
                        draw_markers(v.user_scn, cs, info, args.arrow_m)
                try:
                    v.set_texts((None, None, *status_text(cs, k, t, plan)))
                except Exception:
                    pass
                v.sync()
                if state['paused']:
                    while state['paused'] and v.is_running():
                        time.sleep(0.02)
                        v.sync()
                    clock['wall0'] = time.perf_counter() - clock['wall']
                if not v.is_running():
                    raise _Closed
                lag = clock['wall'] - (time.perf_counter() - clock['wall0'])
                if lag > 0:
                    time.sleep(lag)

            try:
                rows, ended = C.run_trial(cs, plan, callback=cb)
            except _Closed:
                print('[VIEW] window closed')
                return 0
            r = C.score_trial(rows, ended, cols, crit, cs.Tm, sim['settle_s'], plan.k)
            print_result(cs, plan, r)
            try:
                v.set_texts((None, None, 'result', 'HELD' if r['held'] else f'FAIL {r["cause"]} ({r["phase"]})'))
            except Exception:
                pass
            if args.plot:
                save_plot(args.plot, rows, cols, cs, plan, r, crit)
                print(f'[VIEW] plot -> {args.plot}')
            if args.video:
                write_video(args.video, rec['frames'], args.fps)
            while v.is_running():
                v.sync()
                time.sleep(0.05)
        return 0

    r = C.score_trial(rows, ended, cols, crit, cs.Tm, sim['settle_s'], plan.k)
    print_result(cs, plan, r)
    if args.plot:
        save_plot(args.plot, rows, cols, cs, plan, r, crit)
        print(f'[VIEW] plot -> {args.plot}')
    if args.video:
        write_video(args.video, rec['frames'], args.fps)
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(np.round(rows.astype(float), 6).tolist())
        print(f'[VIEW] rows -> {args.csv}')
    return 0


def write_video(path, frames, fps):
    if not frames:
        print('[VIEW] no frames recorded')
        return
    import imageio
    try:
        imageio.mimsave(path, frames, fps=fps, **({'quality': 8} if str(path).endswith('.mp4') else {}))
    except Exception as e:  # imageio-ffmpeg 없음 등
        alt = str(Path(path).with_suffix('.gif'))
        print(f'[VIEW] cannot write {path} ({e}); writing {alt}')
        imageio.mimsave(alt, frames, duration=1000.0 / fps)
        path = alt
    print(f'[VIEW] video -> {path} ({len(frames)} frames)')


def build_parser():
    p = argparse.ArgumentParser(description='실험 C viewer: cell 하나에서 팔 흔들기 시행 하나를 MuJoCo로 본다')
    p.add_argument('cell', help='design0001/r60y15, design0001/r060.0_y015.0, XML 파일 경로 또는 cell 폴더 경로')
    p.add_argument('--cells-dir', default=None, help='grasp_cells 폴더 (기본: 이 파일 옆 grasp_cells)')
    p.add_argument('--mesh-dir', default=None, help='scene.xml의 mesh 절대경로가 없을 때 찾을 …/assets/rb5_850e/meshes')
    p.add_argument('--allow-missing-collision', action='store_true')
    p.add_argument('--dir', default='py', help='손이 처음 움직이는 방향 (px nx py ny pz nz, 대각선 pxpy …)')
    p.add_argument('--mag', type=float, default=1.0, help='손 가속도 최댓값 [g]')
    p.add_argument('--zero', action='store_true', help='외력 0 대조 (팔을 움직이지 않음)')
    p.add_argument('--force', action='store_true', help='계획 충돌(infeasible)이어도 돌려 봄')
    p.add_argument('--batch-stop', action='store_true', help='실험처럼 30 mm 미끄러지거나 접촉을 0.1 s 잃으면 멈춤')
    C.add_sim_args(p)
    C.add_crit_args(p)
    v = p.add_argument_group('보기')
    v.add_argument('--speed', type=float, default=0.25, help='운동·settle 구간 재생 속도 (1 = 실시간)')
    v.add_argument('--hold-speed', type=float, default=1.0, help='유지 구간 재생 속도')
    v.add_argument('--start-delay', type=float, default=2.0, help='창을 연 뒤 시작까지 기다리는 시간 [s] (0이면 Space로 시작)')
    v.add_argument('--viewer-fps', type=float, default=60.0)
    v.add_argument('--distance', type=float, default=0.7)
    v.add_argument('--azimuth', type=float, default=135.0)
    v.add_argument('--elevation', type=float, default=-20.0)
    v.add_argument('--arrow-m', type=float, default=0.15, help='가속도 화살표 길이 [m per g]')
    v.add_argument('--no-markers', action='store_true', help='화살표·계획 위치 표시 끔')
    v.add_argument('--headless', action='store_true', help='창 없이 돌리고 결과·그림·영상만 저장')
    v.add_argument('--video', default=None, help='영상 저장 경로 (.mp4 또는 .gif)')
    v.add_argument('--fps', type=float, default=30.0, help='영상 fps (재생 속도는 --speed/--hold-speed)')
    v.add_argument('--width', type=int, default=1280)
    v.add_argument('--height', type=int, default=720)
    v.add_argument('--plot', default=None, help='시간 응답 그림 저장 경로 (.png)')
    v.add_argument('--csv', default=None, help='(--headless) trace row를 CSV로 저장')
    return p


if __name__ == '__main__':
    sys.exit(main(build_parser().parse_args()))
