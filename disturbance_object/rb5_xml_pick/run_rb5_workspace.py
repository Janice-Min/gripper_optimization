"""Pick every graspable workspace cell with the RB5 and save the state right after lifting.

Inputs
  --xml-dir        design XMLs made by generate_mjcf.py (designNNNN.xml)
  --workspace-dir  output of evaluate_grasp_workspace.py (graspable_points.csv, grasp_config.json)

For every graspable cell (design, r, y, x) the hand XML is attached to the RB5 wrist by
run_rb5_xml.py (same folder, reused unchanged). A cylinder of radius r stands on the stand on
the bench, and the RB5 places the palm so that the cylinder sits exactly where the cell says:
    cylinder centre in palm frame = palm_face + (y + r) * finger_forward + x * lateral
    (design XMLs: finger_forward = palm -x, palm_face = -1 mm  ->  (-0.001 - y - r, x, 0))
Sequence and verdict are those of run_rb5_xml.py: lower -> approach -> tendon closing -> lift -> hold.

Output (--out-dir, default grasp_cells/)
  <design>/r055.0_y020.0/scene.xml     RB5 + hand + cylinder scene; keyframe 'lifted' = state right after the lift
  <design>/r055.0_y020.0/state.npz     exact MuJoCo integration state at that moment (+ qpos, qvel, ctrl)
  <design>/r055.0_y020.0/summary.json  cell, lifted-state object pose and finger forces, final verdict
  rb5_pick_results.csv                 every attempted cell: PASS/FAIL and reason
Only PASS cells are stored; --save-failed keeps the others under _failed/.

Examples
  python run_rb5_workspace.py --dry-run                         # count cells, nothing simulated
  python run_rb5_workspace.py --designs design0001 --viewer     # watch one design in the viewer
  python run_rb5_workspace.py --workers 6                       # all cells, headless, parallel
  python run_rb5_workspace.py --workers 6 --resume              # continue an interrupted run
  python run_rb5_workspace.py --show grasp_cells/design0001/r055.0_y020.0
All argparse options and defaults are at the bottom of this file (same layout as run_rb5_xml.py).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import threading
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
RESULT_FIELDS = ['design_id', 'r_mm', 'y_mm', 'x_mm', 'workspace_grasp_type', 'grasp_local_x_mm',
                 'grasp_local_y_mm', 'tension_N', 'success', 'reason', 'lift_at_snapshot_mm',
                 'final_lift_mm', 'min_lift_in_window_mm', 'bilateral_contact', 'off_support',
                 'four_link_contact', 'saved_to', 'runtime_s']
_BASE = None


# --------------------------------------------------------------------------- planning (no MuJoCo)
def cell_tag(r_mm, y_mm):
    return f'r{r_mm:05.1f}_y{y_mm:05.1f}'


def palm_face_frame(xml_path, palm_body, palm_geom, forward_angle):
    """Finger-forward unit vector f, lateral unit vector n and palm-face offset along f (palm frame)."""
    root = ET.parse(xml_path).getroot()
    palm = root.find(f".//body[@name='{palm_body}']")
    geom = None if palm is None else palm.find(f"geom[@name='{palm_geom}']")
    if geom is None:
        raise ValueError(f'{xml_path}: body {palm_body!r} with child geom {palm_geom!r} not found')
    if any(geom.get(k) for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto')) \
            or geom.get('type') != 'box':
        raise ValueError(f'{xml_path}: palm geom must be an unrotated box (as in design XMLs)')
    pos = np.array([float(v) for v in geom.get('pos', '0 0 0').split()])
    size = np.array([float(v) for v in geom.get('size').split()])
    a = np.radians(forward_angle)
    f = np.array([np.cos(a), np.sin(a)])
    n = np.array([f[1], -f[0]])  # f rotated by -90 deg: +MuJoCo y for f = -x, as in evaluate_grasp_workspace.py
    face = float(pos[:2] @ f + abs(f[0]) * size[0] + abs(f[1]) * size[1])
    return f, n, face


def grasp_xy(cell, frame):
    f, n, face = frame
    xy = f * (face + (cell['y_mm'] + cell['r_mm']) / 1000) + n * cell['x_mm'] / 1000
    return [float(v) for v in xy]


def read_cells(ws_dir, designs, max_per_design):
    path = ws_dir / 'graspable_points.csv'
    if not path.is_file():
        raise SystemExit(f'{path} not found: run evaluate_grasp_workspace.py first (or set --workspace-dir)')
    groups = {}
    with path.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if designs and row['design_id'] not in designs:
                continue
            groups.setdefault(row['design_id'], []).append(
                {'r_mm': float(row['r_mm']), 'y_mm': float(row['y_mm']),
                 'x_mm': float(row.get('x_mm') or 0.0), 'grasp_type': row.get('grasp_type', '')})
    for design, cells in groups.items():
        cells.sort(key=lambda c: (c['r_mm'], c['y_mm']))
        if max_per_design and len(cells) > max_per_design:  # evenly spread subset over (r, y)
            keep = np.unique(np.linspace(0, len(cells) - 1, max_per_design).round().astype(int))
            groups[design] = [cells[i] for i in keep]
    missing = sorted(set(designs or []) - set(groups))
    if missing:
        print(f'[WARN] no graspable cells for: {", ".join(missing)}')
    return dict(sorted(groups.items()))


# --------------------------------------------------------------------------- simulation (MuJoCo)
def load_base():
    """Import run_rb5_xml.py lazily (so --dry-run works without MuJoCo) and extend its Task."""
    global _BASE
    if _BASE is None:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import mujoco
        import run_rb5_xml as base

        class CellTask(base.Task):
            """run_rb5_xml.Task that records the state at the end of the lift motion."""
            initial_z = None
            lifted = None

            def move(self, target, seconds, stage):
                if stage == 'lower' and self.initial_z is None:
                    self.initial_z = float(self.d.xpos[self.objbody, 2])  # = Task.run's initial_z
                super().move(target, seconds, stage)
                if stage == 'lift':
                    self.lifted = capture(self, mujoco)

        _BASE = (base, CellTask, mujoco)
    return _BASE


def capture(task, mujoco):
    m, d = task.m, task.d
    forces, sides, support = task.contacts()
    rp = d.xmat[task.palm].reshape(3, 3)
    obj = d.xpos[task.objbody].copy()
    state, spec = None, None
    try:
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, state, spec)
    except (AttributeError, TypeError):
        state, spec = None, None
    info = {
        'time_s': float(d.time),
        'object_pos_m': obj.tolist(),
        'object_quat': d.xquat[task.objbody].tolist(),
        'palm_pos_m': d.xpos[task.palm].tolist(),
        'object_in_palm_frame_m': (rp.T @ (obj - d.xpos[task.palm])).tolist(),
        'lift_m': None if task.initial_z is None else float(obj[2] - task.initial_z),
        'finger_normal_forces_N': {m.geom(int(g)).name: float(v) for g, v in forces.items()},
        'side_forces_N': [float(s) for s in sides],
        'support_force_N': float(support),
    }
    return {'qpos': d.qpos.copy(), 'qvel': d.qvel.copy(), 'act': d.act.copy(), 'ctrl': d.ctrl.copy(),
            'qfrc_applied': d.qfrc_applied.copy(), 'state': state, 'spec': spec, 'info': info}


def finalize(tmp, base, lifted, cell_info, keep_trace, log_text):
    """Turn run_rb5_xml outputs in tmp into scene.xml (+'lifted' keyframe), state.npz, summary.json."""
    src = tmp / 'rb5_attached_scene.xml'
    if src.exists():
        root = ET.parse(src).getroot()
        if lifted is not None:
            frames = root.find('keyframe')
            if frames is None:
                frames = ET.SubElement(root, 'keyframe')
            attrs = {'name': 'lifted', 'time': f"{lifted['info']['time_s']:.15g}",
                     'qpos': base.values(lifted['qpos']), 'qvel': base.values(lifted['qvel']),
                     'ctrl': base.values(lifted['ctrl'])}
            if lifted['act'].size:
                attrs['act'] = base.values(lifted['act'])
            ET.SubElement(frames, 'key', attrs)
        ET.indent(root, space='  ')
        ET.ElementTree(root).write(tmp / 'scene.xml', encoding='utf-8', xml_declaration=True)
        src.unlink()
    if lifted is not None:
        extra = {} if lifted['state'] is None else {'integration_state': lifted['state'],
                                                     'integration_spec': np.array(lifted['spec'])}
        np.savez(tmp / 'state.npz', time=np.array(lifted['info']['time_s']), qpos=lifted['qpos'],
                 qvel=lifted['qvel'], act=lifted['act'], ctrl=lifted['ctrl'],
                 qfrc_applied=lifted['qfrc_applied'], **extra)
    summary_path = tmp / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    summary = {'cell': cell_info, 'lifted_state': None if lifted is None else lifted['info'],
               'saved_lifted_keyframe': lifted is not None, **summary}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    if not keep_trace and (tmp / 'trace.csv').exists():
        (tmp / 'trace.csv').unlink()
    (tmp / 'log.txt').write_text(log_text, encoding='utf-8')


def run_cell(job):
    t0 = time.perf_counter()
    base, CellTask, _ = load_base()
    a = argparse.Namespace(**job['base_args'])
    a.xml, a.radius, a.grasp_xy, a.auto_grasp = Path(job['xml']), job['r_mm'] / 1000, job['grasp_xy'], False
    tmp = Path(job['tmp_dir'])
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    a.output = tmp
    log = io.StringIO()
    task, result, interrupted = None, {'success': False}, False
    sink = contextlib.nullcontext() if job['viewer'] else contextlib.redirect_stdout(log)
    with sink:
        try:
            model, data, ik, meta, rotation, grasp, home = base.build_scene(a)
            task = CellTask(a, model, data, ik, meta, rotation, grasp, home)
            if job['viewer']:
                from mujoco import viewer as mjviewer
                existing = set(threading.enumerate())
                task.viewer = mjviewer.launch_passive(model, data)
                task.viewer_threads = [t for t in threading.enumerate() if t not in existing]
                base.set_camera(task.viewer.cam, task.viewer.opt, a)
                task.viewer.sync()
            task.preflight()
            result = task.run()
        except KeyboardInterrupt:
            result, interrupted = {'success': False, 'reason': 'interrupted'}, True
        except (ValueError, RuntimeError, OSError) as exc:
            result = {'success': False, 'reason': str(exc)}
        except Exception as exc:  # keep the batch alive; full traceback goes to log.txt
            result = {'success': False, 'reason': f'{type(exc).__name__}: {exc}'}
            log.write(traceback.format_exc())
        finally:
            if task is not None:
                base.save(task, result)
                if task.viewer is not None:
                    task.viewer.close()
                    for thread in task.viewer_threads:
                        thread.join(timeout=a.viewer_shutdown_timeout)

    lifted = None if task is None else task.lifted
    success = bool(result.get('success')) and lifted is not None
    saved = ''
    if not interrupted and (success or job['save_failed']):
        cell_info = {k: job[k] for k in ('design_id', 'r_mm', 'y_mm', 'x_mm', 'workspace_grasp_type')}
        cell_info.update(grasp_local_xy_m=job['grasp_xy'], tension_N=a.tension, source_xml=job['xml'],
                         rb5_pick_success=success)
        finalize(tmp, base, lifted, cell_info, job['save_trace'], log.getvalue())
        dest = Path(job['cell_dir'] if success else job['failed_dir'])
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(dest))
        saved = str(dest)
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    mm = lambda v: '' if v is None else round(float(v) * 1000, 3)
    return {
        'design_id': job['design_id'], 'r_mm': job['r_mm'], 'y_mm': job['y_mm'], 'x_mm': job['x_mm'],
        'workspace_grasp_type': job['workspace_grasp_type'],
        'grasp_local_x_mm': mm(job['grasp_xy'][0]), 'grasp_local_y_mm': mm(job['grasp_xy'][1]),
        'tension_N': a.tension, 'success': int(success),
        'reason': 'interrupted' if interrupted else ('success' if success else result.get('reason', '')),
        'lift_at_snapshot_mm': mm(None if lifted is None else lifted['info']['lift_m']),
        'final_lift_mm': mm(result.get('object_lift_m')),
        'min_lift_in_window_mm': mm(result.get('minimum_lift_in_window_m')),
        'bilateral_contact': result.get('bilateral_contact_in_window', ''),
        'off_support': result.get('off_support_in_window', ''),
        'four_link_contact': result.get('four_link_contact', ''),
        'saved_to': saved, 'runtime_s': round(time.perf_counter() - t0, 2),
    }


def show(cell_dir):
    import mujoco
    from mujoco import viewer as mjviewer
    cell = Path(cell_dir)
    model = mujoco.MjModel.from_xml_path(str(cell / 'scene.xml'))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'lifted')
    if key < 0:
        raise SystemExit(f'{cell}/scene.xml has no keyframe "lifted"')
    mujoco.mj_resetDataKeyframe(model, data, key)
    state_file = cell / 'state.npz'
    if state_file.exists():
        saved = np.load(state_file)
        if 'integration_state' in saved.files:
            mujoco.mj_setState(model, data, saved['integration_state'], int(saved['integration_spec']))
    mujoco.mj_forward(model, data)
    print(f'Showing {cell} at the moment right after the lift (physics keeps running).')
    mjviewer.launch(model, data)


# --------------------------------------------------------------------------- batch driver
def main(args):
    if args.show:
        show(args.show)
        return 0
    xml_dir, ws_dir, out = (p.expanduser().resolve() for p in (args.xml_dir, args.workspace_dir, args.out_dir))
    config_path = ws_dir / 'grasp_config.json'
    ws_cfg = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    # Same physics as the workspace search unless overridden on the command line.
    if args.tension is None:
        args.tension = float(ws_cfg.get('tension', 10.0))
    if args.mass is None and ws_cfg.get('mass') is not None:
        args.mass = float(ws_cfg['mass'])
    if args.object_height is None and ws_cfg.get('half_height') is not None:
        args.object_height = 2 * float(ws_cfg['half_height'])

    groups = read_cells(ws_dir, args.designs, args.max_cells_per_design)
    jobs = []
    done = set()
    results_path = out / 'rb5_pick_results.csv'
    if args.resume and results_path.exists():
        with results_path.open(encoding='utf-8-sig') as f:
            done = {(r['design_id'], cell_tag(float(r['r_mm']), float(r['y_mm'])))
                    for r in csv.DictReader(f) if r['reason'] != 'interrupted'}
    base_args = {k: v for k, v in vars(args).items()}
    base_args.update(auto_grasp_max=.08, auto_grasp_step=.0025, approach_clearance=.001,
                     headless=not args.viewer, exit_after_task=True, video=None, video_fps=30)
    for design, cells in groups.items():
        xml = xml_dir / f'{design}.xml'
        if not xml.is_file():
            print(f'[WARN] {xml} not found; {len(cells)} cells of {design} skipped')
            continue
        frame = palm_face_frame(xml, args.palm_body, args.palm_geom, args.forward_angle)
        for c in cells:
            tag = cell_tag(c['r_mm'], c['y_mm'])
            if (design, tag) in done:
                continue
            jobs.append({'design_id': design, 'xml': str(xml), 'r_mm': c['r_mm'], 'y_mm': c['y_mm'],
                         'x_mm': c['x_mm'], 'workspace_grasp_type': c['grasp_type'],
                         'grasp_xy': grasp_xy(c, frame), 'base_args': base_args,
                         'tmp_dir': str(out / '_tmp' / f'{design}_{tag}'),
                         'cell_dir': str(out / design / tag), 'failed_dir': str(out / '_failed' / design / tag),
                         'viewer': args.viewer, 'save_failed': args.save_failed, 'save_trace': args.save_trace})

    n_cells = sum(len(c) for c in groups.values())
    mass = 'from XML' if args.mass is None else f'{args.mass:g} kg'
    print(f'{len(groups)} design(s), {n_cells} graspable cell(s); {len(jobs)} to run '
          f'({len(done)} already done). Tension {args.tension:g} N, mass {mass}, '
          f'stand {args.support_height * 1000:g} mm.')
    if jobs:
        j = jobs[0]
        print(f'  e.g. {j["design_id"]} r={j["r_mm"]:g} y={j["y_mm"]:g} mm -> cylinder centre in palm frame '
              f'({j["grasp_xy"][0] * 1000:.1f}, {j["grasp_xy"][1] * 1000:.1f}) mm')
    if args.dry_run or not jobs:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / 'rb5_pick_config.json').write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2), encoding='utf-8')
    append = args.resume and results_path.exists() and results_path.stat().st_size > 0
    workers = 1 if args.viewer else max(1, args.workers)
    passed = 0
    t0 = time.perf_counter()
    with results_path.open('a' if append else 'w', newline='', encoding='utf-8' if append else 'utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not append:
            writer.writeheader()
        pool = Pool(workers) if workers > 1 and len(jobs) > 1 else None
        try:
            rows = pool.imap_unordered(run_cell, jobs) if pool is not None else map(run_cell, jobs)
            for k, row in enumerate(rows, 1):
                if row['reason'] == 'interrupted':
                    raise KeyboardInterrupt
                writer.writerow(row)
                f.flush()
                passed += row['success']
                eta = (time.perf_counter() - t0) / k * (len(jobs) - k)
                verdict = (f"PASS lift {row['lift_at_snapshot_mm']} mm" if row['success']
                           else f"FAIL {str(row['reason'])[:90]}")
                print(f"[{k}/{len(jobs)}] {row['design_id']} r={row['r_mm']:g} y={row['y_mm']:g}: {verdict} "
                      f"({row['runtime_s']} s, ETA {eta / 60:.1f} min)", flush=True)
        except KeyboardInterrupt:
            print('Interrupted: finished cells are saved; rerun with --resume to continue.', flush=True)
            if pool is not None:
                pool.terminate()
            return 130
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            shutil.rmtree(out / '_tmp', ignore_errors=True)
    print(f'Done: {passed}/{len(jobs)} cells picked and lifted. States in {out}, table {results_path.name}')
    return 0


# ============================================================================
# USER SETTINGS: ALL argparse options and editable defaults are together BELOW.
# Options shared with run_rb5_xml.py keep its names and defaults.
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- inputs / outputs / batch control
    parser.add_argument('--xml-dir', type=Path, default=HERE/'xml', help='design XMLs from generate_mjcf.py')
    parser.add_argument('--workspace-dir', type=Path, default=HERE/'workspace_results', help='evaluate_grasp_workspace.py output')
    parser.add_argument('--out-dir', type=Path, default=HERE/'grasp_cells', help='where lifted states are saved')
    parser.add_argument('--designs', nargs='+', default=None, help='only these design ids, e.g. design0001 design0042')
    parser.add_argument('--max-cells-per-design', type=int, default=0, help='0 = all cells; N = evenly spread N cells per design')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='parallel processes (headless)')
    parser.add_argument('--resume', action='store_true', help='skip cells already in rb5_pick_results.csv')
    parser.add_argument('--save-failed', action='store_true', help='also keep FAIL cells under _failed/')
    parser.add_argument('--save-trace', action='store_true', help='keep trace.csv (about 0.3 MB per cell)')
    parser.add_argument('--dry-run', action='store_true', help='list the planned cells only (no MuJoCo needed)')
    parser.add_argument('--viewer', action='store_true', help='watch each cell in the MuJoCo viewer (forces --workers 1)')
    parser.add_argument('--show', type=Path, default=None, help='open a saved grasp_cells/<design>/<cell> at its lifted state')
    # --- hand / object (same physics as the workspace search by default)
    parser.add_argument('--palm-body', default='palm', help='Palm body name in input XML')
    parser.add_argument('--palm-geom', default='palm_geom', help='Palm box geom that defines y = 0')
    parser.add_argument('--object-body', default='object_body', help='Object body name in input XML')
    parser.add_argument('--open-key', default='open', help='Initial open-hand keyframe')
    parser.add_argument('--forward-angle', type=float, default=180., help='Finger-forward angle in palm xy [deg]; design XMLs = -x = 180')
    parser.add_argument('--tension', type=float, default=None, help='Tendon tension [N]; None = grasp_config.json of the workspace run (10 N)')
    parser.add_argument('--mass', type=float, default=None, help='Object mass [kg]; None = workspace run / XML (0.3 kg)')
    parser.add_argument('--object-height', type=float, default=None, help='Cylinder full height [m]; None = workspace run / XML (38 mm)')
    parser.add_argument('--object-xy', type=float, nargs=2, default=[.60, 0.], help='Object world xy on the bench [m]')
    parser.add_argument('--support-height', type=float, default=.08, help='Stand height on the bench [m]; avoids wrist/bench collision')
    parser.add_argument('--spawn-clearance', type=float, default=.0005, help='Initial object/support clearance [m]')
    parser.add_argument('--mount-length', type=float, default=.10, help='Flange-to-palm adapter length [m]')
    # --- motion (run_rb5_xml.py defaults)
    parser.add_argument('--open-kp', type=float, default=1., help='Temporary open-hand spring gain [Nm/rad]')
    parser.add_argument('--open-kd', type=float, default=.03, help='Temporary open-hand damping [Nms/rad]')
    parser.add_argument('--open-torque-limit', type=float, default=.3, help='Temporary opening torque cap [Nm]')
    parser.add_argument('--open-release-time', type=float, default=.3, help='Open controller release duration [s]')
    parser.add_argument('--lift', type=float, default=.15, help='Vertical lift distance [m]')
    parser.add_argument('--approach-distance', type=float, default=.12, help='Horizontal pregrasp distance [m]')
    parser.add_argument('--home-height', type=float, default=.22, help='Home height above grasp target [m]')
    parser.add_argument('--settle-time', type=float, default=.6, help='Initial settling time [s]')
    parser.add_argument('--move-time', type=float, default=3., help='Time to lower [s]')
    parser.add_argument('--approach-time', type=float, default=2.5, help='Horizontal approach time [s]')
    parser.add_argument('--preclose-time', type=float, default=.5, help='Settling before closing [s]')
    parser.add_argument('--ramp-time', type=float, default=1.5, help='Tension ramp time [s]')
    parser.add_argument('--grasp-hold', type=float, default=1., help='Hold before lift [s]')
    parser.add_argument('--lift-time', type=float, default=3., help='Lift duration [s]')
    parser.add_argument('--hold-time', type=float, default=2., help='Hold after lift; success verdict window lies in it [s]')
    parser.add_argument('--timestep', type=float, default=.001, help='Physics timestep [s]')
    parser.add_argument('--ik-interval', type=float, default=.01, help='IK update period [s]')
    parser.add_argument('--sample-interval', type=float, default=.01, help='Log and verdict sample interval [s]')
    parser.add_argument('--preflight-samples', type=int, default=40, help='IK/approach checks per segment')
    parser.add_argument('--initial-overlap-tolerance', type=float, default=.0001, help='Open-hand overlap tolerance [m]')
    parser.add_argument('--collision-tolerance', type=float, default=.001, help='Unintended penetration failure threshold [m]')
    parser.add_argument('--contact-threshold', type=float, default=.1, help='Bilateral contact force threshold [N]')
    parser.add_argument('--support-force-threshold', type=float, default=.05, help='Off-support contact force threshold [N]')
    parser.add_argument('--success-ratio', type=float, default=.8, help='Minimum achieved / requested lift')
    parser.add_argument('--verdict-window', type=float, default=1., help='Final success evaluation window [s]')
    # --- viewer (only with --viewer)
    parser.add_argument('--speed', type=float, default=1., help='Viewer playback speed')
    parser.add_argument('--viewer-fps', type=int, default=60, help='Viewer update frequency')
    parser.add_argument('--viewer-shutdown-timeout', type=float, default=3., help='Wait for viewer render-thread cleanup [s]')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=[.30, 0., .28], help='Viewer target [m]')
    parser.add_argument('--camera-distance', type=float, default=1.45, help='Viewer distance [m]')
    parser.add_argument('--camera-azimuth', type=float, default=135., help='Viewer azimuth [deg]')
    parser.add_argument('--camera-elevation', type=float, default=-25., help='Viewer elevation [deg]')
    args = parser.parse_args()
    for name in ('lift', 'approach_distance', 'home_height', 'settle_time', 'move_time', 'approach_time',
                 'preclose_time', 'ramp_time', 'grasp_hold', 'lift_time', 'hold_time', 'timestep', 'ik_interval',
                 'sample_interval', 'contact_threshold', 'support_force_threshold', 'verdict_window', 'speed',
                 'viewer_fps', 'viewer_shutdown_timeout', 'camera_distance', 'open_release_time', 'open_torque_limit'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and > 0')
    for name in ('tension', 'mass', 'object_height'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0 or (name != 'tension' and value == 0)):
            parser.error(f'--{name.replace("_", "-")} must be finite and positive')
    for name in ('support_height', 'spawn_clearance', 'initial_overlap_tolerance', 'collision_tolerance', 'open_kp', 'open_kd'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and >= 0')
    if not np.isfinite(args.mount_length) or args.mount_length <= .012:
        parser.error('--mount-length must be > .012 m')
    if args.open_release_time > args.ramp_time:
        parser.error('--open-release-time must be <= --ramp-time')
    if not 0 < args.success_ratio <= 1 or args.verdict_window > args.hold_time:
        parser.error('Require 0 < success-ratio <= 1 and verdict-window <= hold-time')
    if args.sample_interval > args.verdict_window or args.timestep > args.sample_interval or args.preflight_samples < 2:
        parser.error('Require timestep <= sample-interval <= verdict-window and preflight-samples >= 2')
    if args.max_cells_per_design < 0 or args.workers < 1:
        parser.error('--max-cells-per-design must be >= 0 and --workers >= 1')
    sys.exit(main(args))
