#!/usr/bin/env python3
"""Find which cylinders (radius r) at which palm distances (y) each finger design can grasp.

Adapted from evaluate_workspace.py for the Stage-0 MJCF produced by generate_mjcf.py
(geom names upper_green_link / upper_red_link / lower_green_link / lower_red_link,
object freejoint 'object_free', keyframe 'open').

Report frame (CSV columns)
  x [mm]  lateral offset of the cylinder axis from the gripper mid-plane (MuJoCo +y). Fixed 0.
  y [mm]  gap between the palm front face and the nearest point of the cylinder surface,
          measured along the approach direction (MuJoCo -x). y = 0: cylinder touches the palm.
  r [mm]  cylinder radius. Half-height and mass are kept from the XML (19 mm, 0.3 kg).
  MuJoCo object centre = (x_palm_face - y - r,  x,  z_palm);  x_palm_face = -1 mm for Stage 0.

Trial (same protocol as evaluate_workspace.py)
  reset to keyframe 'open' -> place cylinder -> gravity OFF, tendon tension ramps 0 -> 10 N
  over 0.3 s -> gravity ON -> run to 1.0 s. The last 0.3 s is the hold window.
  success = both fingers touch the cylinder (normal force > 0.01 N) in >= 80 % of hold steps
            AND total object displacement <= 25 mm   (evaluate_workspace.py criterion)
            AND vertical slip after gravity ON <= 5 mm (added: rejects a slowly sliding object)
  out_of_reach     cylinder lies beyond the fully stretched fingers (not simulated)
  initial_overlap  cylinder intersects the open gripper at t = 0 (cannot be approached)

Outputs (in --out-dir)
  grasp_map.csv           every (design, r, y) tested: reason, contact ratio, drift, slip, grasp type
  graspable_points.csv    only the successful (design, r, y) points
  grasp_summary.csv       one row per design: geometry, score, r and y ranges, y list per r
  grasp_config.json       the settings used

Examples
  python evaluate_grasp_workspace.py --xml-dir xml
  python evaluate_grasp_workspace.py --xml xml/design0001.xml
  python evaluate_grasp_workspace.py --xml-dir xml --workers 8 --r-max 80 --y-step 10   # faster, coarser
  python evaluate_grasp_workspace.py --xml-dir xml --resume        # continue an interrupted run
"""
import argparse
import csv
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

# MJCF geom name -> label used in the CSV
CONTACT_GEOMS = {
    'upper_green_link': 'upper_prox', 'upper_red_link': 'upper_dist',
    'lower_green_link': 'lower_prox', 'lower_red_link': 'lower_dist',
    'palm_geom': 'palm',
}
LINKS = ('upper_prox', 'upper_dist', 'lower_prox', 'lower_dist')
NOT_SIMULATED = ('out_of_reach', 'initial_overlap')

MAP_FIELDS = ['design_id', 'r_mm', 'y_mm', 'x_mm', 'obj_center_x_mujoco_mm', 'success', 'reason',
              'bilateral_contact_ratio', 'drift_mm', 'z_slip_mm', 'final_y_mm', 'grasp_type',
              'contacts_at_end']
POINT_FIELDS = ['design_id', 'r_mm', 'y_mm', 'x_mm', 'grasp_type', 'bilateral_contact_ratio',
                'drift_mm', 'z_slip_mm', 'final_y_mm']
SUMMARY_FIELDS = ['design_id', 'L_p_mm', 'L_d_mm', 'L_total_mm', 'L_p_ratio', 'G_mm',
                  'th_open_p_deg', 'th_open_d_deg', 'n_points', 'n_simulated', 'n_success', 'score',
                  'r_min_success_mm', 'r_max_success_mm', 'y_min_success_mm', 'y_max_success_mm',
                  'enveloping_count', 'tip_count', 'other_count', 'graspable_r_mm', 'y_by_r',
                  'runtime_s']


def grid(start, stop, step):  # millimetres, stop included
    return [round(float(v), 6) for v in np.arange(start, stop + step * 0.1, step)]


def _floats(text):
    return [float(t) for t in text.split()]


def design_info(root):
    """Geometry needed for placement, read back from the MJCF itself."""
    def el(tag, name):
        e = root.find(f".//{tag}[@name='{name}']")
        if e is None:
            raise ValueError(f'MJCF has no {tag} named {name!r} (expected Stage-0 naming)')
        return e

    def length(geom):
        ft = _floats(el('geom', geom).get('fromto'))
        return math.dist(ft[:3], ft[3:])

    palm = _floats(el('body', 'palm').get('pos', '0 0 0'))
    pg = el('geom', 'palm_geom')
    gpos, gsize = _floats(pg.get('pos', '0 0 0')), _floats(pg.get('size'))
    base = _floats(el('body', 'upper_link1').get('pos'))
    q = _floats(el('key', 'open').get('qpos'))
    L_p, L_d = length('upper_green_link'), length('upper_red_link')
    return {
        'L_p': L_p, 'L_d': L_d, 'G': 2 * abs(base[1]),
        'th_p': math.degrees(-q[0]), 'th_d': math.degrees(-q[1]),
        'face_x': palm[0] + gpos[0] - gsize[0],          # palm surface facing the fingers
        'pivot_x': palm[0] + base[0],
        'z': palm[2],
        'reach': L_p + L_d + float(el('geom', 'upper_green_link').get('size').split()[0]),
    }


def model_ids(model, tension):
    def oid(kind, name):
        i = mujoco.mj_name2id(model, kind, name)
        if i < 0:
            raise ValueError(f'{name!r} not found in compiled model')
        return i

    obj_joint = oid(mujoco.mjtObj.mjOBJ_JOINT, 'object_free')
    act = oid(mujoco.mjtObj.mjOBJ_ACTUATOR, 'pulley_pull')
    gear = abs(float(model.actuator_gear[act, 0]))
    ctrl = tension / gear
    lo, hi = model.actuator_ctrlrange[act]
    if model.actuator_ctrllimited[act] and not lo - 1e-9 <= ctrl <= hi + 1e-9:
        raise ValueError(f'tension {tension} N needs ctrl={ctrl:g}, outside ctrlrange [{lo:g}, {hi:g}]')
    return {
        'key': oid(mujoco.mjtObj.mjOBJ_KEY, 'open'),
        'obj_geom': oid(mujoco.mjtObj.mjOBJ_GEOM, 'object'),
        'qadr': int(model.jnt_qposadr[obj_joint]),
        'dadr': int(model.jnt_dofadr[obj_joint]),
        'act': act, 'ctrl': ctrl,
        'labels': {oid(mujoco.mjtObj.mjOBJ_GEOM, g): lab for g, lab in CONTACT_GEOMS.items()},
        'gravity': model.opt.gravity.copy(),
    }


def contact_pairs(data):
    """(ncon, 2) geom ids; works with MuJoCo 2.x (geom1/geom2) and 3.x (geom[2])."""
    n = data.ncon
    if n == 0:
        return np.empty((0, 2), dtype=int)
    con = data.contact
    try:
        g = np.asarray(con.geom)
        if g.ndim == 2:
            return g[:n]
    except AttributeError:
        pass
    return np.column_stack((np.asarray(con.geom1)[:n], np.asarray(con.geom2)[:n]))


def initial_overlap(data, obj, tol):
    pairs = contact_pairs(data)
    if len(pairs) == 0:
        return False
    dist = np.asarray(data.contact.dist)[:len(pairs)]
    mask = (pairs[:, 0] == obj) | (pairs[:, 1] == obj)
    return bool(np.any(dist[mask] < -tol))


def touched_links(model, data, ids, f6, min_force):
    obj, out = ids['obj_geom'], set()
    for j, (g1, g2) in enumerate(contact_pairs(data)):
        other = g2 if g1 == obj else g1 if g2 == obj else -1
        label = ids['labels'].get(int(other))
        if label is None or label in out:
            continue
        mujoco.mj_contactForce(model, data, j, f6)
        if f6[0] > min_force:  # normal force [N]
            out.add(label)
    return out


def unstable(data):
    if not np.all(np.isfinite(data.qpos)):
        return True
    try:
        return data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number > 0
    except (AttributeError, IndexError, TypeError):
        return False


def run_trial(model, data, ids, center, r, info, cfg):
    dt = model.opt.timestep
    n = max(1, round(cfg['duration'] / dt))
    hold = min(n, max(1, round(cfg['hold'] / dt)))
    ramp = max(1, round(cfg['ramp'] / dt))
    qa, da = ids['qadr'], ids['dadr']

    mujoco.mj_resetDataKeyframe(model, data, ids['key'])  # open angles from the design JSON
    data.qpos[qa:qa + 3] = center
    data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[da:da + 6] = 0.0
    data.ctrl[:] = 0.0
    model.opt.gravity[:] = 0.0  # grasp first, test gravity after the closing ramp
    mujoco.mj_forward(model, data)
    blank = {'bilateral_contact_ratio': '', 'drift_mm': '', 'z_slip_mm': '', 'final_y_mm': '',
             'grasp_type': '', 'contacts_at_end': ''}
    if initial_overlap(data, ids['obj_geom'], cfg['overlap_tol']):
        model.opt.gravity[:] = ids['gravity']
        return {**blank, 'success': 0, 'reason': 'initial_overlap'}

    p0 = data.qpos[qa:qa + 3].copy()
    z_on = float(p0[2])
    f6 = np.zeros(6)
    hits = env = tip = other = 0
    touched = set()
    for i in range(n):
        data.ctrl[ids['act']] = ids['ctrl'] * min(1.0, (i + 1) / ramp)
        if i == ramp:
            model.opt.gravity[:] = ids['gravity']
            z_on = float(data.qpos[qa + 2])
        mujoco.mj_step(model, data)
        if i < n - hold:
            continue
        touched = touched_links(model, data, ids, f6, cfg['min_force'])
        if any(t.startswith('upper') for t in touched) and any(t.startswith('lower') for t in touched):
            hits += 1
            if all(link in touched for link in LINKS):
                env += 1
            elif 'upper_dist' in touched and 'lower_dist' in touched:
                tip += 1
            else:
                other += 1
    model.opt.gravity[:] = ids['gravity']

    pos = data.qpos[qa:qa + 3].copy()
    ratio = hits / hold
    drift = float(np.linalg.norm(pos - p0))
    slip = z_on - float(pos[2])  # + = dropped
    if unstable(data):
        reason = 'sim_unstable'
    elif ratio < cfg['min_ratio']:
        reason = 'no_bilateral_contact'
    elif abs(slip) > cfg['max_slip']:
        reason = 'slip_under_gravity'
    elif drift > cfg['max_drift']:
        reason = 'excess_drift'
    else:
        reason = 'success'
    kind = max((('enveloping', env), ('tip', tip), ('other', other)), key=lambda t: t[1])[0] if hits else ''
    return {
        'success': int(reason == 'success'), 'reason': reason,
        'bilateral_contact_ratio': round(ratio, 4), 'drift_mm': round(drift * 1000, 3),
        'z_slip_mm': round(slip * 1000, 3),
        'final_y_mm': round((info['face_x'] - (pos[0] + r)) * 1000, 3) if np.isfinite(pos[0]) else '',
        'grasp_type': kind, 'contacts_at_end': '+'.join(sorted(touched)),
    }


def summarize(design_id, info, rows, runtime):
    ok = [row for row in rows if row['success'] == 1]
    radii = sorted({row['r_mm'] for row in ok})
    ys = [row['y_mm'] for row in ok]
    kinds = [row['grasp_type'] for row in ok]
    lt = info['L_p'] + info['L_d']
    return {
        'design_id': design_id,
        'L_p_mm': round(info['L_p'] * 1000, 4), 'L_d_mm': round(info['L_d'] * 1000, 4),
        'L_total_mm': round(lt * 1000, 4), 'L_p_ratio': round(info['L_p'] / lt, 5),
        'G_mm': round(info['G'] * 1000, 4),
        'th_open_p_deg': round(info['th_p'], 4), 'th_open_d_deg': round(info['th_d'], 4),
        'n_points': len(rows),
        'n_simulated': sum(row['reason'] not in NOT_SIMULATED for row in rows),
        'n_success': len(ok), 'score': round(len(ok) / len(rows), 6) if rows else 0,
        'r_min_success_mm': min(radii, default=''), 'r_max_success_mm': max(radii, default=''),
        'y_min_success_mm': min(ys, default=''), 'y_max_success_mm': max(ys, default=''),
        'enveloping_count': kinds.count('enveloping'), 'tip_count': kinds.count('tip'),
        'other_count': kinds.count('other'),
        'graspable_r_mm': ';'.join(f'{v:g}' for v in radii),
        'y_by_r': ' | '.join(f'r{rv:g}: ' + ','.join(f"{row['y_mm']:g}" for row in ok if row['r_mm'] == rv)
                             for rv in radii),
        'runtime_s': round(runtime, 2),
    }


def evaluate_design(job):
    path, cfg = job
    t0 = time.time()
    tree = ET.parse(path)
    root = tree.getroot()
    info = design_info(root)
    obj = root.find(".//geom[@name='object']")
    if obj is None:
        raise ValueError(f'{path}: object geom not found')
    half_h = cfg['half_height'] if cfg['half_height'] else _floats(obj.get('size'))[1]
    if cfg['mass'] is not None:
        obj.set('mass', f"{cfg['mass']:.12g}")
    x_lat = cfg['x_mm'] / 1000
    rows = []
    for r_mm in cfg['r_values']:
        r = r_mm / 1000
        obj.set('size', f'{r:.12g} {half_h:.12g}')
        model = data = ids = None  # compiled once per radius (collision bounds + inertia)
        for y_mm in cfg['y_values']:
            y = y_mm / 1000
            cx = info['face_x'] - y - r
            row = {'design_id': path.stem, 'r_mm': r_mm, 'y_mm': y_mm, 'x_mm': cfg['x_mm'],
                   'obj_center_x_mujoco_mm': round(cx * 1000, 4)}
            if info['face_x'] - y < info['pivot_x'] - info['reach']:
                row.update(success=0, reason='out_of_reach', bilateral_contact_ratio='', drift_mm='',
                           z_slip_mm='', final_y_mm='', grasp_type='', contacts_at_end='')
            else:
                if model is None:
                    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
                    data = mujoco.MjData(model)
                    ids = model_ids(model, cfg['tension'])
                row.update(run_trial(model, data, ids, (cx, x_lat, info['z']), r, info, cfg))
            rows.append(row)
    runtime = time.time() - t0
    return path.stem, rows, summarize(path.stem, info, rows, runtime)


def open_csv(path, fields, append):
    exists = append and path.exists() and path.stat().st_size > 0
    f = path.open('a' if exists else 'w', newline='', encoding='utf-8-sig' if not exists else 'utf-8')
    w = csv.DictWriter(f, fieldnames=fields)
    if not exists:
        w.writeheader()
    return f, w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--xml', type=Path)
    src.add_argument('--xml-dir', type=Path)
    p.add_argument('--out-dir', type=Path, default=Path('workspace_results'))
    p.add_argument('--r-min', type=float, default=10, help='mm')
    p.add_argument('--r-max', type=float, default=100, help='mm (straight fingers enclose up to ~97 mm)')
    p.add_argument('--r-step', type=float, default=5, help='mm')
    p.add_argument('--y-min', type=float, default=0, help='mm, palm face to cylinder surface')
    p.add_argument('--y-max', type=float, default=140, help='mm')
    p.add_argument('--y-step', type=float, default=5, help='mm')
    p.add_argument('--x-mm', type=float, default=0.0, help='lateral offset of cylinder axis (fixed, default 0)')
    p.add_argument('--tension', type=float, default=10.0, help='N, tendon pull (ctrl = tension/|gear|)')
    p.add_argument('--duration', type=float, default=1.0, help='s')
    p.add_argument('--ramp', type=float, default=0.3, help='s, closing ramp with gravity OFF')
    p.add_argument('--hold', type=float, default=0.3, help='s, final evaluation window')
    p.add_argument('--min-ratio', type=float, default=0.8, help='bilateral contact ratio in hold window')
    p.add_argument('--min-force', type=float, default=0.01, help='N, contact counted above this normal force')
    p.add_argument('--max-drift', type=float, default=25, help='mm, total object displacement')
    p.add_argument('--max-slip', type=float, default=5, help='mm, vertical slip after gravity ON')
    p.add_argument('--overlap-tol', type=float, default=0.1, help='mm, penetration counted as initial overlap')
    p.add_argument('--mass-kg', type=float, default=None, help='override object mass (default: XML, 0.3 kg)')
    p.add_argument('--half-height-mm', type=float, default=None, help='override half-height (default: XML, 19 mm)')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--limit', type=int, default=None, help='evaluate only the first N XML files')
    p.add_argument('--resume', action='store_true', help='skip designs already in grasp_summary.csv')
    a = p.parse_args()

    if min(a.r_step, a.y_step, a.duration, a.ramp, a.hold) <= 0:
        p.error('steps, duration, ramp and hold must be positive')
    if a.ramp + a.hold > a.duration + 1e-9:
        p.error('ramp + hold must be <= duration (hold window must be after gravity is ON)')
    if a.r_min <= 0 or a.r_max < a.r_min or a.y_min < 0 or a.y_max < a.y_min:
        p.error('invalid r or y range')

    xmls = sorted(a.xml_dir.glob('*.xml')) if a.xml_dir else [a.xml]
    if a.limit:
        xmls = xmls[:a.limit]
    if not xmls:
        p.error('no XML files found')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    f_map, f_pts, f_sum = (a.out_dir / n for n in ('grasp_map.csv', 'graspable_points.csv', 'grasp_summary.csv'))
    done = set()
    if a.resume and f_sum.exists():
        with f_sum.open(encoding='utf-8-sig') as f:
            done = {row['design_id'] for row in csv.DictReader(f)}
    todo = [x for x in xmls if x.stem not in done]

    cfg = {
        'r_values': grid(a.r_min, a.r_max, a.r_step), 'y_values': grid(a.y_min, a.y_max, a.y_step),
        'x_mm': a.x_mm, 'tension': a.tension, 'duration': a.duration, 'ramp': a.ramp, 'hold': a.hold,
        'min_ratio': a.min_ratio, 'min_force': a.min_force, 'max_drift': a.max_drift / 1000,
        'max_slip': a.max_slip / 1000, 'overlap_tol': a.overlap_tol / 1000,
        'mass': a.mass_kg, 'half_height': None if a.half_height_mm is None else a.half_height_mm / 1000,
    }
    (a.out_dir / 'grasp_config.json').write_text(json.dumps(
        {**cfg, 'frame': 'x = lateral offset (MuJoCo y), y = palm face to cylinder surface (MuJoCo -x)',
         'units': 'r/y/x lists in mm; max_drift/max_slip/overlap_tol/half_height in m; tension in N'},
        indent=2) + '\n', encoding='utf-8')

    n_pts = len(cfg['r_values']) * len(cfg['y_values'])
    print(f'{len(todo)} design(s) to evaluate ({len(done)} skipped by --resume), '
          f'{n_pts} (r, y) points each, workers={a.workers}')
    handles = [open_csv(f, fields, a.resume) for f, fields in
               ((f_map, MAP_FIELDS), (f_pts, POINT_FIELDS), (f_sum, SUMMARY_FIELDS))]
    (fm, wm), (fp, wp), (fs, ws) = handles
    jobs = [(x, cfg) for x in todo]
    t0 = time.time()
    pool = Pool(a.workers) if a.workers > 1 and len(jobs) > 1 else None
    try:
        results = pool.imap(evaluate_design, jobs) if pool is not None else map(evaluate_design, jobs)
        for k, (design_id, rows, summary) in enumerate(results, 1):
            wm.writerows(rows)
            wp.writerows({key: row[key] for key in POINT_FIELDS} for row in rows if row['success'] == 1)
            ws.writerow(summary)
            for f, _ in handles:
                f.flush()
            rng = (f"r {summary['r_min_success_mm']}-{summary['r_max_success_mm']} mm, "
                   f"y {summary['y_min_success_mm']}-{summary['y_max_success_mm']} mm"
                   if summary['n_success'] else 'nothing graspable')
            print(f"[{k}/{len(jobs)}] {design_id}: {summary['n_success']}/{summary['n_points']} graspable "
                  f"({rng}; {summary['runtime_s']} s)", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
        for f, _ in handles:
            f.close()
    print(f'Done in {time.time() - t0:.1f} s. Results in {a.out_dir}/: '
          'grasp_map.csv, graspable_points.csv, grasp_summary.csv')


if __name__ == '__main__':
    main()
