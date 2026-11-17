#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 A — 천천히 커지는 외력에 대한 파지 유지 한계 F_max (RB5 + grasp_cells)

on_rb5의 A/disturb_force.py와 같은 정의를, run_rb5_workspace.py가 저장한 grasp_cells/<design>/<cell>/
(scene.xml, state.npz, summary.json)에 적용한다. 출발점은 실험 B(disturb_impulse_rb5.py)와 같다:
state.npz 복원 → 외력 없이 --pre-s(0.5 s) 안정화 → 이 상태(S0)에서 모든 시행.

  ramp      물체 질량중심에 xfrc_applied로 F(t) = rate·t (기본 1 mg/s, 상한 10 mg), 손 좌표 방향
  F_max     slip / rotation / lost / collision 중 처음 걸린 순간의 F. 상한까지 버티면 censored
  외력 0 대조  같은 S0를 ramp 전체 시간(10 s) 동안 외력 없이. 실패한 cell은 zero_fail (지표에서 제외)
  --refine  hold test: 일정한 힘(0.5 s 동안 부드럽게 올린 뒤 1 s 유지)을 버티는 최대 힘을 bisection

  python disturb_force_rb5.py --designs design0001 --limit 3 --workers 2       # 빠른 시험
  python disturb_force_rb5.py --workers 6                                      # 전체 (끊기면 같은 명령으로 이어서)
  python disturb_force_rb5.py --rescore --slip-mm 30                           # 판정 기준만 바꿔 다시 판정
  python disturb_force_rb5.py --dirs edges --workers 6                         # 대각선 12방향 추가 (새 방향만 돌림)
  python disturb_force_rb5.py --designs design0023 design0111 --refine         # 상위 설계 hold test 추가

셀 입력: 기본은 이 파일 옆 grasp_cells_xml/ (collect_cell_xmls.py가 만든 XML 하나 = cell 하나,
상태는 <key name="lifted">, summary는 <custom>). 없으면 grasp_cells/ (폴더, state.npz). --cells-dir로 지정.
--clean: 시작 전에 results_X 안의 기존 결과를 모두 지움.
disturb_impulse_rb5.py와 같은 폴더에 두어야 한다 (장면 읽기, 손 좌표계, 접촉력 계산을 같이 쓴다).
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
import disturb_impulse_rb5 as B  # noqa: E402

DEFAULT_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
SIM_KEYS = ('rate_mg_s', 'cap_mg', 'pre_s', 'sample_s', 'check_ms', 'stop_mm', 'stop_lost_s', 'lost_n', 'tension')
COLS = ['t', 'F', 'disp', 'rot', 'tilt', 'lost', 'ext', 'side_min', 'dx', 'dy', 'dz', 'hand', 'palm_f', 'side0', 'side1']
GROUPS = ('env', 'tip')


def sim_hash(sim):
    return hashlib.sha1(json.dumps({k: sim[k] for k in SIM_KEYS}, sort_keys=True).encode()).hexdigest()[:12]


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


# ----------------------------------------------------------------------------- simulation
def run_load(m, d, S0, spec, ctx, ref, fworld, force_at, t_end, sim, stop):
    """S0에서 시작해 F = force_at(t) [N]를 fworld 방향으로 t_end 동안. sample_s마다 한 줄.

    stop: dict(slip_mm, lost_s, rot_deg(None이면 끔), rot_mode, collision_n(None이면 끔)) -> ended 'drop'.
    반환 (rows float32[n, len(COLS)+nl], ended: end / drop / unstable)."""
    import mujoco
    dt = m.opt.timestep
    n_total = max(1, int(math.ceil(t_end / dt - 1e-9)))
    every = max(1, int(round(sim['sample_s'] / dt)))
    ce = max(1, int(round(sim['check_ms'] / 1000.0 / dt)))
    p0, R0, ax0, palm0 = ref
    H = ctx.H
    mujoco.mj_setState(m, d, S0, spec)
    d.xfrc_applied[:] = 0.0
    d.qfrc_applied[:] = 0.0
    w0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    obj = ctx.obj
    xf = d.xfrc_applied
    lost_n = sim['lost_n']
    cdt = ce * dt
    rot_stop = stop.get('rot_deg') or 0.0
    tilt_mode = stop.get('rot_mode') == 'tilt'
    col_stop = stop.get('collision_n')
    rows = []
    dmax = rmax = tmax = lostmax = extmax = 0.0
    smin = math.inf
    lsum = np.zeros(ctx.nl)
    lcnt = 0
    streak = [0, 0]
    ended = 'end'
    F = 0.0
    for i in range(n_total):
        F = float(force_at((i + 1) * dt))
        xf[obj, :3] = fworld * F
        mujoco.mj_step(m, d)
        step = i + 1
        at_row = step % every == 0 or step == n_total
        if step % ce and not at_row:
            continue
        p, Rr, ax, pp, lf, ext, palmf = B.measure(m, d, ctx)
        dp = p - p0
        disp = 1000.0 * math.sqrt(dp @ dp)
        rot = B._angle_deg((np.trace(R0.T @ Rr) - 1.0) / 2.0)
        tilt = B._angle_deg(float(ax @ ax0))
        s0, s1 = lf[ctx.side0].sum(), lf[ctx.side1].sum()
        streak[0] = streak[0] + 1 if s0 < lost_n else 0
        streak[1] = streak[1] + 1 if s1 < lost_n else 0
        lost = max(streak) * cdt
        dmax, rmax, tmax = max(dmax, disp), max(rmax, rot), max(tmax, tilt)
        lostmax, extmax, smin = max(lostmax, lost), max(extmax, ext), min(smin, s0, s1)
        lsum += lf
        lcnt += 1
        rv = tilt if tilt_mode else rot
        stop_now = (disp >= stop['slip_mm'] or lost >= stop['lost_s'] - 1e-12
                    or (rot_stop > 0 and rv >= rot_stop) or (col_stop is not None and ext >= col_stop))
        if at_row or stop_now:
            bad = (int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > w0 or not np.isfinite(d.qpos).all())
            dh = H.T @ dp * 1000.0
            rows.append([step * dt, F, dmax, rmax, tmax, lostmax, extmax, smin, dh[0], dh[1], dh[2],
                         1000.0 * float(np.linalg.norm(pp - palm0)), palmf, s0, s1] + list(lsum / max(lcnt, 1)))
            dmax = rmax = tmax = lostmax = extmax = 0.0
            smin = math.inf
            lsum[:] = 0.0
            lcnt = 0
            if bad:
                ended = 'unstable'
                break
            if stop_now:
                ended = 'drop'
                break
    xf[obj, :3] = 0.0
    return np.asarray(rows, dtype=np.float32), ended


def load_trace(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['__meta__']))
        cols = [str(c) for c in z['columns']]
        trials, extra = {}, {}
        for k in z.files:
            if k.endswith('__trace'):
                trials[k[:-7]] = (z[k], str(z[k[:-7] + '__ended']))
            elif '__hold' in k:
                extra[k] = z[k]
    return meta, cols, trials, extra


def save_trace(path, meta, cols, trials, extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {'__meta__': np.array(json.dumps(meta)), 'columns': np.array(cols)}
    for key, (rows, ended) in trials.items():
        arrs[key + '__trace'] = rows
        arrs[key + '__ended'] = np.array(ended)
    arrs.update(extra)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def prepare_cell(job):
    """모델·S0 준비 (실험 B와 같은 출발점)."""
    import mujoco
    sim = job['sim']
    m, d, _, summary, cinfo = B.load_cell(job['cell_dir'], job['mesh_roots'], job['allow_missing_collision'])
    dropped, dropped_col = cinfo['dropped_meshes'], cinfo['dropped_collision_meshes']
    ctx = B.Ctx(m, summary)
    d.qfrc_applied[:] = 0.0
    d.xfrc_applied[:] = 0.0
    if sim['tension'] is not None:
        for a in ctx.motors:
            c = -float(sim['tension']) / m.actuator_gear[a, 0]
            lo, hi = m.actuator_ctrlrange[a]
            m.actuator_ctrlrange[a] = [min(lo, c), max(hi, c)]
            d.ctrl[a] = c
    tension_used = [float(-d.ctrl[a] * m.actuator_gear[a, 0]) for a in ctx.motors]
    dt = m.opt.timestep
    mg = float(m.body_subtreemass[ctx.obj] * np.linalg.norm(m.opt.gravity))
    n_pre = int(round(sim['pre_s'] / dt))
    n_avg = min(n_pre, int(round(0.1 / dt)))
    lsum = np.zeros(ctx.nl)
    for i in range(n_pre):
        mujoco.mj_step(m, d)
        if i >= n_pre - n_avg:
            lsum += B.measure(m, d, ctx)[4]
    start_lf = lsum / n_avg if n_avg else B.measure(m, d, ctx)[4]
    full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    S0 = np.zeros(mujoco.mj_stateSize(m, full))
    mujoco.mj_getState(m, d, S0, full)
    p0, R0, ax0, palm0, _, _, _ = B.measure(m, d, ctx)
    Rpalm = d.xmat[ctx.palm].reshape(3, 3).copy()
    cell_meta = {
        'cell_id': job['cell_id'], 'design': job['design'], 'cell': job['cell'], 'sim_hash': job['sim_hash'],
        'sim': {k: sim[k] for k in SIM_KEYS},
        'r_mm': (summary.get('cell') or {}).get('r_mm'), 'y_mm': (summary.get('cell') or {}).get('y_mm'),
        'x_mm': (summary.get('cell') or {}).get('x_mm'),
        'workspace_grasp_type': (summary.get('cell') or {}).get('workspace_grasp_type'),
        'rb5_four_link_contact': summary.get('four_link_contact'),
        'mass_kg': float(m.body_subtreemass[ctx.obj]), 'mg_N': mg, 'tension_N': tension_used, 'timestep': dt,
        'start_type': B.contact_type(start_lf, ctx, sim['lost_n']),
        'start_link_forces_N': [float(x) for x in start_lf], 'link_names': ctx.link_names,
        'link_side': [int(s) for s in ctx.side], 'hand_frame_world': (Rpalm @ ctx.H).tolist(),
        'dropped_meshes': dropped, 'dropped_collision_meshes': dropped_col, 'mujoco_version': mujoco.__version__,
        'cell_path': cinfo['cell_path'], 'state_source': cinfo['state_source'],
        'summary_source': cinfo['summary_source'],
    }
    return m, d, ctx, S0, full, (p0, R0, ax0, palm0), Rpalm, mg, cell_meta


def hold_test(m, d, S0, spec, ctx, ref, fworld, sim, crit, f_ramp_N, ramp_censored, cap_N, tol_N, rise, hold):
    """일정한 힘의 bisection (on_rb5 A/disturb_force.py refine과 같은 규칙). (F_hold_N, censored, trials)"""
    stop = dict(slip_mm=crit['slip_mm'], lost_s=crit['lost_s'], rot_deg=crit['rot_deg'], rot_mode=crit['rot_mode'],
                collision_n=crit['collision_n'])
    trials = []

    def held(F):
        rows, ended = run_load(m, d, S0, spec, ctx, ref, fworld,
                               (lambda t: F * smooth(t / rise)) if rise > 0 else (lambda t: F),
                               rise + hold, sim, stop)
        ok = ended == 'end'
        trials.append((F, float(ok), float(rows[-1, 0]) if len(rows) else 0.0))
        return ok

    hi = cap_N if ramp_censored else min(cap_N, max(1.25 * f_ramp_N, f_ramp_N + 2 * tol_N))
    if held(hi):
        return hi, True, trials
    lo = 0.0
    while hi - lo > tol_N:
        mid = 0.5 * (lo + hi)
        if held(mid):
            lo = mid
        else:
            hi = mid
    return lo, False, trials



def simulate_cell(job):
    t_start = time.time()
    try:
        sim = job['sim']
        m, d, ctx, S0, spec, ref, Rpalm, mg, cell_meta = prepare_cell(job)
        cols = COLS + [f'f_{n.replace("hand__", "")}' for n in ctx.link_names]
        out = Path(job['trace_path'])
        trials, extra = {}, {}
        if out.is_file() and not job['overwrite']:
            try:
                old_meta, old_cols, trials, extra = load_trace(out)
                if old_meta.get('sim_hash') != job['sim_hash'] or old_cols != cols:
                    trials, extra = {}, {}
                else:
                    cell_meta['refine_criteria'] = old_meta.get('refine_criteria')
            except Exception:
                trials, extra = {}, {}
        rate_N = sim['rate_mg_s'] * mg
        cap_N = sim['cap_mg'] * mg
        T_ramp = cap_N / rate_N
        ramp_stop = dict(slip_mm=sim['stop_mm'], lost_s=sim['stop_lost_s'], rot_deg=None, collision_n=None)
        ran = 0
        for key in job['keys']:
            if key in trials:
                continue
            if key == 'z0':
                rows, ended = run_load(m, d, S0, spec, ctx, ref, np.zeros(3), lambda t: 0.0, T_ramp, sim, ramp_stop)
                ended = 'held' if ended == 'end' else ended
            else:
                fworld = Rpalm @ ctx.H @ np.array(B.ALL_DIRS[key])
                rows, ended = run_load(m, d, S0, spec, ctx, ref, fworld, lambda t: min(rate_N * t, cap_N),
                                       T_ramp, sim, ramp_stop)
                ended = 'cap' if ended == 'end' else ended
            trials[key] = (rows, ended)
            ran += 1
        if job['refine']:
            crit = job['crit']
            for key in job['keys']:
                if key == 'z0' or f'{key}__hold_result' in extra:
                    continue
                j = judge_ramp(*trials[key], cols, crit, cap_N)
                fworld = Rpalm @ ctx.H @ np.array(B.ALL_DIRS[key])
                f_hold, cens, tr = hold_test(m, d, S0, spec, ctx, ref, fworld, sim, crit, j['F_max_N'], j['censored'],
                                             cap_N, job['tol_mg'] * mg,
                                             job['hold']['rise_s'], job['hold']['hold_s'])
                extra[f'{key}__hold_result'] = np.array([f_hold, float(cens)])
                extra[f'{key}__hold_trials'] = np.array(tr, dtype=np.float64)
                ran += len(tr)
            cell_meta['refine_criteria'] = crit
            cell_meta['hold'] = dict(job['hold'], tol_mg=job['tol_mg'])
        save_trace(out, cell_meta, cols, trials, extra)
        return {'cell_id': job['cell_id'], 'ran': ran, 'sec': time.time() - t_start, 'error': None}
    except Exception:
        return {'cell_id': job['cell_id'], 'ran': 0, 'sec': time.time() - t_start, 'error': traceback.format_exc()}


# ----------------------------------------------------------------------------- judging
def judge_ramp(rows, ended, cols, crit, cap_N, types=None, start_type=None):
    """on_rb5 A/disturb_force.py judge_trace와 같은 규칙."""
    c = {n: i for i, n in enumerate(cols)}
    rows = np.asarray(rows, dtype=float)
    n = len(rows)
    F = rows[:, c['F']]
    rv = rows[:, c['tilt' if crit['rot_mode'] == 'tilt' else 'rot']]
    masks = [('slip', rows[:, c['disp']] >= crit['slip_mm']),
             ('rotation', (rv >= crit['rot_deg']) if crit['rot_deg'] > 0 else np.zeros(n, bool)),
             ('lost', rows[:, c['lost']] >= crit['lost_s'] - 1e-9),
             ('collision', rows[:, c['ext']] >= crit['collision_n'])]
    first, cause = n, None
    for name, mk in masks:
        idx = np.flatnonzero(mk)
        if idx.size and idx[0] < first:
            first, cause = int(idx[0]), name
    if cause is not None:
        i, censored = first, False
    elif ended == 'drop':          # 판정 기준이 중단 기준보다 느슨할 때만
        cause, i, censored = 'drop', n - 1, False
    else:                           # cap: 끝까지 버팀 / unstable: 하한값
        cause, i, censored = ended, n - 1, True
    F_max = cap_N if cause == 'cap' else float(F[i])
    disp = rows[:, c['disp']]
    out = dict(F_max_N=F_max, cause=cause, censored=int(censored), ended=ended, t_Fmax_s=float(rows[i, c['t']]),
               max_disp_mm=float(disp.max()), disp_at_Fmax_mm=float(disp[i]), hand_at_Fmax_mm=float(rows[i, c['hand']]),
               max_rot_deg=float(rows[:, c['rot']].max()))
    if types is not None:
        ref = start_type or str(types[0])
        ch = np.flatnonzero(types != ref)
        k = int(ch[0]) if ch.size else None
        out.update(type_start=ref, type_at_Fmax=str(types[i]), type_change_to=str(types[k]) if k is not None else '',
                   F_type_change_N=float(F[k]) if k is not None else math.nan,
                   type_changed_before_Fmax=int(k is not None and k <= i))
    return out


def zero_judge(rows, ended, cols, crit):
    c = {n: i for i, n in enumerate(cols)}
    rows = np.asarray(rows, dtype=float)
    j = judge_ramp(rows, ended, cols, crit, 0.0)
    fail = j['cause'] not in ('held',) and not (j['censored'] and ended == 'held')
    return dict(zero_fail=int(fail), zero_cause='' if not fail else j['cause'],
                zero_t_fail_s=j['t_Fmax_s'] if fail else math.nan,
                zero_max_disp_mm=float(rows[:, c['disp']].max()), zero_max_rot_deg=float(rows[:, c['rot']].max()))


def row_types(rows, cols, side, thr, window=10):
    """링크 접촉력 0.1 s 이동평균으로 줄마다 접촉 형태."""
    li = [i for i, n in enumerate(cols) if n.startswith('f_')]
    f = np.asarray(rows, dtype=float)[:, li]
    cs = np.vstack([np.zeros((1, f.shape[1])), np.cumsum(f, axis=0)])
    n = np.arange(1, len(f) + 1)
    lo = np.maximum(0, n - window)
    avg = (cs[n] - cs[lo]) / (n - lo)[:, None]
    lfc = B._LF(side)
    return np.array([B.contact_type(a, lfc, thr) for a in avg])


def stats(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if not v.size:
        return dict(n=0)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return dict(n=int(v.size), median=float(med), q25=float(q25), q75=float(q75))


def score_all(out_dir, cells_dir, meta, crit, dirs, levels, metric, w_tip, design_index, designs_filter):
    traces = sorted(p for p in out_dir.glob('*/traces/*.npz') if not p.name.endswith('.tmp.npz'))
    per_design, stale = {}, 0
    for tp in traces:
        design = tp.parent.parent.name
        if designs_filter and design not in designs_filter:
            continue
        try:
            got = load_trace(tp)
        except Exception as e:
            print(f'[WARN] cannot read {tp}: {e}')
            continue
        if got[0].get('sim_hash') != meta['sim_hash']:
            stale += 1
            continue
        per_design.setdefault(design, []).append(got)
    pick = B.read_csv(cells_dir / 'rb5_pick_results.csv') or []
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
    dvars, dvar_cols = B.load_design_vars(design_index, designs)
    all_rows, dstat_rows = [], []
    tot = dict(cells=0, success=0, zero_fail=0, incomplete=0, env=0, tip=0)
    hold_warned = False
    for design in designs:
        cells = sorted(per_design.get(design, []), key=lambda g: (g[0].get('r_mm') or 0, g[0].get('y_mm') or 0))
        dir_rows, cell_rows = [], []
        for tmeta, cols, trials, extra in cells:
            mg = tmeta['mg_N']
            cap_N = tmeta['sim']['cap_mg'] * mg
            thr = tmeta['sim']['lost_n']
            base = dict(design=design, cell=B.compact_cell_name(tmeta['cell']), r_mm=tmeta.get('r_mm'), y_mm=tmeta.get('y_mm'),
                        workspace_grasp_type=tmeta.get('workspace_grasp_type'), start_type=tmeta['start_type'])
            z = zero_judge(*trials['z0'], cols, crit) if 'z0' in trials else None
            grp0 = tmeta['start_type'] if tmeta['start_type'] in GROUPS else 'other'
            missing = [dn for dn in dirs if dn not in trials]
            if z is None or missing:
                group = 'incomplete'
            elif z['zero_fail'] and crit['zero_fail'] == 'fail':
                group = 'zero_fail'
            else:
                group = grp0
            if extra and not hold_warned and tmeta.get('refine_criteria') not in (None, crit) and metric == 'hold':
                print('[WARN] hold tests were judged during simulation with '
                      f'{tmeta.get("refine_criteria")}; only the ramp is re-judged.')
                hold_warned = True
            per = {}
            for dn in dirs:
                if dn not in trials:
                    continue
                rows, ended = trials[dn]
                types = row_types(rows, cols, tmeta['link_side'], thr)
                j = judge_ramp(rows, ended, cols, crit, cap_N, types, tmeta['start_type'])
                j['F_max_mg'] = j['F_max_N'] / mg
                j['creep_limited'] = int(bool(z and z['zero_fail'] and not j['censored']
                                              and z['zero_t_fail_s'] <= j['t_Fmax_s'] + 1e-9))
                hr = extra.get(f'{dn}__hold_result')
                if hr is not None:
                    j['F_hold_mg'] = float(hr[0]) / mg
                    j['hold_censored'] = int(hr[1] > 0.5)
                    j['hold_trials'] = int(len(extra.get(f'{dn}__hold_trials', [])))
                if metric == 'hold':
                    j['F_mg'] = j.get('F_hold_mg', math.nan)
                else:
                    j['F_mg'] = j['F_max_mg']
                per[dn] = j
                dir_rows.append({**base, 'group': group, 'dir': dn, **{k: v for k, v in j.items() if k != 'F_max_N'}})
            row = {**base, 'group': group, **(z or {})}
            if per and not missing:
                vals = {dn: per[dn]['F_mg'] for dn in dirs}
                if any(math.isnan(v) for v in vals.values()):
                    row['group'] = group = 'incomplete'
                else:
                    weakest = min(vals, key=vals.get)
                    row.update(F_min_mg=vals[weakest], weakest_dir=weakest, weakest_cause=per[weakest]['cause'],
                               weakest_creep_limited=per[weakest]['creep_limited'],
                               any_type_change=int(any(per[dn].get('type_changed_before_Fmax') for dn in dirs)),
                               any_censored=int(any(per[dn]['censored'] for dn in dirs)))
                    row.update({f'F_{dn}_mg': vals[dn] for dn in dirs})
                    if all('F_hold_mg' in per[dn] for dn in dirs) and metric == 'ramp':
                        row['F_min_hold_mg'] = min(per[dn]['F_hold_mg'] for dn in dirs)
                    for lv in levels:
                        row[f'pass@{lv:g}'] = int(group in GROUPS + ('other',) and vals[weakest] >= lv - 1e-9)
            cell_rows.append(row)
        if cells:
            B.write_csv(out_dir / design / 'force_dirs.csv', dir_rows)
            B.write_csv(out_dir / design / 'force_cells.csv', cell_rows)
        ok = [c for c in cell_rows if c['group'] in GROUPS + ('other',)]
        env = [c for c in ok if c['group'] == 'env']
        tip = [c for c in ok if c['group'] != 'env']
        cap = meta['sim']['cap_mg']

        def area(rs):
            return float(sum(min(c['F_min_mg'], cap) for c in rs))

        row = dict(design=design, ws_cells_attempted=attempted.get(design, ''), rb5_pass_cells=rb5_pass.get(design, ''),
                   traced_cells=len(cell_rows), incomplete=sum(c['group'] == 'incomplete' for c in cell_rows),
                   zero_fail=sum(c['group'] == 'zero_fail' for c in cell_rows), success=len(ok),
                   success_env=len(env), success_tip=len(tip), J_stable=len(env) + w_tip * len(tip),
                   robust_area=area(ok), robust_area_env=area(env), robust_area_tip=area(tip),
                   F_min_median_mg=float(np.median([c['F_min_mg'] for c in ok])) if ok else math.nan,
                   F_min_median_env_mg=float(np.median([c['F_min_mg'] for c in env])) if env else math.nan,
                   F_min_median_tip_mg=float(np.median([c['F_min_mg'] for c in tip])) if tip else math.nan,
                   F_min_mean_mg=float(np.mean([min(c['F_min_mg'], cap) for c in ok])) if ok else math.nan)
        for lv in levels:
            pe = sum(c[f'pass@{lv:g}'] for c in env)
            pt = sum(c[f'pass@{lv:g}'] for c in tip)
            row[f'pass@{lv:g}'] = pe + pt
            row[f'cond@{lv:g}'] = (pe + pt) / len(ok) if ok else math.nan
            row[f'env_pass@{lv:g}'] = pe
            row[f'tip_pass@{lv:g}'] = pt
            row[f'J_robust@{lv:g}'] = pe + w_tip * pt
        okset = {c['cell'] for c in ok}
        dok = [r for r in dir_rows if r['cell'] in okset]
        for cs in ('slip', 'rotation', 'lost', 'collision', 'cap', 'unstable', 'drop'):
            row[f'cause_{cs}'] = sum(r['cause'] == cs for r in dok)
        row['censored_dirs'] = sum(r['censored'] for r in dok)
        row['creep_limited_weakest'] = sum(c.get('weakest_creep_limited', 0) for c in ok)
        row['type_changed_cells'] = sum(c.get('any_type_change', 0) for c in ok)
        held = [c['F_min_hold_mg'] for c in ok if 'F_min_hold_mg' in c]
        if held:
            row['F_min_hold_median_mg'] = float(np.median(held))
            row['hold_cells'] = len(held)
        for grp, rs in (('env', [r for r in dok if r['group'] == 'env']),
                        ('tip', [r for r in dok if r['group'] != 'env'])):
            for dn in dirs:
                st = stats(r['F_mg'] for r in rs if r['dir'] == dn)
                row[f'F_{dn}_median_{grp}_mg'] = st.get('median', math.nan)
                if st['n']:
                    sel = [r for r in rs if r['dir'] == dn]
                    dstat_rows.append(dict(design=design, group=grp, dir=dn, n=st['n'], median_mg=st['median'],
                                           q25_mg=st['q25'], q75_mg=st['q75'],
                                           censored=sum(r['censored'] for r in sel),
                                           creep_limited=sum(r['creep_limited'] for r in sel),
                                           type_changed=sum(r.get('type_changed_before_Fmax', 0) for r in sel),
                                           F_values_mg=' '.join(f'{r["F_mg"]:.3f}' for r in sel)))
        for col in dvar_cols:
            row[col] = (dvars.get(design) or {}).get(col, '')
        all_rows.append(row)
        tot['cells'] += len(cell_rows)
        tot['success'] += len(ok)
        tot['env'] += len(env)
        tot['tip'] += len(tip)
        tot['zero_fail'] += row['zero_fail']
        tot['incomplete'] += row['incomplete']
    if all_rows:
        first = ['design'] + dvar_cols
        B.write_csv(out_dir / 'all_designs.csv', all_rows, first + [k for k in all_rows[0] if k not in first])
    B.write_csv(out_dir / 'direction_stats.csv', dstat_rows)
    by_area = sorted(all_rows, key=lambda r: -r['robust_area'])[:10]
    by_q = sorted([r for r in all_rows if r['success'] >= 5], key=lambda r: -r['F_min_median_mg'])[:10]
    summary = dict(criteria=crit, dirs=dirs, levels=levels, metric=metric, w_tip=w_tip, totals=tot,
                   stale_traces_skipped=stale, design_index=str(design_index) if design_index else None,
                   top10_robust_area=[dict(design=r['design'], robust_area=r['robust_area'], success=r['success'],
                                           F_min_median_mg=r['F_min_median_mg']) for r in by_area],
                   top10_F_min_median_min5cells=[dict(design=r['design'], F_min_median_mg=r['F_min_median_mg'],
                                                      success=r['success'], robust_area=r['robust_area']) for r in by_q])
    with open(out_dir / 'force_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return tot, stale, by_area


# ----------------------------------------------------------------------------- main
def main(a):
    cells_dir = B.default_cells_dir(a.cells_dir)
    out_root = Path(a.out) if a.out else HERE / 'results_A'
    out_dir = (out_root / (f'force_{a.tag}' if a.tag else 'force')).resolve()
    if a.clean:
        if a.rescore or a.dry_run:
            raise SystemExit('--clean은 시뮬레이션을 새로 돌릴 때만 씁니다 (--rescore/--dry-run과 같이 못 씀)')
        B.clean_results(out_root, cells_dir, 'A')
    mesh_roots = ([a.mesh_dir] if a.mesh_dir else []) + [
        str(HERE / 'assets' / 'rb5_850e' / 'meshes'), str(cells_dir.parent / 'assets' / 'rb5_850e' / 'meshes')]
    sim = dict(rate_mg_s=a.rate_mg_s, cap_mg=a.max_mg, pre_s=a.pre_s, sample_s=a.sample_s, check_ms=a.check_ms,
               stop_mm=a.stop_mm, stop_lost_s=a.stop_lost_s, lost_n=a.lost_n, tension=a.tension)
    crit = dict(slip_mm=a.slip_mm, rot_deg=a.rot_deg, rot_mode=a.rot_mode, lost_s=a.lost_s,
                collision_n=a.collision_n, zero_fail=a.zero_fail)
    if crit['slip_mm'] > sim['stop_mm'] or crit['lost_s'] > sim['stop_lost_s']:
        print('[WARN] criteria looser than the stop rule: stopped ramps count as drop failures.')
    meta_path = out_dir / 'force_meta.json'
    meta = B.read_json(meta_path)
    levels = sorted(set(a.levels))
    if a.rescore:
        if not meta:
            raise SystemExit(f'no results to rescore in {out_dir}')
        dirs = B.parse_dirs(a.dirs) if a.dirs else meta['dirs']
    else:
        h = sim_hash(sim)
        if meta and meta['sim_hash'] != h and not a.overwrite:
            diff = {k: (meta['sim'].get(k), sim[k]) for k in SIM_KEYS if meta['sim'].get(k) != sim[k]}
            raise SystemExit(f'{out_dir} holds results with other settings {diff} (old, new).\n'
                             'Use --tag to keep them apart, or --overwrite to replace them.')
        dirs_run = B.parse_dirs(a.dirs)
        dirs = (meta['dirs'] + [x for x in dirs_run if x not in meta['dirs']]) if (meta and meta['sim_hash'] == h) \
            else dirs_run
        meta = dict(sim=sim, sim_hash=h, dirs=dirs, cells_dir=str(cells_dir),
                    created=(meta or {}).get('created') or time.strftime('%Y-%m-%d %H:%M:%S'),
                    updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        cells = B.select_cells(cells_dir, a.designs, a.only, a.limit)
        B.warn_old_xml(cells, 'A')
        keys = ['z0'] + dirs_run
        jobs = []
        for cdir in cells:
            design, cell = B.cell_parts(cdir)
            tp = out_dir / design / 'traces' / f'{cell}.npz'
            need = True
            if tp.is_file() and not a.overwrite:
                try:
                    tm, _, tr, ex = load_trace(tp)
                    if tm.get('sim_hash') == meta['sim_hash']:
                        need = any(k not in tr for k in keys) or (
                            a.refine and any(f'{k}__hold_result' not in ex for k in dirs_run))
                except Exception:
                    pass
            if need:
                jobs.append(dict(cell_id=f'{design}/{cell}', design=design, cell=cell,
                                 cell_dir=str(cdir), trace_path=str(tp), keys=keys, sim=sim, sim_hash=meta['sim_hash'],
                                 mesh_roots=mesh_roots, overwrite=a.overwrite,
                                 allow_missing_collision=a.allow_missing_collision, refine=a.refine, crit=crit,
                                 hold=dict(rise_s=a.rise_s, hold_s=a.hold_s), tol_mg=a.refine_tol_mg))
        print(f'[A] cells: {cells_dir} ({"XML, <key lifted>" if B.is_xml_cells_dir(cells_dir) else "folders, state.npz"})')
        print(f'[A] {len(cells)} cells in {len({B.cell_parts(c)[0] for c in cells})} designs; {len(jobs)} to run '
              f'({len(cells) - len(jobs)} already done). Ramp {a.rate_mg_s:g} mg/s to {a.max_mg:g} mg '
              f'({a.max_mg / a.rate_mg_s:g} s), {len(dirs_run)} dirs + zero control'
              + (', hold test (bisection)' if a.refine else '') + '.')
        print(f'[A] output: {out_dir}')
        if a.dry_run:
            per = (1 + len(dirs_run)) * 0.25 + (len(dirs_run) * 1.5 if a.refine else 0)
            print(f'[A] dry run. Rough estimate {B.hms(len(jobs) * per / max(1, a.workers))} with {a.workers} workers.')
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        t0, done, errors = time.time(), 0, 0

        def report(res):
            nonlocal done, errors
            done += 1
            el = time.time() - t0
            if res['error']:
                errors += 1
                with open(out_dir / '_errors.log', 'a', encoding='utf-8') as f:
                    f.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} {res["cell_id"]}\n{res["error"]}\n')
                print(f'[{done}/{len(jobs)}] {res["cell_id"]}  ERROR (see _errors.log)', flush=True)
            else:
                print(f'[{done}/{len(jobs)}] {res["cell_id"]}  {res["ran"]} runs  {res["sec"]:.1f} s  '
                      f'elapsed {B.hms(el)}  ETA {B.hms(el / done * (len(jobs) - done))}', flush=True)

        if a.workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                report(simulate_cell(j))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(simulate_cell, j) for j in jobs]
                for fu in as_completed(futs):
                    report(fu.result())
        if errors:
            print(f'[A] {errors} cells failed to run; see {out_dir / "_errors.log"}')
    design_index = B.find_design_index(cells_dir, a.designs_index)
    tot, stale, top = score_all(out_dir, cells_dir, meta, crit, dirs, levels, a.metric, a.w_tip, design_index,
                                a.designs if a.rescore else None)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({**meta, 'last_score': dict(criteria=crit, dirs=dirs, levels=levels, metric=a.metric,
                                              time=time.strftime('%Y-%m-%d %H:%M:%S'))}, f, indent=2)
    print(f'[A] scored {tot["cells"]} cells: success {tot["success"]} (env {tot["env"]}, tip {tot["tip"]}), '
          f'zero_fail {tot["zero_fail"]}, incomplete {tot["incomplete"]}'
          + (f'; skipped {stale} traces with other settings' if stale else ''))
    if design_index is None:
        print('[A] designs_index.csv not found; all_designs.csv has no design variables (use --designs-index).')
    if top:
        print('[A] top designs by robust_area: ' + ', '.join(f'{r["design"]} {r["robust_area"]:.2f}' for r in top[:5]))
    print(f'[A] tables in {out_dir}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(description='실험 A: 천천히 커지는 외력에 대한 파지 유지 한계 F_max (grasp_cells에서 시작)')
    g = p.add_argument_group('대상과 출력')
    g.add_argument('--cells-dir', default=None,
                   help='grasp_cells_xml(XML 파일들) 또는 grasp_cells(폴더) (기본: 이 파일 옆 grasp_cells_xml, 없으면 grasp_cells)')
    g.add_argument('--designs', nargs='+', default=None)
    g.add_argument('--only', nargs='+', default=None,
                   help='특정 cell만, 예: design0001/r60y15 (design0001_r60y15, 예전 이름 design0001/r060.0_y015.0도 됨)')
    g.add_argument('--limit', type=int, default=0, help='설계마다 고르게 N개만 (빠른 시험)')
    g.add_argument('--out', default=None, help='결과 상위 폴더 (기본: 이 파일 옆 results_A)')
    g.add_argument('--tag', default=None, help='결과 폴더 force_<tag>')
    g.add_argument('--overwrite', action='store_true')
    g.add_argument('--clean', action='store_true', help='시작 전에 결과 상위 폴더(results_A) 안의 기존 파일을 모두 지움')
    g.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--mesh-dir', default=None)
    g.add_argument('--allow-missing-collision', action='store_true')
    g.add_argument('--designs-index', default=None)
    s = p.add_argument_group('ramp (바꾸면 다른 실험: --tag 사용)')
    s.add_argument('--rate-mg-s', type=float, default=1.0, help='힘 증가 속도 [mg/s] (기본 1 = 0.3 kg에서 2.94 N/s)')
    s.add_argument('--max-mg', type=float, default=10.0, help='ramp 상한 [mg]')
    s.add_argument('--dirs', nargs='+', default=None, help='axes(6, 기본) | edges(12) | all(18) | px nx …')
    s.add_argument('--tension', type=float, default=None, help='텐던 장력 [N] (기본: 저장 상태 그대로)')
    s.add_argument('--pre-s', type=float, default=0.5, help='시작 전 안정화 [s] (실험 B와 같음)')
    s.add_argument('--sample-s', type=float, default=0.01, help='기록 간격 [s]')
    s.add_argument('--check-ms', type=float, default=1.0, help='판정값 계산 간격 [ms]')
    s.add_argument('--stop-mm', type=float, default=30.0, help='이만큼 밀리면 그 방향 중단 [mm]')
    s.add_argument('--stop-lost-s', type=float, default=0.1, help='접촉을 이만큼 잃으면 중단 [s]')
    s.add_argument('--lost-n', type=float, default=0.1, help='접촉 상실 기준 (한쪽 손가락 힘 합) [N]')
    h = p.add_argument_group('hold test (--refine)')
    h.add_argument('--refine', action='store_true', help='방향마다 일정한 힘의 bisection 추가')
    h.add_argument('--rise-s', type=float, default=0.5)
    h.add_argument('--hold-s', type=float, default=1.0)
    h.add_argument('--refine-tol-mg', type=float, default=0.02)
    r = p.add_argument_group('판정 (--rescore로 시뮬레이션 없이 변경)')
    r.add_argument('--rescore', action='store_true')
    r.add_argument('--slip-mm', type=float, default=5.0)
    r.add_argument('--rot-deg', type=float, default=5.0, help='0이면 회전 판정 끔')
    r.add_argument('--rot-mode', choices=('total', 'tilt'), default='total')
    r.add_argument('--lost-s', type=float, default=0.02)
    r.add_argument('--collision-n', type=float, default=0.05)
    r.add_argument('--zero-fail', choices=('fail', 'keep'), default='fail')
    r.add_argument('--levels', type=float, nargs='+', default=DEFAULT_LEVELS, help='pass level [mg]')
    r.add_argument('--metric', choices=('ramp', 'hold'), default='ramp', help='F_min·robust_area에 쓸 값')
    r.add_argument('--w-tip', type=float, default=0.2)
    return p


if __name__ == '__main__':
    sys.exit(main(build_parser().parse_args()))
