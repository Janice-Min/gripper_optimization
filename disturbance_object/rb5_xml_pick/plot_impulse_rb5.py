#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 B 그림 — disturb_impulse_rb5.py 결과(results_B/impulse[_tag]/)로 그린다.

  python plot_impulse_rb5.py                                   # 개요 그림 4장
  python plot_impulse_rb5.py --k 1                             # 설계 산점도의 기준 펄스 크기 [mg]
  python plot_impulse_rb5.py --map design0001 design0042       # 설계별 (r, y) 지도
  python plot_impulse_rb5.py --trace design0001 r060.0_y015.0  # 한 cell의 시간 응답

출력: results_B/impulse[_tag]/plots/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent

# 색: 범주 = env 파랑, tip 주황, other 청록 (순서 고정). 크기(연속) = 파랑 한 색 계열
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#a3a29d', '#e4e3df', '#fcfcfb'
GROUP_COLOR = {'env': '#2a78d6', 'tip': '#eb6834', 'other': '#1baf7a'}
GROUP_MARK = {'env': 'o', 'tip': '^', 'other': 's'}
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seq_blue', BLUES)
ORD = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b', '#082447']   # 작은 k -> 큰 k

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False,
})


def read_csv(path):
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def num(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[PLOT] {path}')


def group_of(c):
    return c['group'] if c.get('group') in ('env', 'tip') else 'other'


# ----------------------------------------------------------------------------- overview
def plot_levels(cells, mags, out):
    succ = [c for c in cells if c['status'] == 'success']
    levels = [0.0] + list(mags)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    groups = [g for g in ('env', 'tip', 'other') if any(group_of(c) == g for c in succ)]
    w = 0.8 / max(1, len(groups))
    x = np.arange(len(levels))
    for i, g in enumerate(groups):
        cnt = [sum(1 for c in succ if group_of(c) == g and abs(num(c['level_min']) - L) < 1e-9) for L in levels]
        bars = ax.bar(x + (i - (len(groups) - 1) / 2) * w, cnt, w * 0.92, color=GROUP_COLOR[g],
                      label=f'{g} ({sum(cnt)})')
        for b, n in zip(bars, cnt):
            if n:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{n}', ha='center', va='bottom',
                        fontsize=7, color=INK2)
    ax.set_xticks(x, [f'{L:g}' for L in levels])
    ax.set_xlabel('level_min [mg]  (largest pulse held in every direction)')
    ax.set_ylabel('cells')
    zf = sum(c['status'] == 'zero_fail' for c in cells)
    ax.set_title(f'Impulse limit per cell: {len(succ)} cells ({zf} zero-force-control failures excluded)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right')
    save(fig, out / 'impulse_levels.png')


def plot_direction_heatmap(dstats, dirs, mags, out):
    groups = [g for g in ('env', 'tip') if any(r['group'] == g for r in dstats)]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(2.2 + 1.1 * len(mags) * len(groups), 0.45 * len(dirs) + 1.4),
                             squeeze=False)
    for ax, g in zip(axes[0], groups):
        M = np.full((len(dirs), len(mags)), np.nan)
        N = np.zeros_like(M)
        for i, dn in enumerate(dirs):
            for j, k in enumerate(mags):
                sel = [r for r in dstats if r['group'] == g and r['dir'] == dn and abs(num(r['k_mg']) - k) < 1e-9]
                n = sum(int(r['n']) for r in sel)
                if n:
                    M[i, j] = sum(int(r['held']) for r in sel) / n
                    N[i, j] = n
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect='auto')
        for i in range(len(dirs)):
            for j in range(len(mags)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{100 * M[i, j]:.0f}%', ha='center', va='center', fontsize=8,
                            color='white' if M[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(mags)), [f'{k:g}' for k in mags])
        ax.set_yticks(range(len(dirs)), dirs)
        ax.set_xlabel('pulse size k [mg]')
        ax.grid(False)
        ax.set_title(f'{g}: share held ({int(N.max()) if N.size else 0} cells)')
    fig.suptitle('Share of trials held, by direction and pulse size (all designs)', fontsize=10, fontweight='bold', color=INK)
    fig.text(0.5, -0.02, '+x toward upper finger · +y out of the hand · -y into the palm · +z up (cylinder axis)',
             ha='center', fontsize=8, color=INK2)
    save(fig, out / 'impulse_direction_heatmap.png')


def pareto(points):
    """x, y 모두 클수록 좋은 점들의 front (x 내림차순)."""
    front, best = [], -math.inf
    for x, y, lab in sorted(points, key=lambda p: (-p[0], -p[1])):
        if y > best:
            front.append((x, y, lab))
            best = y
    return front


def plot_designs(designs, k, top, out):
    key = f'cond@{k:g}'
    pts = [(num(d['success']), num(d.get(key)), d['design']) for d in designs
           if num(d['success'], 0) > 0 and not math.isnan(num(d.get(key)))]
    if not pts:
        print(f'[PLOT] no {key} column; skip impulse_designs.png')
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    ax.scatter(xs, ys, s=18, color=GROUP_COLOR['env'], alpha=0.55, linewidths=0)
    fr = pareto(pts)
    fx, fy = [p[0] for p in fr], [p[1] for p in fr]
    ax.plot(fx, fy, color=INK2, lw=1.2, zorder=3)
    ax.scatter(fx, fy, s=34, facecolor=GROUP_COLOR['env'], edgecolor=SURF, linewidths=1.5, zorder=4)
    ranked = sorted(designs, key=lambda d: -num(d['impact_area'], 0))[:top]
    lab = {d['design'] for d in ranked} | {p[2] for p in fr}
    for x, y, name in pts:
        if name in lab:
            ax.annotate(name.replace('design', 'd'), (x, y), xytext=(4, 3), textcoords='offset points',
                        fontsize=7, color=INK2)
    ax.set_xlabel('success cells (RB5 pick passed and zero-force control held)')
    ax.set_ylabel(f'{key}: share of cells holding {k:g} mg in every direction')
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f'Workspace size vs impulse robustness per design\nline: Pareto front; labels: front + top {top} by impact_area')
    save(fig, out / f'impulse_designs_k{k:g}.png')


def find_vars(designs, wanted):
    if not designs:
        return []
    cols = list(designs[0].keys())
    out = []
    for w in wanted:
        hit = [c for c in cols if c.lower() == w.lower()] or [c for c in cols if c.lower().startswith(w.lower())]
        if hit and all(not math.isnan(num(d.get(hit[0]))) for d in designs if d.get(hit[0]) not in ('', None)):
            out.append(hit[0])
    return out


def plot_vars(designs, var_names, out):
    vs = find_vars(designs, var_names)
    vs = [v for v in vs if sum(not math.isnan(num(d.get(v))) for d in designs) > 2]
    if not vs:
        print('[PLOT] no design-variable columns in all_designs.csv; skip impulse_vars.png '
              '(give --designs-index to disturb_impulse_rb5.py)')
        return
    rows = [('impact_area', 'impact_area [mg]'), ('level_min_mean', 'mean level_min [mg]')]
    fig, axes = plt.subplots(len(rows), len(vs), figsize=(2.7 * len(vs), 2.5 * len(rows)), squeeze=False,
                             sharey='row')
    ok = [d for d in designs if num(d['success'], 0) > 0]
    for j, v in enumerate(vs):
        for i, (col, lab) in enumerate(rows):
            ax = axes[i, j]
            x = np.array([num(d.get(v)) for d in ok])
            y = np.array([num(d.get(col)) for d in ok])
            m = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(x[m], y[m], s=10, color=GROUP_COLOR['env'], alpha=0.5, linewidths=0)
            if m.sum() > 3:
                rx = np.argsort(np.argsort(x[m]))
                ry = np.argsort(np.argsort(y[m]))
                rho = np.corrcoef(rx, ry)[0, 1]
                ax.text(0.02, 0.97, f'Spearman {rho:+.2f}', transform=ax.transAxes, va='top', fontsize=7, color=INK2)
            if i == len(rows) - 1:
                ax.set_xlabel(v)
            if j == 0:
                ax.set_ylabel(lab)
    fig.suptitle('Design variables vs impulse robustness (designs with success cells)', fontsize=10, fontweight='bold', color=INK)
    save(fig, out / 'impulse_vars.png')


# ----------------------------------------------------------------------------- per design
def plot_map(design, cells, pick, mags, out):
    cs = [c for c in cells if c['design'] == design]
    if not cs:
        print(f'[PLOT] no traced cells for {design}')
        return
    levels = [0.0] + list(mags)
    cols = BLUES[1:1 + len(levels)] if len(levels) <= 6 else [SEQ(i / (len(levels) - 1)) for i in range(len(levels))]
    cmap = ListedColormap(cols)
    bounds = [levels[0] - 0.5] + [(a + b) / 2 for a, b in zip(levels[:-1], levels[1:])] + [levels[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fails = [p for p in pick if p.get('design_id') == design and str(p.get('success')).strip() not in ('1', 'True')]
    if fails:
        ax.scatter([num(p['r_mm']) for p in fails], [num(p['y_mm']) for p in fails], s=14, marker='.',
                   color=MUTED, label='RB5 pick failed')
    zf = [c for c in cs if c['status'] == 'zero_fail']
    if zf:
        ax.scatter([num(c['r_mm']) for c in zf], [num(c['y_mm']) for c in zf], s=40, marker='x', color=INK2,
                   label='zero-force control failed')
    sc = None
    for g in ('env', 'tip', 'other'):
        sel = [c for c in cs if c['status'] == 'success' and group_of(c) == g]
        if sel:
            sc = ax.scatter([num(c['r_mm']) for c in sel], [num(c['y_mm']) for c in sel],
                            c=[num(c['level_min']) for c in sel], cmap=cmap, norm=norm, s=70,
                            marker=GROUP_MARK[g], edgecolors=INK2, linewidths=0.6, label=g)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, ticks=levels, pad=0.02)
        cb.set_label('level_min [mg]')
    ax.set_xlabel('r [mm]')
    ax.set_ylabel('y [mm]  (palm face to cylinder surface)')
    ax.set_title(f'{design}: impulse limit per cell (marker = contact type at pulse start)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.18, 1.0), fontsize=8)
    save(fig, out / f'impulse_map_{design}.png')


def plot_trace(res_dir, design, cell, meta, out):
    path = res_dir / design / 'traces' / f'{cell}.npz'
    if not path.is_file():
        print(f'[PLOT] no trace {path}')
        return
    with np.load(path, allow_pickle=False) as z:
        tm = json.loads(str(z['__meta__']))
        cols = [str(c) for c in z['columns']]
        tr = {k[:-7]: z[k] for k in z.files if k.endswith('__trace')}
    ci = {n: i for i, n in enumerate(cols)}
    dirs = [d for d in meta['dirs'] if any(k.startswith(d + '_') for k in tr)]
    mags = meta['mags']
    T = meta['sim']['pulse_ms'] / 1000.0
    ncol = min(3, len(dirs))
    nrow = math.ceil(len(dirs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, dn in zip(axes.flat, dirs):
        if 'z0' in tr:
            r = tr['z0']
            ax.plot(r[:, ci['t']], r[:, ci['disp']], color=MUTED, lw=1.5, label='no force')
        for i, k in enumerate(mags):
            key = f'{dn}_{k:g}mg'
            if key in tr:
                r = tr[key]
                ax.plot(r[:, ci['t']], r[:, ci['disp']], color=ORD[i % len(ORD)], lw=2, label=f'{k:g} mg')
        ax.axhline(5.0, color=INK2, lw=1, ls='--')
        ax.axvspan(1e-3, T, color=GRID, lw=0)
        ax.set_xscale('log')
        ax.set_xlim(4e-3, meta['sim']['pulse_ms'] / 1000 + meta['sim']['after_s'])
        ax.set_ylim(0, 32)
        ax.set_title(dn)
    for ax in axes.flat[len(dirs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('time after pulse start [s] (log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('object displacement in hand [mm]')
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=len(lab), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f'{design}/{cell}  r={tm.get("r_mm")} mm, y={tm.get("y_mm")} mm, start {tm.get("start_type")}; '
                 f'shaded = pulse, dashed = 5 mm slip limit', fontsize=9, color=INK, y=1.08)
    save(fig, out / f'impulse_trace_{design}_{cell}.png')


def main():
    p = argparse.ArgumentParser(description='실험 B 그림')
    p.add_argument('--out', default=None, help='결과 상위 폴더 (기본: 이 파일 옆 results_B)')
    p.add_argument('--tag', default=None)
    p.add_argument('--k', type=float, default=1.0, help='설계 산점도의 펄스 크기 [mg]')
    p.add_argument('--top', type=int, default=8, help='라벨을 붙일 impact_area 상위 설계 수')
    p.add_argument('--vars', nargs='+', default=['L_total', 'L_p_ratio', 'G', 'th_open_p'],
                   help='impulse_vars.png에 쓸 설계 변수 열 이름')
    p.add_argument('--map', nargs='+', default=None, help='(r, y) 지도를 그릴 설계')
    p.add_argument('--trace', nargs=2, metavar=('DESIGN', 'CELL'), default=None)
    p.add_argument('--cells-dir', default=None, help='rb5_pick_results.csv가 있는 grasp_cells (지도에 RB5 실패 표시)')
    a = p.parse_args()
    res = (Path(a.out) if a.out else HERE / 'results_B') / (f'impulse_{a.tag}' if a.tag else 'impulse')
    meta = json.loads((res / 'impulse_meta.json').read_text(encoding='utf-8'))
    last = meta.get('last_score') or {}
    mags = last.get('mags') or meta['mags']
    dirs = last.get('dirs') or meta['dirs']
    out = res / 'plots'
    if a.trace:
        plot_trace(res, a.trace[0], a.trace[1], meta, out)
        return
    cells = []
    for f in sorted(res.glob('*/impulse_cells.csv')):
        cells += read_csv(f)
    if a.map:
        cdir = Path(a.cells_dir) if a.cells_dir else Path(meta.get('cells_dir', HERE / 'grasp_cells'))
        pick = read_csv(cdir / 'rb5_pick_results.csv')
        for d in a.map:
            plot_map(d, cells, pick, mags, out)
        return
    designs = read_csv(res / 'all_designs.csv')
    dstats = read_csv(res / 'direction_stats.csv')
    plot_levels(cells, mags, out)
    plot_direction_heatmap(dstats, dirs, mags, out)
    plot_designs(designs, a.k, a.top, out)
    plot_vars(designs, a.vars, out)


if __name__ == '__main__':
    main()
