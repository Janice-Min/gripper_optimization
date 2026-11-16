#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 A 설계 순위와 그림 — disturb_force_rb5.py 결과(results_A/force[_tag]/)로 만든다.

  python rank_force_designs.py                 # results_A/force
  python rank_force_designs.py --tag edges
  python rank_force_designs.py --b results_B/impulse   # 실험 B와 비교 (기본: 있으면 자동)

출력 (results_A/force/):
  A_design_ranking.csv      robust_area 순위, cell bootstrap 순위 안정성(P_top10, P_top20),
                            F_min 중앙값, pass@k, workspace–품질 Pareto
  A_fig_directions.png      방향별 F_max 분포 (env | tip, 모든 설계)
  A_fig_design_ranking.png  상위 32개 robust_area (bootstrap 범위) + 성공 cell 수 vs F_min 중앙값
  A_vs_B.csv / A_vs_B.png   (B 결과가 있을 때) 같은 cell·방향에서 펄스 크기 k와 A의 F_max 차이별 B 유지 비율,
                            설계 순위 상관
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#a3a29d', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False})
NAMES = {'px': '+x', 'nx': '-x', 'py': '+y out', 'ny': '-y palm', 'pz': '+z up', 'nz': '-z down'}


def rcsv(p):
    try:
        with open(p, newline='', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def num(v, d=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def wcsv(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) and not math.isnan(v) else ('' if isinstance(v, float) else v))
                        for k, v in r.items()})


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return math.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description='실험 A 설계 순위와 그림')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--b', default=None, help='실험 B 결과 폴더 (기본: results_B/impulse가 있으면)')
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--min-cells', type=int, default=5, help='F_min 중앙값 순위에 넣을 최소 성공 cell 수')
    a = ap.parse_args()
    res = (Path(a.out) if a.out else HERE / 'results_A') / (f'force_{a.tag}' if a.tag else 'force')
    designs = rcsv(res / 'all_designs.csv')
    if not designs:
        raise SystemExit(f'no all_designs.csv in {res}')
    # cell별 F_min (성공 cell), zero_fail은 0
    cells = {}
    for d in designs:
        rows = rcsv(res / d['design'] / 'force_cells.csv')
        v = [num(r['F_min_mg']) for r in rows if r['group'] in ('env', 'tip', 'other')]
        v += [0.0] * sum(r['group'] == 'zero_fail' for r in rows)
        cells[d['design']] = np.array(v, float)
    D = [dict(d) for d in designs]
    for d in D:
        for k in ('robust_area', 'F_min_median_mg', 'success', 'success_env', 'pass@1', 'pass@0.5', 'zero_fail'):
            d[k] = num(d.get(k), 0.0 if k != 'F_min_median_mg' else math.nan)
    D.sort(key=lambda d: (-d['robust_area'], -(d['F_min_median_mg'] if not math.isnan(d['F_min_median_mg']) else -1)))
    for i, d in enumerate(D):
        d['rank'] = i + 1
    names = [d['design'] for d in D]
    rng = np.random.default_rng(0)
    B_ = a.boot
    ranks = np.zeros((B_, len(D)))
    lo, hi = [], []
    for b in range(B_):
        sc = np.array([c[rng.integers(0, len(c), len(c))].sum() if len(c) else 0.0 for c in (cells[n] for n in names)])
        sc += rng.uniform(0, 1e-9, len(sc))
        ranks[b] = (-sc).argsort().argsort() + 1
    for n in names:
        c = cells[n]
        s = [c[rng.integers(0, len(c), len(c))].sum() for _ in range(1000)] if len(c) else [0.0]
        lo.append(np.percentile(s, 5))
        hi.append(np.percentile(s, 95))
    for i, d in enumerate(D):
        d['robust_area_p05'], d['robust_area_p95'] = lo[i], hi[i]
        d['P_top10'] = float((ranks[:, i] <= 10).mean())
        d['P_top20'] = float((ranks[:, i] <= 20).mean())
        d['rank_boot_median'] = float(np.median(ranks[:, i]))
        d['rank_boot_p05'], d['rank_boot_p95'] = (float(x) for x in np.percentile(ranks[:, i], [5, 95]))
        d['env_share'] = d['success_env'] / d['success'] if d['success'] else math.nan
    q = [d for d in D if d['success'] >= a.min_cells and not math.isnan(d['F_min_median_mg'])]
    for i, d in enumerate(sorted(q, key=lambda d: -d['F_min_median_mg'])):
        d['rank_quality'] = i + 1
    pts = sorted([(d['success'], d['F_min_median_mg'], d['design']) for d in D if d['success'] > 0],
                 key=lambda p: (-p[0], -p[1]))
    front, best = set(), -1
    for x, y, n in pts:
        if y > best + 1e-12:
            front.add(n)
            best = y
    for d in D:
        d['pareto_ws_quality'] = int(d['design'] in front)
    first = ['rank', 'design', 'robust_area', 'robust_area_p05', 'robust_area_p95', 'P_top10', 'P_top20',
             'rank_boot_median', 'rank_boot_p05', 'rank_boot_p95', 'success', 'success_env', 'success_tip', 'zero_fail',
             'env_share', 'F_min_median_mg', 'rank_quality', 'pareto_ws_quality']
    wcsv(res / 'A_design_ranking.csv', [{k: d.get(k, '') for k in first} | {k: v for k, v in d.items() if k not in first}
                                        for d in D])

    # 그림 1: 방향별 분포
    ds = rcsv(res / 'direction_stats.csv')
    dirs = [x for x in ['pz', 'ny', 'px', 'nx', 'py', 'nz'] if any(r['dir'] == x for r in ds)]
    dirs += sorted({r['dir'] for r in ds} - set(dirs))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, g in zip(axes, ('env', 'tip')):
        data = []
        for dn in dirs:
            v = [float(x) for r in ds if r['group'] == g and r['dir'] == dn for x in r['F_values_mg'].split()]
            data.append(v)
        keep = [i for i, v in enumerate(data) if v]
        if keep:
            col = BLUE if g == 'env' else ORANGE
            bp = ax.boxplot([data[i] for i in keep], positions=keep, widths=0.55, patch_artist=True, showfliers=False,
                            medianprops=dict(color=SURF, lw=1.6), whiskerprops=dict(color=col), capprops=dict(color=col))
            for bx in bp['boxes']:
                bx.set(facecolor=col, edgecolor=col)
            for i in keep:
                ax.text(i, np.median(data[i]), f'{np.median(data[i]):.2f}', ha='center', va='bottom', fontsize=7,
                        color=INK, bbox=dict(facecolor=SURF, edgecolor='none', pad=0.5), zorder=5)
        n = max((len(v) for v in data), default=0)
        ax.set_xticks(range(len(dirs)), [NAMES.get(x, x) for x in dirs])
        ax.axhline(1.0, color=INK2, lw=0.8, ls='--')
        ax.set_title(f'{"Enveloping" if g == "env" else "Tip"} grasps ({n} cells)')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('F_max (ramp) [mg]  (box: quartiles, whiskers 1.5 IQR)')
    fig.suptitle('Holding force per direction, all designs (dashed: 1 mg = object weight)', fontsize=10,
                 fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(res / 'A_fig_directions.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    # 그림 2: 순위
    top = D[:32][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw={'width_ratios': [1, 1.15]})
    y = np.arange(len(top))
    a1.hlines(y, [d['robust_area_p05'] for d in top], [d['robust_area_p95'] for d in top], color=GRID, lw=4)
    a1.scatter([d['robust_area'] for d in top], y, s=36, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.2)
    a1.set_yticks(y, [f'{d["design"].replace("design", "d")}  ({int(d["success_env"])}/{int(d["success"])})' for d in top],
                  fontsize=7.5)
    a1.set_xlabel('robust_area = sum of F_min over success cells [mg]\nbar: 90% bootstrap range over cells')
    a1.set_title('Top 32 by robust_area (labels: env / success cells)')
    a1.grid(axis='y', visible=False)
    ok = [d for d in D if d['success'] > 0]
    a2.scatter([d['success'] for d in ok], [d['F_min_median_mg'] for d in ok], s=14, color=MUTED, alpha=0.6,
               linewidths=0, label='all designs')
    t20 = [d for d in D[:20] if d['success'] > 0]
    a2.scatter([d['success'] for d in t20], [d['F_min_median_mg'] for d in t20], s=40, color=BLUE, edgecolor=SURF,
               linewidth=1.2, zorder=3, label='top 20 by robust_area')
    pf = sorted([d for d in ok if d['pareto_ws_quality']], key=lambda d: d['success'])
    a2.plot([d['success'] for d in pf], [d['F_min_median_mg'] for d in pf], color=INK2, lw=1, drawstyle='steps-pre',
            label='Pareto front')
    for d in pf:
        a2.annotate(d['design'].replace('design', 'd'), (d['success'], d['F_min_median_mg']), xytext=(4, 3),
                    textcoords='offset points', fontsize=7, color=INK2)
    a2.set_xlabel('success cells (RB5 pick passed and held 10 s with no force)')
    a2.set_ylabel('median F_min per cell [mg]')
    a2.set_title('Workspace size vs per-grasp holding force')
    a2.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(res / 'A_fig_design_ranking.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'[RANK] {res / "A_design_ranking.csv"}, A_fig_directions.png, A_fig_design_ranking.png')

    # 실험 B와 비교
    bdir = Path(a.b) if a.b else HERE / 'results_B' / 'impulse'
    if not (bdir / 'all_designs.csv').is_file():
        return
    pairs = []
    for d in designs:
        fa = {(r['cell'], r['dir']): num(r['F_mg']) for r in rcsv(res / d['design'] / 'force_dirs.csv')
              if r['group'] in ('env', 'tip', 'other') and r.get('censored') in ('0', 'False')}
        for r in rcsv(bdir / d['design'] / 'impulse_trials.csv'):
            k = (r['cell'], r['dir'])
            if k in fa and r.get('cell_status') == 'success':
                pairs.append((num(r['k_mg']) - fa[k], int(r['held']), r['start_type']))
    if not pairs:
        return
    diff = np.array([p[0] for p in pairs])
    held = np.array([p[1] for p in pairs])
    edges = [-10, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 1.0, 10]
    rows = []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (diff >= lo_) & (diff < hi_)
        if m.any():
            rows.append(dict(k_minus_FA_from=lo_, k_minus_FA_to=hi_, trials=int(m.sum()), held_rate=float(held[m].mean())))
    bd = {r['design']: r for r in rcsv(bdir / 'all_designs.csv')}
    common = [d for d in designs if d['design'] in bd and num(d.get('success'), 0) > 0]
    rho_area = spearman([num(d['robust_area']) for d in common], [num(bd[d['design']].get('impact_area')) for d in common])
    rows.append(dict(note='design rank correlation (Spearman) A robust_area vs B impact_area', value=rho_area,
                     designs=len(common)))
    wcsv(res / 'A_vs_B.csv', rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.arange(len(rows) - 1)
    ax.bar(xs, [100 * r['held_rate'] for r in rows[:-1]], color=BLUE, width=0.7)
    for x, r in zip(xs, rows[:-1]):
        ax.text(x, 100 * r['held_rate'] + 1, f'{100 * r["held_rate"]:.0f}%\nn={r["trials"]}', ha='center',
                va='bottom', fontsize=7, color=INK2)
    ax.set_xticks(xs, [f'{r["k_minus_FA_from"]:g}..{r["k_minus_FA_to"]:g}'.replace('-10..', '<').replace('..10', '+')
                       for r in rows[:-1]], fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_xlabel('pulse size k minus ramp F_max of the same cell and direction [mg]')
    ax.set_ylabel('B trials held [%]')
    ax.set_title(f'Does the ramp limit predict the 50 ms pulse result?  (design rank rho = {rho_area:.2f})')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(res / 'A_vs_B.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'[RANK] A_vs_B.csv, A_vs_B.png (B: {bdir})')


if __name__ == '__main__':
    main()
