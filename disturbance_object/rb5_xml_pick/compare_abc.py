#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 A·B·C를 함께 비교해 세 외란(천천히 커지는 힘 / 충격 / 팔 흔들기)을 모두 잘 견디는 설계 찾기

  python compare_abc.py results_A/force results_B/impulse results_C/shake
  python compare_abc.py --A a.csv --B b.csv --C c.csv --score mean --agg geomean
  python compare_abc.py results_A/force results_B/impulse results_C/shake --rank-by pareto --top 25

입력은 all_designs.csv 또는 결과 폴더(그 안의 all_designs.csv). 실험 종류는 열 이름으로 알아낸다.
plot_disturb_results.py와 같은 폴더에 두어야 한다 (읽기·그림 함수를 같이 쓴다).

[1] 설계 단위 비교 — all_designs.csv 세 개만 있으면 된다
  s_X        실험마다 같은 종류의 점수 (--score: area(기본) | mean | J@k | pass@k | cond@k)
               area = Σ_성공 cell level_min = 성공 cell 수 × cell당 평균 강건성 (A robust_area, B impact_area, C shake_area)
               mean = cell당 평균 강건성 (workspace 크기와 분리)
  X_n        0~1 정규화 (--norm max: s/max(s) (기본) | rank: 백분위, 0점은 0)
  ABC_score  세 정규화 점수의 결합 (--agg min(기본): 가장 약한 외란이 순위를 정함 | geomean | mean)
  pareto     (A_n, B_n, C_n) 세 축 모두에서 지배당하지 않는 설계 = layer 1. 그다음 front = layer 2 …
  allround   세 실험 모두에서 상위 --top-frac(기본 25 %) 안에 드는 설계
  joint_ub@k = min(pass_A@k, pass_B@k, pass_C@k)
             세 실험에서 '모두' k를 버틴 cell 수의 상한. 같은 cell이 셋 다 버텼는지는 설계 단위 표만으로는 모른다.

[2] cell 단위 비교 — 결과 폴더 안에 <design>/force_cells.csv, impulse_cells.csv, shake_cells.csv가 있으면 자동
  같은 (design, cell)을 세 실험에서 맞춰, 셋 다 성공(외력 0 대조 통과)한 cell만 남긴다.
  joint_level = min(F_min_A [mg], level_min_B [mg], level_min_C [g])
             세 값 모두 '물체 무게 mg의 몇 배 하중'이라 같은 척도다 (C: 손 가속도 k g → 관성력 k·mg).
  joint_pass@k, joint_J@k = env + w_tip·tip, joint_area = Σ joint_level → 정확한 '세 외란 모두' 지표.
  이때 --rank-by auto(기본)는 joint_area로 순위를 매긴다 (cell 파일이 없으면 ABC_score).

출력 (<out>/): ABC_ranking.csv, ABC_summary.txt, (cell 단위면) ABC_joint_cells.csv, 그림
  ABC_1_pairwise          A–B, A–C, B–C 정규화 점수 산점도. 파란 점 = 3축 Pareto front, 고리 = 상위 설계
  ABC_2_scorecard         상위 N 설계 × (A, B, C, ABC_score) heatmap + 성공 cell, Pareto layer, joint
  ABC_3_parallel          평행 좌표: 세 실험에서의 점수 변화 (회색 = 나머지 설계)
  ABC_4_coverage_quality  workspace 크기(성공 cell) vs 세 실험 결합 cell당 강건성 + Pareto front
  ABC_5_correlation       실험 간 Spearman 상관 (area, cell당 평균, 성공 cell 수)
  ABC_6_joint             상위 설계의 joint pass@k (cell 단위면 정확한 값, 아니면 상한)
  ABC_7_workspace_maps    (cell 단위만) 상위 설계의 (r, y) cell 지도: A, B, C, joint level
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_disturb_results as P  # noqa: E402
from disturb_impulse_rb5 import compact_cell_name  # noqa: E402  (r050.0_y005.0 → r50y05)

plt = P.plt
KEYS = ('A', 'B', 'C')
NAMES = {'A': 'A force', 'B': 'B impulse', 'C': 'C shake'}


# ----------------------------------------------------------------------------- loading
def load_inputs(args):
    exps = {}
    for k in KEYS:
        p = getattr(args, k)
        if p:
            exps[k] = P.Exp(p, k)
    for p in args.inputs:
        e = P.Exp(p)
        if e.key in exps:
            raise SystemExit(f'실험 {e.key}가 두 번 들어옴: {exps[e.key].csv}, {e.csv}')
        exps[e.key] = e
    miss = [k for k in KEYS if k not in exps]
    if miss:
        raise SystemExit(f'실험 {", ".join(miss)}의 all_designs.csv가 필요함')
    if args.designs_index:
        for e in exps.values():
            e.add_design_vars(args.designs_index)
    return exps


def normalize(s, how):
    s = s.astype(float)
    if how == 'rank':
        r = s.rank(pct=True, method='average')
        r[s <= 0] = 0.0
        return r
    m = s.max()
    return s / m if m > 0 else s * 0.0


def common_levels(exps):
    sets = [set(e.levels) for e in exps.values()]
    return sorted(set.intersection(*sets))


def merged_table(exps, args):
    designs = sorted(set().union(*[set(e.df.index) for e in exps.values()]))
    T = pd.DataFrame(index=pd.Index(designs, name='design'))
    labels = {}
    lv_common = common_levels(exps)
    for k, e in exps.items():
        sc, col, label, _ = e.score(args.score)
        labels[k] = col
        df = e.df
        T[f'{k}_score'] = sc
        T[f'{k}_success'] = df['success']
        T[f'{k}_env'] = df['success_env']
        T[f'{k}_tip'] = df['success_tip']
        T[f'{k}_area'] = df[e.spec['area']]
        T[f'{k}_mean'] = e.mean_quality()
        for lv in lv_common:
            for c in ('pass', 'env_pass', 'tip_pass'):
                cc = f'{c}@{P.lv_key(lv)}'
                if cc in df:
                    T[f'{k}_{cc}'] = pd.to_numeric(df[cc], errors='coerce')
        if 'incomplete' in df:
            T[f'{k}_incomplete'] = df['incomplete']
        if k == 'C' and 'cells_with_censored_dir' in df:
            T['C_censored_cells'] = df['cells_with_censored_dir']
    # 한 실험에 없는 설계: 다른 실험에서 성공 cell이 0개면 0점(파지 자체가 안 됨), 아니면 incomplete
    T['missing'] = ''
    succ = T[[f'{k}_success' for k in KEYS]]
    no_cells = (succ.fillna(0) == 0).all(axis=1)
    for k in KEYS:
        miss = T[f'{k}_score'].isna()
        fill = miss & no_cells
        for c in [c for c in T.columns if c.startswith(f'{k}_') and c != f'{k}_mean']:
            T.loc[fill, c] = T.loc[fill, c].fillna(0.0)
        T.loc[miss & ~no_cells, 'missing'] += k
    T['coverage'] = succ.min(axis=1)
    for k in KEYS:
        T[f'{k}_n'] = normalize(T[f'{k}_score'].fillna(0.0), args.norm)
        T[f'{k}_pct'] = T[f'{k}_score'].rank(pct=True, method='average')
    N = T[[f'{k}_n' for k in KEYS]].to_numpy(float)
    T['ABC_min'] = N.min(1)
    with np.errstate(divide='ignore'):
        T['ABC_geomean'] = np.where((N > 0).all(1), np.exp(np.log(np.clip(N, 1e-12, None)).mean(1)), 0.0)
    T['ABC_mean'] = N.mean(1)
    T['ABC_score'] = T[f'ABC_{args.agg}']
    # cell당 강건성의 결합 (workspace 크기와 분리)
    Q = np.c_[[normalize(T[f'{k}_mean'].fillna(0.0), 'max') for k in KEYS]].T
    T['quality_ABC'] = {'min': Q.min(1), 'mean': Q.mean(1),
                        'geomean': np.where((Q > 0).all(1), np.exp(np.log(np.clip(Q, 1e-12, None)).mean(1)), 0.0)
                        }[args.agg]
    # Pareto layers (점수가 하나라도 있는 설계만)
    valid = (T['missing'] == '') & (N.max(1) > 0)
    T['pareto_layer'] = np.nan
    T.loc[valid, 'pareto_layer'] = P.pareto_layers(N[valid.to_numpy()])
    q = 1.0 - args.top_frac
    T['allround'] = (valid & (T[[f'{k}_pct' for k in KEYS]] >= q).all(axis=1)).astype(int)
    # joint 상한
    for lv in lv_common:
        lk = P.lv_key(lv)
        cols = [f'{k}_pass@{lk}' for k in KEYS]
        if all(c in T for c in cols):
            T[f'joint_ub@{lk}'] = T[cols].min(axis=1)
            T[f'joint_J_ub@{lk}'] = (T[[f'{k}_env_pass@{lk}' for k in KEYS]].min(axis=1)
                                     + args.w_tip * T[[f'{k}_tip_pass@{lk}' for k in KEYS]].min(axis=1))
    return T, labels, lv_common


# ----------------------------------------------------------------------------- cell level
RC = re.compile(r'r(\d+(?:\.\d+)?)')
YC = re.compile(r'y(\d+(?:\.\d+)?)')


def load_cells(e):
    files = sorted(e.root.glob(f'*/{e.spec["cells_file"]}'))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, encoding='utf-8-sig', dtype={'cell': str, 'design': str}) for f in files],
                   ignore_index=True)
    df['design'] = df['design'].str.strip()
    df['cell'] = df['cell'].str.strip().map(compact_cell_name)   # 예전 이름이 섞여 있어도 세 실험이 맞춰지게
    if e.key == 'A':
        ok = df['group'].isin(['env', 'tip', 'other'])
        lvl = pd.to_numeric(df.get('F_min_mg'), errors='coerce')
    else:
        ok = df['status'].astype(str) == 'success'
        lvl = pd.to_numeric(df['level_min'], errors='coerce')
    out = pd.DataFrame({'design': df['design'], 'cell': df['cell'], 'ok': ok & lvl.notna(), 'level': lvl,
                        'env': df['group'].astype(str) == 'env'})
    for ax_, rx in (('r_mm', RC), ('y_mm', YC)):
        v = pd.to_numeric(df[ax_], errors='coerce') if ax_ in df else pd.Series(np.nan, index=df.index)
        parsed = df['cell'].map(lambda s, rx=rx: float(rx.search(s).group(1)) if rx.search(s) else np.nan)
        out[ax_] = v.fillna(parsed)
    if e.key == 'C':
        out['censored'] = df.get('censored_dirs', pd.Series('', index=df.index)).fillna('').astype(str).str.len() > 0
    return out


def joint_cells(exps):
    cells = {k: load_cells(e) for k, e in exps.items()}
    if any(v is None for v in cells.values()):
        miss = [k for k, v in cells.items() if v is None]
        print(f'[ABC] cell 파일이 없음 ({", ".join(miss)}: <design>/{"/".join(exps[k].spec["cells_file"] for k in miss)})'
              ' → 설계 단위 비교만 (joint는 상한값)')
        return None
    a, b, c = cells['A'], cells['B'], cells['C']
    J = (a.rename(columns={'ok': 'ok_A', 'level': 'F_A_mg', 'env': 'env_A'})
         .merge(b[['design', 'cell', 'ok', 'level', 'env']].rename(
             columns={'ok': 'ok_B', 'level': 'L_B_mg', 'env': 'env_B'}), on=['design', 'cell'], how='outer')
         .merge(c[['design', 'cell', 'ok', 'level', 'env', 'censored']].rename(
             columns={'ok': 'ok_C', 'level': 'L_C_g', 'env': 'env_C', 'censored': 'C_censored'}),
             on=['design', 'cell'], how='outer'))
    # 세 실험 모두에 trace(cell 행)가 있는 cell만 비교 대상
    J['matched'] = J[['ok_A', 'ok_B', 'ok_C']].notna().all(axis=1)
    for k in KEYS:
        J[f'ok_{k}'] = J[f'ok_{k}'].astype('boolean').fillna(False).astype(bool)
    J['ok'] = J['matched'] & J['ok_A'] & J['ok_B'] & J['ok_C']
    J['env'] = J['env_A'].astype('boolean').fillna(J['env_B'].astype('boolean')).fillna(False).astype(bool)
    lv = J[['F_A_mg', 'L_B_mg', 'L_C_g']].astype(float)
    J['joint_level'] = np.where(J['ok'], lv.min(axis=1), np.nan)
    J['C_censored'] = J['C_censored'].astype('boolean').fillna(False).astype(bool)
    J['weakest_exp'] = ''
    okr = J['ok'] & lv.notna().all(axis=1)
    J.loc[okr, 'weakest_exp'] = lv[okr].idxmin(axis=1).map({'F_A_mg': 'A', 'L_B_mg': 'B', 'L_C_g': 'C'})
    return J


def joint_per_design(J, levels, w_tip):
    rows = []
    for d, g in J.groupby('design'):
        ok = g[g['ok']]
        r = dict(design=d, joint_success=len(ok), joint_env=int(ok['env'].sum()),
                 joint_tip=int((~ok['env']).sum()), joint_area=float(ok['joint_level'].sum()),
                 joint_mean=float(ok['joint_level'].mean()) if len(ok) else np.nan,
                 joint_C_censored=int((ok['C_censored']).sum()),
                 joint_weakest_A=int((ok['weakest_exp'] == 'A').sum()),
                 joint_weakest_B=int((ok['weakest_exp'] == 'B').sum()),
                 joint_weakest_C=int((ok['weakest_exp'] == 'C').sum()))
        for lv in levels:
            lk = P.lv_key(lv)
            m = ok['joint_level'] >= lv - 1e-9
            r[f'joint_pass@{lk}'] = int(m.sum())
            r[f'joint_J@{lk}'] = float((m & ok['env']).sum() + w_tip * (m & ~ok['env']).sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index('design')


# ----------------------------------------------------------------------------- ranking
def rank_table(T, args, cell_mode):
    rb = args.rank_by
    if rb == 'auto':
        rb = 'joint_area' if cell_mode else 'agg'
    v = T[T['missing'] == ''].copy()
    if rb == 'agg':
        keys, asc, title = ['ABC_score', 'ABC_geomean', 'ABC_mean', 'coverage'], [False] * 4, f'ABC_score ({args.agg})'
    elif rb == 'pareto':
        keys, asc, title = ['pareto_layer', 'ABC_score', 'ABC_geomean'], [True, False, False], \
            f'Pareto layer, then ABC_score ({args.agg})'
    else:
        if rb not in v:
            raise SystemExit(f'--rank-by {rb!r}: 열이 없음 (예: agg, pareto, joint_area, joint_ub@0.5, ABC_geomean)')
        keys, asc, title = [rb, 'ABC_score', 'ABC_geomean'], [False, False, False], rb
    v = v.sort_values(keys, ascending=asc, na_position='last')
    v.insert(0, 'rank', np.arange(1, len(v) + 1))
    return v, rb, title


# ----------------------------------------------------------------------------- figures
def fig_pairwise(R, labels, args, out):
    pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), layout='constrained')
    front = R['pareto_layer'] == 1
    top = list(R.index[:args.label_top])
    lab = list(dict.fromkeys(top + (list(R.index[front]) if front.sum() <= 10 else [])))
    base = args.baseline if args.baseline in R.index else None
    lab = [d for d in lab if d != base]
    for ax, (p, q) in zip(axs, pairs):
        x, y = R[f'{p}_n'], R[f'{q}_n']
        ax.plot([0, 1], [0, 1], color=P.GRID, lw=1, zorder=0)
        ax.scatter(x[~front], y[~front], s=16, color=P.FAINT, edgecolors='none', zorder=1, label='other designs')
        ax.scatter(x[front], y[front], s=30, color=P.CAT[0], edgecolors='white', linewidths=0.7, zorder=2,
                   label='3-axis Pareto front')
        ax.scatter(x[top], y[top], s=70, facecolors='none', edgecolors=P.INK, linewidths=1.0, zorder=3,
                   label=f'top {len(top)} overall')
        P.place_labels(ax, x[lab].to_numpy(), y[lab].to_numpy(), [f'{P.short(d)}' for d in lab], fontsize=7)
        if base:
            ax.scatter([x[base]], [y[base]], marker='D', s=36, color=P.INK, zorder=4, label='baseline')
            P.place_labels(ax, [x[base]], [y[base]], [f'{P.short(base)} (base)'], fontsize=7, weight='bold')
        rho = R[f'{p}_score'].corr(R[f'{q}_score'], method='spearman')
        ax.set_xlim(-0.03, 1.06)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel(f'{NAMES[p]}: {labels[p]} (normalized)')
        ax.set_ylabel(f'{NAMES[q]}: {labels[q]} (normalized)')
        ax.set_title(f'{p} vs {q}   Spearman ρ = {rho:.2f}', fontsize=9.5)
    axs[0].legend(loc='upper left', fontsize=7.5)
    fig.suptitle(f'A · B · C — how designs rank across the three disturbances (score = {args.score}, '
                 f'norm = {args.norm}; overall rank by {args.rank_title})', x=0.01, ha='left', fontsize=11.5,
                 fontweight='bold')
    fig.savefig(out)
    plt.close(fig)


def fig_scorecard(R, labels, lv_common, args, out, cell_mode):
    rows = list(R.index[:args.top])
    base = args.baseline if args.baseline in R.index else None
    if base and base not in rows:
        rows.append(base)
    n = len(rows)
    jl = P.lv_key(lv_common[0]) if lv_common else None
    jcol = (f'joint_pass@{jl}' if cell_mode else f'joint_ub@{jl}') if jl else None
    heat = ['A_n', 'B_n', 'C_n', 'ABC_score']
    raw = ['A_score', 'B_score', 'C_score', 'ABC_score']
    txt_cols = [('cells', lambda r: f'{int(r.coverage)}'), ('Pareto', lambda r: f'L{int(r.pareto_layer)}'
                                                            if np.isfinite(r.pareto_layer) else '–')]
    if jcol and jcol in R:
        txt_cols.append(((f'joint@{jl}' if cell_mode else f'joint≤@{jl}'), lambda r: f'{int(r[jcol])}'))
    if cell_mode:
        txt_cols.append(('joint area', lambda r: f'{r.joint_area:.2g}'))
    fig, ax = plt.subplots(figsize=(8.6, 0.27 * n + 1.9), layout='constrained')
    M = R.loc[rows, heat].to_numpy(float)
    cm = P.SEQ.copy()
    ax.imshow(M, cmap=cm, vmin=0, vmax=1, aspect='auto', interpolation='nearest',
              extent=(-0.5, len(heat) - 0.5, n - 0.5, -0.5))
    for i, d in enumerate(rows):
        for j, (h, rc) in enumerate(zip(heat, raw)):
            v = R.at[d, rc]
            s = f'{v:.3g}' if h != 'ABC_score' else f'{v:.2f}'
            if h == 'C_n' and 'C_censored_cells' in R and R.at[d, 'C_success'] > 0 and \
                    R.at[d, 'C_censored_cells'] / R.at[d, 'C_success'] >= 0.2:
                s += '*'
            ax.text(j, i, s, ha='center', va='center', fontsize=7.2, color=P.text_color_on(cm(M[i, j])))
        for j, (_, f) in enumerate(txt_cols):
            ax.text(len(heat) + 0.15 + j * 0.95, i, f(R.loc[d]), ha='center', va='center', fontsize=7.5,
                    color=P.INK2)
    ax.set_xlim(-0.5, len(heat) + len(txt_cols) * 0.95 - 0.2)
    ax.set_ylim(n - 0.5, -0.5)
    xt = list(range(len(heat))) + [len(heat) + 0.15 + j * 0.95 for j in range(len(txt_cols))]
    xl = [f'A\n{labels["A"]}', f'B\n{labels["B"]}', f'C\n{labels["C"]}', f'ABC\n{args.agg}'] + \
         [t for t, _ in txt_cols]
    ax.set_xticks(xt, xl, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n), [f'{int(R.at[d, "rank"]):>3}  {d}' for d in rows], fontsize=7.8,
                  fontfamily='monospace')
    for t, d in zip(ax.get_yticklabels(), rows):
        if d == base:
            t.set_fontweight('bold')
        if R.at[d, 'pareto_layer'] == 1:
            t.set_color(P.CAT[0])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n + 1):
        ax.axhline(i - 0.5, color='white', lw=1.4)
    for j in range(len(heat) + 1):
        ax.axvline(j - 0.5, color='white', lw=1.4)
    note = (f'color = normalized score ({args.norm}); numbers = raw score. Blue design name = 3-axis Pareto front. '
            f'cells = min success cells over A/B/C.')
    if 'C_censored_cells' in R:
        note += '\n* ≥20 % of C success cells have arm-limited (censored) directions → C is a lower bound.'
    if jcol:
        note += ('\njoint = cells that held ' + jl + ' in every direction in A, B and C (exact, cell level).'
                 if cell_mode else '\njoint≤ = min(pass_A, pass_B, pass_C) at ' + jl + ': upper bound on cells '
                 'that held in all three (needs *_cells.csv for the exact count).')
    ax.set_xlabel(note, fontsize=7.3, color=P.INK2, loc='left')
    ax.set_title(f'A · B · C scorecard — top {args.top} by {args.rank_title}', pad=26)
    fig.savefig(out)
    plt.close(fig)


def fig_parallel(R, labels, args, out):
    cols = ['A_n', 'B_n', 'C_n', 'ABC_score']
    xs = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8.6, 5.0), layout='constrained')
    for _, r in R.iterrows():
        ax.plot(xs, r[cols].to_numpy(float), color=P.FAINT, lw=0.6, alpha=0.5, zorder=1)
    top = list(R.index[:args.highlight])
    for i, d in enumerate(top):
        ax.plot(xs, R.loc[d, cols].to_numpy(float), color=P.CAT[i], lw=2.2, marker='o', ms=5, zorder=3,
                label=f'#{int(R.at[d, "rank"])} {d}')
    base = args.baseline if args.baseline in R.index else None
    if base:
        ax.plot(xs, R.loc[base, cols].to_numpy(float), color=P.INK, lw=1.6, ls='--', marker='D', ms=4,
                zorder=4, label=f'#{int(R.at[base, "rank"])} {base} (baseline)')
    ax.set_xticks(xs, [f'{NAMES["A"]}\n{labels["A"]}', f'{NAMES["B"]}\n{labels["B"]}',
                       f'{NAMES["C"]}\n{labels["C"]}', f'combined\nABC {args.agg}'])
    for x in xs:
        ax.axvline(x, color='#bdbcb6', lw=0.8, zorder=0)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel(f'normalized score ({args.norm})')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f'A · B · C — top {len(top)} designs across the three disturbances (gray = all others)')
    fig.savefig(out)
    plt.close(fig)


def fig_coverage_quality(R, args, out):
    ok = (R['coverage'] > 0)
    X = R.loc[ok, 'coverage'].to_numpy(float)
    Y = R.loc[ok, 'quality_ABC'].to_numpy(float)
    names = list(R.index[ok])
    front = P.pareto_mask(np.c_[X, Y])
    fig, ax = plt.subplots(figsize=(8.6, 5.6), layout='constrained')
    sca = ax.scatter(X, Y, c=R.loc[ok, 'ABC_score'], cmap=P.SEQ, vmin=0, vmax=max(R['ABC_score'].max(), 1e-9),
                     s=26, edgecolors='white', linewidths=0.7, zorder=3)
    fi = np.flatnonzero(front)
    fi = fi[np.argsort(X[fi])]
    ax.plot(X[fi], Y[fi], drawstyle='steps-pre', color=P.INK, lw=1.1, zorder=2)
    ax.scatter(X[fi], Y[fi], s=60, facecolors='none', edgecolors=P.INK, linewidths=1.1, zorder=4)
    cb = fig.colorbar(sca, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label(f'ABC_score ({args.agg})')
    cb.outline.set_visible(False)
    idx = {d: i for i, d in enumerate(names)}
    lab = [names[i] for i in fi] + [d for d in R.index[:5] if d in idx]
    lab = list(dict.fromkeys(lab))
    base = args.baseline if args.baseline in idx else None
    lab = [d for d in lab if d != base]
    P.place_labels(ax, [X[idx[d]] for d in lab], [Y[idx[d]] for d in lab], [P.short(d) for d in lab])
    if base:
        i = idx[base]
        ax.scatter([X[i]], [Y[i]], marker='D', s=40, color=P.INK, zorder=5)
        P.place_labels(ax, [X[i]], [Y[i]], [f'{P.short(base)} (baseline)'], weight='bold')
    ax.set_xlim(0, X.max() * 1.08 + 0.5)
    ax.set_ylim(0, max(Y.max() * 1.1, 0.05))
    ax.set_xlabel('graspable cells (min over A/B/C of success cells) — workspace size')
    ax.set_ylabel(f'per-cell robustness, combined ({args.agg} of A/B/C mean level_min, each ÷ its max)')
    ax.set_title(f'A · B · C — workspace size vs all-round per-cell robustness '
                 f'(ringed + line = Pareto front, {int(front.sum())} designs)')
    fig.savefig(out)
    plt.close(fig)


def fig_correlation(R, out):
    cols = [('A_success', 'success cells'), ('A_area', 'A area'), ('B_area', 'B area'), ('C_area', 'C area'),
            ('A_mean', 'A mean/cell'), ('B_mean', 'B mean/cell'), ('C_mean', 'C mean/cell')]
    D = R[[c for c, _ in cols]].astype(float)
    C = D.corr(method='spearman').to_numpy()
    lab = [l for _, l in cols]
    fig, ax = plt.subplots(figsize=(6.6, 5.6), layout='constrained')
    im = ax.imshow(C, cmap=P.DIV, vmin=-1, vmax=1)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color=P.text_color_on(P.DIV((C[i, j] + 1) / 2)))
    ax.set_xticks(range(len(lab)), lab, rotation=35, ha='right')
    ax.set_yticks(range(len(lab)), lab)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('Spearman ρ')
    cb.outline.set_visible(False)
    ax.set_title('A · B · C — does robustness to one disturbance predict another?\n'
                 '(area = cells × mean; designs with ≥1 success cell)', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def fig_joint(R, lv_common, args, out, cell_mode):
    rows = list(R.index[:args.top])
    base = args.baseline if args.baseline in R.index else None
    if base and base not in rows:
        rows.append(base)
    lvls = [P.lv_key(lv) for lv in lv_common]
    pref = 'joint_pass' if cell_mode else 'joint_ub'
    lvls = [lk for lk in lvls if f'{pref}@{lk}' in R]
    if not lvls:
        print('[ABC] 공통 크기가 없어 ABC_6_joint 생략')
        return
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.32 * n + 1.8), layout='constrained')
    y = np.arange(n)[::-1].astype(float)
    if base and base == rows[-1] and base not in list(R.index[:args.top]):
        y[-1] -= 0.6
    cover = R.loc[rows, 'joint_success' if cell_mode else 'coverage'].to_numpy(float)
    ax.barh(y, cover, height=0.8, color='#ecebe7', edgecolor='none',
            label='cells graspable in all three' if cell_mode else 'graspable cells (min over A/B/C)')
    shades = [P.BLUE[k] for k in (250, 450, 650, 700)][:len(lvls)] if len(lvls) <= 4 else \
        [P.SEQ(x) for x in np.linspace(0.3, 1, len(lvls))]
    h = 0.75 / len(lvls)
    for j, (lk, colr) in enumerate(zip(lvls, shades)):
        v = R.loc[rows, f'{pref}@{lk}'].to_numpy(float)
        yy = y + 0.375 - h * (j + 0.5)
        ax.barh(yy, v, height=h * 0.92, color=colr, edgecolor='white', linewidth=0.5,
                label=f'held {lk} (mg / mg / g) in every direction of A, B and C')
        for yv, vv in zip(yy, v):
            if vv > 0:
                ax.text(vv + 0.15, yv, f'{int(vv)}', va='center', fontsize=6.5, color=P.INK2)
    ax.set_yticks(y, [f'{int(R.at[d, "rank"]):>3}  {d}' for d in rows], fontsize=7.8, fontfamily='monospace')
    for t, d in zip(ax.get_yticklabels(), rows):
        if d == base:
            t.set_fontweight('bold')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('cells')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.set_title(('A · B · C — cells that survive all three disturbances (exact, cell level)' if cell_mode else
                  'A · B · C — upper bound on cells that survive all three: min(pass_A, pass_B, pass_C)')
                 + f'\ntop {args.top} by {args.rank_title}', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def fig_workspace_maps(J, R, args, out):
    designs = list(R.index[:args.map_designs])
    base = args.baseline if args.baseline in R.index else None
    if base and base not in designs:
        designs.append(base)
    cols = [('F_A_mg', 'A  F_min [mg]', 'ok_A'), ('L_B_mg', 'B  pulse level [mg]', 'ok_B'),
            ('L_C_g', 'C  shake level [g]', 'ok_C'), ('joint_level', 'joint = min(A, B, C)', 'ok')]
    nr = len(designs)
    fig, axs = plt.subplots(nr, 4, figsize=(12.5, 2.5 * nr + 0.9), layout='constrained', squeeze=False,
                            sharex=True, sharey=True)
    vmax = args.map_vmax
    sca = None
    for i, d in enumerate(designs):
        g = J[J['design'] == d]
        for j, (c, title, okc) in enumerate(cols):
            ax = axs[i, j]
            bad = g[~g[okc]]
            ax.scatter(bad['r_mm'], bad['y_mm'], s=22, facecolors='none', edgecolors=P.FAINT, linewidths=0.9,
                       zorder=1)
            gg = g[g[okc]]
            for env, mk in ((True, 'o'), (False, '^')):
                s = gg[gg['env'] == env]
                sca = ax.scatter(s['r_mm'], s['y_mm'], c=np.minimum(s[c].astype(float), vmax), cmap=P.SEQ,
                                 vmin=0, vmax=vmax, marker=mk, s=46, edgecolors=P.INK2, linewidths=0.4, zorder=2)
            if i == 0:
                ax.set_title(title, fontsize=9, fontweight='normal', color=P.INK2)
            if j == 0:
                ax.set_ylabel(f'#{int(R.at[d, "rank"])} {d}\ny [mm]', fontsize=8)
            if i == nr - 1:
                ax.set_xlabel('r [mm]')
            ax.tick_params(labelsize=7)
    cb = fig.colorbar(sca, ax=axs[:, -1].tolist(), shrink=0.5, pad=0.01)
    cb.set_label(f'load held ÷ mg (color capped at {vmax:g})')
    cb.outline.set_visible(False)
    fig.suptitle('A · B · C — robust graspable workspace maps (● env, ▲ tip, ○ not graspable / failed control)',
                 x=0.01, ha='left', fontsize=11, fontweight='bold')
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def write_summary(R, T, lv_common, args, cell_mode, out):
    lines = []
    lines.append(f'score = {args.score}, norm = {args.norm}, agg = {args.agg}, rank by {args.rank_title}')
    lines.append(f'designs ranked: {len(R)}  (excluded as incomplete: {int((T["missing"] != "").sum())})')
    lines.append('')
    show = ['rank', 'A_score', 'B_score', 'C_score', 'ABC_score', 'coverage', 'pareto_layer']
    lk = P.lv_key(lv_common[0]) if lv_common else None
    if cell_mode:
        show += ['joint_success', 'joint_area'] + ([f'joint_pass@{lk}'] if lk else [])
    elif lk:
        show += [f'joint_ub@{lk}']
    lines.append(f'TOP {args.top}')
    lines.append(R[show].head(args.top).to_string(float_format=lambda v: f'{v:.3g}'))
    lines.append('')
    front = R[R['pareto_layer'] == 1].sort_values('ABC_score', ascending=False)
    lines.append(f'3-AXIS PARETO FRONT ({len(front)} designs; no design is at least as good in A, B and C '
                 'and better in one):')
    lines.append(', '.join(f'{d}(#{int(r["rank"])})' for d, r in front.iterrows()))
    lines.append('')
    ar = R[R['allround'] == 1]
    lines.append(f'ALL-ROUND (top {100 * args.top_frac:.0f} % in every one of A, B, C): '
                 + (', '.join(f'{d}(#{int(r["rank"])})' for d, r in ar.iterrows()) or 'none'))
    if args.baseline in R.index:
        b = R.loc[args.baseline]
        lines.append(f'\nBASELINE {args.baseline}: rank {int(b["rank"])} / {len(R)}, A {b.A_score:.3g}, '
                     f'B {b.B_score:.3g}, C {b.C_score:.3g}, Pareto layer {b.pareto_layer:g}')
    lines.append('')
    rho = R[['A_score', 'B_score', 'C_score']].corr(method='spearman')
    lines.append(f'Spearman between experiments: A–B {rho.at["A_score", "B_score"]:.2f}, '
                 f'A–C {rho.at["A_score", "C_score"]:.2f}, B–C {rho.at["B_score", "C_score"]:.2f}')
    zeroC = int((R['C_score'] <= 0).sum())
    if zeroC:
        lines.append(f'{zeroC} of {len(R)} designs score 0 in C (no cell held the smallest shake in all directions);'
                     ' with --agg min they all tie at ABC_score 0.')
    if 'C_censored_cells' in R:
        cs, su = R['C_censored_cells'].sum(), R['C_success'].sum()
        if su:
            lines.append(f'C: {int(cs)} of {int(su)} success cells ({100 * cs / su:.0f} %) have an arm-limited '
                         '(censored) direction; their C level is a lower bound.')
    txt = '\n'.join(lines)
    (out / 'ABC_summary.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)


# ----------------------------------------------------------------------------- main
def build_parser():
    p = argparse.ArgumentParser(description='실험 A·B·C 결과를 합쳐 세 외란을 모두 잘 견디는 설계 찾기')
    p.add_argument('inputs', nargs='*', help='all_designs.csv 또는 결과 폴더 3개 (순서 무관, 열 이름으로 구분)')
    p.add_argument('--A', default=None, help='실험 A all_designs.csv 또는 results_A/force')
    p.add_argument('--B', default=None, help='실험 B all_designs.csv 또는 results_B/impulse')
    p.add_argument('--C', default=None, help='실험 C all_designs.csv 또는 results_C/shake')
    p.add_argument('--out', default='figs_disturb/ABC', help='출력 폴더')
    p.add_argument('--score', default='area', help='실험마다 쓸 점수: area | mean | J@k | pass@k | cond@k')
    p.add_argument('--norm', choices=('max', 'rank'), default='max', help='정규화: s/max | 백분위')
    p.add_argument('--agg', choices=('min', 'geomean', 'mean'), default='min',
                   help='세 점수 결합: min(가장 약한 외란) | geomean | mean')
    p.add_argument('--rank-by', default='auto',
                   help='auto(cell 파일이 있으면 joint_area, 없으면 agg) | agg | pareto | 표의 열 이름')
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--highlight', type=int, default=5, help='평행 좌표에서 색으로 강조할 설계 수')
    p.add_argument('--label-top', type=int, default=8, help='산점도에 이름을 붙일 상위 설계 수')
    p.add_argument('--top-frac', type=float, default=0.25, help='all-round: 세 실험 모두 상위 이 비율 안')
    p.add_argument('--w-tip', type=float, default=0.2, help='joint_J의 tip 가중치')
    p.add_argument('--baseline', default='design0001')
    p.add_argument('--map-designs', type=int, default=4, help='workspace 지도에 그릴 상위 설계 수 (cell 단위)')
    p.add_argument('--map-vmax', type=float, default=2.0, help='workspace 지도 색 상한 [mg]')
    p.add_argument('--no-cells', action='store_true', help='cell 파일이 있어도 설계 단위로만 비교')
    p.add_argument('--designs-index', default=None, help='설계 변수 표 (ranking CSV에 붙임)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    P.setup_style()
    exps = load_inputs(args)
    for k in KEYS:
        print(f'[ABC] {k}: {exps[k].csv} ({len(exps[k].df)} designs)')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T, labels, lv_common = merged_table(exps, args)
    inc = T[T['missing'] != '']
    if len(inc):
        print(f'[ABC] 한 실험에만 빠진 설계 {len(inc)}개는 순위에서 뺌: '
              + ', '.join(f'{d}(no {m})' for d, m in inc['missing'].items()))
    for k in KEYS:
        c = f'{k}_incomplete'
        if c in T and (T[c] > 0).any():
            print(f'[ABC] 주의: {k}에서 incomplete cell이 있는 설계 {int((T[c] > 0).sum())}개 (시뮬레이션이 덜 끝남)')
    J = None if args.no_cells else joint_cells(exps)
    cell_mode = J is not None
    if cell_mode:
        jd = joint_per_design(J, lv_common, args.w_tip)
        T = T.join(jd)
        for c in jd.columns:
            T[c] = T[c].fillna(0)
        J.to_csv(out / 'ABC_joint_cells.csv', index=False, encoding='utf-8-sig', float_format='%.4g')
        um = int((~J['matched']).sum())
        if um:
            print(f'[ABC] 세 실험에서 짝이 안 맞는 cell {um}개 (한 실험에만 trace가 있음) → joint에서 제외')
    R, rb, args.rank_title = rank_table(T, args, cell_mode)
    print(f'[ABC] rank by {args.rank_title}' + (' (cell-level joint)' if cell_mode else ' (design-level)'))
    first = ['rank', 'A_score', 'B_score', 'C_score', 'A_n', 'B_n', 'C_n', 'ABC_score', 'ABC_min', 'ABC_geomean',
             'ABC_mean', 'pareto_layer', 'allround', 'coverage', 'quality_ABC']
    first += [c for c in R.columns if c.startswith('joint')]
    dv = []
    for e in exps.values():
        dv += [c for c in e.dvar_cols if c not in dv]
    Rout = R[[c for c in first if c in R] + [c for c in R.columns if c not in first]].copy()
    for c in dv:
        for e in exps.values():
            if c in e.df:
                Rout[c] = e.df[c].reindex(Rout.index)
                break
    Rout.to_csv(out / 'ABC_ranking.csv', encoding='utf-8-sig', float_format='%.4g')
    fig_pairwise(R, labels, args, out / 'ABC_1_pairwise.png')
    fig_scorecard(R, labels, lv_common, args, out / 'ABC_2_scorecard.png', cell_mode)
    fig_parallel(R, labels, args, out / 'ABC_3_parallel.png')
    fig_coverage_quality(R, args, out / 'ABC_4_coverage_quality.png')
    fig_correlation(R[R['coverage'] > 0], out / 'ABC_5_correlation.png')
    fig_joint(R, lv_common, args, out / 'ABC_6_joint.png', cell_mode)
    if cell_mode:
        fig_workspace_maps(J, R, args, out / 'ABC_7_workspace_maps.png')
    print()
    write_summary(R, T, lv_common, args, cell_mode, out)
    print(f'\n[ABC] 표·그림: {out.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
