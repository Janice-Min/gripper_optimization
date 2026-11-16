#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
외란 실험 A/B/C 결과 그림 — all_designs.csv로 설계 순위·Pareto·생존 곡선·방향·실패 원인

disturb_force_rb5.py(A) / disturb_impulse_rb5.py(B) / disturb_shake_rb5.py(C)가 쓴 all_designs.csv를
읽어 실험마다 그림과 순위표를 만든다. 실험 종류는 열 이름으로 알아낸다
(robust_area → A, impact_area → B, shake_area → C). 결과 폴더를 주면 그 안의 all_designs.csv를 쓴다.

  python plot_disturb_results.py results_A/force results_B/impulse results_C/shake
  python plot_disturb_results.py results_C/shake/all_designs.csv --score J@0.5 --top 25
  python plot_disturb_results.py results_A/force --designs-index designs_index.csv

그림 (<out>/<A|B|C>/):
  1_ranking       상위 N 설계의 점수 막대 (env/tip 성분이 있으면 쌓아서). 기준 설계(--baseline)는 항상 표시
  2_pareto        성공 cell 수(workspace 크기) × cell당 평균 강건성. 점선은 area(= 수 × 평균)가 같은 곡선
  3_survival      외란 크기 k에서 모든 방향을 버틴 cell 수(pass@k). 상위 설계 + 기준 설계, 나머지는 회색
  4_directions    방향별 강건성 heatmap (A: 방향별 F_max 중앙값 env/tip, B·C: k에서 버틴 cell 비율)
  5_failures      실패 원인(slip/rotation/lost/collision…)과 실패 시점(B·C) 비율
  6_design_vars   설계 변수 vs 점수 (all_designs.csv에 설계 변수 열이 있거나 --designs-index를 줄 때)
표: <X>_ranking.csv (순위, 점수, 2_pareto의 Pareto front 여부)

점수 (--score):
  area      Σ_성공 cell level_min (A robust_area [mg], B impact_area [mg], C shake_area [g]) — 각 실험 스크립트의 순위 기준
  mean      cell당 평균 level_min (workspace 크기와 분리한 품질)
  J@k       env_pass@k + w_tip·tip_pass@k (모든 방향에서 k를 버틴 cell; tip은 w_tip배, 기본 0.2)
  pass@k, cond@k, 또는 all_designs.csv의 아무 열 이름
필요: numpy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgb  # noqa: E402

# ----------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
FAINT = '#d3d2cc'          # 배경 설계(회색 점·선)
NAN_BG = '#f3f2ef'         # heatmap 빈 칸
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BLUE = {100: '#cde2fb', 150: '#b7d3f6', 200: '#9ec5f4', 250: '#86b6ef', 300: '#6da7ec', 350: '#5598e7',
        400: '#3987e5', 450: '#2a78d6', 500: '#256abf', 550: '#1c5cab', 600: '#184f95', 650: '#104281',
        700: '#0d366b'}
ENV_C, TIP_C = BLUE[550], BLUE[250]
SEQ = LinearSegmentedColormap.from_list('seq_blue', [BLUE[k] for k in (100, 200, 300, 400, 500, 600, 700)])
DIV = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0a3a2', '#f0efec', '#9ec5f4', '#2a78d6'])
CAUSE_COLORS = {'slip': CAT[0], 'rotation': CAT[1], 'lost': CAT[2], 'collision': CAT[3], 'unstable': CAT[4],
                'drop': CAT[5]}
PHASE_COLORS = [BLUE[600], BLUE[400], BLUE[200]]
ARM_C = '#b5b4ad'


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.size': 9,
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.edgecolor': '#bdbcb6', 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.labelsize': 9,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 8,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': INK,
    })


def short(design):
    m = re.match(r'design0*(\d+)$', str(design))
    return f'd{int(m.group(1)):04d}' if m else str(design)


def suptitle(fig, title, sub=None):
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    if sub:
        fig.text(0.01, 0.955 if fig.get_figheight() > 4 else 0.93, sub, ha='left', va='top', fontsize=8.5,
                 color=INK2)


def text_color_on(hex_or_rgb):
    r, g, b = to_rgb(hex_or_rgb)
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def place_labels(ax, xs, ys, texts, fontsize=7.5, color=INK, weight='normal'):
    """간단한 겹침 회피: 후보 위치를 차례로 시도해 이미 놓은 라벨·축 밖과 겹치지 않는 첫 자리에 둔다."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    placed = getattr(ax, '_placed_bbs', [])
    offs = [(5, 5), (5, -5), (-5, 5), (-5, -5), (8, 0), (-8, 0), (0, 9), (0, -9), (12, 12), (12, -12),
            (-12, 12), (-12, -12), (20, 4), (-20, 4), (20, -6), (-20, -6), (0, 18), (0, -18)]
    out = []
    for x, y, s in zip(xs, ys, texts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        chosen = None
        for dx, dy in offs:
            ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
            va = 'bottom' if dy > 0 else ('top' if dy < 0 else 'center')
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va,
                            fontsize=fontsize, color=color, fontweight=weight, annotation_clip=False,
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=0, shrinkB=2)
                            if abs(dx) + abs(dy) > 16 else None)
            bb = t.get_window_extent(rend).expanded(1.04, 1.12)
            inside = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
            if inside and not any(bb.overlaps(p) for p in placed):
                chosen = t
                placed.append(bb)
                break
            t.remove()
        if chosen is None:
            t = ax.annotate(s, (x, y), xytext=offs[0], textcoords='offset points', fontsize=fontsize, color=color,
                            fontweight=weight, annotation_clip=False)
            placed.append(t.get_window_extent(rend))
            chosen = t
        out.append(chosen)
    ax._placed_bbs = placed
    return out


# ----------------------------------------------------------------------------- experiments
EXPS = {
    'A': dict(title='A · slow force ramp', area='robust_area', mean='F_min_mean_mg', median='F_min_median_mg',
              J='J_robust', unit='mg', what='F_min = weakest-direction F_max',
              area_parts=('robust_area_env', 'robust_area_tip'),
              causes=('slip', 'rotation', 'lost', 'collision', 'unstable', 'drop'), phases=(),
              cells_file='force_cells.csv'),
    'B': dict(title='B · impulse (50 ms pulse)', area='impact_area', mean='level_min_mean',
              median='level_min_median', J='J_impulse', unit='mg',
              what='level_min = largest pulse held in every direction', area_parts=None,
              causes=('slip', 'rotation', 'lost', 'collision', 'unstable'), phases=('pulse', 'transient', 'delayed'),
              cells_file='impulse_cells.csv'),
    'C': dict(title='C · arm shake', area='shake_area', mean='level_min_mean', median='level_min_median',
              J='J_shake', unit='g', what='level_min = largest hand acceleration held in every direction',
              area_parts=None, causes=('slip', 'rotation', 'lost', 'collision', 'unstable'),
              phases=('motion', 'transient', 'delayed'), cells_file='shake_cells.csv'),
}
DIR_ORDER = ['px', 'nx', 'py', 'ny', 'pz', 'nz']
DIR_LABEL = {'px': '+x\njaw up', 'nx': '−x\njaw low', 'py': '+y\nout', 'ny': '−y\npalm', 'pz': '+z\nup',
             'nz': '−z\ndown'}


def lv_key(lv):
    return f'{float(lv):g}'


def resolve_csv(p):
    p = Path(p)
    if p.is_dir():
        if (p / 'all_designs.csv').is_file():
            return p / 'all_designs.csv'
        hits = sorted(p.glob('*/all_designs.csv'))
        if len(hits) == 1:
            return hits[0]
        raise SystemExit(f'{p}: all_designs.csv를 찾지 못함 (또는 여러 개: {[str(h) for h in hits]})')
    if not p.is_file():
        raise SystemExit(f'{p}: 파일 없음')
    return p


def detect_exp(df):
    for k, s in EXPS.items():
        if s['area'] in df.columns:
            return k
    raise SystemExit('실험 종류를 알 수 없음 (robust_area / impact_area / shake_area 열이 없음)')


class Exp:
    def __init__(self, path, key=None):
        self.csv = resolve_csv(path)
        self.root = self.csv.parent
        df = pd.read_csv(self.csv, encoding='utf-8-sig')
        df['design'] = df['design'].astype(str).str.strip()
        self.df = df.set_index('design')
        self.key = key or detect_exp(self.df)
        self.spec = EXPS[self.key]
        self.levels = sorted({float(c.split('@', 1)[1]) for c in self.df.columns if c.startswith('pass@')})
        cols = list(self.df.columns)
        # 실험 스크립트는 설계 변수 열을 design 바로 뒤(ws_cells_attempted 앞)에 둔다
        self.dvar_cols = cols[:cols.index('ws_cells_attempted')] if 'ws_cells_attempted' in cols else []
        self.w_tip = self._infer_w_tip()
        for c in ('success', 'success_env', 'success_tip', self.spec['area']):
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce').fillna(0.0)

    def _infer_w_tip(self):
        ws = []
        for lv in self.levels:
            J, e, t = (f'{self.spec["J"]}@{lv_key(lv)}', f'env_pass@{lv_key(lv)}', f'tip_pass@{lv_key(lv)}')
            if all(c in self.df for c in (J, e, t)):
                sel = self.df[t] > 0
                ws += list(((self.df[J] - self.df[e]) / self.df[t])[sel])
        return float(np.median(ws)) if ws else 0.2

    def add_design_vars(self, path):
        dv = load_design_index(path, list(self.df.index))
        if dv is None:
            return
        new = [c for c in dv.columns if c not in self.df.columns]
        self.df = self.df.join(dv[new])
        self.dvar_cols = list(dict.fromkeys(self.dvar_cols + new))

    @property
    def title(self):
        return self.spec['title']

    def mean_quality(self):
        return pd.to_numeric(self.df[self.spec['mean']], errors='coerce')

    def score(self, which):
        """(series, column, label, components) — components: [(series, name, color)] 이면 막대를 쌓을 수 있음."""
        s, df = self.spec, self.df
        u = s['unit']
        comps = None
        if which == 'area':
            col = s['area']
            label = f'{col} = Σ level_min over success cells [{u}·cell]'
            if s['area_parts'] and all(p in df for p in s['area_parts']):
                comps = [(df[s['area_parts'][0]], 'env cells', ENV_C), (df[s['area_parts'][1]], 'tip cells', TIP_C)]
        elif which == 'mean':
            col = s['mean']
            label = f'{col}: mean {s["what"]} per success cell [{u}]'
        elif which == 'median':
            col = s['median']
            label = f'{col}: median {s["what"]} per success cell [{u}]'
        else:
            m = re.match(r'^(J|pass|cond|env_pass|tip_pass)@([0-9.]+)$', which)
            if m:
                kind, lv = m.group(1), lv_key(m.group(2))
                col = f'{s["J"]}@{lv}' if kind == 'J' else f'{kind}@{lv}'
                e, t = f'env_pass@{lv}', f'tip_pass@{lv}'
                if kind == 'J':
                    label = f'{col} = env_pass + {self.w_tip:g}·tip_pass at {lv} {u} (all directions held)'
                    if e in df and t in df:
                        comps = [(df[e], 'env cells', ENV_C), (self.w_tip * df[t], f'{self.w_tip:g} × tip cells', TIP_C)]
                elif kind == 'pass':
                    label = f'{col}: success cells that held {lv} {u} in every direction'
                    if e in df and t in df:
                        comps = [(df[e], 'env cells', ENV_C), (df[t], 'tip cells', TIP_C)]
                elif kind == 'cond':
                    label = f'{col}: fraction of success cells that held {lv} {u} in every direction'
                else:
                    label = col
            else:
                col, label = which, which
        if col not in df:
            avail = [s['area'], s['mean']] + [c for c in df.columns if '@' in c][:12]
            raise SystemExit(f'[{self.key}] 열 {col!r} 없음. 예: {avail}')
        ser = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        if comps:
            comps = [(pd.to_numeric(c, errors='coerce').fillna(0.0), n, k) for c, n, k in comps]
        return ser, col, label, comps

    def ranked(self, sc):
        """점수 내림차순, 같으면 cell당 평균 강건성, 성공 cell 수 순."""
        q = self.mean_quality().fillna(0.0)
        order = np.lexsort((-self.df['success'].to_numpy(), -q.to_numpy(), -sc.to_numpy()))
        return list(self.df.index[order])


def load_design_index(path, designs):
    if not path:
        return None
    rows = pd.read_csv(path, encoding='utf-8-sig')
    designs = set(designs)
    key = None
    for c in rows.columns:
        vals = set(rows[c].astype(str).str.strip())
        if vals & designs:
            key = c
            rows['_d'] = rows[c].astype(str).str.strip()
            break
    if key is None:
        for c in rows.columns:
            try:
                ids = rows[c].map(lambda v: f'design{int(float(v)):04d}')
            except (TypeError, ValueError):
                continue
            if set(ids) & designs:
                key = c
                rows['_d'] = ids
                break
    if key is None:
        print(f'[WARN] {path}: 설계 이름과 맞는 열을 못 찾음')
        return None
    out = rows.drop(columns=[key]).set_index('_d')
    num = out.apply(pd.to_numeric, errors='coerce')
    keep = [c for c in num.columns if num[c].notna().mean() > 0.8 and num[c].nunique() > 1]
    return num[keep]


def pareto_mask(P):
    """모든 열을 최대화할 때 지배당하지 않는 점."""
    P = np.asarray(P, float)
    if len(P) == 0:
        return np.zeros(0, bool)
    ge = (P[None, :, :] >= P[:, None, :]).all(-1)
    gt = (P[None, :, :] > P[:, None, :]).any(-1)
    return ~(ge & gt).any(1)


def pareto_layers(P):
    P = np.asarray(P, float)
    layer = np.zeros(len(P), int)
    left = np.arange(len(P))
    k = 1
    while left.size:
        m = pareto_mask(P[left])
        layer[left[m]] = k
        left = left[~m]
        k += 1
    return layer


# ----------------------------------------------------------------------------- figures
def fig_ranking(e, sc, col, label, comps, args, out):
    order = e.ranked(sc)
    rank = sc.rank(ascending=False, method='min').astype(int)
    rows = order[:args.top]
    base = args.baseline if args.baseline in e.df.index else None
    if base and base not in rows:
        rows = rows + [base]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.2, 0.26 * n + 1.5), layout='constrained')
    ypos = np.arange(n)[::-1].astype(float)
    if base and base == rows[-1] and base not in order[:args.top]:
        ypos[-1] -= 0.6
    vals = sc.loc[rows].to_numpy()
    if comps:
        left = np.zeros(n)
        for ser, name, color in comps:
            v = ser.loc[rows].to_numpy()
            ax.barh(ypos, v, left=left, height=0.62, color=color, edgecolor='white', linewidth=0.8, label=name)
            left += v
    else:
        ax.barh(ypos, vals, height=0.62, color=BLUE[450], edgecolor='white', linewidth=0.8)
    xmax = max(vals.max(), 1e-9)
    for y, d, v in zip(ypos, rows, vals):
        r = e.df.loc[d]
        ax.text(v + xmax * 0.012, y, f'{v:.3g}   n={int(r.success)} ({int(r.success_env)} env / '
                f'{int(r.success_tip)} tip)', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(ypos, [f'{rank[d]:>3}  {d}' for d in rows], fontsize=8, fontfamily='monospace')
    for t, d in zip(ax.get_yticklabels(), rows):
        if d == base:
            t.set_fontweight('bold')
            t.set_color(INK)
    ax.set_xlim(0, xmax * 1.42)
    ax.set_xlabel(label)
    ax.grid(axis='y', visible=False)
    if comps:
        ax.legend(loc='lower right', ncols=len(comps))
    ax.set_title(f'{e.title} — design ranking by {col} (top {args.top}'
                 + (f', baseline {base} in bold' if base else '') + ')')
    fig.savefig(out)
    plt.close(fig)


def fig_pareto(e, sc, args, out):
    df = e.df
    x = df['success'].astype(float)
    y = e.mean_quality()
    ok = (x > 0) & y.notna()
    X, Y = x[ok].to_numpy(), y[ok].to_numpy()
    names = list(df.index[ok])
    envf = (df['success_env'][ok] / df['success'][ok]).to_numpy()
    front = pareto_mask(np.c_[X, Y])
    fig, ax = plt.subplots(figsize=(8.2, 5.6), layout='constrained')
    xmax, ymax = X.max() * 1.08 + 0.5, Y.max() * 1.12
    # area가 같은 곡선 (area = 성공 cell 수 × 평균)
    area = X * Y
    ticks = matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, area.max())
    xs = np.linspace(0.5, xmax, 300)
    for a in ticks[1:]:
        yy = a / xs
        m = yy <= ymax
        ax.plot(xs[m], yy[m], color=FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if m.any():
            ax.text(xs[m][-1], yy[m][-1], f' area {a:g}', fontsize=7, color=MUTED, va='center', ha='left')
    sca = ax.scatter(X, Y, c=envf, cmap=SEQ, vmin=0, vmax=1, s=26, edgecolors='white', linewidths=0.7, zorder=3)
    fi = np.flatnonzero(front)
    fi = fi[np.argsort(X[fi])]
    ax.plot(X[fi], Y[fi], drawstyle='steps-pre', color=INK, lw=1.1, zorder=2)
    ax.scatter(X[fi], Y[fi], s=60, facecolors='none', edgecolors=INK, linewidths=1.1, zorder=4)
    cb = fig.colorbar(sca, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label('env fraction of success cells')
    cb.outline.set_visible(False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('success cells (graspable after zero-force control) — workspace size')
    ax.set_ylabel(f'{e.spec["mean"]}: mean {e.spec["what"]} [{e.spec["unit"]}]')
    lab = {names[i] for i in fi}
    top_area = [d for d in e.ranked(e.score('area')[0]) if d in names][:3]
    lab |= set(top_area)
    base = args.baseline if args.baseline in names else None
    idx = {d: i for i, d in enumerate(names)}
    ll = sorted(lab - {base}, key=lambda d: -area[idx[d]])
    place_labels(ax, [X[idx[d]] for d in ll], [Y[idx[d]] for d in ll], [short(d) for d in ll])
    if base:
        i = idx[base]
        ax.scatter([X[i]], [Y[i]], marker='D', s=40, color=INK, zorder=5)
        place_labels(ax, [X[i]], [Y[i]], [f'{short(base)} (baseline)'], weight='bold')
    ax.set_title(f'{e.title} — workspace size vs per-cell robustness '
                 f'(ringed + line = Pareto front, {int(front.sum())} designs)')
    fig.savefig(out)
    plt.close(fig)
    pf = pd.Series(False, index=df.index)
    pf.loc[[names[i] for i in fi]] = True
    return pf


def fig_survival(e, sc, args, out):
    df = e.df
    lvls = e.levels
    cols = ['success'] + [f'pass@{lv_key(lv)}' for lv in lvls]
    M = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy()
    ylab = 'cells that held every direction (pass@k)'
    if args.survival == 'cond':
        M = M / np.where(M[:, :1] > 0, M[:, :1], np.nan)
        ylab = 'fraction of success cells that held every direction (cond@k)'
    xs = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8.2, 5.0), layout='constrained')
    for row in M:
        ax.plot(xs, row, color=FAINT, lw=0.6, alpha=0.55, zorder=1)
    if args.highlight_designs:
        top = [d for d in args.highlight_designs if d in df.index]
        ttl = 'chosen designs'
    else:
        top = e.ranked(sc)[:args.highlight]
        ttl = f'top {len(top)} by score'
    for i, d in enumerate(top):
        r = M[df.index.get_loc(d)]
        ax.plot(xs, r, color=CAT[i % len(CAT)], lw=2, marker='o', ms=4.5, zorder=3, label=f'{d}')
    base = args.baseline if args.baseline in df.index else None
    if base:
        ax.plot(xs, M[df.index.get_loc(base)], color=INK, lw=1.6, ls='--', marker='D', ms=4, zorder=4,
                label=f'{base} (baseline)')
    ax.set_xticks(xs, ['0\n(none)'] + [lv_key(lv) for lv in lvls])
    ax.set_xlabel(f'disturbance level k [{e.spec["unit"]}]')
    ax.set_ylabel(ylab)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', title=ttl, title_fontsize=8)
    ax.set_title(f'{e.title} — robust workspace vs disturbance level (gray = all other designs)')
    fig.savefig(out)
    plt.close(fig)


def draw_heat(ax, M, rows, cols, vmin, vmax, fmt, cmap=SEQ, log=False):
    Mm = np.ma.masked_invalid(M)
    cm = cmap.with_extremes(bad=NAN_BG)
    norm = LogNorm(vmin, vmax, clip=True) if log else Normalize(vmin, vmax, clip=True)
    im = ax.imshow(Mm, cmap=cm, norm=norm, aspect='auto', interpolation='nearest')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '–', ha='center', va='center', fontsize=7, color=MUTED)
                continue
            rgb = cm(float(norm(max(v, vmin) if log else v)))
            ax.text(j, i, fmt(v), ha='center', va='center', fontsize=6.8, color=text_color_on(rgb))
    ax.set_xticks(range(len(cols)), cols, fontsize=7.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which='minor', color='white', lw=1.5)
    ax.tick_params(which='minor', length=0)
    return im


def found_dirs(columns, pattern):
    ds = []
    for c in columns:
        m = re.match(pattern, c)
        if m and m.group(1) not in ds and m.group(1) != 'min':
            ds.append(m.group(1))
    return sorted(ds, key=lambda d: (DIR_ORDER.index(d) if d in DIR_ORDER else 99, d))


def fig_directions(e, sc, args, out):
    df = e.df
    rows = e.ranked(sc)[:args.top]
    base = args.baseline if args.baseline in df.index else None
    if base and base not in rows:
        rows = rows + [base]
    rlab = [f'{d} n={int(df.at[d, "success"]):>2}' for d in rows]
    log = False
    if e.key == 'A':
        dirs = found_dirs(df.columns, r'^F_(\w+?)_median_(env|tip)_mg$')
        panels = [('env cells: median F_max per direction [mg]', [f'F_{d}_median_env_mg' for d in dirs]),
                  ('tip cells: median F_max per direction [mg]', [f'F_{d}_median_tip_mg' for d in dirs])]
        allv = np.concatenate([pd.to_numeric(df[c], errors='coerce').to_numpy(float)
                               for _, cs in panels for c in cs if c in df])
        cap = float(np.nanmax(allv)) if np.isfinite(allv).any() else 10.0   # ramp 상한 (--max-mg, 기본 10)
        fmt = lambda v: f'{v:.0f}+' if v >= cap - 1e-6 else (f'{v:.2g}' if v < 10 else f'{v:.0f}')  # noqa: E731
        vmin, vmax, log = 0.2, cap, True
        cblab = f'F_max [mg] (log color; "{cap:.0f}+" = held to the ramp cap)'
    else:
        dirs = found_dirs(df.columns, r'^held_(\w+?)@')
        dl = args.dir_levels or [lv for lv in e.levels if lv in (0.5, 1.0)] or e.levels[:2]
        panels = [(f'fraction of success cells that held {lv_key(lv)} {e.spec["unit"]}',
                   [f'held_{d}@{lv_key(lv)}' for d in dirs]) for lv in dl]
        fmt = lambda v: f'{100 * v:.0f}'  # noqa: E731
        vmin, vmax, cblab = 0.0, 1.0, 'held fraction (numbers in %)'
    if not dirs:
        print(f'[{e.key}] 방향별 열이 없어 4_directions 생략')
        return
    clab = [DIR_LABEL.get(d, d) for d in dirs]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 1.6, 0.27 * len(rows) + 1.9),
                            layout='constrained', squeeze=False)
    im = None
    for k, (ax, (ptitle, cs)) in enumerate(zip(axs[0], panels)):
        M = np.array([[pd.to_numeric(df.at[d, c], errors='coerce') if c in df else np.nan for c in cs]
                      for d in rows], float)
        im = draw_heat(ax, M, rlab if k == 0 else [''] * len(rows), clab, vmin, vmax, fmt, log=log)
        ax.set_title(ptitle, fontsize=8.5, fontweight='normal', color=INK2)
        if base and base == rows[-1] and base not in e.ranked(sc)[:args.top]:
            ax.axhline(len(rows) - 1.5, color=INK, lw=0.8)
    cb = fig.colorbar(im, ax=axs[0].tolist(), shrink=0.6, pad=0.01)
    if log:
        tk = [t for t in (0.25, 0.5, 1, 2, 5, 10, 20) if vmin <= t <= vmax]
        cb.set_ticks(tk, labels=[f'{t:g}' for t in tk])
        cb.minorticks_off()
    cb.set_label(cblab)
    cb.outline.set_visible(False)
    sub = ('direction = hand frame; C uses the hand\'s first-move direction' if e.key == 'C'
           else 'direction = load direction in hand frame')
    fig.suptitle(f'{e.title} — robustness by direction, top {args.top} by score\n({sub}; n = success cells; '
                 f'– = no cells of that type)',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.savefig(out)
    plt.close(fig)


def fig_failures(e, sc, args, out):
    df = e.df
    s = e.spec
    rows = e.ranked(sc)[:args.top]
    base = args.baseline if args.baseline in df.index else None
    if base and base not in rows:
        rows = rows + [base]
    labels = ['ALL designs'] + rows
    causes = [c for c in s['causes'] if f'cause_{c}' in df and df[f'cause_{c}'].sum() > 0]
    C = np.array([[df[f'cause_{c}'].sum() for c in causes]] + [[df.at[d, f'cause_{c}'] for c in causes]
                                                              for d in rows], float)
    phases = [p for p in s['phases'] if f'fail_{p}' in df and df[f'fail_{p}'].sum() > 0]
    has_arm = e.key == 'C' and 'arm_limited' in df
    P = None
    if phases:
        P = np.array([[df[f'fail_{p}'].sum() for p in phases]] + [[df.at[d, f'fail_{p}'] for p in phases]
                                                                 for d in rows], float)
        if has_arm:
            arm = np.r_[df['arm_limited'].sum(), [df.at[d, 'arm_limited'] for d in rows]].astype(float)
            P = np.c_[P, arm]
    n = len(labels)
    ncol = 2 if P is not None else 1
    fig, axs = plt.subplots(1, ncol, figsize=(5.2 * ncol + 1.2, 0.25 * n + 1.9), layout='constrained',
                            squeeze=False)
    y = np.arange(n)[::-1].astype(float)
    y[1:] -= 0.5

    def stack(ax, Mx, names, colors, hatches=None):
        tot = Mx.sum(1)
        frac = Mx / np.where(tot > 0, tot, np.nan)[:, None]
        left = np.zeros(n)
        for j, (nm, colr) in enumerate(zip(names, colors)):
            v = np.nan_to_num(frac[:, j])
            ax.barh(y, v, left=left, height=0.66, color=colr, edgecolor='white', linewidth=0.8, label=nm,
                    hatch=(hatches[j] if hatches else None))
            left += v
        for yy, t in zip(y, tot):
            ax.text(1.01, yy, f'n={int(t)}', va='center', fontsize=7, color=INK2)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100%'])
        ax.grid(axis='y', visible=False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06 if n > 12 else -0.1), ncols=len(names),
                  fontsize=7.5)

    ax = axs[0, 0]
    stack(ax, C, causes, [CAUSE_COLORS[c] for c in causes])
    ax.set_yticks(y, labels, fontsize=7.5, fontfamily='monospace')
    ax.get_yticklabels()[0].set_fontweight('bold')
    unit_txt = 'failed ramps (cell × direction)' if e.key == 'A' else 'failed trials (cell × direction × k)'
    ax.set_title(f'failure cause — share of {unit_txt}', fontsize=8.5, fontweight='normal', color=INK2)
    if P is not None:
        ax2 = axs[0, 1]
        names = list(phases) + (['arm-limited (not judged)'] if has_arm else [])
        colors = PHASE_COLORS[:len(phases)] + ([ARM_C] if has_arm else [])
        stack(ax2, P, names, colors, hatches=[None] * len(phases) + (['///'] if has_arm else []))
        ax2.set_yticks(y, [''] * n)
        ax2.set_title('when it failed' + (' (+ arm could not follow the command)' if has_arm else ''),
                      fontsize=8.5, fontweight='normal', color=INK2)
    extra = ''
    if e.key == 'A' and 'cause_cap' in df:
        extra = f' · held to cap (censored, not failures): {int(df["cause_cap"].sum())} directions'
    fig.suptitle(f'{e.title} — how grasps fail (top {args.top} by score){extra}', x=0.01, ha='left',
                 fontsize=10.5, fontweight='bold')
    fig.savefig(out)
    plt.close(fig)


def fig_design_vars(e, sc, col, args, out):
    df = e.df
    vars_ = [c for c in e.dvar_cols if pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.8
             and pd.to_numeric(df[c], errors='coerce').nunique() > 1][:12]
    if not vars_:
        print(f'[{e.key}] 설계 변수 열이 없어 6_design_vars 생략 (--designs-index designs_index.csv로 추가 가능)')
        return False
    nc = min(4, len(vars_))
    nr = math.ceil(len(vars_) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.6 * nr + 0.6), layout='constrained', squeeze=False)
    top = set(e.ranked(sc)[:10])
    base = args.baseline if args.baseline in df.index else None
    for ax, v in zip(axs.flat, vars_):
        xv = pd.to_numeric(df[v], errors='coerce')
        m = xv.notna()
        is_top = df.index.isin(list(top))
        ax.scatter(xv[m & ~is_top], sc[m & ~is_top], s=12, color=FAINT, edgecolors='none')
        ax.scatter(xv[m & is_top], sc[m & is_top], s=22, color=CAT[0], edgecolors='white', linewidths=0.6,
                   label='top 10')
        if base and m.get(base, False):
            ax.scatter([xv[base]], [sc[base]], marker='D', s=26, color=INK, label='baseline')
        rho = pd.Series(xv[m]).corr(sc[m], method='spearman')
        ax.set_title(f'{v}   ρ={rho:+.2f}', fontsize=8.5, fontweight='normal', color=INK2)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(vars_):]:
        ax.set_visible(False)
    axs.flat[0].legend(loc='best', fontsize=7)
    axs.flat[0].set_ylabel(col)
    fig.suptitle(f'{e.title} — design variables vs {col} (ρ = Spearman)', x=0.01, ha='left', fontsize=10.5,
                 fontweight='bold')
    fig.savefig(out)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- driver
def ranking_table(e, sc, col, pf):
    df = e.df
    order = e.ranked(sc)
    keep = ['success', 'success_env', 'success_tip', e.spec['area'], e.spec['mean'], e.spec['median']]
    keep += [c for lv in e.levels for c in (f'pass@{lv_key(lv)}', f'{e.spec["J"]}@{lv_key(lv)}') if c in df]
    for c in ('zero_fail', 'cells_with_censored_dir', 'arm_limited', 'infeasible'):
        if c in df:
            keep.append(c)
    t = df.loc[order, list(dict.fromkeys([c for c in keep if c in df]))].copy()
    t.insert(0, 'score', sc.loc[order])
    t.insert(0, 'score_col', col)
    t.insert(0, 'rank', sc.loc[order].rank(ascending=False, method='min').astype(int).to_numpy())
    t['pareto_size_vs_quality'] = pf.reindex(order).fillna(False).astype(int)
    for c in e.dvar_cols:
        t[c] = df.loc[order, c]
    return t


def run_one(path, args, key=None):
    e = Exp(path, key)
    if args.designs_index:
        e.add_design_vars(args.designs_index)
    sc, col, label, comps = e.score(args.score)
    out = Path(args.out) / e.key
    out.mkdir(parents=True, exist_ok=True)
    print(f'[{e.key}] {e.csv}  ({len(e.df)} designs, levels {e.levels}, w_tip {e.w_tip:g}) → {out}')
    fig_ranking(e, sc, col, label, comps, args, out / f'{e.key}_1_ranking.png')
    pf = fig_pareto(e, sc, args, out / f'{e.key}_2_pareto.png')
    fig_survival(e, sc, args, out / f'{e.key}_3_survival.png')
    fig_directions(e, sc, args, out / f'{e.key}_4_directions.png')
    fig_failures(e, sc, args, out / f'{e.key}_5_failures.png')
    fig_design_vars(e, sc, col, args, out / f'{e.key}_6_design_vars.png')
    t = ranking_table(e, sc, col, pf)
    t.to_csv(out / f'{e.key}_ranking.csv', encoding='utf-8-sig', index_label='design', float_format='%.4g')
    show = ['rank', 'score', 'success', 'success_env', 'success_tip', e.spec['mean']]
    print(t[show].head(10).to_string(float_format=lambda v: f'{v:.3g}'))
    if e.key == 'C' and 'cells_with_censored_dir' in e.df:
        cen, suc = e.df['cells_with_censored_dir'].sum(), e.df['success'].sum()
        if suc and cen / suc > 0.1:
            print(f'[C] 주의: 성공 cell {int(suc)}개 중 {int(cen)}개({100 * cen / suc:.0f} %)에 arm_limited로 '
                  'censored된 방향이 있음 → 그 cell의 level은 하한값 (파지 실패가 아니라 팔이 가속도를 못 낸 것)')
    return e


def build_parser():
    p = argparse.ArgumentParser(description='외란 실험 A/B/C 결과 그림 (all_designs.csv)')
    p.add_argument('inputs', nargs='+', help='all_designs.csv 또는 결과 폴더 (results_A/force 등). 여러 개 가능')
    p.add_argument('--out', default='figs_disturb', help='그림 폴더 (기본 figs_disturb/<A|B|C>/)')
    p.add_argument('--score', default='area', help='순위 점수: area | mean | median | J@k | pass@k | cond@k | 열 이름')
    p.add_argument('--top', type=int, default=20, help='순위·heatmap에 보일 설계 수')
    p.add_argument('--highlight', type=int, default=5, help='생존 곡선에서 색으로 강조할 상위 설계 수')
    p.add_argument('--highlight-designs', nargs='+', default=None,
                   help='생존 곡선에서 강조할 설계를 직접 지정 (A·B·C 그림에서 같은 설계가 같은 색)')
    p.add_argument('--baseline', default='design0001', help='항상 표시할 기준 설계')
    p.add_argument('--survival', choices=('pass', 'cond'), default='pass', help='생존 곡선 y: cell 수 | 비율')
    p.add_argument('--dir-levels', type=float, nargs='+', default=None, help='B·C 방향 heatmap의 크기 (기본 0.5 1)')
    p.add_argument('--designs-index', default=None, help='설계 변수 표 (designs_index.csv)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_style()
    for path in args.inputs:
        run_one(path, args)
    print(f'그림: {Path(args.out).resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
