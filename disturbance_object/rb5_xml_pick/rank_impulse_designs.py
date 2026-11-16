#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 B 설계 순위 — results_B/impulse[_tag]/all_designs.csv, direction_stats.csv로
  B_design_ranking.csv (impact_area 순위, bootstrap 순위 안정성, Pareto), 그림 2장을 만든다.

  python rank_impulse_designs.py                 # results_B/impulse
  python rank_impulse_designs.py --tag p20
"""
import argparse, os, re
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); ap.add_argument('--tag', default=None)
ap.add_argument('--boot', type=int, default=4000); ap.add_argument('--tier', type=float, default=None,
    help='top tier 기준 impact_area (기본: 18번째 설계의 값)')
A = ap.parse_args()
RES = (Path(A.out) if A.out else HERE / 'results_B') / (f'impulse_{A.tag}' if A.tag else 'impulse')
os.chdir(RES)
d = pd.read_csv('all_designs.csv', encoding='utf-8-sig')
ks=sorted(float(c.split('@')[1]) for c in d.columns if re.fullmatch(r'pass@[0-9.]+', c)); dk=list(np.diff([0.0]+ks))
dirs=sorted({c.split('_')[1].split('@')[0] for c in d.columns if c.startswith('held_')})
# per-design cell level_min multiset (success cells) + zero_fail cells as 0
def levels(r):
    n=int(r.success); p=[int(r[f'pass@{k:g}']) for k in ks]
    cnt=[n-p[0]]+[p[i]-p[i+1] for i in range(len(p)-1)]+[p[-1]]
    return np.repeat([0.0]+ks, cnt)
L={row.design: levels(row) for _, row in d.iterrows()}
for r in d.itertuples():
    assert abs(L[r.design].sum()-r.impact_area)<1e-9
d['dir_mean_level']=sum(dk[i]*d[[f'held_{dn}@{k:g}' for dn in dirs]].fillna(0).mean(axis=1) for i,k in enumerate(ks)).where(d.success>0)
d['env_share']=(d.success_env/d.success).where(d.success>0)
d['zero_fail_rate']=(d.zero_fail/d.traced_cells).where(d.traced_cells>0)
# ranking: impact_area, then pass@1, then dir_mean_level, then success
d=d.sort_values(['impact_area','pass@1','dir_mean_level','success'],ascending=False).reset_index(drop=True)
d['rank']=np.arange(1,len(d)+1)
d['rank_tied']=d.impact_area.rank(ascending=False,method='min').astype(int)
# bootstrap: resample traced cells (success + zero_fail as 0) per design
rng=np.random.default_rng(0); B=A.boot
names=d.design.tolist(); cells=[np.concatenate([L[n], np.zeros(int(d.loc[i,'zero_fail']))]) for i,n in enumerate(names)]
top10=np.zeros(len(d)); top20=np.zeros(len(d)); ranks=np.zeros((B,len(d)))
for b in range(B):
    sc=np.array([c[rng.integers(0,len(c),len(c))].sum() if len(c) else 0.0 for c in cells])
    sc=sc+rng.uniform(0,1e-6,len(sc))   # random tie-break
    rk=(-sc).argsort().argsort()+1; ranks[b]=rk
    top10+=rk<=10; top20+=rk<=20
d['P_top10']=top10/B; d['P_top20']=top20/B
d['rank_boot_median']=np.median(ranks,axis=0); d['rank_boot_p05']=np.percentile(ranks,5,axis=0); d['rank_boot_p95']=np.percentile(ranks,95,axis=0)
ia_lo=[];ia_hi=[]
for c in cells:
    if len(c)==0: ia_lo.append(0); ia_hi.append(0); continue
    s=[c[rng.integers(0,len(c),len(c))].sum() for _ in range(2000)]; ia_lo.append(np.percentile(s,5)); ia_hi.append(np.percentile(s,95))
d['impact_area_p05']=ia_lo; d['impact_area_p95']=ia_hi
# Pareto: success (workspace) vs level_min_mean (quality)
pts=d[d.success>0][['design','success','level_min_mean']].values
front=set(); best=-1
for nm,x,y in sorted(pts,key=lambda p:(-p[1],-p[2])):
    if y>best+1e-12: front.add(nm); best=y
d['pareto_ws_quality']=d.design.isin(front).astype(int)
cols=['rank','rank_tied','design','impact_area','impact_area_p05','impact_area_p95','P_top10','P_top20','rank_boot_median','rank_boot_p05','rank_boot_p95',
      'success','success_env','success_tip','zero_fail','zero_fail_rate','env_share','level_min_mean','level_min_median']+[f'pass@{k:g}' for k in ks]+[f'cond@{k:g}' for k in ks]+[
      f'J_impulse@{k:g}' for k in ks]+['dir_mean_level','pareto_ws_quality','ws_cells_attempted','rb5_pass_cells']
dvars=[c for c in d.columns if c not in cols and not re.match(r'(held_|env_pass|tip_pass|fail_|cause_|incomplete|traced|success_other|type_changed|level_min_mean)', c)]
cols=cols[:3]+dvars+cols[3:]
d[cols].to_csv('B_design_ranking.csv',index=False,encoding='utf-8-sig',float_format='%.4g')
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(d[['rank','design','impact_area','P_top10','P_top20','success','success_env','level_min_mean']].head(20).to_string(index=False))
print('pareto', sorted(front))
print('sum P_top10 over top10/top20 listed:', d.P_top10[:10].sum().round(2), d.P_top20[:20].sum().round(2))
print('designs with P_top20>=0.5:', (d.P_top20>=0.5).sum(), ' P_top10>=0.5:', (d.P_top10>=0.5).sum())

# ---------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#a3a29d', '#e4e3df', '#fcfcfb'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False})
s = pd.read_csv('direction_stats.csv', encoding='utf-8-sig')
a = s.groupby(['group', 'dir', 'k_mg'])[['n', 'held']].sum().reset_index(); a['rate'] = a.held / a.n
names = {'pz': '+z up', 'ny': '-y into palm', 'px': '+x', 'nx': '-x', 'py': '+y pull out', 'nz': '-z down'}
order = [o for o in ['pz', 'ny', 'px', 'nx', 'py', 'nz'] if o in set(a.dir)]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
for ax, g in zip(axes, ['env', 'tip']):
    n = int(a[(a.group == g)].n.max())
    for i, dn in enumerate(order):
        sel = a[(a.group == g) & (a.dir == dn)].sort_values('k_mg')
        ax.plot(sel.k_mg, 100 * sel.rate, color=CAT[i], lw=2, marker='o', ms=5, label=names[dn])
    ends = sorted([(100 * a[(a.group == g) & (a.dir == dn) & (a.k_mg == a.k_mg.max())].rate.iloc[0], dn) for dn in order])
    y_prev = -99
    for y, dn in ends:   # direct labels at the right end, nudged apart
        y_lab = max(y, y_prev + 6); y_prev = y_lab
        ax.text(a.k_mg.max() * 1.03, y_lab, names[dn], fontsize=7.5, color=INK2, va='center')
    ax.set_xticks(sorted(a.k_mg.unique())); ax.set_xlim(a.k_mg.min() - 0.1, a.k_mg.max() * 1.25); ax.set_ylim(-3, 103)
    ax.set_xlabel('pulse size k [mg]  (50 ms, m = 0.3 kg)')
    ax.set_title(f'{"Enveloping" if g == "env" else "Tip"} grasps ({n} cells)')
axes[0].set_ylabel('trials held for 10 s [%]')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05), fontsize=8)
fig.tight_layout()
fig.savefig('B_fig_direction_survival.png', dpi=170, bbox_inches='tight'); plt.close(fig)

r = pd.read_csv('B_design_ranking.csv', encoding='utf-8-sig')
top = r.head(32).iloc[::-1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw={'width_ratios': [1, 1.15]})
y = np.arange(len(top))
TIER = A.tier if A.tier is not None else float(r.impact_area.iloc[min(17, len(r) - 1)])
n_tier = int((r.impact_area >= TIER).sum())
tier = top.impact_area >= TIER
ax1.hlines(y, top.impact_area_p05, top.impact_area_p95, color=GRID, lw=4)
ax1.scatter(top.impact_area, y, s=36, color=np.where(tier, CAT[0], MUTED), zorder=3, edgecolor=SURF, linewidth=1.2)
ax1.set_yticks(y, [f'{d.replace("design", "d")}  ({int(e)}/{int(s_)})' for d, e, s_ in zip(top.design, top.success_env, top.success)], fontsize=7.5)
ax1.set_xlabel('impact_area = sum of level_min over success cells [mg]\nbar: 90% bootstrap range over cells')
ax1.set_title('Top 32 by impact_area (labels: env / success cells)')
ax1.axhline(len(top) - n_tier - 0.5, color=INK2, lw=0.8, ls='--')
ax1.text(top.impact_area_p95.max(), len(top) - n_tier - 0.35, f'top tier above ({n_tier})', fontsize=7.5, color=INK2, va='top', ha='right')
ax1.grid(axis='y', visible=False)
ok = r[r.success > 0]
ax2.scatter(ok.success, ok.level_min_mean, s=14, color=MUTED, alpha=0.6, linewidths=0, label='all designs')
tt = ok[ok.impact_area >= TIER]
ax2.scatter(tt.success, tt.level_min_mean, s=40, color=CAT[0], edgecolor=SURF, linewidth=1.2, zorder=3, label=f'top tier (impact_area >= {TIER:g})')
pf = ok[ok.pareto_ws_quality == 1].sort_values('success')
ax2.plot(pf.success, pf.level_min_mean, color=INK2, lw=1, zorder=2, drawstyle='steps-pre', label='Pareto front')
for _, p in pf.iterrows():
    ax2.annotate(p.design.replace('design', 'd'), (p.success, p.level_min_mean), xytext=(4, 3), textcoords='offset points', fontsize=7, color=INK2)
ax2.set_xlabel('success cells (RB5 pick passed and held 10 s with no force)')
ax2.set_ylabel('mean level_min per cell [mg]')
ax2.set_title('Workspace size vs per-grasp impulse robustness')
ax2.legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig('B_fig_design_ranking.png', dpi=170, bbox_inches='tight'); plt.close(fig)
print(f'[RANK] {RES / "B_design_ranking.csv"}, B_fig_direction_survival.png, B_fig_design_ranking.png')
