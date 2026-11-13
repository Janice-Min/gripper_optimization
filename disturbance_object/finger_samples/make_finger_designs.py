#!/usr/bin/env python3
"""Stage-0 based finger-design sampler: 4 geometric factors only.

Varied factors (same ranges as generate_designs.py)
    L_total     100 .. 150 mm        total finger length L_p + L_d
    L_p_ratio   0.3 .. 0.7           L_p / L_total
    G           50  .. G_max mm      pivot gap (G_max default 100 mm)
    th_open_p   10  .. 45 deg        proximal open angle

Fixed factors (all inside the DOE ranges)
    th_open_d    = 0 deg             distal link collinear with the proximal link: each finger is
                                     one straight line at the open pose (phi = th_open_p).
                                     The original factor phi_fraction = (th_open_p - 10)/(th_open_p + 20)
                                     therefore follows th_open_p; it is listed in designs_index.csv.
    -- values below are read back from gripper_stage0.xml --
    k_p          = 0.0107245099833   N*m/rad
    k_d          = 0.000500423768821 N*m/rad
    preload_p_N  = 0.107245099833 N  -> tau0_p = 5.362e-4 N*m (Stage-0 springref offset 0.05 rad)
    preload_d_N  = 0.0500423768821 N -> tau0_d = 2.502e-4 N*m (Stage-0 springref offset 0.5 rad)
    (preload [N] -> torque [N*m] with the assumed 5 mm moment arm, as in generate_designs.py)

Sampling
    lhs  (default) maximin Latin hypercube with centred strata: every factor takes N
         distinct, evenly spaced values (no two designs share a value of any factor)
         and the 4-D points are spread out (best of --candidates random LHS).
    grid full factorial over --levels, e.g. --levels 6 5 6 8.
    The Stage-0 dimensions (150 mm, 0.5, 75 mm, 30 deg) are added as design0001 unless
    --no-baseline is given. With th_open_d = 0 it differs from gripper_stage0.xml only in
    the distal open angle (joint2/joint4 range, springref, keyframe).

Each sample is written as designNNNN.json with exactly the nine keys read by
generate_mjcf.py: L_p, L_d, G [m]; th_open_p, th_open_d [deg]; k_p, k_d [N*m/rad];
tau0_p, tau0_d [N*m].  designs_index.csv and DOE_configuration.json describe the set.

Examples
    python make_finger_designs.py                          # 300 LHS + baseline -> designs/
    python make_finger_designs.py --n 500 --out-dir designs --overwrite
    python make_finger_designs.py --method grid --levels 6 5 6 8
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

SEED = 20261113

# Varied factors: name, unit, low, high (G high is replaced by --gmax-mm)
FACTORS = [
    ('L_total_mm', 'mm', 100.0, 150.0),
    ('L_p_ratio', '-', 0.3, 0.7),
    ('G_mm', 'mm', 50.0, 100.0),
    ('th_open_p_deg', 'deg', 10.0, 45.0),
]
BASELINE = {'L_total_mm': 150.0, 'L_p_ratio': 0.5, 'G_mm': 75.0, 'th_open_p_deg': 30.0}

# Fixed factors = gripper_stage0.xml values
STAGE0_FIXED = {
    'th_open_d': 0.0,  # deg, straight finger (distal collinear with proximal)
    'k_p': 0.0107245099833,
    'k_d': 0.000500423768821,
    'preload_p_N': 0.107245099833,
    'preload_d_N': 0.0500423768821,
}
K_MIN, K_MAX = 0.0005, 0.02
PRELOAD_MAX_N = 10.0
JSON_KEYS = ('L_p', 'L_d', 'G', 'th_open_p', 'th_open_d', 'k_p', 'k_d', 'tau0_p', 'tau0_d')


def sig(x, digits=12):
    """Round to significant digits (keeps tiny stiffness/torque values exact)."""
    return float(f'{float(x):.{digits}g}')


# --------------------------------------------------------------------------- sampling
def centred_lhs(n, d, rng):
    """Centred Latin hypercube in [0,1]^d: column j is a permutation of (i+0.5)/n."""
    return np.column_stack([(rng.permutation(n) + 0.5) / n for _ in range(d)])


def min_pair_distance(u):
    diff = u[:, None, :] - u[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def maximin_lhs(n, d, rng, candidates):
    best, best_d = None, -1.0
    for _ in range(max(1, candidates)):
        u = centred_lhs(n, d, rng)
        dist = min_pair_distance(u) if n > 1 else 1.0
        if dist > best_d:
            best, best_d = u, dist
    return best, best_d


def full_grid(levels):
    axes = [np.linspace(0.0, 1.0, k) if k > 1 else np.array([0.5]) for k in levels]
    return np.array(list(itertools.product(*axes)), dtype=float)


# --------------------------------------------------------------------------- mapping
def to_physical(u, gmax_mm):
    lows = np.array([f[2] for f in FACTORS])
    highs = np.array([f[3] for f in FACTORS])
    highs[2] = gmax_mm
    return lows + (highs - lows) * u


def design_json(L_total_mm, ratio, G_mm, th_p, fixed, radius_mm):
    th_d = fixed['th_open_d']
    lt = L_total_mm / 1000.0
    arm = radius_mm / 1000.0
    return {
        'L_p': sig(lt * ratio),
        'L_d': sig(lt * (1.0 - ratio)),
        'G': sig(G_mm / 1000.0),
        'th_open_p': sig(th_p),
        'th_open_d': sig(th_d),
        'k_p': sig(fixed['k_p']),
        'k_d': sig(fixed['k_d']),
        'tau0_p': sig(fixed['preload_p_N'] * arm),
        'tau0_d': sig(fixed['preload_d_N'] * arm),
    }


def validate(d, gmax_mm, radius_mm):
    """Same limits as generate_designs.validate_sample and generate_mjcf.update_xml."""
    eps = 2e-8
    lt = d['L_p'] + d['L_d']
    tp, td = d['th_open_p'], d['th_open_d']
    checks = [
        (0.1 - eps <= lt <= 0.15 + eps, 'L_total'),
        (0.3 - eps <= d['L_p'] / lt <= 0.7 + eps, 'L_p_ratio'),
        (0.05 - eps <= d['G'] <= gmax_mm / 1000 + eps, 'G'),
        (d['G'] <= lt + eps, 'G > finger length'),
        (10.0 <= tp <= 45.0, 'th_open_p'),
        (10.0 - tp - 1e-7 <= td <= 30.0 + 1e-7, 'th_open_d'),
        (tp + td <= tp + 30.0 + eps, 'phi'),
        (K_MIN - 1e-9 <= d['k_p'] <= K_MAX + 1e-9, 'k_p'),
        (K_MIN - 1e-9 <= d['k_d'] <= K_MAX + 1e-9, 'k_d'),
        (0.0 <= d['tau0_p'] <= PRELOAD_MAX_N * radius_mm / 1000 + 1e-9, 'tau0_p'),
        (0.0 <= d['tau0_d'] <= PRELOAD_MAX_N * radius_mm / 1000 + 1e-9, 'tau0_d'),
    ]
    bad = [name for ok, name in checks if not ok]
    if bad:
        raise ValueError(f'Design violates {bad}: {d}')


# --------------------------------------------------------------------------- main
def build(args):
    rng = np.random.default_rng(args.seed)
    if args.method == 'lhs':
        u, dmin = maximin_lhs(args.n, len(FACTORS), rng, args.candidates)
        methods = ['LHS'] * len(u)
        sampling_note = (f'maximin centred LHS, n={args.n}, best of {args.candidates} '
                         f'candidates, min normalised pair distance={dmin:.4f}')
    else:
        u = full_grid(args.levels)
        methods = ['grid'] * len(u)
        sampling_note = f'full factorial grid, levels={args.levels}'
    x = to_physical(u, args.gmax_mm)

    rows = []
    if not args.no_baseline:
        rows.append(('baseline_stage0', [BASELINE[f[0]] for f in FACTORS], [None] * 4))
    rows += [(m, list(map(float, xi)), list(map(float, ui))) for m, xi, ui in zip(methods, x, u)]
    return rows, sampling_note


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='designs')
    p.add_argument('--method', choices=('lhs', 'grid'), default='lhs')
    p.add_argument('--n', type=int, default=300, help='number of LHS samples (baseline not counted)')
    p.add_argument('--candidates', type=int, default=200, help='random LHS tried for maximin')
    p.add_argument('--levels', type=int, nargs=4, default=[6, 5, 6, 8], metavar=('L', 'RATIO', 'G', 'TH'),
                   help='grid levels for L_total, L_p_ratio, G, th_open_p')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--gmax-mm', type=float, default=100.0, help='upper bound of G, 50..100 mm')
    p.add_argument('--radius-mm', type=float, default=5.0, help='assumed tendon moment arm for preload -> torque')
    p.add_argument('--no-baseline', action='store_true', help='do not add the Stage-0 design as design0001')
    p.add_argument('--overwrite', action='store_true', help='delete old design*.json in --out-dir first')
    for key, val in STAGE0_FIXED.items():
        p.add_argument('--' + key.replace('_', '-'), dest=key, type=float, default=val,
                       help=f'fixed value (default: {val})')
    a = p.parse_args()

    if not 50.0 <= a.gmax_mm <= 100.0:
        p.error('--gmax-mm must be within 50..100 mm (G <= minimum finger length 100 mm)')
    if a.method == 'lhs' and a.n < 1:
        p.error('--n must be >= 1')
    if a.method == 'grid' and min(a.levels) < 1:
        p.error('--levels must be >= 1')
    if not 0.0 <= a.th_open_d <= 30.0:
        # DOE rule 10 - th_open_p <= th_open_d <= 30 must hold for every th_open_p in 10..45
        p.error('--th-open-d must be within 0..30 deg (0 = straight finger)')
    fixed = {k: getattr(a, k) for k in STAGE0_FIXED}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    old = sorted(out.glob('design[0-9]*.json'))
    if old and not a.overwrite:
        p.error(f'{out} already has {len(old)} design*.json files; use --overwrite or another --out-dir')
    for f in old:
        f.unlink()

    rows, sampling_note = build(a)
    width = max(4, len(str(len(rows))))
    seen = {}
    index = []
    for i, (method, xphys, ucoded) in enumerate(rows, 1):
        L_total_mm, ratio, G_mm, th_p = xphys
        d = design_json(L_total_mm, ratio, G_mm, th_p, fixed, a.radius_mm)
        validate(d, a.gmax_mm, a.radius_mm)
        key = tuple(d[k] for k in JSON_KEYS)
        if key in seen:
            if method == 'baseline_stage0' or seen[key] == 'baseline_stage0':
                continue  # an LHS/grid point that coincides with the baseline: keep only one
            raise ValueError(f'Duplicate design at sample {i}: {d}')
        seen[key] = method
        design_id = f'design{len(index) + 1:0{width}d}'
        (out / f'{design_id}.json').write_text(json.dumps(d, indent=2) + '\n', encoding='utf-8')
        lt = d['L_p'] + d['L_d']
        index.append({
            'design_id': design_id, 'filename': f'{design_id}.json', 'method': method,
            'L_total_mm': sig(lt * 1000), 'L_p_ratio': sig(d['L_p'] / lt), 'G_mm': sig(d['G'] * 1000),
            'th_open_p_deg': d['th_open_p'], 'th_open_d_deg': d['th_open_d'],
            'L_p_mm': sig(d['L_p'] * 1000), 'L_d_mm': sig(d['L_d'] * 1000),
            'phi_open_deg': sig(d['th_open_p'] + d['th_open_d']),
            'phi_fraction': sig((d['th_open_p'] + d['th_open_d'] - 10.0) / (d['th_open_p'] + 20.0)),
            'preload_p_N': fixed['preload_p_N'],
            'preload_d_N': fixed['preload_d_N'],
            **{f'u_{FACTORS[j][0]}': ('' if ucoded[j] is None else sig(ucoded[j])) for j in range(4)},
            **d,
        })

    with (out / 'designs_index.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)

    config = {
        'number_of_designs': len(index),
        'sampling': sampling_note,
        'seed': a.seed,
        'varied_factors': {name: {'unit': unit, 'min': lo, 'max': (a.gmax_mm if name == 'G_mm' else hi)}
                           for name, unit, lo, hi in FACTORS},
        'fixed_factors': fixed,
        'fixed_factor_source': 'th_open_d: user choice (0 = straight finger); k_p, k_d, preloads: '
                               'gripper_stage0.xml stiffness and springref offsets',
        'th_open_d_rule': f'th_open_d fixed at {a.th_open_d:g} deg; phi = th_open_p + th_open_d; '
                          'phi_fraction = (phi - 10)/(th_open_p + 20) varies per design',
        'joint_moment_arm_mm_assumed': a.radius_mm,
        'baseline_included': not a.no_baseline,
        'json_units': {'L_p': 'm', 'L_d': 'm', 'G': 'm', 'th_open_p': 'deg', 'th_open_d': 'deg',
                       'k_p': 'N*m/rad', 'k_d': 'N*m/rad', 'tau0_p': 'N*m', 'tau0_d': 'N*m'},
    }
    (out / 'DOE_configuration.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    print(f'Generated {len(index)} design JSON files in {out} ({sampling_note})')
    for name, unit, lo, hi in FACTORS:
        col = [r[name] for r in index]
        print(f'  {name:14s} {min(col):9.4f} .. {max(col):9.4f} {unit}  ({len(set(col))} distinct values)')


if __name__ == '__main__':
    main()
