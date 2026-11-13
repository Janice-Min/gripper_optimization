#!/usr/bin/env python3
"""Modify only design-dependent attributes in gripper_stage0.xml.

Usage:
  python generate_mjcf.py --design_input designs
  python generate_mjcf.py --design_input designs/design001.json
  python generate_mjcf.py --design_input designs --template gripper_stage0.xml

JSON: L_p,L_d,G in meters; th_open_p,th_open_d in degrees;
      k_p,k_d in N*m/rad; tau0_p,tau0_d in N*m (0..0.05).
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def update_xml(template, design):
    """Return Stage-0 MJCF tree with only design-dependent values edited."""
    Lp, Ld, G = (float(design[k]) for k in ('L_p', 'L_d', 'G'))
    tp, td = (float(design[k]) for k in ('th_open_p', 'th_open_d'))
    kp, kd = (float(design[k]) for k in ('k_p', 'k_d'))
    t0p, t0d = (float(design[k]) for k in ('tau0_p', 'tau0_d'))
    if not all(map(math.isfinite, (Lp, Ld, G, tp, td, kp, kd, t0p, t0d))):
        raise ValueError('Design includes non-finite values')
    if not (Lp > 0.01 and Ld > 0.01 and G > 0.02):
        raise ValueError('L_p, L_d and G must be positive (meters)')
    if not (10 <= tp <= 45 and 10 - tp - 1e-7 <= td <= 30 + 1e-7):
        raise ValueError('Open angle range violated (degrees)')
    if not (0.0005 - 1e-9 <= kp <= 0.02 + 1e-9 and
            0.0005 - 1e-9 <= kd <= 0.02 + 1e-9 and
            0 <= t0p <= 0.0500001 and 0 <= t0d <= 0.0500001):
        raise ValueError('Stiffness or tau0 out of DOE range (tau0 is N*m)')

    root = ET.parse(template, parser=ET.XMLParser(
        target=ET.TreeBuilder(insert_comments=True))).getroot()

    def node(tag, name):
        elem = root.find(f".//{tag}[@name='{name}']")
        if elem is None:
            raise ValueError(f'Template missing {tag} name={name!r}')
        return elem

    def attr(tag, name, key, value):
        node(tag, name).set(key, value)

    def fmt(*xyz):
        return ' '.join(f'{v:.12g}' for v in xyz)

    # Palm and bases: follow original Y-axis symmetry.
    for name, side in [('upper_link1', 1), ('lower_link1', -1)]:
        attr('body', name, 'pos', fmt(0, side * G / 2, 0))
    for name, side in [('joint1_pin', 1), ('joint3_pin', -1),
                       ('joint1_wrap', 1), ('joint3_wrap', -1)]:
        attr('geom', name, 'pos', fmt(0, side * G / 2, 0))
    for name, side in [('joint1_side', 1), ('joint3_side', -1)]:
        attr('site', name, 'pos', fmt(0, side * (G / 2 - 0.012), 0))
    for name, side in [('palm_up', 1), ('palm_lo', -1)]:
        attr('site', name, 'pos', fmt(0.008, side * (G / 2 - 0.006), 0))
    palm_size = node('geom', 'palm_geom').get('size').split()
    palm_size[1] = f'{G / 2 + 0.01:.12g}'
    attr('geom', 'palm_geom', 'size', ' '.join(palm_size))

    # Link lengths and tendon attachment points, preserving Stage-0 offsets.
    for name in ('upper_green_link', 'lower_green_link'):
        attr('geom', name, 'fromto', fmt(0, 0, 0, -Lp, 0, 0))
    for name in ('upper_red_link', 'lower_red_link'):
        attr('geom', name, 'fromto', fmt(0, 0, 0, -Ld, 0, 0))
    for name in ('upper_link2', 'lower_link2'):
        attr('body', name, 'pos', fmt(-Lp, 0, 0))
    for name in ('joint2_pin', 'joint4_pin'):
        attr('geom', name, 'pos', fmt(-Lp, 0, 0))
    for name, y in [('upper_prox_top', -0.004), ('lower_prox_top', 0.004)]:
        attr('site', name, 'pos', fmt(-Lp + 0.004, y, 0))
    for name, y in [('upper_tip', -0.004), ('lower_tip', 0.004)]:
        attr('site', name, 'pos', fmt(-Ld + 0.002, y, 0))

    # Joint angles: upper finger negative; lower finger mirrored positive.
    # springref is offset to create outward spring torque tau0 at the open pose.
    p, d = math.radians(tp), math.radians(td)
    for name, q, k, tau, limit in (
        ('joint1', -p, kp, t0p, fmt(-p, 1.4)),
        ('joint2', -d, kd, t0d, fmt(-d, 1.5)),
        ('joint3', +p, kp, t0p, fmt(-1.4, p)),
        ('joint4', +d, kd, t0d, fmt(-1.5, d)),
    ):
        sign = -1 if name in ('joint1', 'joint2') else 1
        for key, value in [('range', limit), ('stiffness', f'{k:.12g}'),
                           ('springref', f'{q + sign * tau / k:.12g}')]:
            attr('joint', name, key, value)

    # Keep the object's original 7 freejoint coordinates unchanged.
    key = node('key', 'open')
    qpos = key.get('qpos').split()
    if len(qpos) != 11:
        raise ValueError('Expected the 4 hinges + 7 freejoint coordinates of Stage 0')
    qpos[:4] = [f'{v:.12g}' for v in (-p, -d, p, d)]
    key.set('qpos', ' '.join(qpos))
    return root


def find_path(path):
    if path.exists():
        return path
    alternate = Path(__file__).resolve().parent / path
    if alternate.exists():
        return alternate
    raise FileNotFoundError(f'File/folder not found: {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--design_input', type=Path, required=True)
    ap.add_argument('--template', type=Path, default=Path('gripper_stage0.xml'))
    ap.add_argument('--output', type=Path, help='XML file (single input) or directory (folder input)')
    a = ap.parse_args()
    source = find_path(a.design_input)
    template = find_path(a.template)
    batch = source.is_dir()
    inputs = sorted(source.glob('design[0-9]*.json')) if batch else [source]
    if not inputs:
        ap.error(f'No design###.json files found in {source}')
    out = a.output or ((source.parent / 'xml') if batch else source.with_suffix('.xml'))
    if batch:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    for file in inputs:
        design = json.loads(file.read_text(encoding='utf-8-sig'))
        tree = ET.ElementTree(update_xml(template, design))
        dest = out / (file.stem + '.xml') if batch else out
        tree.write(dest, encoding='utf-8', xml_declaration=True)
    print(f'Generated {len(inputs)} Stage-0 XML file(s) -> {out}')


if __name__ == '__main__':
    main()
