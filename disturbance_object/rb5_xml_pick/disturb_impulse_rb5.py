#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실험 B — 짧은 충격(펄스)에 대한 파지 유지 (RB5 + grasp_cells)

run_rb5_workspace.py가 저장한 grasp_cells/<design>/<cell>/ 의 '들어 올린 직후' 상태(state.npz)를
복원해 잠시 안정시킨 뒤, 물체 질량중심에 손 좌표계 방향의 짧은 펄스 F = k*mg 를 T 동안 주고,
펄스 후 10 s 동안 놓치지 않고 계속 잡고 있는지(held) 판정한다. 외력 0 대조도 같이 돌린다.

  python disturb_impulse_rb5.py --designs design0001 --limit 3 --dirs py nx     # 빠른 시험
  python disturb_impulse_rb5.py --workers 6                                     # 전체 (끊기면 같은 명령으로 이어서)
  python disturb_impulse_rb5.py --rescore --slip-mode residual                  # 시뮬레이션 없이 재판정
  python disturb_impulse_rb5.py --view design0001/r60y15 --dir py --mag 1        # 한 시행을 viewer로

셀 입력: 기본은 이 파일 옆 grasp_cells_xml/ (collect_cell_xmls.py가 만든 XML 하나 = cell 하나,
상태는 <key name="lifted">, summary는 <custom>). 없으면 grasp_cells/ (폴더, state.npz). --cells-dir로 지정.
--clean: 시작 전에 results_X 안의 기존 결과를 모두 지움.
시뮬레이션 결과는 cell마다 traces/<cell>.npz 하나에 저장하고, 표(CSV)는 항상 trace에서 다시 계산한다.
cell 이름은 입력이 폴더(r050.0_y005.0)든 XML이든 짧은 이름 r50y05로 통일한다 (A·C도 같음).
그래서 판정 기준만 바꿀 때는 --rescore로 충분하고, 중단돼도 같은 명령으로 끝난 cell을 건너뛴다.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- directions
# 손 좌표계 (workspace 평가와 같은 x, y 정의):
#   +x : jaw 방향, upper 손가락 쪽 (palm +y)      -x : lower 손가락 쪽
#   +y : 손 밖으로 빼내기 (손가락이 뻗는 방향)    -y : palm 쪽으로 밀기
#   +z : 원기둥 축 = palm z (이 장착에서는 world 위쪽)
AXES = {'px': (1, 0, 0), 'nx': (-1, 0, 0), 'py': (0, 1, 0),
        'ny': (0, -1, 0), 'pz': (0, 0, 1), 'nz': (0, 0, -1)}


def _edge_dirs():
    out = {}
    for a, b in (('x', 'y'), ('x', 'z'), ('y', 'z')):
        for sa in 'pn':
            for sb in 'pn':
                v = [0.0, 0.0, 0.0]
                v['xyz'.index(a)] = 1.0 if sa == 'p' else -1.0
                v['xyz'.index(b)] = 1.0 if sb == 'p' else -1.0
                out[f'{sa}{a}{sb}{b}'] = tuple(x / math.sqrt(2.0) for x in v)
    return out


EDGES = _edge_dirs()
ALL_DIRS = {**AXES, **EDGES}

DEFAULT_MAGS = [0.5, 1.0, 2.0, 3.0]
# 이 값이 다르면 trace의 의미가 달라진다 (같은 출력 폴더에 섞지 않는다)
SIM_KEYS = ('pulse_ms', 'shape', 'pre_s', 'after_s', 'settle_s', 'sample_ms', 'coarse_s',
            'check_ms', 'stop_mm', 'stop_lost_s', 'lost_n', 'tension')
BASE_COLS = ['t', 'disp', 'rot', 'tilt', 'lost', 'ext', 'side_min',
             'dx', 'dy', 'dz', 'hand', 'palm_f', 'side0', 'side1']
GROUPS = ('env', 'tip', 'other')


def trial_key(direction, k):
    return f'{direction}_{k:g}mg'


def parse_dirs(tokens):
    if not tokens:
        return list(AXES)
    out = []
    for tok in tokens:
        if tok == 'axes':
            out += list(AXES)
        elif tok == 'edges':
            out += list(EDGES)
        elif tok == 'all':
            out += list(ALL_DIRS)
        elif tok in ALL_DIRS:
            out.append(tok)
        else:
            raise SystemExit(f'unknown direction {tok!r}; use axes, edges, all or {", ".join(ALL_DIRS)}')
    seen = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def sim_hash(sim):
    blob = json.dumps({k: sim[k] for k in SIM_KEYS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- scene loading
def _as_path(s):
    if '\\' in s or (len(s) > 1 and s[1] == ':'):
        return PureWindowsPath(s)
    return Path(s)


def _resolve_file(file_attr, meshdir, roots):
    cands = [file_attr]
    if meshdir:
        cands.append(os.path.join(meshdir, file_attr))
    parts = _as_path(file_attr).parts
    low = [p.lower() for p in parts]
    if 'meshes' in low:
        i = len(low) - 1 - low[::-1].index('meshes')
        rel = parts[i + 1:]
        for r in roots:
            cands.append(str(Path(r, *rel)))
    for c in cands:
        try:
            if os.path.isfile(c):
                return os.path.abspath(c)
        except OSError:
            pass
    return None


def load_scene_xml(scene_path, mesh_roots, allow_missing_collision=False):
    """scene.xml을 읽어 mesh 경로를 찾아 바꾼 XML 문자열을 돌려준다.

    scene.xml의 mesh 경로는 절대 경로다. 폴더를 옮겼으면 --mesh-dir(…/assets/rb5_850e/meshes)에서
    같은 하위 경로를 찾는다. 시각용 mesh(contype 0)를 못 찾으면 빼고, 충돌용 mesh를 못 찾으면 멈춘다.
    """
    root = ET.parse(scene_path).getroot()
    comp = root.find('compiler')
    meshdir = comp.get('meshdir') if comp is not None else None
    asset = root.find('asset')
    missing = {}
    if asset is not None:
        for el in asset.findall('mesh'):
            f = el.get('file')
            if not f:
                continue
            r = _resolve_file(f, meshdir, mesh_roots)
            if r:
                el.set('file', r)
            else:
                missing[el.get('name')] = el
    if comp is not None:
        comp.attrib.pop('meshdir', None)
    dropped_collision = []
    if missing:
        for parent in root.iter():
            for g in list(parent):
                if g.tag == 'geom' and g.get('mesh') in missing:
                    visual = ('visual' in (g.get('class') or '')) or (
                        g.get('contype') == '0' and g.get('conaffinity') == '0')
                    if not visual:
                        dropped_collision.append(g.get('mesh'))
                    parent.remove(g)
        for el in missing.values():
            asset.remove(el)
        if dropped_collision and not allow_missing_collision:
            raise FileNotFoundError(
                f'collision mesh files not found ({", ".join(sorted(set(dropped_collision)))}). '
                f'Give --mesh-dir <rb5_xml_pick>/assets/rb5_850e/meshes')
    return ET.tostring(root, encoding='unicode'), sorted(missing), sorted(set(dropped_collision))


# ----------------------------------------------------------------------------- cells (폴더 또는 XML 하나)
# grasp_cells/<design>/<cell>/        scene.xml + state.npz + summary.json   (상태: state.npz)
# grasp_cells_xml/<design>_rYYyZZ.xml collect_cell_xmls.py가 만든 XML 하나  (상태: <key name="lifted">,
#                                     summary·cell 이름: <custom><text name="cell_summary"/"cell_id">)
STATE_KEY = 'lifted'
_CELL_ID_RE = re.compile(r'<text\s+name="cell_id"\s+data="([^"]+)"')
# cell 이름 두 가지: 폴더 이름 r050.0_y005.0 (grasp_cells) / 짧은 이름 r50y05 (collect_cell_xmls.py의 XML 파일 이름).
# 결과(trace 파일 이름, CSV의 cell 열)는 입력이 폴더든 XML이든 항상 짧은 이름으로 저장한다.
_LONG_CELL_RE = re.compile(r'^r(-?\d+(?:\.\d+)?)_y(-?\d+(?:\.\d+)?)$')
_SHORT_CELL_RE = re.compile(r'^r(m?\d+(?:p\d+)?)y(m?\d+(?:p\d+)?)$')


def _cell_num(v):
    """collect_cell_xmls.py의 num()과 같은 규칙: 50.0 → '50', 5.0 → '05', 52.5 → '52p5'."""
    v = float(v)
    if v.is_integer():
        return f'{int(v):02d}'
    return f'{v:g}'.replace('.', 'p').replace('-', 'm')


def compact_cell_name(name):
    """'r050.0_y005.0' → 'r50y05'. 이미 짧은 이름이거나 형식이 다르면 그대로."""
    name = str(name).strip()
    mt = _LONG_CELL_RE.match(name)
    if mt:
        return f'r{_cell_num(mt.group(1))}y{_cell_num(mt.group(2))}'
    return name


def cell_ry(name):
    """cell 이름(두 형식 모두) → (r_mm, y_mm). 해석이 안 되면 (inf, inf) (정렬용)."""
    name = str(name).strip()
    mt = _LONG_CELL_RE.match(name)
    if mt:
        return float(mt.group(1)), float(mt.group(2))
    mt = _SHORT_CELL_RE.match(name)
    if mt:
        def f(s):
            return float(s.replace('m', '-').replace('p', '.'))
        return f(mt.group(1)), f(mt.group(2))
    return math.inf, math.inf


def is_xml_cell(p):
    return Path(p).suffix.lower() == '.xml'


def is_xml_cells_dir(cells_dir):
    return Path(cells_dir).is_dir() and any(Path(cells_dir).glob('*.xml'))


@functools.lru_cache(maxsize=None)
def _raw_cell_parts(p):
    """cell 경로 -> (design, 원래 cell 이름). 폴더면 폴더 이름, XML이면 <custom> cell_id, 없으면 파일 이름."""
    p = Path(p)
    if not is_xml_cell(p):
        return p.parent.name, p.name
    mt = _CELL_ID_RE.search(p.read_text(encoding='utf-8', errors='ignore'))
    if mt:
        parts = mt.group(1).replace('\\', '/').strip('/').split('/')
        if len(parts) >= 2:
            return parts[-2], parts[-1]
    design, _, cell = p.stem.partition('_')
    return design, (cell or p.stem)


def cell_parts(p):
    """cell 경로 -> (design, cell). cell은 항상 짧은 이름(r50y05) — 결과 폴더·CSV에 쓰는 이름."""
    design, cell = _raw_cell_parts(p)
    return design, compact_cell_name(cell)


def cell_label(p):
    return '/'.join(cell_parts(p))


def _normalize_cid(cid):
    """'design0001/r060.0_y015.0', 'design0001/r60y15', 'design0001_r60y15(.xml)' → 'design0001/r60y15'."""
    cid = str(cid).replace('\\', '/').strip().strip('/')
    if cid.lower().endswith('.xml'):
        cid = cid[:-4]
    parts = cid.split('/')
    if len(parts) >= 2:
        return f'{parts[-2]}/{compact_cell_name(parts[-1])}'
    design, sep, cell = cid.partition('_r')
    if sep:
        return f'{design}/{compact_cell_name("r" + cell)}'
    return cid


@functools.lru_cache(maxsize=None)
def _xml_cell_map(cells_dir):
    out = {}
    for f in sorted(Path(cells_dir).glob('*.xml')):
        out[cell_label(f)] = f
        out['/'.join(_raw_cell_parts(f))] = f
        out[f.stem] = f
        out[f.name] = f
    return out


def resolve_cell(cells_dir, cid):
    """'design0001/r60y15', 'design0001/r060.0_y015.0', 'design0001_r60y15', 'design0001_r60y15.xml' 모두 받는다."""
    cells_dir = Path(cells_dir)
    cid = str(cid).replace('\\', '/')
    if is_xml_cells_dir(cells_dir):
        mp = _xml_cell_map(str(cells_dir))
        f = mp.get(cid) or mp.get(Path(cid).name) or mp.get(_normalize_cid(cid))
        if f is None:
            raise SystemExit(f'cell {cid!r} not found in {cells_dir}')
        return f
    p = cells_dir / cid
    if (p / 'state.npz').is_file():
        return p
    # 짧은 이름으로 준 경우: 설계 폴더 안에서 같은 (r, y) 폴더를 찾는다
    norm = _normalize_cid(cid)
    if '/' in norm:
        design, cell = norm.split('/', 1)
        ddir = cells_dir / design
        if ddir.is_dir():
            for c in sorted(ddir.iterdir()):
                if c.is_dir() and compact_cell_name(c.name) == cell and (c / 'state.npz').is_file():
                    return c
    raise SystemExit(f'no state.npz for cell {cid!r} in {cells_dir}')


def _xml_summary(path):
    """XML 안의 cell_summary. 없으면 같은 폴더 index.csv의 몇 가지 값만."""
    root = ET.parse(path).getroot()
    for el in root.iter('text'):
        if el.get('name') == 'cell_summary' and el.get('data'):
            return json.loads(el.get('data')), 'xml'
    for row in read_csv(Path(path).parent / 'index.csv') or []:
        if row.get('file') == Path(path).name:
            num = lambda v: float(v) if v not in (None, '') else None  # noqa: E731
            flag = lambda v: {'True': True, 'False': False}.get(str(v), v)  # noqa: E731
            return ({'cell': {'r_mm': num(row.get('r_mm')), 'y_mm': num(row.get('y_mm')),
                              'workspace_grasp_type': row.get('workspace_grasp_type') or None},
                     'four_link_contact': flag(row.get('four_link_contact')),
                     'success': flag(row.get('success'))}, 'index.csv')
    return {}, 'none'


def load_cell(cell_path, mesh_roots, allow_missing_collision=False):
    """cell 하나를 읽어 파지 상태로 복원. 반환 (m, d, xml, summary, info).

    폴더: scene.xml + state.npz(mj_setState) + summary.json
    XML : 파일 하나 + <key name="lifted">(mj_resetDataKeyframe) + <custom> summary
    """
    import mujoco
    p = Path(cell_path)
    if not p.exists():  # cells_dir/design0001/r060.0_y015.0 처럼 원래 id로 만든 경로 (view 스크립트 등)
        p = resolve_cell(p.parent.parent, f'{p.parent.name}/{p.name}')
    xml_mode = is_xml_cell(p)
    scene = p if xml_mode else p / 'scene.xml'
    xml, dropped, dropped_col = load_scene_xml(scene, mesh_roots, allow_missing_collision)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if xml_mode:
        summary, ssrc = _xml_summary(p)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, STATE_KEY)
        if kid < 0:
            raise ValueError(f'{p.name}: <key name="{STATE_KEY}"> 없음')
        mujoco.mj_resetDataKeyframe(m, d, kid)
        state_src = f'key:{STATE_KEY}'
    else:
        summary, ssrc = (read_json(p / 'summary.json', {}) or {}), 'summary.json'
        with np.load(p / 'state.npz', allow_pickle=False) as z:
            spec = int(z['integration_spec']) if 'integration_spec' in z.files else int(
                mujoco.mjtState.mjSTATE_INTEGRATION)
            state = np.array(z['integration_state'], dtype=float)
        if mujoco.mj_stateSize(m, spec) != state.size:
            raise ValueError(f'state size {state.size} != model {mujoco.mj_stateSize(m, spec)} '
                             f'(scene.xml과 state.npz가 다른 모델?)')
        mujoco.mj_setState(m, d, state, spec)
        state_src = 'state.npz'
    info = {'cell_path': str(p), 'state_source': state_src, 'summary_source': ssrc,
            'dropped_meshes': dropped, 'dropped_collision_meshes': dropped_col}
    return m, d, xml, summary, info


def warn_old_xml(cells, tag):
    """summary가 안 들어간(예전 collect_cell_xmls.py로 만든) XML이 있으면 알림."""
    old = [p for p in cells if is_xml_cell(p) and 'name="cell_summary"' not in Path(p).read_text(encoding='utf-8', errors='ignore')]
    if old:
        print(f'[{tag}] [WARN] {len(old)} XML files have no <custom> cell_summary (e.g. {Path(old[0]).name}): '
              'hand/object names and hand frame fall back to defaults. '
              'Re-run collect_cell_xmls.py to embed summary.json.')


def default_cells_dir(explicit=None):
    """--cells-dir가 없으면 grasp_cells_xml(있으면)을 먼저, 그다음 grasp_cells."""
    if explicit:
        return Path(explicit).resolve()
    for p in (HERE / 'grasp_cells_xml', HERE.parent / 'grasp_cells_xml', HERE / 'grasp_cells', HERE.parent / 'grasp_cells'):
        if p.is_dir():
            return p.resolve()
    return (HERE / 'grasp_cells').resolve()


def clean_results(out_root, cells_dir, tag):
    """--clean: 결과 상위 폴더(results_X) 안의 기존 파일을 모두 지운다."""
    out_root, cells_dir = Path(out_root).resolve(), Path(cells_dir).resolve()
    if (out_root in (HERE, HERE.parent, Path(out_root.anchor)) or out_root in HERE.parents
            or out_root == cells_dir or out_root in cells_dir.parents):
        raise SystemExit(f'[{tag}] --clean refused: {out_root} is not a results folder')
    if not out_root.is_dir():
        return
    items = list(out_root.iterdir())
    if not items:
        return
    nfile = sum(1 for _ in out_root.rglob('*') if _.is_file())
    print(f'[{tag}] --clean: removing {nfile} files in {out_root} ({", ".join(sorted(i.name for i in items)[:8])}'
          + (' …' if len(items) > 8 else '') + ')')
    for it in items:
        if it.is_dir() and not it.is_symlink():
            shutil.rmtree(it)
        else:
            it.unlink()


def _name(m, objtype, *cands):
    import mujoco
    for c in cands:
        if c is None:
            continue
        for n in (c, f'hand__{c}'):
            i = mujoco.mj_name2id(m, objtype, n)
            if i >= 0:
                return i, n
    raise KeyError(f'none of {cands} found')


class Ctx:
    """모델에서 쓰는 id와 손 좌표계."""

    def __init__(self, m, summary):
        import mujoco
        B, G, A = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_ACTUATOR
        self.palm, self.palm_name = _name(m, B, summary.get('detected_palm_body'), 'palm')
        self.obj, self.obj_name = _name(m, B, summary.get('detected_object_body'), 'object_body')
        self.obj_geom, _ = _name(m, G, summary.get('object_geom'), 'object')
        groups = summary.get('finger_geom_groups') or [
            [n for n in ('upper_green_link', 'upper_red_link')], [n for n in ('lower_green_link', 'lower_red_link')]]
        self.link_names, self.link_ids, sides = [], [], []
        for s, grp in enumerate(groups[:2]):
            for g in grp:
                gid, gname = _name(m, G, g)
                self.link_names.append(gname)
                self.link_ids.append(gid)
                sides.append(s)
        self.nl = len(self.link_ids)
        self.side = np.array(sides)
        self.side0 = np.flatnonzero(self.side == 0)
        self.side1 = np.flatnonzero(self.side == 1)
        self.link_idx = -np.ones(m.ngeom, dtype=int)
        for i, gid in enumerate(self.link_ids):
            self.link_idx[gid] = i
        self.is_obj = np.zeros(m.ngeom, dtype=bool)
        self.is_obj[m.geom_bodyid == self.obj] = True
        # palm subtree = 손
        in_hand = np.zeros(m.nbody, dtype=bool)
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == self.palm:
                    in_hand[b] = True
                    break
                p = m.body_parentid[p]
        self.is_hand = in_hand[m.geom_bodyid]
        self.motors = []
        for mn in summary.get('motors') or ['pulley_pull']:
            try:
                self.motors.append(_name(m, A, mn)[0])
            except KeyError:
                pass
        # 손 좌표계 (palm 좌표로 표현): x = jaw, y = 손가락이 뻗는 방향, z = palm z
        a = math.radians(float(summary.get('forward_angle_deg', 180.0)))
        fy = np.array([math.cos(a), math.sin(a), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        fx = np.cross(fy, z)
        self.H = np.column_stack([fx, fy, z])
        self.buf = np.zeros(6)


def measure(m, d, ctx):
    """물체의 palm 기준 위치·자세, 링크별 법선 접촉력, 외부 접촉력."""
    import mujoco
    Rp = d.xmat[ctx.palm].reshape(3, 3)
    pp = d.xpos[ctx.palm]
    p = Rp.T @ (d.xpos[ctx.obj] - pp)
    Rr = Rp.T @ d.xmat[ctx.obj].reshape(3, 3)
    axis = Rp.T @ d.geom_xmat[ctx.obj_geom].reshape(3, 3)[:, 2]
    lf = np.zeros(ctx.nl)
    ext = 0.0
    palmf = 0.0
    n = d.ncon
    if n:
        c = d.contact
        g1, g2 = c.geom1, c.geom2
        io1, io2 = ctx.is_obj[g1], ctx.is_obj[g2]
        idx = np.flatnonzero(io1 ^ io2)
        if idx.size:
            adr = c.efc_address
            for i in idx:
                if adr[i] < 0:
                    continue
                mujoco.mj_contactForce(m, d, int(i), ctx.buf)
                fn = ctx.buf[0]
                other = g2[i] if io1[i] else g1[i]
                li = ctx.link_idx[other]
                if li >= 0:
                    lf[li] += fn
                elif ctx.is_hand[other]:
                    palmf += fn
                else:
                    ext += fn
    return p, Rr, axis, pp.copy(), lf, ext, palmf


def contact_type(lf, ctx, thr):
    on = lf > thr
    s0 = lf[ctx.side0].sum() > thr
    s1 = lf[ctx.side1].sum() > thr
    if on.all():
        return 'env'
    if s0 and s1:
        return 'tip'
    if s0 or s1:
        return 'one_side'
    return 'none'


def _angle_deg(cosv):
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


# ----------------------------------------------------------------------------- simulation
def row_schedule(sim, dt):
    T = sim['pulse_ms'] / 1000.0
    n_total = int(round((T + sim['after_s']) / dt))
    fine_end = min(n_total, int(round((T + sim['settle_s']) / dt)))
    fs = max(1, int(round(sim['sample_ms'] / 1000.0 / dt)))
    cs = max(1, int(round(sim['coarse_s'] / dt)))
    ends = list(range(fs, fine_end + 1, fs))
    if not ends or ends[-1] != fine_end:
        ends.append(fine_end)
    s = fine_end + cs
    while s < n_total:
        ends.append(s)
        s += cs
    if ends[-1] != n_total:
        ends.append(n_total)
    return [e for e in ends if e > 0], n_total


def run_trial(m, d, S0, spec, ctx, ref, fworld, F, sim, ends, n_total):
    """S0에서 시작해 한 시행. 반환: (rows[n_rows, ncols] float32, ended)."""
    import mujoco
    dt = m.opt.timestep
    T = sim['pulse_ms'] / 1000.0
    n_pulse = int(round(T / dt)) if F > 0 else 0
    ce = max(1, int(round(sim['check_ms'] / 1000.0 / dt)))
    lost_n, stop_mm, stop_lost = sim['lost_n'], sim['stop_mm'], sim['stop_lost_s']
    halfsine = sim['shape'] == 'halfsine'
    p0, R0, ax0, palm0 = ref
    H = ctx.H
    mujoco.mj_setState(m, d, S0, spec)
    d.xfrc_applied[:] = 0.0
    d.qfrc_applied[:] = 0.0
    w0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    xf = d.xfrc_applied
    obj = ctx.obj
    end_set = set(ends)
    rows = []
    # row accumulators
    dmax = rmax = tmax = lostmax = extmax = 0.0
    smin = math.inf
    lsum = np.zeros(ctx.nl)
    lcnt = 0
    streak = [0, 0]
    cdt = ce * dt
    step = 0
    ended = 'complete'
    vec = fworld * F
    last = None
    while step < n_total:
        if step < n_pulse:
            amp = math.sin(math.pi * (step + 0.5) * dt / T) if halfsine else 1.0
            xf[obj, :3] = vec * amp
        elif step == n_pulse and F > 0:
            xf[obj, :3] = 0.0
        mujoco.mj_step(m, d)
        step += 1
        at_end = step in end_set
        if step % ce and not at_end:
            continue
        p, Rr, ax, pp, lf, ext, palmf = measure(m, d, ctx)
        dp = p - p0
        disp = 1000.0 * math.sqrt(dp @ dp)
        rot = _angle_deg((np.trace(R0.T @ Rr) - 1.0) / 2.0)
        tilt = _angle_deg(float(ax @ ax0))
        s0 = lf[ctx.side0].sum()
        s1 = lf[ctx.side1].sum()
        streak[0] = streak[0] + 1 if s0 < lost_n else 0
        streak[1] = streak[1] + 1 if s1 < lost_n else 0
        lost = max(streak) * cdt
        dmax = max(dmax, disp)
        rmax = max(rmax, rot)
        tmax = max(tmax, tilt)
        lostmax = max(lostmax, lost)
        extmax = max(extmax, ext)
        smin = min(smin, s0, s1)
        lsum += lf
        lcnt += 1
        stop = disp >= stop_mm or lost >= stop_lost - 1e-12
        if at_end or stop:
            bad = (int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > w0
                   or not np.isfinite(d.qpos).all())
            dh = H.T @ dp * 1000.0
            last = [step * dt, dmax, rmax, tmax, lostmax, extmax, smin,
                    dh[0], dh[1], dh[2], 1000.0 * float(np.linalg.norm(pp - palm0)), palmf, s0, s1]
            rows.append(last + list(lsum / max(lcnt, 1)))
            dmax = rmax = tmax = lostmax = extmax = 0.0
            smin = math.inf
            lsum[:] = 0.0
            lcnt = 0
            if bad:
                ended = 'unstable'
                break
            if stop:
                ended = 'stop_disp' if disp >= stop_mm else 'stop_lost'
                break
    xf[obj, :3] = 0.0
    return np.asarray(rows, dtype=np.float32), ended


def read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def load_trace(path):
    """trace npz -> (meta dict, columns, {key: (rows, ended)})"""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['__meta__']))
        cols = [str(c) for c in z['columns']]
        trials = {}
        for k in z.files:
            if k.endswith('__trace'):
                key = k[:-7]
                trials[key] = (z[k], str(z[key + '__ended']))
    return meta, cols, trials


def save_trace(path, meta, cols, trials):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {'__meta__': np.array(json.dumps(meta)), 'columns': np.array(cols)}
    for key, (rows, ended) in trials.items():
        arrs[key + '__trace'] = rows
        arrs[key + '__ended'] = np.array(ended)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def simulate_cell(job):
    """worker: 한 cell의 필요한 시행을 돌려 trace 파일에 저장(기존 시행에 추가)."""
    t_start = time.time()
    cell_id = job['cell_id']
    try:
        import mujoco
        sim = job['sim']
        m, d, _, summary, cinfo = load_cell(job['cell_dir'], job['mesh_roots'], job['allow_missing_collision'])
        dropped, dropped_col = cinfo['dropped_meshes'], cinfo['dropped_collision_meshes']
        ctx = Ctx(m, summary)
        applied =float(np.abs(d.qfrc_applied).sum() + np.abs(d.xfrc_applied).sum())
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        # 텐던 장력: 기본은 저장 상태의 ctrl 그대로, --tension이면 ctrl = -T/gear
        tension = sim['tension']
        if tension is not None:
            for a in ctx.motors:
                gear = m.actuator_gear[a, 0]
                c = -float(tension) / gear
                lo, hi = m.actuator_ctrlrange[a]
                m.actuator_ctrlrange[a] = [min(lo, c), max(hi, c)]
                d.ctrl[a] = c
        tension_used = [float(-d.ctrl[a] * m.actuator_gear[a, 0]) for a in ctx.motors]
        dt = m.opt.timestep
        g = float(np.linalg.norm(m.opt.gravity))
        mass = float(m.body_subtreemass[ctx.obj])
        mg = mass * g
        # 안정화 (외력 없음). 마지막 0.1 s의 링크 접촉력 평균으로 시작 접촉 형태를 정한다
        n_pre = int(round(sim['pre_s'] / dt))
        n_avg = min(n_pre, int(round(0.1 / dt)))
        lsum = np.zeros(ctx.nl)
        for i in range(n_pre):
            mujoco.mj_step(m, d)
            if i >= n_pre - n_avg:
                lsum += measure(m, d, ctx)[4]
        start_lf = lsum / max(n_avg, 1) if n_avg else measure(m, d, ctx)[4]
        start_type = contact_type(start_lf, ctx, sim['lost_n'])
        full = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        S0 = np.zeros(mujoco.mj_stateSize(m, full))
        mujoco.mj_getState(m, d, S0, full)
        p0, R0, ax0, palm0, _, _, _ = measure(m, d, ctx)
        Rpalm = d.xmat[ctx.palm].reshape(3, 3).copy()
        ref = (p0, R0, ax0, palm0)
        ends, n_total = row_schedule(sim, dt)
        cols = BASE_COLS + [f'f_{n.replace("hand__", "")}' for n in ctx.link_names]
        out = Path(job['trace_path'])
        trials = {}
        if out.is_file() and not job['overwrite']:
            try:
                old_meta, old_cols, trials = load_trace(out)
                if old_meta.get('sim_hash') != job['sim_hash'] or old_cols != cols:
                    trials = {}
            except Exception:
                trials = {}
        todo = [k for k in job['keys'] if k not in trials]
        mono = job.get('monotonic_crit')
        failed_dirs = set()
        T = sim['pulse_ms'] / 1000.0
        ran = 0
        # 방향마다 작은 펄스부터 (--monotonic이면 실패한 방향의 큰 펄스는 생략)
        def _order(key):
            if key == 'z0':
                return ('', 0.0)
            dn, ks = key.rsplit('_', 1)
            return (dn, float(ks[:-2]))
        if mono:
            for key in sorted(trials, key=_order):
                if key != 'z0' and not score_trial(*trials[key], cols, mono, T, sim['settle_s'])['held']:
                    failed_dirs.add(_order(key)[0])
        for key in sorted(todo, key=_order):
            direction, k = _order(key)
            if mono and direction in failed_dirs:
                continue
            fworld = np.zeros(3) if key == 'z0' else Rpalm @ ctx.H @ np.array(ALL_DIRS[direction])
            rows, ended = run_trial(m, d, S0, full, ctx, ref, fworld, k * mg, sim, ends, n_total)
            trials[key] = (rows, ended)
            ran += 1
            if mono and key != 'z0' and not score_trial(rows, ended, cols, mono, T, sim['settle_s'])['held']:
                failed_dirs.add(direction)
        meta = {
            'cell_id': cell_id, 'design': job['design'], 'cell': job['cell'],
            'sim_hash': job['sim_hash'], 'sim': {k: sim[k] for k in SIM_KEYS},
            'r_mm': (summary.get('cell') or {}).get('r_mm'),
            'y_mm': (summary.get('cell') or {}).get('y_mm'),
            'x_mm': (summary.get('cell') or {}).get('x_mm'),
            'workspace_grasp_type': (summary.get('cell') or {}).get('workspace_grasp_type'),
            'rb5_four_link_contact': summary.get('four_link_contact'),
            'mass_kg': mass, 'mg_N': mg, 'tension_N': tension_used, 'timestep': dt,
            'start_type': start_type, 'start_link_forces_N': [float(x) for x in start_lf],
            'link_names': ctx.link_names, 'link_side': [int(s) for s in ctx.side],
            'hand_frame_world': (Rpalm @ ctx.H).tolist(),
            'object_in_palm_mm': (1000.0 * p0).tolist(),
            'applied_force_in_state': applied,
            'dropped_meshes': dropped, 'dropped_collision_meshes': dropped_col,
            'cell_path': cinfo['cell_path'], 'state_source': cinfo['state_source'],
            'summary_source': cinfo['summary_source'],
            'mujoco_version': mujoco.__version__,
            'source_mujoco_version': summary.get('source_mujoco_version'),
        }
        save_trace(out, meta, cols, trials)
        return {'cell_id': cell_id, 'ran': ran, 'sec': time.time() - t_start, 'error': None}
    except Exception:
        return {'cell_id': cell_id, 'ran': 0, 'sec': time.time() - t_start,
                'error': traceback.format_exc()}


# ----------------------------------------------------------------------------- scoring
def score_trial(rows, ended, cols, crit, T, settle):
    c = {n: i for i, n in enumerate(cols)}
    n = len(rows)
    res = {'held': False, 'cause': 'no_data', 'phase': '', 't_fail_s': math.nan, 'ended': ended}
    if n == 0:
        return res
    t = rows[:, c['t']].astype(float)
    tstart = np.concatenate([[0.0], t[:-1]])
    after = tstart >= T + settle - 1e-9
    slip = rows[:, c['disp']] >= crit['slip_mm']
    if crit['slip_mode'] == 'residual':
        slip &= after
    rotv = rows[:, c['tilt' if crit['rot_mode'] == 'tilt' else 'rot']]
    masks = [('slip', slip),
             ('rotation', rotv >= crit['rot_deg']),
             ('lost', rows[:, c['lost']] >= crit['lost_s'] - 1e-9),
             ('collision', rows[:, c['ext']] >= crit['collision_n'])]
    first, cause = n, None
    for name, mk in masks:
        idx = np.flatnonzero(mk)
        if idx.size and idx[0] < first:
            first, cause = int(idx[0]), name
    if cause is None and ended != 'complete':
        cause = {'stop_disp': 'slip', 'stop_lost': 'lost', 'unstable': 'unstable'}.get(ended, ended)
        first = n - 1
    held = cause is None
    res.update(held=held, cause='' if held else cause)
    if not held:
        tf = float(t[first])
        res['t_fail_s'] = tf
        res['phase'] = 'pulse' if tf <= T + 1e-9 else ('transient' if tf <= T + settle + 1e-9 else 'delayed')
    disp = rows[:, c['disp']]
    res['max_disp_mm'] = float(disp.max())
    res['resid_disp_mm'] = float(disp[after].max()) if after.any() else math.nan
    res['max_rot_deg'] = float(rows[:, c['rot']].max())
    res['max_tilt_deg'] = float(rows[:, c['tilt']].max())
    res['max_hand_mm'] = float(rows[:, c['hand']].max())
    res['end_dx_mm'], res['end_dy_mm'], res['end_dz_mm'] = (float(rows[-1, c[k]]) for k in ('dx', 'dy', 'dz'))
    res['min_side_N'] = float(rows[:, c['side_min']].min())
    res['max_ext_N'] = float(rows[:, c['ext']].max())
    lcols = [i for i, nm in enumerate(cols) if nm.startswith('f_')]
    res['_end_lf'] = rows[-1, lcols].astype(float)
    return res


def fmt(v, nd=4):
    if isinstance(v, float):
        if math.isnan(v):
            return ''
        return f'{v:.{nd}g}' if abs(v) < 1e4 else f'{v:.1f}'
    return v


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, '')) for k in fields})
    os.replace(tmp, path)


def read_csv(path):
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


class _LF:
    """contact_type에 넘길 최소 ctx."""

    def __init__(self, side):
        side = np.asarray(side)
        self.side0 = np.flatnonzero(side == 0)
        self.side1 = np.flatnonzero(side == 1)


def find_design_index(cells_dir, explicit):
    if explicit:
        return Path(explicit)
    for p in (HERE / 'designs_index.csv', HERE / 'designs' / 'designs_index.csv',
              cells_dir.parent / 'designs_index.csv', cells_dir.parent / 'designs' / 'designs_index.csv',
              HERE.parent / 'designs_index.csv', HERE.parent / 'designs' / 'designs_index.csv'):
        if p.is_file():
            return p
    return None


def load_design_vars(path, designs):
    rows = read_csv(path) if path else None
    if not rows:
        return {}, []
    keycol = None
    for col in rows[0]:
        vals = {str(r[col]).strip() for r in rows}
        if vals & set(designs):
            keycol = col
            break
    if keycol is None:  # design0001 <-> 1 같은 경우
        for col in rows[0]:
            try:
                vals = {f'design{int(float(r[col])):04d}' for r in rows}
            except (TypeError, ValueError):
                continue
            if vals & set(designs):
                out = {f'design{int(float(r[col])):04d}': {k: v for k, v in r.items() if k != col} for r in rows}
                return out, [k for k in rows[0] if k != col]
        return {}, []
    return ({str(r[keycol]).strip(): {k: v for k, v in r.items() if k != keycol} for r in rows},
            [k for k in rows[0] if k != keycol])


def score_all(out_dir, cells_dir, meta, crit, mags, dirs, designs_filter, w_tip, design_index):
    T = meta['sim']['pulse_ms'] / 1000.0
    settle = meta['sim']['settle_s']
    trace_files = sorted(out_dir.glob('*/traces/*.npz'))
    trace_files = [p for p in trace_files if not p.name.endswith('.tmp.npz')]
    per_design = {}
    stale = 0
    for tp in trace_files:
        design = tp.parent.parent.name
        if designs_filter and design not in designs_filter:
            continue
        try:
            tmeta, cols, trials = load_trace(tp)
        except Exception as e:
            print(f'[WARN] cannot read {tp}: {e}')
            continue
        if tmeta.get('sim_hash') != meta['sim_hash']:
            stale += 1
            continue
        per_design.setdefault(design, []).append((tmeta, cols, trials))

    pick = read_csv(cells_dir / 'rb5_pick_results.csv') or []
    attempted, rb5_pass = {}, {}
    for r in pick:
        dsg = r.get('design_id')
        attempted[dsg] = attempted.get(dsg, 0) + 1
        if str(r.get('success')).strip() in ('1', 'True', 'true'):
            rb5_pass[dsg] = rb5_pass.get(dsg, 0) + 1
    # 아직 안 돌린 설계는 빼고, RB5 pick에서 성공 cell이 0개인 설계는 0으로 넣는다
    nopass = {dsg for dsg in attempted if not rb5_pass.get(dsg)}
    if designs_filter:
        nopass &= set(designs_filter)
    designs = sorted(set(per_design) | nopass)
    dvars, dvar_cols = load_design_vars(design_index, designs)

    all_rows, dir_stats = [], []
    totals = {'cells': 0, 'success': 0, 'zero_fail': 0, 'incomplete': 0, 'trials': 0}
    for design in designs:
        cells = sorted(per_design.get(design, []), key=lambda x: (x[0].get('r_mm') or 0, x[0].get('y_mm') or 0))
        trial_rows, cell_rows = [], []
        for tmeta, cols, trials in cells:
            lfctx = _LF(tmeta['link_side'])
            thr = tmeta['sim']['lost_n']
            base = {'design': design, 'cell': compact_cell_name(tmeta['cell']), 'r_mm': tmeta.get('r_mm'),
                    'y_mm': tmeta.get('y_mm'), 'workspace_grasp_type': tmeta.get('workspace_grasp_type'),
                    'start_type': tmeta['start_type']}
            z = score_trial(*trials['z0'], cols, crit, T, settle) if 'z0' in trials else None
            zero_ok = bool(z and z['held'])
            group = tmeta['start_type'] if tmeta['start_type'] in ('env', 'tip') else 'other'
            if z is None:
                status = 'incomplete'
            elif not zero_ok and crit['zero_fail'] == 'fail':
                status = 'zero_fail'
            else:
                status = 'success'
            levels, complete = {}, z is not None
            for dn in dirs:
                lvl = 0.0
                broke = False
                for k in mags:
                    key = trial_key(dn, k)
                    if key not in trials:
                        if not broke and not crit['monotonic']:
                            complete = False
                        broke = True
                        continue
                    r = score_trial(*trials[key], cols, crit, T, settle)
                    end_type = contact_type(r.pop('_end_lf'), lfctx, thr)
                    trial_rows.append({**base, 'dir': dn, 'k_mg': k, 'impulse_Ns': k * tmeta['mg_N'] * T * (
                        2 / math.pi if tmeta['sim']['shape'] == 'halfsine' else 1.0),
                        'held': int(r['held']), 'cause': r['cause'], 'phase': r['phase'],
                        't_fail_s': r['t_fail_s'], 'ended': r['ended'], 'max_disp_mm': r['max_disp_mm'],
                        'resid_disp_mm': r['resid_disp_mm'], 'max_rot_deg': r['max_rot_deg'],
                        'max_tilt_deg': r['max_tilt_deg'], 'end_dx_mm': r['end_dx_mm'],
                        'end_dy_mm': r['end_dy_mm'], 'end_dz_mm': r['end_dz_mm'],
                        'max_hand_mm': r['max_hand_mm'], 'min_side_N': r['min_side_N'],
                        'max_ext_N': r['max_ext_N'], 'end_type': end_type,
                        'type_changed': int(r['held'] and end_type != tmeta['start_type']),
                        'cell_status': status})
                    if r['held'] and not broke:
                        lvl = k
                    else:
                        broke = True
                levels[dn] = lvl
            if not complete and status == 'success':
                status = 'incomplete'
            lmin = min(levels.values()) if levels else math.nan
            weakest = '|'.join(dn for dn in dirs if levels.get(dn) == lmin)
            crow = {**base, 'group': group, 'status': status, 'zero_held': int(zero_ok) if z else '',
                    'zero_cause': (z['cause'] if z else ''), 'zero_max_disp_mm': z['max_disp_mm'] if z else math.nan,
                    'zero_max_rot_deg': z['max_rot_deg'] if z else math.nan,
                    'level_min': lmin, 'weakest_dir': weakest,
                    'mass_kg': tmeta['mass_kg'], 'tension_N': tmeta['tension_N'][0] if tmeta['tension_N'] else '',
                    'rb5_four_link_contact': tmeta.get('rb5_four_link_contact')}
            for dn in dirs:
                crow[f'level_{dn}'] = levels.get(dn, math.nan)
            for k in mags:
                crow[f'pass@{k:g}'] = int(status == 'success' and lmin >= k - 1e-12)
            cell_rows.append(crow)

        ddir = out_dir / design
        trial_fields = None
        if cells:
            write_csv(ddir / 'impulse_trials.csv', trial_rows, trial_fields)
            write_csv(ddir / 'impulse_cells.csv', cell_rows)
        succ = [c for c in cell_rows if c['status'] == 'success']
        ok_cells = {c['cell'] for c in succ}
        row = {'design': design, 'ws_cells_attempted': attempted.get(design, ''),
               'rb5_pass_cells': rb5_pass.get(design, ''), 'traced_cells': len(cell_rows),
               'incomplete': sum(c['status'] == 'incomplete' for c in cell_rows),
               'zero_fail': sum(c['status'] == 'zero_fail' for c in cell_rows),
               'success': len(succ),
               'success_env': sum(c['group'] == 'env' for c in succ),
               'success_tip': sum(c['group'] == 'tip' for c in succ),
               'success_other': sum(c['group'] == 'other' for c in succ)}
        lm = [c['level_min'] for c in succ]
        row['impact_area'] = float(sum(lm))
        row['level_min_median'] = float(np.median(lm)) if lm else math.nan
        row['level_min_mean'] = float(np.mean(lm)) if lm else math.nan
        for k in mags:
            pe = sum(c['level_min'] >= k - 1e-12 and c['group'] == 'env' for c in succ)
            pt = sum(c['level_min'] >= k - 1e-12 and c['group'] != 'env' for c in succ)
            row[f'pass@{k:g}'] = pe + pt
            row[f'cond@{k:g}'] = (pe + pt) / len(succ) if succ else math.nan
            row[f'env_pass@{k:g}'] = pe
            row[f'tip_pass@{k:g}'] = pt
            row[f'J_impulse@{k:g}'] = pe + w_tip * pt
        tr_ok = [t for t in trial_rows if t['cell'] in ok_cells]
        for ph in ('pulse', 'transient', 'delayed'):
            row[f'fail_{ph}'] = sum(t['phase'] == ph for t in tr_ok)
        for cs in ('slip', 'rotation', 'lost', 'collision', 'unstable'):
            row[f'cause_{cs}'] = sum(t['cause'] == cs for t in tr_ok)
        row['type_changed'] = sum(t['type_changed'] for t in tr_ok)
        for dn in dirs:
            for k in mags:
                sel = [t['held'] for t in tr_ok if t['dir'] == dn and t['k_mg'] == k]
                row[f'held_{dn}@{k:g}'] = float(np.mean(sel)) if sel else math.nan
                for grp in GROUPS:
                    gsel = [t['held'] for t in tr_ok if t['dir'] == dn and t['k_mg'] == k and
                            (t['start_type'] if t['start_type'] in ('env', 'tip') else 'other') == grp]
                    if gsel:
                        dir_stats.append({'design': design, 'group': grp, 'dir': dn, 'k_mg': k,
                                          'n': len(gsel), 'held': int(sum(gsel)),
                                          'held_rate': float(np.mean(gsel))})
        for col in dvar_cols:
            row[col] = (dvars.get(design) or {}).get(col, '')
        all_rows.append(row)
        totals['cells'] += len(cell_rows)
        totals['success'] += len(succ)
        totals['zero_fail'] += row['zero_fail']
        totals['incomplete'] += row['incomplete']
        totals['trials'] += len(trial_rows)

    # 설계 변수 열을 design 바로 뒤로
    if all_rows:
        first = ['design'] + dvar_cols
        fields = first + [k for k in all_rows[0] if k not in first]
        write_csv(out_dir / 'all_designs.csv', all_rows, fields)
    write_csv(out_dir / 'direction_stats.csv', dir_stats,
              ['design', 'group', 'dir', 'k_mg', 'n', 'held', 'held_rate'])
    ranked = sorted(all_rows, key=lambda r: -r['impact_area'])[:10]
    summary = {'criteria': crit, 'mags': mags, 'dirs': dirs, 'w_tip': w_tip, 'totals': totals,
               'stale_traces_skipped': stale, 'design_index': str(design_index) if design_index else None,
               'top10_impact_area': [{'design': r['design'], 'impact_area': r['impact_area'],
                                      'success': r['success'], 'level_min_median': r['level_min_median']}
                                     for r in ranked]}
    with open(out_dir / 'impulse_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return totals, stale, ranked


# ----------------------------------------------------------------------------- viewer
def view_trial(args, sim, cells_dir, mesh_roots):
    import mujoco
    import mujoco.viewer
    cpath = resolve_cell(cells_dir, args.view)
    m, d, _, summary, _ = load_cell(cpath, mesh_roots, args.allow_missing_collision)
    ctx = Ctx(m, summary)
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    if sim['tension'] is not None:
        for a in ctx.motors:
            c = -sim['tension'] / m.actuator_gear[a, 0]
            m.actuator_ctrlrange[a, 1] = max(m.actuator_ctrlrange[a, 1], c)
            d.ctrl[a] = c
    mg = float(m.body_subtreemass[ctx.obj] * np.linalg.norm(m.opt.gravity))
    dt = m.opt.timestep
    T = sim['pulse_ms'] / 1000.0
    n_pre, n_pulse = int(round(sim['pre_s'] / dt)), int(round(T / dt))
    n_total = n_pre + int(round((T + sim['after_s']) / dt))
    F = args.mag * mg
    print(f'[VIEW] {args.view}: settle {sim["pre_s"]} s, then {args.dir} {args.mag:g} mg = {F:.2f} N '
          f'for {sim["pulse_ms"]:g} ms, hold {sim["after_s"]:g} s. Close the window to quit.')
    fworld = None
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = d.xpos[ctx.obj]
        v.cam.distance = 0.6
        step = 0
        wall0 = time.time()
        while v.is_running() and step < n_total:
            if step == n_pre:
                fworld = d.xmat[ctx.palm].reshape(3, 3) @ ctx.H @ np.array(ALL_DIRS[args.dir])
            if fworld is not None and n_pre <= step < n_pre + n_pulse:
                amp = math.sin(math.pi * (step - n_pre + 0.5) * dt / T) if sim['shape'] == 'halfsine' else 1.0
                d.xfrc_applied[ctx.obj, :3] = fworld * F * amp
            else:
                d.xfrc_applied[ctx.obj, :3] = 0
            mujoco.mj_step(m, d)
            step += 1
            if step % 10 == 0:
                v.sync()
                lag = step * dt / args.speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
        while v.is_running():
            v.sync()
            time.sleep(0.05)


# ----------------------------------------------------------------------------- main
def select_cells(cells_dir, designs, only, limit):
    """cell 경로 목록 (폴더 또는 XML 파일). design/cell 이름은 cell_parts()로."""
    out = []
    if not cells_dir.is_dir():
        raise SystemExit(f'cells dir not found: {cells_dir}')
    if only:
        return [resolve_cell(cells_dir, c) for c in only]
    if is_xml_cells_dir(cells_dir):
        per = {}
        for f in sorted(cells_dir.glob('*.xml')):
            per.setdefault(cell_parts(f)[0], []).append(f)
        for design in sorted(per):
            if designs and design not in designs:
                continue
            cells = sorted(per[design], key=lambda f: (cell_ry(cell_parts(f)[1]), cell_parts(f)[1]))
            if limit and len(cells) > limit:
                idx = sorted(set(np.linspace(0, len(cells) - 1, limit).round().astype(int)))
                cells = [cells[i] for i in idx]
            out += cells
        return out
    for ddir in sorted(cells_dir.iterdir()):
        if not ddir.is_dir() or ddir.name.startswith(('_', '.')):
            continue
        if designs and ddir.name not in designs:
            continue
        cells = [c for c in sorted(ddir.iterdir())
                 if c.is_dir() and (c / 'state.npz').is_file() and (c / 'scene.xml').is_file()]
        if limit and len(cells) > limit:
            idx = sorted(set(np.linspace(0, len(cells) - 1, limit).round().astype(int)))
            cells = [cells[i] for i in idx]
        out += cells
    return out


def hms(s):
    s = int(max(0, s))
    return f'{s // 3600}h{s % 3600 // 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s'


def main(args):
    cells_dir = default_cells_dir(args.cells_dir)
    out_root = Path(args.out) if args.out else HERE / 'results_B'
    out_dir = (out_root / (f'impulse_{args.tag}' if args.tag else 'impulse')).resolve()
    if args.clean:
        if args.rescore or args.view or args.dry_run:
            raise SystemExit('--clean은 시뮬레이션을 새로 돌릴 때만 씁니다 (--rescore/--view/--dry-run과 같이 못 씀)')
        clean_results(out_root, cells_dir, 'B')
    mesh_roots = [str(Path(p)) for p in ([args.mesh_dir] if args.mesh_dir else [])] + [
        str(HERE / 'assets' / 'rb5_850e' / 'meshes'), str(cells_dir.parent / 'assets' / 'rb5_850e' / 'meshes')]
    sim = {'pulse_ms': args.pulse_ms, 'shape': args.shape, 'pre_s': args.pre_s, 'after_s': args.after_s,
           'settle_s': args.settle_s, 'sample_ms': args.sample_ms, 'coarse_s': args.coarse_s,
           'check_ms': args.check_ms, 'stop_mm': args.stop_mm, 'stop_lost_s': args.stop_lost_s,
           'lost_n': args.lost_n, 'tension': args.tension}
    if args.view:
        if args.dir not in ALL_DIRS:
            raise SystemExit(f'--dir must be one of {", ".join(ALL_DIRS)}')
        return view_trial(args, sim, cells_dir, mesh_roots)

    meta_path = out_dir / 'impulse_meta.json'
    meta = read_json(meta_path)
    if args.rescore:
        if not meta:
            raise SystemExit(f'no results to rescore in {out_dir}')
    else:
        h = sim_hash(sim)
        if meta and meta['sim_hash'] != h and not args.overwrite:
            diff = {k: (meta['sim'].get(k), sim[k]) for k in SIM_KEYS if meta['sim'].get(k) != sim[k]}
            raise SystemExit(f'{out_dir} holds results with other settings {diff} (old, new).\n'
                             f'Use --tag to keep them apart, or --overwrite to replace them.')
        mags_run = sorted(set(args.mags or DEFAULT_MAGS))
        dirs_run = parse_dirs(args.dirs)
        if meta and meta['sim_hash'] == h:
            mags_all = sorted(set(meta['mags']) | set(mags_run))
            dirs_all = meta['dirs'] + [dn for dn in dirs_run if dn not in meta['dirs']]
        else:
            mags_all, dirs_all = mags_run, dirs_run
        meta = {'sim': sim, 'sim_hash': h, 'mags': mags_all, 'dirs': dirs_all,
                'cells_dir': str(cells_dir), 'created': (meta or {}).get('created') or time.strftime('%Y-%m-%d %H:%M:%S'),
                'updated': time.strftime('%Y-%m-%d %H:%M:%S')}
    # 표는 이 폴더에 지금까지 돌린 크기·방향 전부로 만든다. --rescore에서만 --mags/--dirs로 고를 수 있다
    if args.rescore:
        mags = sorted(set(args.mags)) if args.mags else meta['mags']
        dirs = parse_dirs(args.dirs) if args.dirs else meta['dirs']
    else:
        mags, dirs = meta['mags'], meta['dirs']
    crit = {'slip_mm': args.slip_mm, 'slip_mode': args.slip_mode, 'rot_deg': args.rot_deg,
            'rot_mode': args.rot_mode, 'lost_s': args.lost_s, 'collision_n': args.collision_n,
            'zero_fail': args.zero_fail, 'monotonic': bool(args.monotonic or meta.get('monotonic_used'))}
    if args.monotonic and not args.rescore:
        meta['monotonic_used'] = True
    if crit['slip_mm'] > meta['sim']['stop_mm']:
        print(f'[WARN] --slip-mm {crit["slip_mm"]} > stop {meta["sim"]["stop_mm"]} mm: trials stopped there count as slip.')
    if crit['lost_s'] > meta['sim']['stop_lost_s']:
        print(f'[WARN] --lost-s {crit["lost_s"]} > stop {meta["sim"]["stop_lost_s"]} s: trials stopped there count as lost.')

    if not args.rescore:
        cells = select_cells(cells_dir, args.designs, args.only, args.limit)
        warn_old_xml(cells, 'B')
        keys = ['z0'] + [trial_key(dn, k) for dn in dirs_run for k in mags_run]
        jobs = []
        for cdir in cells:
            design, cell = cell_parts(cdir)
            tp = out_dir / design / 'traces' / f'{cell}.npz'
            need = keys
            if tp.is_file() and not args.overwrite:
                try:
                    tm, _, tr = load_trace(tp)
                    if tm.get('sim_hash') == meta['sim_hash']:
                        need = [k for k in keys if k not in tr]
                except Exception:
                    pass
            if need:
                jobs.append({'cell_id': f'{design}/{cell}', 'design': design,
                             'cell': cell, 'cell_dir': str(cdir), 'trace_path': str(tp),
                             'keys': keys, 'sim': sim, 'sim_hash': meta['sim_hash'],
                             'mesh_roots': mesh_roots, 'overwrite': args.overwrite,
                             'monotonic_crit': crit if crit['monotonic'] else None,
                             'allow_missing_collision': args.allow_missing_collision})
        n_trials = sum(len(keys) for _ in jobs)
        n_designs = len({cell_parts(c)[0] for c in cells})
        print(f'[B] cells: {cells_dir} ({"XML, <key lifted>" if is_xml_cells_dir(cells_dir) else "folders, state.npz"})')
        print(f'[B] {len(cells)} cells in {n_designs} designs; {len(jobs)} to run '
              f'({len(cells) - len(jobs)} already done). Per cell: zero control + {len(dirs_run)} dirs x '
              f'{len(mags_run)} mags = {len(keys)} trials of {sim["pulse_ms"]:g} ms + {sim["after_s"]:g} s.')
        print(f'[B] output: {out_dir}')
        if args.dry_run:
            est = n_trials * 0.45 / max(1, args.workers)
            print(f'[B] dry run. Rough estimate {hms(est)} with {args.workers} workers (about 0.4-0.5 s per trial).')
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        errors = []
        t0 = time.time()
        done = 0

        def report(res):
            nonlocal done
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            if res['error']:
                errors.append(res)
                print(f'[{done}/{len(jobs)}] {res["cell_id"]}  ERROR (see _errors.log)', flush=True)
                with open(out_dir / '_errors.log', 'a', encoding='utf-8') as f:
                    f.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} {res["cell_id"]}\n{res["error"]}\n')
            else:
                print(f'[{done}/{len(jobs)}] {res["cell_id"]}  {res["ran"]} trials  {res["sec"]:.1f} s  '
                      f'elapsed {hms(el)}  ETA {hms(eta)}', flush=True)

        if args.workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                report(simulate_cell(j))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(simulate_cell, j) for j in jobs]
                try:
                    for fu in as_completed(futs):
                        report(fu.result())
                except KeyboardInterrupt:
                    print('[B] interrupted; finished cells are saved. Run the same command to continue.')
                    for fu in futs:
                        fu.cancel()
                    raise
        if errors:
            print(f'[B] {len(errors)} cells failed to run; details in {out_dir / "_errors.log"}')

    design_index = find_design_index(cells_dir, args.designs_index)
    totals, stale, ranked = score_all(out_dir, cells_dir, meta, crit, mags, dirs,
                                      args.designs if args.rescore else None, args.w_tip, design_index)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({**meta, 'last_score': {'criteria': crit, 'mags': mags, 'dirs': dirs,
                                          'time': time.strftime('%Y-%m-%d %H:%M:%S')}}, f, indent=2)
    print(f'[B] scored {totals["cells"]} cells: success {totals["success"]}, zero_fail {totals["zero_fail"]}, '
          f'incomplete {totals["incomplete"]}' + (f'; skipped {stale} traces with other settings' if stale else ''))
    if design_index is None:
        print('[B] designs_index.csv not found; all_designs.csv has no design variables (use --designs-index).')
    if ranked:
        print('[B] top designs by impact_area: ' + ', '.join(
            f'{r["design"]} {r["impact_area"]:g}' for r in ranked[:5]))
    print(f'[B] tables in {out_dir}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(description='실험 B: 짧은 충격(펄스) 후 파지 유지 판정 (grasp_cells 상태에서 시작)')
    g = p.add_argument_group('대상과 출력')
    g.add_argument('--cells-dir', default=None,
                   help='grasp_cells_xml(XML 파일들) 또는 grasp_cells(폴더) (기본: 이 파일 옆 grasp_cells_xml, 없으면 grasp_cells)')
    g.add_argument('--designs', nargs='+', default=None, help='설계 이름 (기본: 전부)')
    g.add_argument('--only', nargs='+', default=None,
                   help='특정 cell만, 예: design0001/r60y15 (design0001_r60y15, 예전 이름 design0001/r060.0_y015.0도 됨)')
    g.add_argument('--limit', type=int, default=0, help='설계마다 (r, y) 순서로 고르게 N개만 (빠른 시험)')
    g.add_argument('--out', default=None, help='결과 상위 폴더 (기본: 이 파일 옆 results_B)')
    g.add_argument('--tag', default=None, help='결과 폴더 impulse_<tag> (설정이 다른 실험을 따로 보관)')
    g.add_argument('--overwrite', action='store_true', help='설정이 다른 기존 결과를 덮어씀')
    g.add_argument('--clean', action='store_true', help='시작 전에 결과 상위 폴더(results_B) 안의 기존 파일을 모두 지움')
    g.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    g.add_argument('--dry-run', action='store_true', help='cell·시행 수만 출력')
    g.add_argument('--mesh-dir', default=None, help='scene.xml의 mesh 절대경로가 없을 때 찾을 …/assets/rb5_850e/meshes')
    g.add_argument('--allow-missing-collision', action='store_true', help='충돌 mesh를 못 찾아도 빼고 진행 (확인용)')
    g.add_argument('--designs-index', default=None, help='설계 변수 표 designs_index.csv (기본: 자동 탐색)')
    s = p.add_argument_group('충격 (바꾸면 다른 실험: --tag 사용)')
    s.add_argument('--pulse-ms', type=float, default=50.0, help='펄스 길이 T [ms]')
    s.add_argument('--shape', choices=('rect', 'halfsine'), default='rect', help='펄스 모양 (halfsine 충격량 = 2/pi 배)')
    s.add_argument('--mags', type=float, nargs='+', default=None, help='펄스 크기 k [mg] (기본 0.5 1 2 3)')
    s.add_argument('--dirs', nargs='+', default=None, help='axes(6) | edges(12) | all(18) | px nx py ny pz nz pxpy …')
    s.add_argument('--tension', type=float, default=None, help='텐던 장력 [N] (기본: 저장 상태 그대로, 10 N)')
    s.add_argument('--pre-s', type=float, default=0.5, help='펄스 전 안정화 시간 [s] (외력 없음)')
    s.add_argument('--after-s', type=float, default=10.0, help='펄스 후 유지 시간 [s]')
    s.add_argument('--settle-s', type=float, default=0.5, help='transient 구간 / residual slip 시작 [s]')
    s.add_argument('--sample-ms', type=float, default=5.0, help='펄스~settle 구간 기록 간격 [ms]')
    s.add_argument('--coarse-s', type=float, default=0.1, help='그 뒤 기록 간격 [s]')
    s.add_argument('--check-ms', type=float, default=1.0, help='판정값 계산 간격 [ms]')
    s.add_argument('--stop-mm', type=float, default=30.0, help='이만큼 미끄러지면 시행 중단 [mm]')
    s.add_argument('--stop-lost-s', type=float, default=0.1, help='접촉을 이만큼 잃으면 시행 중단 [s]')
    s.add_argument('--lost-n', type=float, default=0.1, help='한쪽 손가락 접촉력 합이 이 값 미만이면 접촉 상실 [N]')
    r = p.add_argument_group('판정 (--rescore로 시뮬레이션 없이 변경 가능)')
    r.add_argument('--rescore', action='store_true', help='시뮬레이션 없이 trace로 다시 판정')
    r.add_argument('--slip-mm', type=float, default=5.0, help='손 기준 물체 변위 한계 [mm]')
    r.add_argument('--slip-mode', choices=('total', 'residual'), default='total',
                   help='residual: 펄스 후 settle-s 이후의 변위만 셈')
    r.add_argument('--rot-deg', type=float, default=5.0, help='손 기준 물체 회전 한계 [deg]')
    r.add_argument('--rot-mode', choices=('total', 'tilt'), default='total', help='tilt: 원기둥 축 기울기만')
    r.add_argument('--lost-s', type=float, default=0.02, help='접촉 상실 지속 한계 [s]')
    r.add_argument('--collision-n', type=float, default=0.05, help='물체가 손 밖의 물체(팔·받침대)와 닿는 힘 한계 [N]')
    r.add_argument('--zero-fail', choices=('fail', 'keep'), default='fail', help='외력 0 대조 실패 cell 처리')
    r.add_argument('--monotonic', action='store_true', help='한 방향에서 실패하면 더 큰 펄스를 실패로 봄(시행 생략)')
    r.add_argument('--w-tip', type=float, default=0.2, help='J_impulse의 tip 가중치')
    v = p.add_argument_group('보기')
    v.add_argument('--view', default=None, help='한 cell을 viewer로: design0001/r60y15')
    v.add_argument('--dir', default='py', help='--view 방향')
    v.add_argument('--mag', type=float, default=1.0, help='--view 크기 [mg]')
    v.add_argument('--speed', type=float, default=1.0, help='--view 재생 속도')
    return p


if __name__ == '__main__':
    sys.exit(main(build_parser().parse_args()))
