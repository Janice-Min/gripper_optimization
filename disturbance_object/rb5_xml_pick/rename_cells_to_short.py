#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미 저장된 A·B(·C) 결과의 cell 이름을 짧은 형식으로 한 번에 바꾼다 (재시뮬레이션 없음).

  r050.0_y005.0  →  r50y05      (collect_cell_xmls.py의 XML 파일 이름, 실험 C 결과와 같은 형식)

바꾸는 것 (결과 폴더마다, 예: results_A/force):
  <design>/traces/<cell>.npz   파일 이름만 (내용 그대로 → git이 rename으로 인식, 저장소 크기 증가 없음).
                               안의 __meta__에 남은 예전 cell 이름은 실험 코드가 읽을 때 짧은 이름으로 바꿔 쓴다.
                               --update-meta: __meta__(cell, cell_id)도 고쳐 다시 저장 (파일 내용이 바뀜 → git 용량 증가)
  <design>/*.csv 등            'cell' 열이 있는 CSV의 그 열만 (force_cells/dirs, impulse_cells/trials, …)
이미 짧은 이름인 것은 건드리지 않고, 같은 이름의 파일을 덮어쓰지 않으므로 여러 번 실행해도 안전하다.

  python rename_cells_to_short.py                         # 미리보기만 (results_A/*, results_B/*, results_C/* 전부)
  python rename_cells_to_short.py --apply                 # 실제로 바꿈
  python rename_cells_to_short.py results_A/force results_B/impulse --apply

바꾼 뒤에는 순위·그림·비교를 다시 만든다 (CSV는 이미 바뀌어 있어서 --rescore는 필요 없다):
  python rank_force_designs.py ; python rank_impulse_designs.py
  python compare_abc.py results_A/force results_B/impulse results_C/shake
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disturb_impulse_rb5 import compact_cell_name  # noqa: E402  (같은 폴더, 실험 코드와 같은 규칙)


def find_result_dirs(args_dirs):
    if args_dirs:
        return [Path(p).resolve() for p in args_dirs]
    out = []
    for top in ('results_A', 'results_B', 'results_C'):
        root = HERE / top
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and any(d.glob('*/traces/*.npz')):
                out.append(d)
    return out


def load_all(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def same_arrays(a, b):
    """__meta__를 뺀 모든 배열이 같은가 (중간에 끊긴 이전 실행의 흔적인지 확인)."""
    ka, kb = set(a) - {'__meta__'}, set(b) - {'__meta__'}
    return ka == kb and all(np.array_equal(a[k], b[k]) for k in ka)


def rewrite_meta(path):
    """trace 안의 __meta__ cell/cell_id를 짧은 이름으로 (배열은 그대로, 파일은 다시 압축 저장). 바뀌었으면 True."""
    arrs = load_all(path)
    meta = json.loads(str(arrs['__meta__']))
    design = meta.get('design') or path.parent.parent.name
    cell = compact_cell_name(meta.get('cell', path.stem))
    if meta.get('cell') == cell and meta.get('cell_id') == f'{design}/{cell}':
        return False
    meta['cell'], meta['cell_id'] = cell, f'{design}/{cell}'
    arrs['__meta__'] = np.array(json.dumps(meta))
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)
    return True


def migrate_trace(old, apply, update_meta):
    """반환: (상태, 새 경로). 상태: 'renamed' | 'short' | 'cleaned' | 'conflict'

    기본은 파일 이름만 바꾼다(내용 그대로 → git에서 rename으로 잡혀 저장소가 커지지 않음).
    trace 안 __meta__의 예전 cell 이름은 실험 코드가 읽을 때 짧은 이름으로 바꿔 쓴다.
    --update-meta면 __meta__도 고쳐서 다시 저장한다(파일 내용이 바뀜)."""
    new_stem = compact_cell_name(old.stem)
    if new_stem == old.stem:
        if apply and update_meta:
            rewrite_meta(old)
        return 'short', None
    new = old.with_name(new_stem + '.npz')
    if new.exists():
        if same_arrays(load_all(old), load_all(new)):
            if apply:
                old.unlink()           # 같은 결과가 새 이름으로 이미 있음 → 예전 파일만 정리
                if update_meta:
                    rewrite_meta(new)
            return 'cleaned', new
        return 'conflict', new
    if apply:
        os.rename(old, new)            # 대상이 있으면 실패하므로 덮어쓰지 않는다
        if update_meta:
            rewrite_meta(new)
    return 'renamed', new


def migrate_csv(path, apply):
    """'cell' 열만 짧은 이름으로. 나머지 값·열 순서·인코딩은 그대로. 반환: 바뀐 행 수."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows or 'cell' not in rows[0]:
        return 0
    ci = rows[0].index('cell')
    n = 0
    for r in rows[1:]:
        if ci < len(r):
            new = compact_cell_name(r[ci])
            if new != r[ci]:
                r[ci] = new
                n += 1
    if n and apply:
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(rows)
        os.replace(tmp, path)
    return n


def main():
    ap = argparse.ArgumentParser(description='저장된 결과의 cell 이름을 r50y05 형식으로 (재시뮬레이션 없음)')
    ap.add_argument('dirs', nargs='*', help='결과 폴더 (기본: 이 파일 옆 results_A/*, results_B/*, results_C/*)')
    ap.add_argument('--apply', action='store_true', help='실제로 바꿈 (없으면 미리보기만)')
    ap.add_argument('--update-meta', action='store_true',
                    help='trace 안 __meta__의 cell 이름도 고쳐 다시 저장 (기본: 파일 이름만 바꿈)')
    ap.add_argument('--show', type=int, default=3, help='폴더마다 보여 줄 예시 수')
    a = ap.parse_args()

    dirs = find_result_dirs(a.dirs)
    if not dirs:
        sys.exit('결과 폴더를 못 찾음 (traces/*.npz가 있는 results_X/<name>)')
    mode = '적용' if a.apply else '미리보기 (--apply로 실제 적용)'
    print(f'[rename] {mode}')
    total = dict(renamed=0, short=0, cleaned=0, conflict=0, csv_files=0, csv_rows=0)
    conflicts = []
    for rd in dirs:
        cnt = dict(renamed=0, short=0, cleaned=0, conflict=0)
        shown = 0
        traces = sorted(p for p in rd.glob('*/traces/*.npz') if not p.name.endswith('.tmp.npz'))
        for tp in traces:
            st, new = migrate_trace(tp, a.apply, a.update_meta)
            cnt[st] += 1
            if st == 'conflict':
                conflicts.append((tp, new))
            elif st == 'renamed' and shown < a.show:
                print(f'    {tp.parent.parent.name}/traces/{tp.name}  →  {new.name}')
                shown += 1
        csv_files = csv_rows = 0
        for cp in sorted(rd.rglob('*.csv')):
            if 'traces' in cp.parts:
                continue
            n = migrate_csv(cp, a.apply)
            if n:
                csv_files += 1
                csv_rows += n
        print(f'  {rd.parent.name}/{rd.name}: trace {len(traces)}개 중 이름 변경 {cnt["renamed"]}, '
              f'이미 짧은 이름 {cnt["short"]}, 중복 정리 {cnt["cleaned"]}, 충돌 {cnt["conflict"]} | '
              f'CSV {csv_files}개 파일 {csv_rows}행')
        for k in cnt:
            total[k] += cnt[k]
        total['csv_files'] += csv_files
        total['csv_rows'] += csv_rows
    if conflicts:
        print('\n[rename] 충돌: 같은 cell의 두 이름 파일이 모두 있고 내용이 다름 → 둘 다 그대로 둠. 어느 쪽을 쓸지 확인 필요:')
        for o, n in conflicts[:20]:
            print(f'    {o}  /  {n.name}')
        if len(conflicts) > 20:
            print(f'    … 외 {len(conflicts) - 20}개')
    print(f'\n[rename] 합계: trace 이름 변경 {total["renamed"]}, 중복 정리 {total["cleaned"]}, 충돌 {total["conflict"]}, '
          f'CSV {total["csv_files"]}개 파일 {total["csv_rows"]}행' + ('' if a.apply else '  (미리보기, 아무것도 안 바뀜)'))
    return 1 if conflicts else 0


if __name__ == '__main__':
    sys.exit(main())
