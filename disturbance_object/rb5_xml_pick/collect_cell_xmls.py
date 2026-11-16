#!/usr/bin/env python3
"""grasp_cells/<design>/<cell>/scene.xml 을 한 폴더로 모아 design0001_r50y05.xml 형식으로 저장.

각 scene.xml에는 파지 성공 상태가 <key name="lifted">로 이미 들어 있어서 state.npz 없이 XML만으로 충분.
summary.json(손·물체 이름, 손 좌표계, r·y 등)은 실험 A/B/C 코드가 쓰므로 XML 끝에 <custom>으로 넣는다:
  <custom>
    <text name="cell_id" data="design0001/r050.0_y005.0"/>    ← 결과 폴더의 cell 이름 (기존과 같음)
    <text name="cell_summary" data="{...summary.json...}"/>
  </custom>
(<custom>은 물리에 영향 없음. 나머지 내용은 원본 그대로.)
rb5_pick_results.csv(실험 표의 ws_cells_attempted / rb5_pass_cells), rb5_pick_config.json(view_pick_disturb.py)이
있으면 같이 복사한다.

사용:
  python collect_cell_xmls.py grasp_cells                 # → grasp_cells_xml/ (기존 xml은 지우고 새로 만듦)
  python collect_cell_xmls.py grasp_cells --out D:/cells_xml --check   # mujoco로 열어서 lifted key 확인

열기:
  python view_cell.py grasp_cells_xml/design0001_r50y05.xml
  python -m mujoco.viewer --mjcf grasp_cells_xml/design0001_r50y05.xml  → Key 1 (lifted) → Load key
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

CELL_RE = re.compile(r"^r(-?[\d.]+)_y(-?[\d.]+)$")


def num(v):
    """50.0 → '50', 5.0 → '05', 52.5 → '52p5'"""
    if float(v).is_integer():
        return f"{int(v):02d}"
    return f"{v:g}".replace(".", "p").replace("-", "m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="grasp_cells 폴더 (또는 design 폴더 하나)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: <root>_xml)")
    ap.add_argument("--check", action="store_true", help="만든 XML을 mujoco로 열어 lifted key 확인")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else root.with_name(root.name + "_xml")
    out.mkdir(parents=True, exist_ok=True)
    old = list(out.glob("*.xml"))
    for f in old:  # 이전에 만든 것과 섞이지 않게
        f.unlink()

    scenes = sorted(p for p in root.rglob("scene.xml")
                    if not any(part.startswith("_") for part in p.relative_to(root).parts))
    rows, seen, skipped = [], {}, []
    for scene in scenes:
        cell, design = scene.parent, scene.parent.parent.name
        mt = CELL_RE.match(cell.name)
        if not mt:
            skipped.append(f"{cell} (폴더 이름이 r..._y... 형식이 아님)")
            continue
        r, y = float(mt.group(1)), float(mt.group(2))
        text = scene.read_text(encoding="utf-8")
        if 'name="lifted"' not in text:
            skipped.append(f"{cell} (lifted key 없음)")
            continue
        i = text.rfind("</mujoco>")
        if i < 0:
            skipped.append(f"{cell} (</mujoco> 없음)")
            continue
        name = f"{design}_r{num(r)}y{num(y)}.xml"
        if name in seen:
            sys.exit(f"이름 충돌: {name} ← {seen[name]} / {cell}")
        seen[name] = cell

        s = {}
        if (cell / "summary.json").exists():
            s = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
        block = ("  <custom>\n"
                 f"    <text name=\"cell_id\" data={quoteattr(f'{design}/{cell.name}')}/>\n"
                 f"    <text name=\"cell_summary\" data={quoteattr(json.dumps(s, ensure_ascii=False, separators=(',', ':')))}/>\n"
                 "  </custom>\n")
        (out / name).write_text(text[:i] + block + text[i:], encoding="utf-8")

        c = s.get("cell", {})
        rows.append({
            "file": name, "cell_id": f"{design}/{cell.name}", "design": design, "r_mm": r, "y_mm": y,
            "workspace_grasp_type": c.get("workspace_grasp_type", ""),
            "four_link_contact": s.get("four_link_contact", ""),
            "success": s.get("success", ""),
        })

    if rows:
        with open(out / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    copied = []
    for extra in ("rb5_pick_results.csv", "rb5_pick_config.json"):  # 실험 표 / view_pick_disturb.py 집기 설정
        if (root / extra).is_file():
            shutil.copy2(root / extra, out / extra)
            copied.append(extra)

    for msg in skipped:
        print("[SKIP]", msg)
    print(f"{len(rows)}개 XML → {out}  (" + ", ".join(["index.csv"] + copied) + " 포함"
          + (f", 이전 xml {len(old)}개 지움" if old else "") + ")")

    if a.check and rows:
        import mujoco
        bad = 0
        for row in rows:
            try:
                m = mujoco.MjModel.from_xml_path(str(out / row["file"]))
                d = mujoco.MjData(m)
                mujoco.mj_resetDataKeyframe(m, d, m.key("lifted").id)
            except Exception as e:
                bad += 1
                print("[FAIL]", row["file"], str(e).splitlines()[0])
        print(f"check: {len(rows) - bad}/{len(rows)} 정상")


if __name__ == "__main__":
    main()
