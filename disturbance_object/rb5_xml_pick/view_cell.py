#!/usr/bin/env python3
"""cell XML을 파지 상태(lifted key)로 바로 열기.

  python view_cell.py grasp_cells_xml/design0001_r50y05.xml
  python view_cell.py grasp_cells_xml/design0001_r50y05.xml task_home   # 다른 key

Reload 버튼을 누르면 다시 파지 상태로 돌아옴 (Reset은 key 없는 기본 자세).
"""
import sys

import mujoco
import mujoco.viewer


def load(path, key):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    mujoco.mj_forward(m, d)
    return m, d


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용: python view_cell.py <cell.xml> [key=lifted]")
    path = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "lifted"
    mujoco.viewer.launch(loader=lambda: load(path, key))
