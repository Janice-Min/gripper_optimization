"""Run the RB5 pick-and-lift (run_pick_lift.py) for one workspace-map cell.

The cell (r, x, y) uses the map convention of gripper_core: object radius r, object
center at hand (x, y+r). The scene file is edited in memory only: the cylinder radius
and the 'grasp' site, which the arm IK drives to the object center.

  python grasp/rb5_cell.py 35 0 5                (from on_rb5/; scene: python common/build_scene.py)
  python grasp/rb5_cell.py 40 20 25 --scene common/rb5_best_design_scene.xml
"""
import argparse
import contextlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from run_pick_lift import PickLift


def scene_for_cell(scene, r_mm, x_mm, y_mm):
    tree = ET.parse(scene)
    root = tree.getroot()
    obj = root.find(".//geom[@name='obj_g']")
    obj.set('size', f"{r_mm/1000:.6g} {obj.get('size').split()[1]}")
    root.find(".//site[@name='grasp']").set('pos', f'{x_mm/1000:.6g} {(y_mm+r_mm)/1000:.6g} 0')
    return root


def run_cell(r_mm, x_mm, y_mm, scene=str(layout.COMMON/'rb5_best_design_scene.xml'), tension=20.0, lift=0.15):
    root = scene_for_cell(scene, r_mm, x_mm, y_mm)
    # Keep relative mesh paths valid when the XML is compiled from a string.
    meshdir = root.find('compiler').get('meshdir')
    if meshdir:
        root.find('compiler').set('meshdir', str((Path(scene).resolve().parent/meshdir)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    sim = PickLift(model, data)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = sim.run(tension, lift)
    except RuntimeError as exc:
        result = {'success': False, 'error': str(exc), 'stage': sim.stage}
    rot = data.xmat[model.body('obj').id].reshape(3, 3)
    result['cell'] = [float(r_mm), float(x_mm), float(y_mm)]
    result['end_tilt_deg'] = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('r_mm', type=float)
    p.add_argument('x_mm', type=float)
    p.add_argument('y_mm', type=float)
    p.add_argument('--scene', default=str(layout.COMMON/'rb5_best_design_scene.xml'))
    args = p.parse_args()
    result = run_cell(args.r_mm, args.x_mm, args.y_mm, args.scene)
    print(json.dumps(result, indent=2, default=float))
    print('PASS' if result['success'] else 'FAIL')


if __name__ == '__main__':
    main()
