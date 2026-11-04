"""Rebuild the standalone RB5 + original tendon hand + cylinder scene.

Inputs in sources/ are retained verbatim. No network or Class_IR install needed.
Run: python build_scene.py

compose() and add_home_keyframe() are also used by rb5_eval.py, so the optimizer
mounts every candidate hand exactly the way this script mounts best_design_fine.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
MESHDIR = 'assets/rb5_850e/meshes'
HOME_POS = [0.46, 0.0, 0.28]
HOME_SEED = [0.5457, -0.492, 2.2764, 1.3572, -1.0251, 0.0]


def load_sources(root=ROOT):
    """Return (robot, hand, scene) roots from sources/."""
    return tuple(ET.parse(Path(root) / 'sources' / name).getroot()
                 for name in ('rb5_850e.xml', 'best_design_fine.xml', 'task_scene.xml'))


def compose(robot, hand, scene, meshdir=MESHDIR):
    """Mount the hand on the RB5 inside the task scene.

    `scene` is modified in place and returned; `robot` and `hand` are only copied from.
    Pass a fresh parse or a deepcopy of task_scene.xml for every call.
    """
    scene.set('model', 'RB5 with best_design_fine tendon hand')
    scene.find('compiler').set('meshdir', str(meshdir))
    opt = scene.find('option')
    opt.set('timestep', '0.001')
    opt.set('gravity', '0 0 -9.81')
    opt.set('iterations', '100')
    opt.set('tolerance', '1e-10')
    scene.find('visual/global').set('azimuth', '135')
    scene.find('visual/global').set('elevation', '-25')
    scene.find('statistic').set('center', '0.3 0 0.25')

    # Keep the original laboratory furniture, room and decorative shelf objects.
    # Remove every tabletop task object and its obsolete sensors.
    wb = scene.find('worldbody')
    for body in list(wb.findall('body')):
        wb.remove(body)
    scene.remove(scene.find('sensor'))
    defaults = scene.find('default')
    for cls in list(defaults):
        if cls.get('class') in ('glass', 'bulb_wall'):
            defaults.remove(cls)
    defaults.extend(deepcopy(list(robot.find('default'))))
    # Scope the hand's defaults so they cannot change RB5 joints or scene geoms.
    hd = ET.SubElement(defaults, 'default', {'class': 'hand'})
    hd.extend(deepcopy(list(hand.find('default'))))
    scene.find('asset').extend(deepcopy(list(robot.find('asset'))))
    rb = deepcopy(robot.find('worldbody/body'))
    wb.append(rb)
    flange = rb.find('.//body[@name="link6"]')
    tcp = flange.find('site[@name="tcp"]')
    mount = ET.SubElement(flange, 'body', {
        'name': 'tool_mount', 'pos': tcp.get('pos'), 'quat': tcp.get('quat')})
    ET.SubElement(mount, 'geom', {
        'name': 'mount_flange', 'type': 'cylinder', 'size': '0.035 0.006',
        'pos': '0 0 0.006', 'mass': '0.04', 'rgba': '0.25 0.28 0.32 1'})
    ET.SubElement(mount, 'geom', {
        'name': 'mount_adapter', 'type': 'box', 'size': '0.018 0.025 0.0415',
        'pos': '0 0 0.0535', 'mass': '0.06', 'rgba': '0.36 0.40 0.45 1'})
    # Original hand coordinates: x = jaw width, +y = approach, z = cylinder axis.
    # Rotate +90 deg about flange x: hand +y becomes flange +z.
    hb = ET.SubElement(mount, 'body', {
        'name': 'hand', 'pos': '0 0 0.10', 'quat': '0.707106781187 0.707106781187 0 0',
        'childclass': 'hand'})
    for element in hand.find('worldbody'):
        # 'table' is the mocap support plane of the rb5 cell protocol. The RB5 scene has
        # its own bench, and a mocap body cannot be nested inside the hand anyway.
        if element.tag in ('light', 'camera') or element.get('name') in ('obj', 'table'):
            continue
        hb.append(deepcopy(element))
    # The arm IK drives the 'grasp' site to the cylinder center, so put the site where
    # the hand XML places its object: the map cell (x, y+r) chosen by the optimizer.
    src_obj = hand.find('worldbody/body[@name="obj"]')
    ox, oy = (float(v) for v in src_obj.get('pos').split()[:2])
    ET.SubElement(hb, 'site', {'name': 'grasp', 'pos': f'{ox:.6g} {oy:.6g} 0',
                            'size': '0.002', 'rgba': '0.2 0.8 1 0', 'group': '4'})
    obj = deepcopy(src_obj)
    halfheight = float(obj.find('geom').get('size').split()[1])
    obj.set('pos', f'0.60 0 {halfheight+0.0005:.6g}')
    obj.find('geom').set('class', 'hand')
    obj.find('geom').set('rgba', '0.22 0.65 0.36 1')
    wb.append(obj)

    contact = ET.SubElement(scene, 'contact')
    contact.extend(deepcopy(list(robot.find('contact'))))
    for exc in hand.find('contact'):
        new = deepcopy(exc)
        # "world" was the fixed palm in the source. Never exclude fingers/table.
        for attr in ('body1', 'body2'):
            if new.get(attr) == 'world':
                new.set(attr, 'hand')
        contact.append(new)
    scene.append(deepcopy(hand.find('tendon')))
    actuator = ET.SubElement(scene, 'actuator')
    actuator.extend(deepcopy(list(robot.find('actuator'))))
    actuator.extend(deepcopy(list(hand.find('actuator'))))
    # Physical tension = -gear * ctrl = 0.5 * ctrl; max 20 N.
    custom = ET.SubElement(scene, 'custom')
    ET.SubElement(custom, 'numeric', {'name': 'grasp_plane_z', 'data': '0.060'})
    ET.SubElement(custom, 'numeric', {'name': 'default_tension_N', 'data': '20'})
    return scene


def home_arm_q(model):
    """Arm joint angles of the 'home' pose for a compiled scene."""
    from run_pick_lift import ArmIK, HAND_ROTATION
    return ArmIK(model).solve(HOME_POS, HAND_ROTATION, HOME_SEED)


def add_home_keyframe(scene, model):
    """Append the 'home' keyframe (all 4 fingers open, free cylinder at rest)."""
    from run_pick_lift import ArmIK
    q = home_arm_q(model)
    qpos = model.qpos0.copy()
    qpos[ArmIK(model).qadr] = q
    ctrl = [*q, 0.0]
    keys = ET.SubElement(scene, 'keyframe')
    ET.SubElement(keys, 'key', {'name': 'home',
        'qpos': ' '.join(f'{v:.12g}' for v in qpos),
        'ctrl': ' '.join(f'{v:.12g}' for v in ctrl)})
    return q


def build(output=None):
    robot, hand, scene = load_sources()
    scene = compose(robot, hand, scene)
    output = Path(output) if output else ROOT / 'rb5_best_design_scene.xml'
    ET.indent(scene, space='  ')
    ET.ElementTree(scene).write(output, encoding='utf-8', xml_declaration=True)

    # A convenient whole-scene keyframe, including all 4 fingers and free cylinder.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(output))
    add_home_keyframe(scene, model)
    ET.indent(scene, space='  ')
    ET.ElementTree(scene).write(output, encoding='utf-8', xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(output))
    print(f'Built and compiled: {output.name}')
    return output


if __name__ == '__main__':
    build()
