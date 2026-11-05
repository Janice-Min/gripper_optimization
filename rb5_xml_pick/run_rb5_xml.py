"""Attach a design001-style tendon hand to an RB5 and physically pick/lift its object.
All user-editable defaults / argparse options are at the very bottom of this file.
SI units: metre, kilogram, second, radian, newton. No object weld or upward force.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import time
import threading
import xml.etree.ElementTree as ET

try:
    import mujoco
    import numpy as np
except ImportError as exc:
    raise SystemExit('Install dependencies: python -m pip install -r requirements.txt') from exc


def values(x):
    return ' '.join(f'{float(v):.15g}' for v in np.asarray(x).ravel())


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t**3 * (10 - 15*t + 6*t*t)


def quaternion(rotation):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation, dtype=float).ravel())
    return q


def descendant(model, body, ancestor):
    while body:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def prefix_xml(root, prefix):
    # Names and references must change together, including tendon wrap sites/classes.
    references = {'class', 'childclass', 'body', 'body1', 'body2', 'joint', 'joint1',
                  'joint2', 'geom', 'geom1', 'geom2', 'site', 'site1', 'site2', 'sidesite',
                  'tendon', 'tendon1', 'tendon2', 'material', 'mesh', 'texture', 'hfield',
                  'target', 'actuator', 'objname', 'refname', 'cranksite', 'slidersite'}
    for e in root.iter():
        for key in list(e.attrib):
            if key == 'name' or key in references:
                if e.get(key) != 'world':
                    e.set(key, prefix + e.get(key))


def source_model(args):
    path = args.xml.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'XML not found: {path}')
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.open_key)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key_id)
    else:
        print(f'[INFO] No keyframe {args.open_key!r}; using finger spring reference angles.')
    mujoco.mj_forward(m, d)
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.palm_body)
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.object_body)
    if palm < 1 or obj < 1:
        raise ValueError(f'Require body {args.palm_body!r} and body {args.object_body!r}; '
                         'override --palm-body / --object-body for renamed XMLs.')
    hand_joints = [i for i in range(m.njnt) if descendant(m, int(m.jnt_bodyid[i]), palm)]
    if not hand_joints or any(m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE for i in hand_joints):
        raise ValueError('The palm must be fixed and its finger joints must be named hinge joints.')
    if key_id < 0:
        for j in hand_joints:
            qa = int(m.jnt_qposadr[j])
            d.qpos[qa] = m.qpos_spring[qa]
            if m.jnt_limited[j]:
                d.qpos[qa] = np.clip(d.qpos[qa], *m.jnt_range[j])
        mujoco.mj_forward(m, d)
    if m.body_jntnum[palm]:
        raise ValueError('The palm itself must be fixed, as in design001.xml.')
    object_joints = [i for i in range(m.njnt) if descendant(m, int(m.jnt_bodyid[i]), obj)]
    if len(object_joints) != 1 or m.jnt_type[object_joints[0]] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('The object must have exactly one freejoint.')
    object_geoms = [i for i in range(m.ngeom) if descendant(m, int(m.geom_bodyid[i]), obj)]
    if len(object_geoms) != 1 or m.geom_type[object_geoms[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError('design001 format requires one cylinder geom on the object body.')
    og = object_geoms[0]
    if m.geom_contype[og] == 0 and m.geom_conaffinity[og] == 0:
        raise ValueError('Object collision must be enabled.')
    if m.neq:
        raise ValueError('Equality constraints are outside the design001 format; remove them first.')
    for i in hand_joints:
        if not m.joint(i).name:
            raise ValueError('Every finger joint needs a name.')
    # Direct palm children identify the two fingers; all collidable descendant geoms count.
    branches = [b for b in range(1, m.nbody) if m.body_parentid[b] == palm
                and any(descendant(m, int(m.jnt_bodyid[j]), b) for j in hand_joints)]
    if len(branches) != 2:
        raise ValueError('Expected two articulated finger branches below palm.')
    groups = [[i for i in range(m.ngeom) if descendant(m, int(m.geom_bodyid[i]), b)
               and (m.geom_contype[i] or m.geom_conaffinity[i])] for b in branches]
    if any(not g for g in groups) or any(not m.geom(i).name for g in groups for i in g):
        raise ValueError('Both fingers need named collidable geoms.')
    if m.nu == 0 or any(m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON
                        or m.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE
                        or m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
                        or m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE
                        or m.actuator_gainprm[i, 0] != 1 for i in range(m.nu)):
        raise ValueError('Use stateless tendon motor actuator(s), as in design001.xml.')
    if any(not m.actuator(i).name or abs(m.actuator_gear[i, 0]) < 1e-12 for i in range(m.nu)):
        raise ValueError('Tendon motors need a name and a nonzero gear.')
    rp = d.xmat[palm].reshape(3, 3)
    rel = rp.T @ (d.geom_xpos[og] - d.xpos[palm])
    object_rotation = rp.T @ d.geom_xmat[og].reshape(3, 3)
    if abs(object_rotation[2, 2]) < .999:
        raise ValueError('Cylinder axis must be parallel to palm z, as in design001.xml.')
    if args.grasp_xy is not None:
        rel[:2] = args.grasp_xy
    source_rel = rel.copy()
    # The hand's -x points toward the object in design001. An explicit option
    # allows the same hierarchy with a different finger-forward direction.
    a = np.radians(args.forward_angle)
    forward = np.array([np.cos(a), np.sin(a), 0.])
    hand_rotation = np.array([[np.cos(a), np.sin(a), 0.],
                              [-np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    adjustment = 0.
    if args.auto_grasp and args.grasp_xy is None:
        check = mujoco.MjData(m)
        check.qpos[:] = d.qpos
        oq = int(m.jnt_qposadr[object_joints[0]])
        original_q = check.qpos[oq:oq+3].copy()
        original_size = m.geom_size[og].copy()
        if args.radius is not None:
            m.geom_size[og, 0] = args.radius
        if args.object_height is not None:
            m.geom_size[og, 1] = args.object_height/2
        hand_geoms = [i for i in range(m.ngeom) if descendant(m, int(m.geom_bodyid[i]), palm)
                      and (m.geom_contype[i] or m.geom_conaffinity[i])]
        found = False
        try:
            for shift in np.arange(0., args.auto_grasp_max+args.auto_grasp_step/2, args.auto_grasp_step):
                good = True
                for off in np.linspace(0., args.approach_distance, args.preflight_samples):
                    check.qpos[oq:oq+3] = original_q + rp @ (forward*(shift+off))
                    mujoco.mj_forward(m, check)
                    if any(mujoco.mj_geomDistance(m, check, og, g, 1., None) < args.approach_clearance
                           for g in hand_geoms):
                        good = False
                        break
                if good:
                    adjustment = float(shift)
                    rel += forward*shift
                    found = True
                    break
        finally:
            m.geom_size[og] = original_size
        if not found:
            raise ValueError('No collision-free open-hand approach in the automatic search range. '
                             'Adjust --grasp-xy / --radius / --auto-grasp-max.')
        if adjustment > 0:
            print(f'[AUTO] Grasp center shifted {adjustment*1000:.1f} mm along finger-forward '
                  f'to avoid initial overlap: {np.round(rel[:2], 5).tolist()} m.', flush=True)
    # Canonicalize includes, degrees and implicit orientations through MuJoCo.
    # For ordinary radian design001 inputs retain original numeric precision.
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    if root.find('.//include') is not None or compiler is None or compiler.get('angle', 'degree') != 'radian':
        canonical = args.output / 'source_expanded.xml'
        mujoco.mj_saveLastXML(str(canonical), m)
        root = ET.parse(canonical).getroot()
    compiler = root.find('compiler')
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            directory = (compiler.get('meshdir', '') if asset.tag == 'mesh'
                         else compiler.get('texturedir', ''))
            asset.set('file', str((path.parent / directory / asset.get('file')).resolve()))
    meta = dict(source_xml=str(path), source_mujoco_version=mujoco.__version__,
                radius_m=float(m.geom_size[og, 0]), height_m=float(2*m.geom_size[og, 1]),
                mass_kg=float(m.body_mass[obj]), source_grasp_local_m=rel.tolist(),
                original_grasp_local_m=source_rel.tolist(), auto_grasp_shift_m=adjustment,
                open_key_used=args.open_key if key_id >= 0 else 'springref',
                hand_joints={m.joint(j).name: float(d.qpos[m.jnt_qposadr[j]]) for j in hand_joints},
                finger_geom_groups=[[m.geom(i).name for i in g] for g in groups],
                object_geom=m.geom(og).name,
                motors=[m.actuator(i).name for i in range(m.nu)])
    return root, m, d, meta, hand_rotation, forward


class ArmIK:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        names = ('base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3')
        self.qa = np.array([model.joint(n).qposadr[0] for n in names])
        self.dofs = np.array([model.joint(n).dofadr[0] for n in names])
        self.act = np.array([model.actuator(n).id for n in names])
        self.ranges = np.array([model.joint(n).range for n in names])
        self.site = model.site('task_grasp').id
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def pose(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, p, rotation, seed, max_iter=200):
        q = np.asarray(seed, dtype=float).copy()
        goal = quaternion(rotation)
        for _ in range(max_iter):
            xyz, rot = self.pose(q)
            inv, delta, angle = np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_negQuat(inv, quaternion(rot))
            mujoco.mju_mulQuat(delta, goal, inv)
            mujoco.mju_quat2Vel(angle, delta, 1.)
            err = np.r_[np.asarray(p) - xyz, .25 * angle]
            if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(angle) < 1e-4:
                return q
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            jac = np.vstack([self.jp[:, self.dofs], .25*self.jr[:, self.dofs]])
            deltaq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6*np.eye(6), err)
            deltaq *= min(1., .12 / max(np.max(np.abs(deltaq)), 1e-12))
            q = np.clip(q + deltaq, self.ranges[:, 0]+.001, self.ranges[:, 1]-.001)
        raise ValueError(f'IK target unreachable: {np.round(p, 4).tolist()}, '
                         f'position error={np.linalg.norm(err[:3])*1000:.2f} mm; '
                         'adjust --object-xy, --lift, or --approach-distance.')


def build_scene(args):
    rootdir = Path(__file__).resolve().parent
    hand, sm, sd, meta, rotation, forward = source_model(args)
    robot = ET.parse(rootdir/'sources/rb5_850e.xml').getroot()
    scene = ET.parse(rootdir/'sources/task_scene.xml').getroot()
    scene.set('model', 'RB5 XML hand pick and lift')
    scene.find('compiler').set('meshdir', str(rootdir/'assets/rb5_850e/meshes'))
    for e in robot.findall('asset/mesh'):
        e.set('file', str(rootdir/'assets/rb5_850e/meshes'/e.get('file')))
    opt = scene.find('option')
    opt.set('timestep', str(args.timestep))
    opt.set('gravity', '0 0 -9.81')
    opt.set('iterations', '100')
    opt.set('tolerance', '1e-10')
    world = scene.find('worldbody')
    for b in list(world.findall('body')):
        world.remove(b)
    scene.remove(scene.find('sensor'))
    defs = scene.find('default')
    defs.extend(deepcopy(list(robot.find('default'))))
    prefix_xml(hand, 'hand__')
    hd = ET.SubElement(defs, 'default', {'class': 'hand__root'})
    if hand.find('default') is not None:
        hd.extend(deepcopy(list(hand.find('default'))))
    asset = scene.find('asset')
    asset.extend(deepcopy(list(robot.find('asset'))))
    if hand.find('asset') is not None:
        asset.extend(deepcopy(list(hand.find('asset'))))
    rb = deepcopy(robot.find('worldbody/body'))
    world.append(rb)
    flange = rb.find('.//body[@name="link6"]')
    tcp = flange.find('site[@name="tcp"]')
    mount = ET.SubElement(flange, 'body', {'name': 'task_mount', 'pos': tcp.get('pos'), 'quat': tcp.get('quat')})
    ET.SubElement(mount, 'geom', name='task_flange', type='cylinder', size='.035 .006',
                  pos='0 0 .006', mass='.04', rgba='.25 .28 .32 1')
    ET.SubElement(mount, 'geom', name='task_adapter', type='box',
                  size=values([.018, .025, (args.mount_length-.012)/2]),
                  pos=values([0, 0, (args.mount_length+.012)/2]), mass='.06', rgba='.36 .40 .45 1')
    palm_source = hand.find(f'.//body[@name="hand__{args.palm_body}"]')
    if palm_source is None:
        raise ValueError('Palm body must be present in XML.')
    palm = deepcopy(palm_source)
    # Construct mount axes: forward -> flange +z, hand z -> flange -y.
    lateral = np.cross(forward, [0., 0., 1.])
    local_to_mount = np.column_stack([lateral, [0., 0., 1.], forward]).T
    local_to_mount[1] *= -1  # rows: lateral, -hand z, forward; right-handed
    for attr in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
        palm.attrib.pop(attr, None)
    palm.set('pos', values([0, 0, args.mount_length]))
    palm.set('quat', values(quaternion(local_to_mount)))
    palm.set('childclass', palm.get('childclass', 'hand__root'))
    mount.append(palm)
    rel = np.array(meta['source_grasp_local_m'])
    ET.SubElement(palm, 'site', name='task_grasp', pos=values(rel), size='.002', group='4', rgba='0 1 1 0')
    obj = deepcopy(hand.find(f'.//body[@name="hand__{args.object_body}"]'))
    obj.set('childclass', obj.get('childclass', 'hand__root'))
    geom = obj.find('geom')
    # Cylinder/body offsets are accepted only when directly on the free body.
    og = sm.geom(meta['object_geom']).id
    if sm.geom_bodyid[og] != sm.body(args.object_body).id:
        raise ValueError('Put the cylinder geom directly on object_body.')
    radius = args.radius if args.radius is not None else meta['radius_m']
    height = args.object_height if args.object_height is not None else meta['height_m']
    mass = args.mass if args.mass is not None else meta['mass_kg']
    for attr in ('fromto', 'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
        geom.attrib.pop(attr, None)
    geom.set('pos', '0 0 0')
    geom.set('size', values([radius, height/2]))
    geom.set('mass', str(mass))
    # Explicit inertial, if supplied, must follow mass/size overrides too.
    old_inertia = obj.find('inertial')
    if old_inertia is not None:
        obj.remove(old_inertia)
    for attr in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
        obj.attrib.pop(attr, None)
    xyz = np.r_[args.object_xy, args.support_height + height/2 + args.spawn_clearance]
    obj.set('pos', values(xyz))
    world.append(obj)
    if args.support_height > 0:
        ET.SubElement(world, 'geom', name='task_support', type='cylinder',
                      pos=values([*args.object_xy, args.support_height/2]),
                      size=values([radius*.85, args.support_height/2]),
                      rgba='.55 .59 .64 1', friction='1 .005 .0001')
    contact = ET.SubElement(scene, 'contact')
    contact.extend(deepcopy(list(robot.find('contact'))))
    if hand.find('contact') is not None:
        for e in hand.find('contact'):
            if 'world' in e.attrib.values():
                raise ValueError('World contact references are outside design001 format.')
            contact.append(deepcopy(e))
    if hand.find('tendon') is None:
        raise ValueError('Missing tendon section.')
    scene.append(deepcopy(hand.find('tendon')))
    actuator = ET.SubElement(scene, 'actuator')
    actuator.extend(deepcopy(list(robot.find('actuator'))))
    for original in hand.find('actuator'):
        motor = deepcopy(original)
        motor.set('class', motor.get('class', 'hand__root'))
        i = sm.actuator(motor.get('name').removeprefix('hand__')).id
        control = -args.tension / sm.actuator_gear[i, 0]
        lo, hi = sm.actuator_ctrlrange[i] if sm.actuator_ctrllimited[i] else [0., 0.]
        motor.set('ctrlrange', values([min(lo, 0., control), max(hi, 0., control, 1e-9)]))
        motor.set('ctrllimited', 'true')
        if sm.actuator_forcelimited[i]:
            flo, fhi = sm.actuator_forcerange[i]
            motor.set('forcerange', values([min(flo, control), max(fhi, control)]))
        actuator.append(motor)
    scene_path = args.output/'rb5_attached_scene.xml'
    ET.indent(scene, space='  ')
    ET.ElementTree(scene).write(scene_path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for name, q in meta['hand_joints'].items():
        data.qpos[model.joint('hand__'+name).qposadr[0]] = q
    ik = ArmIK(model)
    grasp = np.r_[args.object_xy, args.support_height+height/2]
    home = grasp + np.array([-args.approach_distance, 0., args.home_height])
    seed = [.5457, -.492, 2.2764, 1.3572, -1.0251, 0.]
    q = ik.solve(home, rotation, seed)
    data.qpos[ik.qa] = q
    data.ctrl[ik.act] = q
    mujoco.mj_forward(model, data)
    key = ET.SubElement(ET.SubElement(scene, 'keyframe'), 'key', name='task_home',
                        qpos=values(data.qpos), ctrl=values(data.ctrl))
    ET.indent(scene, space='  ')
    ET.ElementTree(scene).write(scene_path, encoding='utf-8', xml_declaration=True)
    meta.update(radius_m=radius, height_m=height, mass_kg=mass,
                tension_per_motor_N=args.tension, commanded_lift_m=args.lift,
                support_height_m=args.support_height, scene_xml=str(scene_path),
                opening_assist_enabled=args.open_kp > 0 or args.open_kd > 0,
                opening_assist_release_s=args.open_release_time,
                friction_unchanged=True, object_welded=False,
                tendon_ctrl=(-args.tension/model.actuator_gear[[model.actuator('hand__'+n).id for n in meta['motors']], 0]).tolist(),
                hand_rotation_world=rotation.tolist(), initial_hand_qpos=meta['hand_joints'])
    return model, data, ik, meta, rotation, grasp, home


class Task:
    def __init__(self, args, model, data, ik, meta, rotation, grasp, home):
        self.args, self.m, self.d, self.ik, self.meta = args, model, data, ik, meta
        self.rotation, self.grasp, self.pos = rotation, grasp, home.copy()
        self.qcmd = data.qpos[ik.qa].copy()
        self.motors = np.array([model.actuator('hand__'+n).id for n in meta['motors']])
        self.obj = model.geom('hand__'+meta['object_geom']).id
        self.objbody = model.body('hand__'+args.object_body).id
        self.groups = [[model.geom('hand__'+n).id for n in g] for g in meta['finger_geom_groups']]
        self.fingers = self.groups[0]+self.groups[1]
        self.palm = model.body('hand__'+args.palm_body).id
        self.hand_geoms = [i for i in range(model.ngeom) if descendant(model, int(model.geom_bodyid[i]), self.palm)
                           and (model.geom_contype[i] or model.geom_conaffinity[i])]
        self.supports = {model.geom('bench_top').id}
        if args.support_height > 0:
            self.supports.add(model.geom('task_support').id)
        self.rows, self.history = [], deque()
        self.viewer, self.recorder = None, None
        self.viewer_threads = []
        self.stage, self.frames = 'initialize', 0
        self.max_penetration, self.max_tracking = 0., 0.
        self.wall0, self.time0 = time.perf_counter(), data.time
        self.normal = np.zeros(6)
        self.check_collisions = True
        self.open_fraction = 1.
        self.hand_qadr = np.array([model.joint('hand__'+n).qposadr[0] for n in meta['hand_joints']])
        self.hand_dof = np.array([model.joint('hand__'+n).dofadr[0] for n in meta['hand_joints']])
        self.open_q = np.array(list(meta['hand_joints'].values()))

    def contacts(self):
        forces = {g: 0. for g in self.fingers}
        support = 0.
        for i, c in enumerate(self.d.contact):
            if self.obj not in (c.geom1, c.geom2):
                continue
            other = int(c.geom2 if c.geom1 == self.obj else c.geom1)
            mujoco.mj_contactForce(self.m, self.d, i, self.normal)
            n = max(0., float(self.normal[0]))
            if other in forces:
                forces[other] += n
            if other in self.supports:
                support += n
        side = [sum(forces[g] for g in group) for group in self.groups]
        return forces, side, support

    def step(self):
        self.d.qfrc_applied[self.hand_dof] = self.open_fraction*np.clip(
            self.args.open_kp*(self.open_q-self.d.qpos[self.hand_qadr])
            - self.args.open_kd*self.d.qvel[self.hand_dof],
            -self.args.open_torque_limit, self.args.open_torque_limit)
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.frames += 1
        if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all() or any(w.number for w in self.d.warning):
            raise RuntimeError('MuJoCo numerical instability/warning; decrease --timestep.')
        if self.check_collisions:
            for c in self.d.contact:
                if self.obj in (c.geom1, c.geom2):
                    continue
                # Finger-to-finger contact is permitted; arm/table/palm collisions are not.
                if c.geom1 in self.fingers and c.geom2 in self.fingers:
                    continue
                depth = max(0., -float(c.dist))
                self.max_penetration = max(self.max_penetration, depth)
                if depth > self.args.collision_tolerance:
                    names = [self.m.geom(int(g)).name or self.m.body(self.m.geom_bodyid[g]).name
                             for g in (c.geom1, c.geom2)]
                    raise RuntimeError(f'Unintended collision {names}: {depth*1000:.2f} mm.')
        if self.frames % max(1, round(self.args.sample_interval/self.m.opt.timestep)) == 0:
            fs, side, support = self.contacts()
            p = self.d.xpos[self.objbody].copy()
            self.max_tracking = max(self.max_tracking, float(np.linalg.norm(self.d.site_xpos[self.ik.site]-self.pos)))
            row = [float(self.d.time), self.stage, *p, *self.d.site_xpos[self.ik.site],
                   *[fs[g] for g in self.fingers], *side, support]
            self.rows.append(row)
            self.history.append((float(self.d.time), float(p[2]), side, support))
            while self.history and self.history[0][0] < self.d.time-self.args.verdict_window:
                self.history.popleft()
        if self.recorder:
            self.recorder.capture(self)
        if self.viewer is not None and self.frames % max(1, round(1/(self.args.viewer_fps*self.m.opt.timestep))) == 0:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
            delay = (self.d.time-self.time0)/self.args.speed-(time.perf_counter()-self.wall0)
            if delay > 0:
                time.sleep(delay)

    def hold(self, seconds, stage):
        self.stage = stage
        print(f'[{self.d.time:6.2f}s] {stage}', flush=True)
        for _ in range(max(1, round(seconds/self.m.opt.timestep))):
            self.step()

    def move(self, target, seconds, stage):
        self.stage = stage
        print(f'[{self.d.time:6.2f}s] {stage}', flush=True)
        begin = self.pos.copy()
        count = max(1, round(seconds/self.m.opt.timestep))
        interval = max(1, round(self.args.ik_interval/self.m.opt.timestep))
        for i in range(count):
            if i % interval == 0 or i == count-1:
                self.pos = begin+smooth((i+1)/count)*(target-begin)
                self.qcmd = self.ik.solve(self.pos, self.rotation, self.qcmd)
                self.d.ctrl[self.ik.act] = self.qcmd
            self.step()
        self.pos = target.copy()

    def preflight(self):
        # Check IK, initial overlap and the entire OPEN-hand approach on scratch data.
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = self.d.qpos
        start = self.pos.copy()
        q = self.qcmd.copy()
        behind = self.grasp + np.array([-self.args.approach_distance, 0., 0.])
        for segment, target in enumerate((behind, self.grasp, self.grasp + [0., 0., self.args.lift])):
            for t in np.linspace(0, 1, self.args.preflight_samples):
                p = start + t*(target-start)
                q = self.ik.solve(p, self.rotation, q)
                scratch.qpos[self.ik.qa] = q
                mujoco.mj_forward(self.m, scratch)
                # Object/finger overlap checked during approach only; lift is closed-hand.
                for c in scratch.contact:
                    if self.obj in (c.geom1, c.geom2):
                        continue
                    if c.geom1 in self.fingers and c.geom2 in self.fingers:
                        continue
                    if c.dist < -self.args.collision_tolerance:
                        names = [self.m.geom(int(g)).name or self.m.body(self.m.geom_bodyid[g]).name
                                 for g in (c.geom1, c.geom2)]
                        raise ValueError(f'Planned arm path collides: {names}. Adjust --approach-distance, '
                                         '--object-xy, --home-height, or --support-height.')
                if segment == 2:
                    continue
                for g in self.hand_geoms:
                    distance = mujoco.mj_geomDistance(self.m, scratch, self.obj, g, 1., None)
                    if distance < -self.args.initial_overlap_tolerance:
                        raise ValueError(f'Open-hand approach overlaps the cylinder by {-distance*1000:.2f} mm '
                                         f'at {self.m.geom(g).name}. Adjust --grasp-xy or --radius; '
                                         'the XML geometry is not automatically changed.')
            start = np.asarray(target).copy()
        print('[OK] XML attached; open-hand approach and IK checked.', flush=True)

    def run(self):
        self.wall0, self.time0 = time.perf_counter(), self.d.time
        self.hold(self.args.settle_time, 'settle')
        initial_z = float(self.d.xpos[self.objbody, 2])
        self.move(self.grasp+[-self.args.approach_distance, 0., 0.], self.args.move_time, 'lower')
        self.move(self.grasp, self.args.approach_time, 'approach')
        self.hold(self.args.preclose_time, 'settle before closing')
        self.stage = 'close'
        print(f'[{self.d.time:6.2f}s] close: {self.args.tension:g} N per tendon', flush=True)
        targets = -self.args.tension/self.m.actuator_gear[self.motors, 0]
        count = max(1, round(self.args.ramp_time/self.m.opt.timestep))
        for i in range(count):
            self.open_fraction = 1.-smooth(min(1., (i+1)/count*self.args.ramp_time/self.args.open_release_time))
            self.d.ctrl[self.motors] = targets*smooth((i+1)/count)
            self.step()
        self.open_fraction = 0.
        self.hold(self.args.grasp_hold, 'grasp hold')
        forces, sides, _ = self.contacts()
        print('Finger geom normal forces [N]:', {self.m.geom(g).name: round(v, 3) for g, v in forces.items()}, flush=True)
        if min(sides) < self.args.contact_threshold:
            raise RuntimeError('No bilateral finger contact. Lift cancelled; adjust grasp position, size or tension.')
        self.move(self.grasp+[0., 0., self.args.lift], self.args.lift_time, 'lift')
        self.hold(self.args.hold_time, 'lift hold')
        samples = list(self.history)
        min_lift = min(r[1]-initial_z for r in samples)
        bilateral = all(min(r[2]) >= self.args.contact_threshold for r in samples)
        off_table = all(r[3] < self.args.support_force_threshold for r in samples)
        fs, sides, support = self.contacts()
        success = bool(min_lift >= self.args.success_ratio*self.args.lift and bilateral and off_table)
        return dict(success=success, initial_object_z_m=initial_z,
                    object_lift_m=float(self.d.xpos[self.objbody, 2]-initial_z),
                    minimum_lift_in_window_m=min_lift, bilateral_contact_in_window=bilateral,
                    off_support_in_window=off_table, support_force_N=support,
                    final_link_forces_N={self.m.geom(g).name: f for g, f in fs.items()},
                    four_link_contact=all(f >= self.args.contact_threshold for f in fs.values()),
                    reason='success' if success else 'insufficient lift / lost contact / support contact')


class Recorder:
    def __init__(self, model, args):
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        self.r = mujoco.Renderer(model, height=720, width=1280)
        self.writer = imageio.get_writer(str(args.video), fps=args.video_fps, codec='libx264', quality=7)
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        set_camera(self.camera, self.options, args)
        self.next = 0.
        self.fps = args.video_fps

    def capture(self, task):
        if task.d.time+1e-9 < self.next:
            return
        self.r.update_scene(task.d, camera=self.camera, scene_option=self.options)
        self.writer.append_data(self.r.render())
        self.next += 1/self.fps

    def close(self):
        self.writer.close()
        self.r.close()


def set_camera(camera, options, args):
    camera.lookat[:] = args.camera_lookat
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    options.geomgroup[3] = 0
    options.sitegroup[4] = 0


def save(task, result):
    result.update(task.meta, stage=task.stage, simulation_time_s=float(task.d.time),
                  max_unintended_penetration_m=task.max_penetration,
                  max_tracking_error_m=task.max_tracking,
                  mujoco_warning_count=int(sum(w.number for w in task.d.warning)))
    (task.args.output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (task.args.output/'trace.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'stage', 'obj_x', 'obj_y', 'obj_z', 'grasp_x', 'grasp_y', 'grasp_z',
                    *[task.m.geom(g).name+'_N' for g in task.fingers], 'side1_N', 'side2_N', 'support_N'])
        w.writerows(task.rows)


def main(args):
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    task = None
    result = {'success': False}
    try:
        model, data, ik, meta, rotation, grasp, home = build_scene(args)
        task = Task(args, model, data, ik, meta, rotation, grasp, home)
        if not args.headless:
            from mujoco import viewer as mjviewer
            existing_threads = set(threading.enumerate())
            task.viewer = mjviewer.launch_passive(model, data)
            task.viewer_threads = [t for t in threading.enumerate() if t not in existing_threads]
            set_camera(task.viewer.cam, task.viewer.opt, args)
            task.viewer.sync()
        if args.video:
            task.recorder = Recorder(model, args)
        print(f'Object: r={meta["radius_m"]*1000:g} mm, height={meta["height_m"]*1000:g} mm, mass={meta["mass_kg"]:g} kg')
        print(f'Gear-compensated motor ctrl: {-args.tension/model.actuator_gear[task.motors, 0]}')
        task.preflight()
        result = task.run()
    except KeyboardInterrupt:
        result = dict(success=False, reason='viewer closed or interrupted')
    except (ValueError, RuntimeError, OSError) as exc:
        result = dict(success=False, reason=str(exc))
    finally:
        if task:
            save(task, result)
            if task.recorder:
                task.recorder.close()
                task.recorder = None
        else:
            (args.output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print('PASS' if result['success'] else 'FAIL', flush=True)
    # Keep the finished pose visible. Pause physics so it does not drop after the verdict.
    if task and task.viewer:
        try:
            if not args.exit_after_task:
                print('Task finished. Close the MuJoCo viewer to exit (final state is paused).', flush=True)
                while task.viewer.is_running():
                    task.viewer.sync()
                    time.sleep(1/args.viewer_fps)
        except KeyboardInterrupt:
            pass
        finally:
            task.viewer.close()
            # launch_passive renders on a daemon thread. Let it release GLFW/OpenGL
            # before Python exits, especially with --exit-after-task.
            for thread in task.viewer_threads:
                thread.join(timeout=args.viewer_shutdown_timeout)
    return 0 if result['success'] else 1


# ============================================================================
# USER SETTINGS: ALL argparse options and editable defaults are together BELOW.
# Change default=... here, or override any option on the command line.
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--xml', type=Path, default=Path(__file__).resolve().parent/'design001.xml', help='Input gripper XML')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results', help='Generated scene, trace and summary folder')
    parser.add_argument('--palm-body', default='palm', help='Palm body name in input XML')
    parser.add_argument('--object-body', default='object_body', help='Object body name in input XML')
    parser.add_argument('--open-key', default='open', help='Initial open-hand keyframe')
    parser.add_argument('--forward-angle', type=float, default=180., help='Finger-forward angle in palm xy [deg]; design001 = -x = 180')
    parser.add_argument('--auto-grasp', action=argparse.BooleanOptionalAction, default=True, help='Automatically shift XML grasp position forward to avoid overlap; explicit --grasp-xy disables this')
    parser.add_argument('--auto-grasp-max', type=float, default=.08, help='Maximum automatic forward shift [m]')
    parser.add_argument('--auto-grasp-step', type=float, default=.0025, help='Automatic grasp search step [m]')
    parser.add_argument('--approach-clearance', type=float, default=.001, help='Automatic open-hand geometric clearance [m]')
    parser.add_argument('--grasp-xy', type=float, nargs=2, default=None, metavar=('X', 'Y'), help='Object center in palm coordinates [m]; None reads XML')
    parser.add_argument('--radius', type=float, default=None, help='Cylinder radius [m]; None reads XML')
    parser.add_argument('--object-height', type=float, default=None, help='Cylinder full height [m]; None reads XML')
    parser.add_argument('--mass', type=float, default=None, help='Object mass [kg]; None reads XML')
    parser.add_argument('--object-xy', type=float, nargs=2, default=[.60, 0.], help='Object world xy [m]')
    parser.add_argument('--support-height', type=float, default=.08, help='Cylinder stand height above bench [m]; avoids wrist/bench collision for short cylinders')
    parser.add_argument('--spawn-clearance', type=float, default=.0005, help='Initial object/support clearance [m]')
    parser.add_argument('--mount-length', type=float, default=.10, help='Flange-to-palm adapter length [m]')
    parser.add_argument('--open-kp', type=float, default=1., help='Temporary open-hand spring gain [Nm/rad]; disabled during grasp/lift')
    parser.add_argument('--open-kd', type=float, default=.03, help='Temporary open-hand damping [Nms/rad]')
    parser.add_argument('--open-torque-limit', type=float, default=.3, help='Temporary opening torque cap [Nm]')
    parser.add_argument('--open-release-time', type=float, default=.3, help='Open controller release duration at closing [s]')
    parser.add_argument('--tension', type=float, default=20., help='Actual contraction tension per tendon [N]; ctrl=-T/gear; limits expanded to allow this value')
    parser.add_argument('--lift', type=float, default=.15, help='Vertical lift distance [m]')
    parser.add_argument('--approach-distance', type=float, default=.12, help='Horizontal pregrasp distance [m]')
    parser.add_argument('--home-height', type=float, default=.22, help='Home height above grasp target [m]')
    parser.add_argument('--settle-time', type=float, default=.6, help='Initial settling time [s]')
    parser.add_argument('--move-time', type=float, default=3., help='Time to lower [s]')
    parser.add_argument('--approach-time', type=float, default=2.5, help='Horizontal approach time [s]')
    parser.add_argument('--preclose-time', type=float, default=.5, help='Settling before closing [s]')
    parser.add_argument('--ramp-time', type=float, default=1.5, help='Tension ramp time [s]')
    parser.add_argument('--grasp-hold', type=float, default=1., help='Hold before lift [s]')
    parser.add_argument('--lift-time', type=float, default=3., help='Lift duration [s]')
    parser.add_argument('--hold-time', type=float, default=2., help='Final hold duration [s]')
    parser.add_argument('--timestep', type=float, default=.001, help='Physics timestep [s]')
    parser.add_argument('--ik-interval', type=float, default=.01, help='IK update period [s]')
    parser.add_argument('--sample-interval', type=float, default=.01, help='Log and verdict sample interval [s]')
    parser.add_argument('--preflight-samples', type=int, default=40, help='IK/approach checks per segment')
    parser.add_argument('--initial-overlap-tolerance', type=float, default=.0001, help='Open-hand overlap tolerance [m]')
    parser.add_argument('--collision-tolerance', type=float, default=.001, help='Unintended penetration failure threshold [m]')
    parser.add_argument('--contact-threshold', type=float, default=.1, help='Bilateral contact force threshold [N]')
    parser.add_argument('--support-force-threshold', type=float, default=.05, help='Off-support contact force threshold [N]')
    parser.add_argument('--success-ratio', type=float, default=.8, help='Minimum achieved / requested lift')
    parser.add_argument('--verdict-window', type=float, default=1., help='Final success evaluation window [s]')
    parser.add_argument('--headless', action='store_true', help='Disable viewer for automated physics validation')
    parser.add_argument('--exit-after-task', action='store_true', help='Close viewer automatically after completion')
    parser.add_argument('--speed', type=float, default=1., help='Viewer playback speed')
    parser.add_argument('--viewer-shutdown-timeout', type=float, default=3., help='Wait for viewer render-thread cleanup [s]')
    parser.add_argument('--viewer-fps', type=int, default=60, help='Viewer update frequency')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=[.30, 0., .28], help='Viewer target [m]')
    parser.add_argument('--camera-distance', type=float, default=1.45, help='Viewer distance [m]')
    parser.add_argument('--camera-azimuth', type=float, default=135., help='Viewer azimuth [deg]')
    parser.add_argument('--camera-elevation', type=float, default=-25., help='Viewer elevation [deg]')
    parser.add_argument('--video', type=Path, default=None, help='Optional MP4 path; requires imageio and imageio-ffmpeg')
    parser.add_argument('--video-fps', type=int, default=30, help='Recording frame rate')
    args = parser.parse_args()
    for name in ('lift', 'approach_distance', 'home_height', 'settle_time', 'move_time', 'approach_time',
                 'preclose_time', 'ramp_time', 'grasp_hold', 'lift_time', 'hold_time', 'timestep', 'ik_interval',
                 'sample_interval', 'contact_threshold', 'support_force_threshold', 'verdict_window',
                 'speed', 'viewer_fps', 'video_fps', 'viewer_shutdown_timeout', 'camera_distance', 'open_release_time', 'open_torque_limit', 'auto_grasp_max', 'auto_grasp_step'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and > 0')
    for name in ('radius', 'object_height', 'mass'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and > 0')
    for name in ('tension', 'support_height', 'spawn_clearance', 'initial_overlap_tolerance', 'collision_tolerance', 'open_kp', 'open_kd', 'approach_clearance'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and >= 0')
    for name in ('object_xy', 'camera_lookat', 'grasp_xy', 'forward_angle', 'camera_azimuth', 'camera_elevation'):
        if getattr(args, name) is not None and not np.isfinite(getattr(args, name)).all():
            parser.error(f'--{name.replace("_", "-")} must contain finite values')
    if not np.isfinite(args.mount_length) or args.mount_length <= .012:
        parser.error('--mount-length must be > .012 m')
    if args.open_release_time > args.ramp_time:
        parser.error('--open-release-time must be <= --ramp-time')
    if not 0 < args.success_ratio <= 1 or args.verdict_window > args.hold_time:
        parser.error('Require 0 < success-ratio <= 1 and verdict-window <= hold-time')
    if args.sample_interval > args.verdict_window or args.timestep > args.sample_interval or args.preflight_samples < 2:
        parser.error('Require timestep <= sample-interval <= verdict-window and preflight-samples >= 2')
    sys.exit(main(args))
