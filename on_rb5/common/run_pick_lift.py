"""RB5 tendon-driven physical grasp and lift. Run with --headless for validation."""
from pathlib import Path
import argparse
import csv
import json
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
ARM_JOINTS = ('base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3')
FINGERS = ('R_gp', 'R_gd', 'L_gp', 'L_gd')
# hand x = world -y, hand y = world +x, hand z = world +z.
HAND_ROTATION = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10.0 - 15.0*u + 6.0*u*u)


class ArmIK:
    """Damped Jacobian IK on separate scratch data, never on live simulation state."""
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.dof = np.array([model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.site = model.site('grasp').id
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self.ranges = np.array([model.joint(n).range for n in ARM_JOINTS])

    def pose(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos, rot, seed, max_iter=150):
        q = np.asarray(seed, dtype=float).copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, np.asarray(rot).ravel())
        for _ in range(max_iter):
            p, r = self.pose(q)
            cur, inv, delta = np.zeros(4), np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(cur, r.ravel())
            mujoco.mju_negQuat(inv, cur)
            mujoco.mju_mulQuat(delta, target_quat, inv)
            dw = np.zeros(3)
            mujoco.mju_quat2Vel(dw, delta, 1.0)
            error = np.r_[np.asarray(pos)-p, 0.25*dw]
            if np.linalg.norm(error[:3]) < 2e-6 and np.linalg.norm(dw) < 2e-5:
                return q
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            j = np.vstack((self.jp[:, self.dof], 0.25*self.jr[:, self.dof]))
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-7*np.eye(6), error)
            dq *= min(1.0, 0.15/max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q+dq, self.ranges[:, 0]+0.001, self.ranges[:, 1]-0.001)
        raise RuntimeError(f'IK failed at {pos}: position error {np.linalg.norm(error[:3]):.6f} m')


class PickLift:
    def __init__(self, model, data, viewer=None, speed=1.0):
        self.m, self.d, self.viewer = model, data, viewer
        self.speed = speed
        self.ik = ArmIK(model)
        self.armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
        self.pull = model.actuator('pulley_pull').id
        self.fg = [model.geom(n).id for n in FINGERS]
        self.obj = model.geom('obj_g').id
        self.objbody = model.body('obj').id
        self.table = model.geom('bench_top').id
        self.qcmd = data.qpos[self.ik.qadr].copy()
        self.pos, _ = self.ik.pose(self.qcmd)
        self.stage = 'init'
        self.rows = []
        self.frame = 0
        self.wall_start = time.perf_counter()
        self.sim_start = data.time
        self.callbacks = []
        self.record_trace = True
        self.max_unintended_penetration = 0.0
        self.worst_collision = None
        self.max_tracking_error = 0.0
        # Fingers share one plane and may meet tip to tip around small objects.
        # That is allowed; finger-palm, arm and table collisions are still checked.
        self.allow_finger_finger = True

    def forces(self):
        force = np.zeros(4)
        table = 0.0
        f = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            if self.obj not in pair:
                continue
            other = pair[1] if pair[0] == self.obj else pair[0]
            mujoco.mj_contactForce(self.m, self.d, i, f)
            if other in self.fg:
                force[self.fg.index(other)] += max(0.0, f[0])
            if other == self.table:
                table += max(0.0, f[0])
        return force, table

    def step(self):
        mujoco.mj_step(self.m, self.d)
        # Update derived positions and contact forces to the same timestamp.
        mujoco.mj_forward(self.m, self.d)
        if not np.all(np.isfinite(self.d.qpos)) or np.max(np.abs(self.d.qvel)) > 1e4:
            raise RuntimeError('Simulation became unstable')
        self.frame += 1
        if self.record_trace and self.frame % 10 == 0:
            fs, tf = self.forces()
            command_pos, _ = self.ik.pose(self.qcmd)
            self.max_tracking_error = max(self.max_tracking_error, float(
                np.linalg.norm(self.d.site_xpos[self.ik.site]-command_pos)))
            for c in self.d.contact:
                if self.obj in (c.geom1, c.geom2):
                    continue
                if self.allow_finger_finger and c.geom1 in self.fg and c.geom2 in self.fg:
                    continue
                penetration = max(0.0, -float(c.dist))
                if penetration > self.max_unintended_penetration:
                    self.max_unintended_penetration = penetration
                    self.worst_collision = [self.m.geom(g).name or self.m.body(self.m.geom_bodyid[g]).name
                                            for g in (c.geom1, c.geom2)]
                if penetration > 0.001:
                    raise RuntimeError(f'Unintended collision: {self.worst_collision}')
            self.rows.append([self.d.time, self.stage, *self.d.xpos[self.objbody],
                              *self.d.site_xpos[self.ik.site], *fs, tf,
                              -self.m.actuator_gear[self.pull, 0]*self.d.ctrl[self.pull]])
        for cb in self.callbacks:
            cb(self)
        if self.viewer is not None and self.frame % 16 == 0:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
            remaining = (self.d.time-self.sim_start)/self.speed - (time.perf_counter()-self.wall_start)
            if remaining > 0:
                time.sleep(remaining)

    def hold(self, duration, stage=None):
        if stage:
            self.stage = stage
            print(f'[{self.d.time:5.2f}s] {stage}', flush=True)
        for _ in range(round(duration/self.m.opt.timestep)):
            self.step()

    def move(self, target, duration, stage):
        self.stage = stage
        print(f'[{self.d.time:5.2f}s] {stage}', flush=True)
        p0 = self.pos.copy()
        steps = round(duration/self.m.opt.timestep)
        for i in range(steps):
            if i % 10 == 0 or i == steps-1:
                p = p0 + smooth((i+1)/steps)*(np.asarray(target)-p0)
                self.qcmd = self.ik.solve(p, HAND_ROTATION, self.qcmd)
                self.d.ctrl[self.armact] = self.qcmd
            self.step()
        self.pos = np.asarray(target).copy()

    def close(self, tension, duration=1.5):
        self.stage = 'close'
        print(f'[{self.d.time:5.2f}s] close: tendon {tension:.1f} N', flush=True)
        gain = -self.m.actuator_gear[self.pull, 0]
        target = tension/gain
        steps = round(duration/self.m.opt.timestep)
        for i in range(steps):
            self.d.ctrl[self.pull] = target*smooth((i+1)/steps)
            self.step()

    def run(self, tension=20.0, lift=0.15):
        self.wall_start = time.perf_counter()
        self.hold(0.6, 'settle')
        initial_z = float(self.d.xpos[self.objbody, 2])
        xy = self.d.xpos[self.objbody, :2].copy()
        z = float(self.m.numeric('grasp_plane_z').data[0])
        # Lower behind the cylinder first, then enter horizontally with open fingers.
        self.move([xy[0]-0.14, xy[1], z], 3.0, 'lower behind cylinder')
        self.move([xy[0], xy[1], z], 2.5, 'approach cylinder')
        self.hold(0.5, 'settle at grasp')
        self.close(tension)
        self.hold(1.0, 'grasp hold')
        fs, _ = self.forces()
        print('Link normal forces [N]:', np.round(fs, 3), flush=True)
        if fs[:2].sum() < 0.10 or fs[2:].sum() < 0.10:
            raise RuntimeError('No bilateral finger contact; lift cancelled')
        self.move([xy[0], xy[1], z+lift], 3.0, 'lift')
        self.hold(2.0, 'lift hold')
        fs, tf = self.forces()
        dz = float(self.d.xpos[self.objbody, 2])-initial_z
        recent = [r for r in self.rows if r[0] >= self.d.time-1.0]
        min_height = min(r[4]-initial_z for r in recent)
        bilateral = all(sum(r[8:10]) > 0.10 and sum(r[10:12]) > 0.10 for r in recent)
        off_table = all(r[12] < 0.05 for r in recent)
        warning_count = int(np.sum(self.d.warning.number))
        success = bool(min_height > 0.8*lift and bilateral and off_table
                       and self.max_unintended_penetration < 0.0001 and warning_count == 0)
        tilt = np.arccos(np.clip(self.d.xmat[self.objbody].reshape(3, 3)[2, 2], -1, 1))
        return {'success': success, 'mujoco_version': mujoco.__version__,
                'initial_object_center_z_m': initial_z, 'final_object_center_z_m': float(self.d.xpos[self.objbody, 2]),
                'object_lift_m': dz, 'commanded_lift_m': lift,
                'last_second_min_lift_m': min_height, 'last_second_bilateral_contact': bilateral,
                'last_second_off_table': off_table, 'final_link_normal_forces_N': fs.tolist(),
                'final_table_normal_force_N': tf, 'tendon_tension_N': tension,
                'final_cylinder_tilt_deg': float(np.degrees(tilt)),
                'max_unintended_penetration_m': self.max_unintended_penetration,
                'max_grasp_tracking_error_m': self.max_tracking_error,
                'mujoco_warning_count': warning_count,
                'simulation_time_s': float(self.d.time)}


class VideoRecorder:
    """Optional offscreen rendering; no effect on simulated time or physics."""
    def __init__(self, model, path, fps=30):
        import imageio.v2 as imageio
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=720, width=1280)
        self.writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
        self.fps, self.next_time = fps, 0.0
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0.3, 0.0, 0.25]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.45, 135, -25
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0
        self.opt.sitegroup[4] = 0

    def capture(self, sim):
        if sim.d.time+1e-9 < self.next_time:
            return
        self.renderer.update_scene(sim.d, camera=self.cam, scene_option=self.opt)
        self.writer.append_data(self.renderer.render())
        self.next_time += 1.0/self.fps

    def close(self):
        self.writer.close()
        self.renderer.close()


def save_results(output, sim, result):
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (output/'trace.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s','stage','obj_x_m','obj_y_m','obj_z_m',
                         'grasp_x_m','grasp_y_m','grasp_z_m',*['N_'+n for n in FINGERS],
                         'table_N','tension_N'])
        writer.writerows(sim.rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', type=Path, default=ROOT/'rb5_best_design_scene.xml')
    p.add_argument('--headless', action='store_true', help='No viewer; run as fast as possible')
    p.add_argument('--tension', type=float, default=20.0, help='Tendon tension [N], 0..20')
    p.add_argument('--lift', type=float, default=0.15, help='Lift distance [m]')
    p.add_argument('--speed', type=float, default=1.0, help='Viewer playback rate')
    import layout
    p.add_argument('--output', type=Path, default=layout.PICK_LIFT_RESULTS)
    p.add_argument('--video', type=Path, help='Optional MP4 path; install requirements-video.txt first')
    args = p.parse_args()
    if not 0 <= args.tension <= 20 or not 0.03 <= args.lift <= 0.25 or args.speed <= 0:
        p.error('Use tension 0.18 N, lift 0.03..0.25 m, speed > 0')
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    viewer = None
    if not args.headless:
        # Import here to avoid shadowing the top-level mujoco module.
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.30, 0.0, 0.25]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.opt.geomgroup[3] = 0  # Hide arm collision meshes and tendon wrap surfaces.
        viewer.opt.sitegroup[4] = 0
    sim = PickLift(model, data, viewer, args.speed)
    recorder = None
    exit_code = 0
    try:
        if args.video:
            recorder = VideoRecorder(model, args.video)
            sim.callbacks.append(recorder.capture)
        try:
            result = sim.run(args.tension, args.lift)
        except RuntimeError as exc:
            result = {'success': False, 'error': str(exc), 'stage': sim.stage,
                      'simulation_time_s': float(data.time), 'mujoco_version': mujoco.__version__}
        save_results(args.output, sim, result)
        print(json.dumps(result, indent=2))
        print('PASS' if result['success'] else 'FAIL')
        exit_code = 0 if result['success'] else 1
        if recorder:
            sim.callbacks.remove(recorder.capture)
            recorder.close()
            recorder = None
        sim.record_trace = False
        if viewer is not None and result['success']:
            print('Close the viewer window to exit.')
            while viewer.is_running():
                sim.step()
    except KeyboardInterrupt:
        print('Viewer closed.')
    finally:
        if recorder:
            recorder.close()
        if viewer is not None:
            viewer.close()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
