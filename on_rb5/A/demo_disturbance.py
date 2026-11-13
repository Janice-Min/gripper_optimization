"""MuJoCo demo of disturbance experiment A: grasp, lift, then push until the object drops.

Give a design XML (stage0 format like ../xml/design075.xml, or the RB5 hand format) or a
label of the run (candidate_02, ...) and a direction. The script mounts the hand on the RB5
exactly like the experiment (config_used.json of the run: tension 20 N, object, table),
runs the pick-and-lift, then ramps a force on the cylinder's center of mass in that
direction (hand frame, same as the experiment) until the object drops, and shows it in the
MuJoCo viewer with an arrow for the force.

  python A/demo_disturbance.py --xml ..\\xml\\design075.xml --dir +x                  (from on_rb5/)
  python A/demo_disturbance.py --xml ..\\xml\\design075.xml --dir px --cell 60 0 35
  python A/demo_disturbance.py --label candidate_02 --dir -y --skip-lift
  python A/demo_disturbance.py --xml ..\\xml\\design075.xml --dir +x --headless --video demo_075_px.mp4
(+x/-x: jaw width, toward the R / L finger; +y: out of the hand (pull-out), -y: into the
palm; +z/-z: up / down along the cylinder axis. Write -x as nx or --dir=-x: argparse reads a
bare -x as an option.)

Which cell (object radius r, position x, y in the hand)?
  --cell R X Y    explicit [mm]
  otherwise, if disturb_force.py has results for this design: the cell that held the most
  in this direction (the cell behind the peak graphs); otherwise the best cell of the
  design's workspace map (references/<label>_workspace.csv); otherwise --cell is required.
--skip-lift starts from the saved post-lift state of disturb_collect.py (no pick-and-lift).

The ramp is the experiment's (default 1 N/s; F is on top of gravity) but runs on to
--cap-mg (default 30 m g) so the object really drops. The console prints
  F_max   the force at the experiment's failure criterion (slip 5 mm, rotation 5 deg or
          lost contact 20 ms relative to the hand, from the run's config), and
  F_drop  the force when the object is clearly gone (moved --drop-mm or no contact 0.1 s),
and, if known, the experiment's F_max of the same cell and direction for comparison.

Video (needs pip install imageio imageio-ffmpeg; on a computer without a display add
--headless): --video FILE.mp4, --record all|disturb, --camera close|wide.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from gripper_core import JOINTS, LINKS, build_xml, read_json
from rb5_eval import best_cell, hand_config, validate_rb5_config
from eval_design_rb5 import apply_patch, check_conversion, load_design, make_arm, read_rows, stage0_conversion
from disturb_collect import cell_name, label_dir, load_label
import disturb_force as DF
from run_pick_lift import ARM_JOINTS, PickLift, VideoRecorder

ARROW_M_PER_N = 0.02    # arrow length per newton
ARROW_RGBA = np.array([0.89, 0.29, 0.28, 1.0], dtype=np.float32)


# ---------------------------------------------------------------- design and cell
def design_from_xml(path, run, cfg):
    """(design, stage0 patch or None, hand template) of a design XML, as eval_design_rb5.py builds it."""
    source = Path(path).read_text(encoding="utf-8-sig")
    tags = {e.get("name") for e in ET.fromstring(source).iter() if e.get("name")}
    if "R_gp" in tags:
        design, _ = load_design(source, cfg)
        return design, None, source
    if "upper_green_link" in tags:
        template = (run/"source_design168.xml").read_text(encoding="utf-8-sig")
        design, patch = stage0_conversion(source, template, cfg)
        check_conversion(source, apply_patch(build_xml(template, design, hand_config(cfg), radius_mm=40), patch))
        return design, patch, template
    raise SystemExit(f"{path} is neither in the RB5 hand format (R_gp, ...) nor the stage0 format (upper_green_link, ...).")


def choose_cell(run, label, direction, given):
    """((r, x, y) [mm], why, experiment row of that cell and direction or None)."""
    rows = []
    path = DF.force_dir(run, label)/"force_dirs.csv"
    if path.is_file() and path.stat().st_size:
        rows = [r for r in read_rows(path) if r["dir"] == direction and r["group"] in ("env", "tip")]
    if given:
        cell = tuple(float(v) for v in given)
        hit = next((r for r in rows if (r["r_mm"], r["x_mm"], r["y_mm"]) == cell), None)
        return cell, "given with --cell", hit
    if rows:
        key = "F_hold_mg" if rows[0].get("F_hold_mg") is not None else "F_max_mg"
        b = max(rows, key=lambda r: r[key])
        return ((b["r_mm"], b["x_mm"], b["y_mm"]),
                f"held the most in {direction} in the experiment ({'hold' if key == 'F_hold_mg' else 'ramp'} "
                f"{b[key]:.2f} m g, {b['group']})", b)
    for p in (run/"references"/f"{label}_workspace.csv", run/"final"/f"{label}_workspace.csv"):
        if p.is_file():
            c = best_cell(read_rows(p))
            return (c["r_mm"], c["x_mm"], c["y_mm"]), "best cell of the workspace map (no disturbance results yet)", None
    raise SystemExit(f"No disturbance results or workspace map for '{label}' in {run}. Give the cell: --cell R X Y")


# ---------------------------------------------------------------- drawing
def add_arrow(scene, arrow):
    """Append the force arrow to an mjvScene (viewer.user_scn or a Renderer's scene)."""
    if arrow["F"] <= 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), ARROW_RGBA)
    # Drawn outside the cylinder, pointing at it: the tip touches the surface opposite the push.
    length = max(0.02, ARROW_M_PER_N*arrow["F"])
    end = np.asarray(arrow["at"], dtype=np.float64)-arrow["dir"]*arrow["radius"]
    start = end-arrow["dir"]*length
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, start, end)
    scene.ngeom += 1


def status_lines(arrow, info):
    return (f"{info['name']}  dir {info['dir']}  cell r={info['r']:g} x={info['x']:g} y={info['y']:g} mm",
            f"t {arrow['t']:5.2f} s   F {arrow['F']:5.2f} N = {arrow['F']/info['mg']:5.2f} m g   "
            f"slip {arrow['disp']:5.1f} mm   rot {arrow['rot']:4.1f} deg   {arrow['note']}")


class DemoRecorder(VideoRecorder):
    """VideoRecorder with the force arrow, a status line and an optional close-up camera."""

    def __init__(self, model, path, arrow, info, camera="close", fps=30):
        super().__init__(model, path, fps)
        self.arrow, self.info, self.camera, self.active = arrow, info, camera, True
        self.site = model.site("grasp").id

    def capture(self, sim):
        if not self.active or sim.d.time+1e-9 < self.next_time:
            return
        if self.camera == "close" and sim.stage in ("disturb", "after drop"):
            self.cam.lookat[:] = sim.d.site_xpos[self.site]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.55, 150, -18
        self.renderer.update_scene(sim.d, camera=self.cam, scene_option=self.opt)
        if sim.stage in ("disturb", "after drop"):
            add_arrow(self.renderer.scene, self.arrow)
        frame = self.renderer.render()
        if sim.stage in ("disturb", "after drop"):
            frame = self.overlay(frame)
        else:
            frame = self.overlay(frame, f"{self.info['name']}: {sim.stage}")
        self.writer.append_data(frame)
        self.next_time += 1.0/self.fps

    def overlay(self, frame, text=None):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return frame
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=22)  # Pillow >= 10.1
        except (ImportError, TypeError):
            font = None
        lines = [text] if text else list(status_lines(self.arrow, self.info))
        draw.rectangle((0, 0, img.width, 30*len(lines)+12), fill=(252, 252, 251))
        for i, line in enumerate(lines):
            draw.text((14, 8+30*i), line, fill=(11, 11, 11), font=font)
        return np.asarray(img)


# ---------------------------------------------------------------- disturbance
def disturb(model, data, sim, d_hand, args, crit, mg, info, arrow, viewer):
    """Ramp until the object drops (or the cap holds for 2 s). Returns (trace, ended)."""
    ob, og, site = model.body("obj").id, model.geom("obj_g").id, model.site("grasp").id
    links = {model.geom(n).id: i for i, n in enumerate(LINKS)}
    armact = np.array([model.actuator(n).id for n in ARM_JOINTS])
    qa = np.array([model.joint(n).qposadr[0] for n in JOINTS])
    q0 = data.qpos[qa].copy()
    dt = model.opt.timestep
    every = max(1, round(read_json(args.run/"config_used.json")["simulation"]["sample_interval_s"]/dt))
    rs0, h0 = data.site_xmat[site].reshape(3, 3).copy(), data.site_xpos[site].copy()
    p0 = rs0.T @ (data.xpos[ob]-h0)
    r0 = rs0.T @ data.xmat[ob].reshape(3, 3)
    direction = rs0 @ np.asarray(d_hand, float)
    sim.stage = "disturb"
    sim.qcmd = data.ctrl[armact].copy()  # The arm keeps holding its last command.
    arrow.update(dir=direction, F=0.0, t=0.0, note="")
    cap = args.cap_mg*mg
    rows, t0, lost_since, held_cap, ended, failed, onset = [], data.time, None, None, "cap", None, None

    def sample(force):
        rs = data.site_xmat[site].reshape(3, 3)
        p = rs.T @ (data.xpos[ob]-data.site_xpos[site])
        r = rs.T @ data.xmat[ob].reshape(3, 3)
        rot = math.degrees(math.acos(np.clip((np.trace(r0.T @ r)-1)/2, -1, 1)))
        tilt = math.degrees(math.acos(np.clip(float(r0[:, 2] @ r[:, 2]), -1, 1)))
        f = DF.link_forces(model, data, og, links)
        hand = 1000*float(np.linalg.norm(data.site_xpos[site]-h0))
        rows.append([data.time-t0, force, *(1000*(p-p0)), rot, *f, *np.degrees(data.qpos[qa]-q0), hand, tilt])
        return 1000*float(np.linalg.norm(p-p0)), rot, min(f[0]+f[1], f[2]+f[3]) > DF.BILATERAL_N

    sample(0.0)
    i = 0
    try:
        while True:
            i += 1
            t = i*dt
            force = min(args.rate*t, cap)
            data.xfrc_applied[ob, :3] = force*direction
            arrow.update(at=data.xpos[ob].copy(), F=force, t=t)
            sim.step()
            if i % every:
                continue
            disp, rot, bilateral = sample(force)
            arrow.update(disp=disp, rot=rot)
            onset = None if bilateral else (len(rows)-1 if onset is None else onset)
            if failed is None:  # First criterion of the experiment = F_max (as DF.failure_events).
                cause = ("slip" if disp >= crit["slip_mm"] else
                         "rotation" if crit["rot_deg"] > 0 and rot >= crit["rot_deg"] else
                         "lost" if onset is not None and rows[-1][0]-rows[onset][0] >= crit["grace_s"]-1e-9 else None)
                if cause:
                    k = onset if cause == "lost" else len(rows)-1
                    failed = (rows[k][1], cause, rows[k][0])
                    arrow["note"] = f"F_max {failed[0]/mg:.2f} m g ({cause})"
                    print(f"[{t:5.2f} s] F_max = {failed[0]:.3f} N = {failed[0]/mg:.2f} m g  ({cause})", flush=True)
            lost_since = None if bilateral else (data.time if lost_since is None else lost_since)
            if disp >= args.drop_mm or (lost_since is not None and data.time-lost_since >= 0.1-1e-9):
                ended = "drop"
                arrow["note"] += f"   dropped at {force/mg:.2f} m g"
                print(f"[{t:5.2f} s] object dropped: F = {force:.3f} N = {force/mg:.2f} m g "
                      f"(moved {disp:.1f} mm{'' if bilateral else ', contact lost'})", flush=True)
                break
            if force >= cap-1e-12:
                held_cap = t if held_cap is None else held_cap
                if t-held_cap >= 2.0:
                    print(f"[{t:5.2f} s] still holding at the cap {args.cap_mg:g} m g after 2 s; stopping", flush=True)
                    break
    except RuntimeError as exc:
        ended = "unintended_collision" if str(exc).startswith("Unintended collision") else "unstable"
        print(f"Stopped: {exc}", flush=True)
    data.xfrc_applied[:] = 0
    arrow["F"] = 0.0
    if ended == "drop" and args.after_s > 0:  # Let it fall, without force.
        sim.stage = "after drop"
        try:
            for _ in range(round(args.after_s/dt)):
                sim.step()
        except RuntimeError:
            pass
    return np.asarray(rows), ended, failed


class ViewerOverlay:
    """PickLift callback: arrow and status text in the passive viewer, close-up camera."""

    def __init__(self, viewer, model, arrow, info, camera):
        self.v, self.arrow, self.info, self.camera = viewer, arrow, info, camera
        self.site = model.site("grasp").id
        self.closed_up = False

    def __call__(self, sim):
        if sim.frame % 16:
            return
        with self.v.lock():
            self.v.user_scn.ngeom = 0
            if sim.stage in ("disturb", "after drop"):
                add_arrow(self.v.user_scn, self.arrow)
                if self.camera == "close":
                    self.v.cam.lookat[:] = sim.d.site_xpos[self.site]
                    if not self.closed_up:
                        self.v.cam.distance, self.v.cam.azimuth, self.v.cam.elevation = 0.55, 150, -18
                        self.closed_up = True
                l1, l2 = status_lines(self.arrow, self.info)
            else:
                l1, l2 = f"{self.info['name']}  dir {self.info['dir']}", sim.stage
            try:
                self.v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, l1, l2))
            except Exception:  # Older MuJoCo without set_texts.
                pass


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=layout.DEFAULT_RUN)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--xml", type=Path, help="Design XML (stage0 or RB5 hand format)")
    src.add_argument("--label", help="Design of the run: final/ (candidate_02, ...) or references/ (design075, ...)")
    p.add_argument("--dir", required=True, choices=[*DF.DIRECTIONS, *map(DF.key, DF.DIRECTIONS)])
    p.add_argument("--cell", type=float, nargs=3, metavar=("R_MM", "X_MM", "Y_MM"))
    p.add_argument("--skip-lift", action="store_true", help="Start from the saved post-lift state (disturb_collect.py)")
    p.add_argument("--rate", type=float, default=1.0, help="Force rate [N/s] (default 1, as the experiment)")
    p.add_argument("--cap-mg", type=float, default=30.0, help="Ramp end [m g] (default 30)")
    p.add_argument("--drop-mm", type=float, default=30.0, help="Object counts as dropped after moving this far [mm]")
    p.add_argument("--after-s", type=float, default=1.5, help="Simulate on this long after the drop [s]")
    p.add_argument("--speed", type=float, default=1.0, help="Viewer playback rate (2 = twice real time)")
    p.add_argument("--camera", choices=("close", "wide"), default="close", help="Close-up on the hand while pushing")
    p.add_argument("--headless", action="store_true", help="No viewer (e.g. only --video)")
    p.add_argument("--video", type=Path, help="Write an MP4 (pip install imageio imageio-ffmpeg)")
    p.add_argument("--record", choices=("all", "disturb"), default="all", help="Video of everything or only the push")
    p.add_argument("--trace", type=Path, help="Write the push as CSV (columns as disturb_force.py traces)")
    args = p.parse_args()
    args.run = args.run.resolve()
    cfg = read_json(args.run/"config_used.json")
    validate_rb5_config(cfg)
    names = {DF.key(d): d for d in DF.DIRECTIONS}
    direction = names.get(args.dir, args.dir)
    label = args.label or args.xml.stem
    (r, x, y), why, exp_row = choose_cell(args.run, label, direction, args.cell)
    sc = cfg["simulation"]
    crit = dict(slip_mm=sc["hold_displacement_mm"], rot_deg=sc["hold_rotation_deg"], rot_mode="total",
                grace_s=sc.get("lost_contact_grace_s", 0.02))
    print(f"{label}, direction {direction}, cell r={r:g} x={x:g} y={y:g} mm ({why})")

    if args.skip_lift:
        out = label_dir(args.run, label)
        state = out/"states"/f"{cell_name(r, x, y)}.npz"
        if not state.is_file():
            raise SystemExit(f"No saved state {state}. Run without --skip-lift (or grasp/disturb_collect.py first).")
        model, data, st = DF.open_cell(out, dict(state_file=f"states/{state.name}"), cfg)
        DF.set_state(model, data, st["state"])
    else:
        if args.xml:
            design, patch, template = design_from_xml(args.xml, args.run, cfg)
        else:
            design, patch, template, _ = load_label(args.run, label)
        arm = make_arm(cfg, patch)
        _, model, q_home = arm.prepare(template, design, r)
        data = mujoco.MjData(model)
        arm.reset_cell(model, data, q_home, r, x, y)
    mg = float(model.body_mass[model.body("obj").id])*sc["gravity_m_s2"]
    info = dict(name=label, dir=direction, r=r, x=x, y=y, mg=mg)
    arrow = dict(at=np.zeros(3), dir=np.zeros(3), F=0.0, t=0.0, disp=0.0, rot=0.0, note="", radius=r/1000)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer  # Imported here so the top-level mujoco stays the module.
        viewer = mjviewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.30, 0.0, 0.25]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.45, 135, -25
        viewer.opt.geomgroup[3] = 0  # Hide arm collision meshes and tendon wrap surfaces.
        viewer.opt.sitegroup[4] = 0
    sim = PickLift(model, data, viewer, args.speed)
    if viewer is not None:
        sim.callbacks.append(ViewerOverlay(viewer, model, arrow, info, args.camera))
    recorder = None
    if args.video:
        recorder = DemoRecorder(model, args.video, arrow, info, args.camera)
        recorder.active = args.record == "all" or args.skip_lift
        sim.callbacks.append(recorder.capture)
    try:
        if not args.skip_lift:
            try:
                result = sim.run(float(sc["tendon_tension_N"]), sc["lift_mm"]/1000)
            except RuntimeError as exc:
                raise SystemExit(f"Pick-and-lift failed in this cell: {exc}. Try another --cell.")
            if not result["success"]:
                print("Warning: the pick-and-lift did not pass in this cell (lift height, contact or table); "
                      "pushing anyway.")
        if recorder and not recorder.active:
            recorder.active, recorder.next_time = True, data.time  # Start the clip now, not at t = 0.
        trace, ended, failed = disturb(model, data, sim, DF.DIRECTIONS[direction], args, crit, mg, info, arrow, viewer)
        print("\nResult")
        if failed:
            print(f"  F_max  = {failed[0]:.3f} N = {failed[0]/mg:.2f} m g   (first criterion: {failed[1]}, "
                  f"t = {failed[2]:.2f} s after the push started)")
        else:
            print(f"  F_max  > {trace[-1, 1]/mg:.2f} m g   (no criterion met)")
        print(f"  ended  = {ended}" + (f" at F = {trace[-1, 1]/mg:.2f} m g" if ended == "drop" else ""))
        if exp_row:
            exp = f"ramp {exp_row['F_max_mg']:.2f} m g ({exp_row['cause']})"
            if exp_row.get("F_hold_mg") is not None:
                exp += f", hold test {exp_row['F_hold_mg']:.2f} m g"
            print(f"  experiment, same cell and direction: {exp}")
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            with args.trace.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(DF.TRACE_COLUMNS)
                w.writerows(trace.tolist())
            print(f"  trace: {args.trace}")
        if recorder:
            sim.callbacks.remove(recorder.capture)
            recorder.close()
            recorder = None
            print(f"  video: {args.video}")
        if viewer is not None:
            print("Close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Viewer closed.")
    finally:
        if recorder:
            recorder.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
