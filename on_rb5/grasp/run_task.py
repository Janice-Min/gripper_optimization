"""Pick-and-lift task on the RB5 with a hand from an optimize_rb5_cmaes.py run.

The design, config and template are read from the run folder. The scene, start pose
and verdict are the same as in the workspace map (rb5_eval.py), so a cell marked
env / tip / fail in the map behaves the same here.

Cell convention as in the map: object radius r, object center at hand (x, y+r),
y = gap between palm face and object surface.

  python grasp/run_task.py                              # final best design, its best cell (from on_rb5/)
  python grasp/run_task.py --cell 30 20 45              # r=30 mm, x=20, y=45
  python grasp/run_task.py --label baseline             # same task with design168
  python grasp/run_task.py --headless --video task.mp4
  python grasp/run_task.py --run grasp/results_rb5_cmaes   # another run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
import layout  # noqa: E402  (puts common/, grasp/, A/, B/, C/ on the import path)
from gripper_core import read_json
from rb5_eval import CellMonitor, Rb5Arm, best_cell, empty_row, run_result, validate_rb5_config
from run_pick_lift import PickLift, VideoRecorder, save_results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=layout.DEFAULT_RUN,
                   help="Output folder of optimize_rb5_cmaes.py (default grasp/results)")
    p.add_argument("--label", help="Design in <run>/final, e.g. baseline, candidate_01, design001 (default: best)")
    p.add_argument("--cell", type=float, nargs=3, metavar=("R_MM", "X_MM", "Y_MM"),
                   help="Object radius and map cell (default: the design's best successful cell)")
    p.add_argument("--headless", action="store_true", help="No viewer; run as fast as possible")
    p.add_argument("--speed", type=float, default=1.0, help="Viewer playback rate")
    p.add_argument("--video", type=Path, help="Optional MP4 path (needs imageio[ffmpeg])")
    args = p.parse_args()
    run = args.run.resolve()
    cfg = read_json(run/"config_used.json")
    validate_rb5_config(cfg)
    label, design, _, rows, template = run_result(run, args.label)
    if args.cell:
        r, x, y = args.cell
    else:
        cell = best_cell(rows)
        r, x, y = cell["r_mm"], cell["x_mm"], cell["y_mm"]
    if y <= 0 or r <= 0 or x < 0:
        raise SystemExit("Use r > 0, x >= 0 and y > 0 (y is the gap between palm and object)")
    print(f"Design {label}: L={design['length_mm']:.1f} mm, Lp/L={design['lp_ratio']:.3f}, "
          f"G={design['gap_mm']:.1f} mm, theta_p={design['theta_p_deg']:.1f} deg")
    print(f"Object r={r:g} mm at map cell x={x:g}, y={y:g} (center {x:g}, {y+r:g} mm in the hand frame)")
    mapped = [row for row in rows if (row["r_mm"], row["x_mm"], row["y_mm"]) == (r, x, y)]
    if mapped:
        m = mapped[0]
        print("Workspace map says:", f"success ({m['contact_type']})" if m["success"] else f"fail ({m['reason']})")
    else:
        print("This cell is not on the evaluated grid.")

    arm = Rb5Arm(cfg)
    _, model, q_home = arm.prepare(template, design, r)
    data = mujoco.MjData(model)
    arm.reset_cell(model, data, q_home, r, x, y)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.30, 0.0, 0.25]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.45, 135, -25
        viewer.opt.geomgroup[3] = 0  # Hide arm collision meshes and tendon wrap surfaces.
        viewer.opt.sitegroup[4] = 0
    sim = PickLift(model, data, viewer, args.speed)
    monitor = CellMonitor(model, cfg, arm.opts)
    sim.callbacks.append(monitor)
    recorder = None
    out = run/"task"/f"{label}_r{r:g}_x{x:g}_y{y:g}"
    try:
        if args.video:
            recorder = VideoRecorder(model, args.video)
            sim.callbacks.append(recorder.capture)
        result, error = None, None
        try:
            result = sim.run(arm.tension, arm.lift)
        except RuntimeError as exc:
            error = exc
        row = arm.finish_row(empty_row(r, x, y), model, data, sim, monitor, result, error)
        summary = result or {"success": False, "error": str(error), "stage": sim.stage,
                             "simulation_time_s": float(data.time), "mujoco_version": mujoco.__version__}
        save_results(out, sim, dict(summary, design_label=label, design=design, verdict=row))
        print(json.dumps({k: row[k] for k in ("contact_type", "reason", "m_F", "R_prox_N", "R_dist_N",
                                             "L_prox_N", "L_dist_N", "palm_N", "hold_drift_mm",
                                             "hold_rotation_deg", "min_lift_mm")}, indent=2, default=float))
        print(f"TASK {'SUCCESS (' + row['contact_type'] + ')' if row['success'] else 'FAIL (' + row['reason'] + ')'}")
        print("Saved:", out)
        if recorder:
            sim.callbacks.remove(recorder.capture)
            recorder.close()
            recorder = None
        if viewer is not None:
            print("Close the viewer window to exit.")
            sim.record_trace = False
            sim.callbacks.clear()
            while viewer.is_running():
                if row["success"]:
                    sim.step()
                else:  # Freeze the failed state for inspection.
                    viewer.sync()
                    time.sleep(0.03)
    except KeyboardInterrupt:
        print("Viewer closed.")
    finally:
        if recorder:
            recorder.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
