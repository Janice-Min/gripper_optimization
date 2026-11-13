"""Move the old flat on_rb5/ (code + runs/) into common/ grasp/ A/ B/ C/.

Run once, after the new code folders (common/, grasp/, A/, B/, C/) have been unpacked
into on_rb5/ next to this file:

  python reorganize.py            # preview: what would move where, conflicts, leftovers
  python reorganize.py --apply    # do it

What moves where (nothing is overwritten; an existing destination is reported and skipped):
  sources/ assets/ config_rb5*.json design168.xml design001.xml check.xml
  rb5_best_design_scene.xml                      -> common/
  results/ (run_pick_lift.py output)             -> grasp/results/pick_lift/
  rb5_agreement_best_design_fine.csv             -> grasp/results/
  runs/<run>/ (final/, references/, plots/, ...) -> grasp/results/<run>/
  runs/<run>/disturbance/<label>/
      cells.csv states/ hands/ meta.json         -> grasp/results/<run>/grasp_states/<label>/
      traces/ force_*                            -> A/results/<run>/<label>/
      impulse*/                                  -> B/results/<run>/impulse*/<label>/
  runs/<run>/disturbance/ all_designs.csv graspable_designs.csv summary.* _validation/ _logs/
                                                 -> A/results/<run>/
  runs/<run>/disturbance/impulse*_all_designs.csv -> B/results/<run>/impulse*/all_designs.csv
  runs/<run>/plots/disturbance/impulse*/         -> B/results/<run>/impulse*/plots/
  runs/<run>/plots/disturbance/ (the rest)       -> A/results/<run>/plots/
Viewer scenes (*_rb5_scene.xml, best_rb5_scene*.xml) get their relative meshdir fixed for the
new depth. Saved grasp states and their hands/*.xml are moved byte for byte (the scene
signature checked by A and B stays valid).

Old top-level code (gripper_core.py, disturb_force.py, ..., EXPERIMENT_A/B.md) is deleted
when it is identical to the GitHub version (commit 36159177) or to the experiment-B files
sent before; a file with local changes goes to _old_code/ instead, so nothing is lost.
Any other file is left where it is and listed at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

ON_RB5 = Path(__file__).resolve().parent

CODE = {  # old top-level file -> folder of its new version
    "gripper_core.py": "common", "rb5_eval.py": "common", "run_pick_lift.py": "common",
    "build_scene.py": "common", "eval_design_rb5.py": "common",
    "optimize_rb5_cmaes.py": "grasp", "add_reference.py": "grasp", "plot_workspace.py": "grasp",
    "compare_designs_rb5.py": "grasp", "run_task.py": "grasp", "rb5_cell.py": "grasp",
    "disturb_collect.py": "grasp",
    "disturb_force.py": "A", "disturb_validate.py": "A", "batch_disturbance.py": "A",
    "plot_disturbance.py": "A", "compare_force.py": "A", "demo_disturbance.py": "A",
    "EXPERIMENT_A.md": "A",
    "disturb_impulse.py": "B", "plot_impulse.py": "B", "EXPERIMENT_B.md": "B",
}
# sha256[:16] of the known versions (CRLF -> LF, no BOM): GitHub 36159177 and experiment B as sent.
KNOWN = {'EXPERIMENT_A.md': ('7e9bc5462cbf6eff',), 'EXPERIMENT_B.md': ('2a4d0e339216b3c3',),
         'add_reference.py': ('b1407741721fb852',), 'batch_disturbance.py': ('78bc48f588190577',),
         'build_scene.py': ('75a996138800b863',), 'compare_designs_rb5.py': ('2df6775ff0859d21',),
         'compare_force.py': ('3e930da4cd02c555',), 'demo_disturbance.py': ('9d76c0226993d789',),
         'disturb_collect.py': ('ea02b2d9625d6d2f',), 'disturb_force.py': ('b4589440703b0ed6',),
         'disturb_impulse.py': ('677dcb4d07ad5ec4',), 'disturb_validate.py': ('2197a674b597641f',),
         'eval_design_rb5.py': ('ed9b1e76d79865b7',), 'gripper_core.py': ('344184d06f4df98f',),
         'optimize_rb5_cmaes.py': ('0f5a776f08fb37dc',), 'plot_disturbance.py': ('eefd010ef703a9db',),
         'plot_impulse.py': ('41eefa817b6ab827',), 'plot_workspace.py': ('c9f56c5af6556d39',),
         'rb5_cell.py': ('1a903d48afb3e605',), 'rb5_eval.py': ('3424ab0c638b0eda',),
         'run_pick_lift.py': ('e1ae7e13ea67fce5',), 'run_task.py': ('adb9d3eacfca9b9b',)}
DATA = {  # old top-level entry -> new place
    "sources": "common/sources", "assets": "common/assets", "config_rb5_arm.json": "common/config_rb5_arm.json",
    "config_rb5.json": "common/config_rb5.json", "design168.xml": "common/design168.xml",
    "design001.xml": "common/design001.xml", "check.xml": "common/check.xml",
    "rb5_best_design_scene.xml": "common/rb5_best_design_scene.xml",
    "results": "grasp/results/pick_lift",
    "rb5_agreement_best_design_fine.csv": "grasp/results/rb5_agreement_best_design_fine.csv",
}
STATE_PARTS = ("cells.csv", "states", "hands", "meta.json")
MESHES = ON_RB5/"common"/"assets"/"rb5_850e"/"meshes"
NEW_CODE = ("common/layout.py", "grasp/disturb_collect.py", "A/disturb_force.py", "B/disturb_impulse.py")


def digest(path):
    b = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
    return hashlib.sha256(b).hexdigest()[:16]


class Plan:
    def __init__(self):
        self.moves, self.deletes, self.conflicts, self.kept_old = [], [], [], []

    def move(self, src, dst, kind):
        self.moves.append((Path(src), Path(dst), kind))


def plan_runs(plan, runs):
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        name = run.name
        g, a, b = ON_RB5/"grasp"/"results"/name, ON_RB5/"A"/"results"/name, ON_RB5/"B"/"results"/name
        dist = run/"disturbance"
        for entry in sorted(run.iterdir()):
            if entry.name == "disturbance" or entry.name == "plots":
                continue
            plan.move(entry, g/entry.name, "grasp run")
        plots = run/"plots"
        if plots.is_dir():
            for entry in sorted(plots.iterdir()):
                if entry.name != "disturbance":
                    plan.move(entry, g/"plots"/entry.name, "grasp run")
            pdist = plots/"disturbance"
            if pdist.is_dir():
                for entry in sorted(pdist.iterdir()):
                    if entry.is_dir() and entry.name.startswith("impulse"):
                        plan.move(entry, b/entry.name/"plots", "B plots")
                    else:
                        plan.move(entry, a/"plots"/entry.name, "A plots")
        if not dist.is_dir():
            continue
        for entry in sorted(dist.iterdir()):
            if entry.is_file():
                m = re.fullmatch(r"(impulse(?:_.+)?)_all_designs\.csv", entry.name)
                if m:
                    plan.move(entry, b/m.group(1)/"all_designs.csv", "B table")
                else:
                    plan.move(entry, a/entry.name, "A table")
            elif entry.name.startswith("_"):
                plan.move(entry, a/entry.name, "A table")
            else:  # One design.
                label = entry.name
                for part in sorted(entry.iterdir()):
                    if part.name in STATE_PARTS:
                        plan.move(part, g/"grasp_states"/label/part.name, "grasp state")
                    elif part.is_dir() and part.name.startswith("impulse"):
                        plan.move(part, b/part.name/label, "B result")
                    else:  # traces/, force_*, anything else of A.
                        plan.move(part, a/label/part.name, "A result")


def make_plan():
    plan = Plan()
    for name, folder in CODE.items():
        old = ON_RB5/name
        if not old.is_file():
            continue
        if digest(old) in KNOWN.get(name, ()):
            plan.deletes.append(old)
        else:
            plan.kept_old.append(old)
            plan.move(old, ON_RB5/"_old_code"/name, "old code with local changes")
    for name, new in DATA.items():
        if (ON_RB5/name).exists():
            plan.move(ON_RB5/name, ON_RB5/new, "common / grasp data")
    if (ON_RB5/"runs").is_dir():
        plan_runs(plan, ON_RB5/"runs")
    return plan


def do_move(src, dst, conflicts):
    """Move src to dst; merge into an existing folder, never overwrite a file."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    elif src.is_dir() and dst.is_dir():
        for child in sorted(src.iterdir()):
            do_move(child, dst/child.name, conflicts)
        if not any(src.iterdir()):
            src.rmdir()
    else:
        conflicts.append((src, dst))


def fix_meshdir(root):
    """Relative meshdir of moved viewer scenes -> common/assets/rb5_850e/meshes from the new place."""
    n = 0
    for path in root.rglob("*.xml"):
        if "grasp_states" in path.parts:
            continue  # Hand XMLs of the saved states: never touched.
        text = path.read_text(encoding="utf-8-sig")
        m = re.search(r'meshdir="([^"]*)"', text)
        if not m or os.path.isabs(m.group(1)) or not m.group(1).replace("\\", "/").endswith("assets/rb5_850e/meshes"):
            continue
        new = Path(os.path.relpath(MESHES, path.parent)).as_posix()
        if new != m.group(1):
            path.write_text(text.replace(m.group(0), f'meshdir="{new}"', 1), encoding="utf-8")
            n += 1
    return n


def remove_empty(path):
    if path.is_dir():
        for child in path.iterdir():
            remove_empty(child)
        if not any(path.iterdir()):
            path.rmdir()


def rel(p):
    return Path(os.path.relpath(p, ON_RB5)).as_posix()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Move the files (default: preview only)")
    p.add_argument("--list", action="store_true", help="Preview every single move, not only a summary")
    args = p.parse_args()
    missing = [c for c in NEW_CODE if not (ON_RB5/c).is_file()]
    if missing:
        raise SystemExit(f"New code not found ({', '.join(missing)}). Unpack the zip into on_rb5/ first.")
    plan = make_plan()
    exists = [(s, d) for s, d, _ in plan.moves if d.exists() and not (s.is_dir() and d.is_dir())]

    kinds = {}
    for s, d, k in plan.moves:
        kinds.setdefault(k, []).append((s, d))
    print(f"on_rb5: {ON_RB5}\n")
    print(f"Old code identical to a known version, to delete: {len(plan.deletes)}")
    if plan.kept_old:
        print(f"Old code WITH LOCAL CHANGES -> _old_code/ (compare with the new version and merge by hand): "
              f"{[x.name for x in plan.kept_old]}")
    for k, items in kinds.items():
        print(f"{k}: {len(items)} moves")
        for s, d in (items if args.list else items[:4]):
            print(f"    {rel(s)}  ->  {rel(d)}")
        if not args.list and len(items) > 4:
            print("    ... (--list shows all)")
    if exists:
        print(f"\nDestination already exists, will be SKIPPED ({len(exists)}):")
        for s, d in exists[:20]:
            print(f"    {rel(s)}  ->  {rel(d)}")
    if not args.apply:
        print("\nPreview only. Run with --apply to move the files.")
        return

    conflicts = []
    for s, d, _ in plan.moves:
        if s.exists():
            do_move(s, d, conflicts)
    for old in plan.deletes:
        old.unlink()
    shutil.rmtree(ON_RB5/"__pycache__", ignore_errors=True)
    n_mesh = sum(fix_meshdir(r) for r in (ON_RB5/"grasp"/"results",) if r.is_dir())
    remove_empty(ON_RB5/"runs")
    left = sorted(rel(x) for x in ON_RB5.iterdir()
                  if x.name not in ("common", "grasp", "A", "B", "C", "_old_code", "reorganize.py", ".gitignore"))
    print(f"\nDone. Deleted {len(plan.deletes)} old code files, moved {len(plan.moves)} entries, "
          f"fixed meshdir in {n_mesh} viewer scenes.")
    if conflicts:
        print(f"Not moved because the destination exists ({len(conflicts)}):")
        for s, d in conflicts:
            print(f"    {rel(s)}  ->  {rel(d)}")
    if left:
        print(f"Left in on_rb5/ (not part of the layout, decide yourself): {left}")
    print("\nCheck (from on_rb5/; none of these simulates over saved results):\n"
          "  python A/batch_disturbance.py --only-table              # A tables from A/results/\n"
          "  python A/demo_disturbance.py --label design075 --dir py --skip-lift --headless\n"
          "                                                          # opens a saved grasp state\n"
          "  python B/disturb_impulse.py --rescore                   # if B results exist\n"
          "Then: git add -A  (git sees the moves as renames)")


if __name__ == "__main__":
    main()
