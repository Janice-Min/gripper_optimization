"""Folder layout of on_rb5: code and results of every experiment.

  on_rb5/
  ├── common/        shared code (gripper_core, rb5_eval, run_pick_lift, build_scene, eval_design_rb5),
  │                  RB5 sources/ and assets/, config_rb5_arm.json, design168.xml / design001.xml
  ├── grasp/         workspace map and CMA-ES                 results/  (config_used.json, final/, references/,
  │                                                                     plots/, ...: the grasp run)
  ├── grasp_cells/   grasp success cells, state after lifting  <label>/  (cells.csv, states/, hands/, meta.json)
  │                  (grasp/disturb_collect.py; A, B and C all start from these states)
  ├── A/             slowly increasing force                   results/  (<label>/traces, force_*, tables, plots/)
  ├── B/             short pulse (impact)                      results/impulse[_<tag>]/<label>/
  └── C/             arm shaking                               results/
--run is the grasp run, grasp/results by default. Another run (e.g. grasp/results_rb5_cmaes, an
earlier search) gets its own folders: grasp_cells_rb5_cmaes/, A/results_rb5_cmaes/, ...

Every script outside common/ starts with
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
    import layout
which puts common/, grasp/, A/, B/ and C/ on the import path (module names are unique).
"""
from __future__ import annotations

import sys
from pathlib import Path

ON_RB5 = Path(__file__).resolve().parents[1]
COMMON = ON_RB5/"common"
GRASP = ON_RB5/"grasp"
EXP_A = ON_RB5/"A"
EXP_B = ON_RB5/"B"
EXP_C = ON_RB5/"C"
XML_DIR = ON_RB5.parent/"xml"  # design001.xml ... design204.xml (stage0 format)

for _d in (COMMON, GRASP, EXP_A, EXP_B, EXP_C):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))

GRASP_RESULTS = GRASP/"results"  # The grasp run (was runs/rb5_cmaes_final).
DEFAULT_RUN = GRASP_RESULTS
PICK_LIFT_RESULTS = GRASP_RESULTS/"pick_lift"  # run_pick_lift.py output


def run_suffix(run):
    """'' for the default run grasp/results, '_<name>' for another run (grasp/results_<name> or any folder)."""
    p = Path(run).resolve()
    if p == DEFAULT_RUN.resolve():
        return ""
    return "_" + (p.name[len("results_"):] if p.name.startswith("results_") else p.name)


def states_root(run):
    """Grasp success cells (saved states) of every design: grasp_cells/<label>/."""
    return ON_RB5/f"grasp_cells{run_suffix(run)}"


def a_root(run):
    """Experiment A results: A/results/."""
    return EXP_A/f"results{run_suffix(run)}"


def b_root(run):
    """Experiment B results: B/results/ (one folder per setting inside)."""
    return EXP_B/f"results{run_suffix(run)}"


def c_root(run):
    """Experiment C results: C/results/."""
    return EXP_C/f"results{run_suffix(run)}"
