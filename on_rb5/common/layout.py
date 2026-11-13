"""Folder layout of on_rb5: code and results of every experiment.

  on_rb5/
  ├── common/   shared code (gripper_core, rb5_eval, run_pick_lift, build_scene, eval_design_rb5),
  │             RB5 sources/ and assets/, configs, design168.xml / design001.xml
  ├── grasp/    workspace map, CMA-ES, grasp states   results/<run>/            (final/, references/, plots/, ...)
  │                                                    results/<run>/grasp_states/<label>/  (cells.csv, states/, hands/)
  ├── A/        slowly increasing force               results/<run>/<label>/    (traces/, force_*)
  ├── B/        short pulse (impact)                  results/<run>/impulse[_<tag>]/<label>/
  └── C/        arm shaking                           results/<run>/
<run> is the name of the grasp run (default rb5_cmaes_final). A, B and C start from the grasp
states of that run, so their results are filed under the same name.

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

GRASP_RESULTS = GRASP/"results"
DEFAULT_RUN = GRASP_RESULTS/"rb5_cmaes_final"
PICK_LIFT_RESULTS = GRASP_RESULTS/"pick_lift"  # run_pick_lift.py output


def states_root(run):
    """Grasp states of every design of a run: <run>/grasp_states/<label>/."""
    return Path(run)/"grasp_states"


def a_root(run):
    """Experiment A results of a run: A/results/<run name>/."""
    return EXP_A/"results"/Path(run).name


def b_root(run):
    """Experiment B results of a run: B/results/<run name>/ (one folder per setting inside)."""
    return EXP_B/"results"/Path(run).name


def c_root(run):
    """Experiment C results of a run: C/results/<run name>/."""
    return EXP_C/"results"/Path(run).name
