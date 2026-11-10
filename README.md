# gripper_optimization

텐던 구동 2지 그리퍼의 설계를 MuJoCo 시뮬레이션으로 최적화하는 코드입니다.
그리퍼를 RB5-850e 팔에 달고 책상 위 원기둥을 집어 들어 올리는 동작으로 Graspable Workspace를 평가합니다.
CMA-ES로 설계 변수 4개(손가락 총길이, L_p 비율, 피벗 간격 G, θ_open,p)를 최적화합니다.

```
gripper_optimization/
├── on_rb5/         RB5 장착 상태의 workspace 평가와 CMA-ES 설계 최적화
└── rb5_xml_pick/   그리퍼 XML 하나를 RB5에 달아 집고 들어 올리는 단독 실행 패키지
```

workspace map의 cell `(r, x, y)`는 원기둥 반지름 r, 손 좌표계에서의 물체 중심 `(x, y + r)`입니다.
y는 palm 면과 물체 표면 사이 거리이고, 손이 대칭이라 x ≥ 0만 시뮬레이션합니다. 단위는 mm입니다.

---

## on_rb5/

### 코드

| 파일 | 설명 |
|---|---|
| `gripper_core.py` | 공통 모듈. 설계 변수와 범위, 설계값으로 손 MJCF 생성(`build_xml`), 평가 격자, 접촉 판정(env / tip), 목적함수 J |
| `build_scene.py` | `sources/`의 RB5·작업 장면·손을 합쳐 `rb5_best_design_scene.xml` 생성 |
| `run_pick_lift.py` | RB5 pick-and-lift 시뮬레이션: IK로 접근 → 텐던 파지 → 상승 → 유지 |
| `rb5_cell.py` | map의 cell 하나에 대해 `run_pick_lift.py` 실행 |
| `rb5_eval.py` | 설계 하나의 workspace를 RB5 위에서 평가. cell마다 pick-and-lift를 끝까지 돌려 판정하고 J 계산. `--rescore`로 저장된 CSV 재판정 |
| `optimize_rb5_cmaes.py` | CMA-ES 최적화. `rb5_coarse` 격자로 탐색하고 상위 후보 3개는 `rb5_fine` 격자로 재평가. 세대마다 checkpoint 저장, `--resume`으로 이어서 실행 |
| `eval_design_rb5.py` | 다른 설계 XML을 최적화 run과 같은 설정·격자로 평가해 `<run>/references/`에 저장. design168 형식과 design001(stage0) 형식 모두 읽음 |
| `add_reference.py` | 설계 XML을 같은 설정·격자로 평가해 `<run>/final/`에 추가 |
| `plot_workspace.py` | run 결과의 workspace map, 기준 설계 대비 증감, 후보 비교 그림 |
| `compare_designs_rb5.py` | 두 설계의 map·증감·요약 비교 그림과 CSV를 `<run>/plots/`에 저장 (기본: design001 vs 최종 best) |
| `run_task.py` | run 폴더의 설계로 특정 cell의 pick-and-lift를 viewer로 재생 |

### 설정·입력

| 파일 | 설명 |
|---|---|
| `config_rb5_arm.json` | 최적화·평가 설정: 변수 범위, 목적함수(w_tip = 0.2 등), 격자, 시뮬레이션, CMA-ES, RB5 장착 옵션 |
| `config_rb5.json` | `rb5_arm` 항목이 없는 이전 설정. 현재 스크립트 기본값으로는 쓰지 않음 |
| `design168.xml` | 최적화 기준 설계(baseline). `R_` / `L_` 이름 형식 |
| `design001.xml` | 비교용 설계. stage0 형식 |
| `check.xml` | RB5 + 손 통합 장면 확인용. 코드에서 읽지 않음 |
| `rb5_agreement_best_design_fine.csv` | best_design_fine의 cell별 결과를 손 단독 시뮬레이션과 RB5 시뮬레이션 사이에서 비교한 표 |
| `sources/` | RB5 모델(`rb5_850e.xml`), 작업 장면(`task_scene.xml`), 장착할 손(`best_design_fine.xml`), Class_IR 라이선스 |
| `assets/rb5_850e/` | RB5 mesh(collision STL, visual OBJ)와 라이선스 |

### 결과

| 폴더 | 설명 |
|---|---|
| `results/` | `run_pick_lift.py` 실행 결과 (`summary.json`, `trace.csv`) |
| `runs/rb5_cmaes/` | 먼저 돌린 CMA-ES run. 탐색 기록만 있고 최종 재평가는 없음 |
| `runs/rb5_cmaes_final/` | 최종 CMA-ES run |

`runs/rb5_cmaes_final/`의 구성:

| 파일·폴더 | 설명 |
|---|---|
| `history.csv` | 평가한 설계 전체 기록 |
| `best_design.json`, `best_design_hand.xml`, `best_rb5_scene.xml` | 탐색 격자(coarse)에서의 best 설계와 손·장면 XML |
| `best_design_fine*`, `best_rb5_scene_fine.xml` | fine 격자 재평가 1위(candidate_02)와 손·장면 XML |
| `final/` | baseline(design168)과 상위 후보 3개(candidate_01~03)의 fine 격자 결과, 비교 요약 `comparison.json` |
| `references/` | `eval_design_rb5.py`로 평가한 design001 |
| `plots/` | workspace map, design001 대비 비교 그림 |
| `task/` | `run_task.py` 실행 결과 |
| `viewer_cell*.json` | best 설계의 대표 성공 cell |
| `config_used.json`, `environment.json`, `checkpoint.pkl` | 사용한 설정, 실행 환경, 재시작용 checkpoint |

### 실행 예시

Python 3.11.9, MuJoCo 3.3.7, cma 4.5.0, numpy 2.4.6 (Windows)에서 실행했습니다.

```powershell
python -m pip install mujoco numpy cma matplotlib
python build_scene.py                                   # run_pick_lift.py, rb5_cell.py용 장면 생성
python optimize_rb5_cmaes.py --workers 8 --out runs/rb5_cmaes_final
python eval_design_rb5.py --run runs/rb5_cmaes_final --xml design001.xml --workers 8
python compare_designs_rb5.py --run runs/rb5_cmaes_final
python run_task.py --run runs/rb5_cmaes_final --label candidate_02
```

---

## rb5_xml_pick/

`on_rb5/` 없이 단독으로 실행되는 패키지입니다. 자세한 사용법과 옵션은 [`README_KO.md`](rb5_xml_pick/README_KO.md)에 있습니다.

| 파일·폴더 | 설명 |
|---|---|
| `run_rb5_xml.py` | design001 형식의 그리퍼 XML을 RB5 손목에 달아 접근 → 파지 → 150 mm 상승 → 2 s 유지. 옵션은 파일 맨 아래 argparse에 있음 |
| `design001.xml`, `design168.xml`, `best_design_fine.xml` | 입력 XML 예시 |
| `candidate_02.json` | candidate_02의 설계값과 평가 결과 |
| `results/` | 실행 결과: 통합 장면 XML, `summary.json`, `trace.csv` |
| `validation/` | 입력 형식·실패 경우 검사 결과 (요약은 `verification.json`) |
| `design001_pick_lift.mp4` | design001 실행 영상 |
| `requirements.txt`, `requirements-video.txt` | 의존성 (Python 3.12, MuJoCo 3.17.0에서 검증), 영상 저장용 추가 의존성 |
| `sources/`, `assets/` | RB5 모델·작업 장면·mesh와 라이선스 |

```powershell
python -m pip install -r requirements.txt
python run_rb5_xml.py --xml design001.xml
```

---

RB5 kinematics·mesh는 Rainbow Robotics [`rbpodo_ros2`](https://github.com/RainbowRobotics/rbpodo_ros2)의 `rb5_850e` 기반입니다.
라이선스는 각 폴더의 `assets/rb5_850e/LICENSE`, `sources/Class_IR_LICENSE`를 참고하세요.
