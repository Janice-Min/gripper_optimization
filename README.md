# gripper_optimization

텐던 구동 2지 그리퍼의 설계를 MuJoCo 시뮬레이션으로 최적화하고 평가하는 코드입니다.
그리퍼를 RB5-850e 팔에 달고 책상 위 원기둥을 집어 들어 올리는 동작으로 두 가지를 잽니다.

- **Graspable Workspace**: 어떤 크기·위치의 원기둥까지 잡을 수 있는가 (CMA-ES 최적화 대상)
- **외란 실험 A**: 잡은 원기둥을 각 방향으로 얼마나 세게 밀거나 당겨도 놓치지 않는가 (방향별 F_max)

```
gripper_optimization/
├── on_rb5/         RB5 장착 상태의 workspace 평가, CMA-ES 설계 최적화, 외란 실험 A
├── rb5_xml_pick/   그리퍼 XML 하나를 RB5에 달아 집고 들어 올리는 단독 실행 패키지
└── xml/            평가할 설계 XML (design001 ~ design204, stage0 형식)
```

workspace map의 cell `(r, x, y)`는 원기둥 반지름 r, 손 좌표계에서의 물체 중심 `(x, y + r)`입니다.
y는 palm 면과 물체 표면 사이 거리이고, 손이 대칭이라 x ≥ 0만 시뮬레이션합니다. 단위는 mm입니다.

---

## on_rb5/ — workspace 평가와 최적화

### 코드

| 파일 | 설명 |
|---|---|
| `gripper_core.py` | 공통 모듈. 설계 변수와 범위, 설계값으로 손 MJCF 생성(`build_xml`), 평가 격자, 접촉 판정(env / tip), 목적함수 J |
| `build_scene.py` | `sources/`의 RB5·작업 장면·손을 합쳐 `rb5_best_design_scene.xml` 생성 |
| `run_pick_lift.py` | RB5 pick-and-lift 시뮬레이션: IK로 접근 → 텐던 파지 → 상승 → 유지 |
| `rb5_cell.py` | map의 cell 하나에 대해 `run_pick_lift.py` 실행 |
| `rb5_eval.py` | 설계 하나의 workspace를 RB5 위에서 평가. cell마다 pick-and-lift를 끝까지 돌려 판정하고 J 계산. `--rescore`로 저장된 CSV 재판정 |
| `optimize_rb5_cmaes.py` | CMA-ES 최적화. `rb5_coarse` 격자로 탐색하고 상위 후보 3개는 `rb5_fine` 격자로 재평가. 세대마다 checkpoint 저장, `--resume`으로 이어서 실행 |
| `eval_design_rb5.py` | 다른 설계 XML을 최적화 run과 같은 설정·격자로 평가해 `<run>/references/`에 저장. design168 형식과 design001(stage0) 형식 모두 읽음 (아래 "stage0 변환 검사" 참고) |
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

### stage0 변환 검사 (`eval_design_rb5.py`)

stage0 형식 XML(`../xml/design***.xml`)은 run의 RB5 손 템플릿 위에 다시 만들어지고, RB5에 올리기 전에 원래 XML과 같은지 검사합니다.
검사 항목은 손가락 값(L_p, L_d, G, θ, 강성, preload), 열린 자세의 site 위치, 그리고 **손가락 운동 범위 전체(5×5 자세)의 텐던 moment arm**입니다(허용 0.05 mm).
예전 검사(0.3 N으로 당긴 뒤 관절 각도 비교)는 preload가 있는 관절이 open stop에 눌리는 깊이(joint limit solref 차이) 때문에 204개 중 168개를 잘못 실패로 판정해 batch가 1단계에서 멈췄습니다. 지금은 204개 모두 통과합니다.

---

## on_rb5/ — 외란 실험 A

실험 설계·검증 결과·결정 사항은 [`on_rb5/EXPERIMENT_A.md`](on_rb5/EXPERIMENT_A.md)에 있습니다. 이 절은 코드와 실행 방법을 설명합니다.

### 무엇을 재나

1. 설계마다 workspace map(792 cell)을 평가하고, 성공 cell을 다시 돌려 **들어 올린 직후('lift hold' 끝)의 전체 시뮬레이션 상태**를 저장합니다.
2. 그 상태에서 원기둥의 **질량 중심에 외력**(`xfrc_applied`)을 0부터 1 N/s로 키우며 6방향으로 밀거나 당깁니다.
   팔은 마지막 관절 명령을 유지하고, 텐던은 장력 20 N을 유지합니다(장력 고정). 중력은 켜져 있으므로 F는 무게에 **더해서** 버티는 외력입니다.
3. 아래 기준 중 하나에 처음 걸린 순간의 힘이 그 cell·그 방향의 **F_max**입니다.

| 기준 | 내용 (기본값은 run의 config) |
|---|---|
| slip | 손(grasp site) 기준 물체 변위 ≥ 5 mm |
| rotation | 손 기준 물체 회전 ≥ 5° (`--rot-mode tilt`면 원기둥 축 기울기만) |
| lost | 한쪽 손가락 접촉력 합 < 0.1 N 상태가 20 ms 이상 |

**방향 (손 좌표계, workspace map과 같음)**

| 방향 | 의미 |
|---|---|
| `+x` / `-x` | jaw 폭 방향. +x는 R 손가락 쪽 |
| `+y` / `-y` | 손 밖으로 빼내기(pull-out) / palm 쪽으로 밀기 |
| `+z` / `-z` | 원기둥 축 = 손가락 평면 밖, 위 / 아래 (world z) |

명령줄에서 `-x`처럼 `-`로 시작하는 방향은 옵션으로 읽히므로 `px nx py ny pz nz`로 씁니다(또는 `--dir=-x`).

**단위와 용어**

| 용어 | 뜻 |
|---|---|
| m g | 물체 무게 단위. m = 0.1 kg이므로 1 m g = 0.981 N |
| F_max | cell 하나, 방향 하나의 유지 한계 |
| F_min | cell 하나의 6방향 F_max 중 최솟값(가장 약한 방향) |
| censored | 상한(기본 10 m g)까지 버틴 값. 실제 값은 더 큼 |
| ramp / hold | ramp: 1 N/s로 키운 F_max. hold(`--refine`): 0.5 s에 걸쳐 올린 일정한 힘을 1 s 유지할 수 있는 최댓값(bisection, 허용 0.02 m g). hold ≈ 0.85 × ramp이고 순위는 같음 |
| 외력 0 대조 | 같은 상태를 외력 없이 10 s 둠. 여기서도 기준에 걸리는 성공 cell은 **실패로 간주**(그룹 `zero_fail`) |
| 그룹 | `env`, `tip`(성공 cell), `hold_mixed`(들었지만 한쪽만 감싼 cell, 참고용), `zero_fail` |

**설계 단위 지표**

| 지표 | 뜻 |
|---|---|
| `success` (env/tip) | 외력 0 대조까지 통과한 성공 cell 수 |
| `J_stable` | 그 cell들로 다시 계산한 workspace J (`J_rerun`은 대조 전) |
| `pass_k mg` | 모든 방향에서 F ≥ k m g인 cell 수 |
| `cond_k mg` | pass / success (workspace 크기와 분리한 파지 품질) |
| `J_robust_k mg` | 통과 cell로 계산한 J |
| `robust_area` | Σ min(F_min, 상한). workspace 크기 × 강건성 |
| `F_min_median_mg` | 성공 cell의 F_min 중앙값. workspace 크기와 무관한 품질 |

### 실행 순서

`on_rb5/`에서 실행합니다. 경로에 공백이 있으면 따옴표로 감싸거나 `..\xml`처럼 상대 경로를 씁니다.

```powershell
# 0) 방법 검증 (선택): ramp 속도·timestep·hold test·외력 0 대조 비교
python disturb_validate.py --labels candidate_02 design001 --cells-per-group 6 --workers 8

# 1+2) 설계 전부: workspace -> 상태 저장 -> 외란 (8 workers로 약 5~6시간, 중단해도 같은 명령으로 이어서)
python batch_disturbance.py --xml-dir ..\xml --workers 8
#      final/ 설계(candidate_02 등)는 ..\xml에 없으므로 따로 (상태가 이미 있으면 첫 줄 생략)
python disturb_collect.py --labels candidate_02 --workers 8
python disturb_force.py --labels candidate_02 --workers 8

# 3) 전체 표·그림
python batch_disturbance.py --only-table
python plot_disturbance.py --top-union 10 --map candidate_02

# 4) 상위 설계 정밀 평가: hold test 추가 (순위는 ramp 그대로, 절대값은 hold)
python disturb_force.py --labels design160 design042 design033 --refine --workers 8 --no-summary
python batch_disturbance.py --only-table

# 5) 고른 설계 비교
python compare_force.py --labels design074 design075 design078 candidate_02 --coverage   # 공통으로 잡는 반지름
python compare_force.py --labels design074 design075 design078 candidate_02 --r 40 50    # 같은 물체 크기에서 방향별 F
python compare_force.py --labels design074 design075 design078 candidate_02 --peak --plot-all   # 설계별 최고값과 모든 cell

# 6) 시연: 잡고 들어 올린 뒤 떨어질 때까지 밀기 (viewer)
python demo_disturbance.py --xml ..\xml\design075.xml --dir px
```

판정 기준만 바꿀 때는 시뮬레이션을 다시 하지 않습니다(trace가 저장되어 있음).

```powershell
python disturb_force.py --rescore --slip-mm 30          # 미끄러져도 30 mm 안이면 버틴 것으로
python disturb_force.py --rescore --rot-mode tilt       # 원기둥 축 기울기만 회전으로
python disturb_force.py --rescore --zero-fail keep      # 외력 0 대조 실패 cell도 성공으로 유지
python batch_disturbance.py --only-table                # 표 다시 만들기
```

### 스크립트

#### `disturb_collect.py` — 1단계: 파지 상태 저장

설계의 성공 cell(env, tip)과 hold_mixed cell을 workspace 평가와 똑같이 다시 돌려(같은 장착, IK 경로, 판정), 원래 판정이 재현되는지 확인하고 'lift hold' 끝의 상태(`mj_getState`, `mjSTATE_INTEGRATION`)를 저장합니다.
2단계는 매번 이 상태를 복원해서 시작하므로 모든 방향이 같은 초기 상태에서 출발합니다.

```powershell
python disturb_collect.py --labels candidate_02 --workers 8
python disturb_collect.py --labels candidate_02 --limit 5        # 빠른 시험
```

| 옵션 | 뜻 |
|---|---|
| `--labels` | `all`(기본) 또는 설계 이름. final/(baseline, candidate_02 …)과 references/(design001 …) |
| `--extra-reasons` | 성공 외에 포함할 실패 cell (기본 `hold_mixed`) |
| `--limit N` | 설계마다 처음 N개 cell만 |

출력 `<run>/disturbance/<label>/`: `cells.csv`, `states/*.npz`, `hands/hand_r*.xml`, `meta.json`.
상태 파일에는 장면 서명(손 XML, sources, MuJoCo 버전)이 들어 있어서, 다른 장면이나 버전에서는 2단계가 거부합니다. 원래 평가와 같은 컴퓨터·같은 MuJoCo 버전에서 돌리세요.

#### `disturb_force.py` — 2단계: 방향별 F_max

저장한 상태마다 외력 0 대조 1번과 6방향 ramp를 돌리고, 판정해서 표와 요약을 만듭니다.

```powershell
python disturb_force.py --labels candidate_02 --workers 8
python disturb_force.py --labels design001 --limit 2 --dirs nz py        # 빠른 시험
python disturb_force.py --labels design160 --refine                      # hold test 추가
python disturb_force.py --rescore --slip-mm 10 --rot-deg 0               # 재판정만
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--rate` | 1 | 힘 증가율 [N/s] |
| `--max-mg` | 10 | ramp 상한 [m g]. 상한까지 버티면 censored |
| `--dirs` | 6방향 | `px nx py ny pz nz` |
| `--stop-mm`, `--stop-lost-s` | 30 mm, 0.1 s | 물체가 확실히 빠지면 ramp를 멈추는 기준(재판정 가능 범위) |
| `--no-zero-control` | | 외력 0 대조 생략 |
| `--refine` | | hold test(bisection). 계산량 약 5배 |
| `--rise-s`, `--hold-s`, `--refine-tol-mg` | 0.5 s, 1 s, 0.02 | hold test 설정 |
| `--slip-mm`, `--rot-deg`, `--rot-mode`, `--grace-s` | config | 실패 기준 (재판정에도 사용) |
| `--zero-fail` | `fail` | 외력 0 대조 실패 cell 처리. `keep`이면 표시만 |
| `--metric` | `ramp` | pass level, F_min, robust_area에 쓸 값 (`hold`는 `--refine` 필요) |
| `--levels` | 0.25 … 5 | pass level [m g] |
| `--rescore` | | 시뮬레이션 없이 저장된 trace만 다시 판정 |
| `--no-summary` | | 공유 summary.json/csv를 고치지 않음 |

출력 `<run>/disturbance/<label>/`:

| 파일 | 내용 |
|---|---|
| `traces/*.npz` | cell마다 방향별 trace(시간, 힘, 손 기준 변위·회전, link 접촉력, 관절 변화, 손 이동, 기울기), 외력 0 대조, hold test 시행 |
| `force_dirs.csv` | cell × 방향 한 줄: F_max, 원인, censored, 접촉 형태 변화(tip→env 등), creep 여부, hold 값 |
| `force_cells.csv` | cell 한 줄: 방향별 F, F_min, 가장 약한 방향, pass level, 외력 0 대조 결과 |
| `force_summary.json` | 설계 요약: 성공 cell, J_stable, robust_area, level별 통과, env/tip 방향별 통계 |

#### `disturb_validate.py` — 0단계: 방법 검증

설계마다 그룹(env/tip/hold_mixed)별로 cell 몇 개를 골라 같은 방향을 여러 설정으로 재고 기준 ramp(1 N/s)와 비교합니다.
비교 항목은 ramp 속도(0.5, 2 N/s), timestep 절반, hold test, 외력 0 대조입니다.

```powershell
python disturb_validate.py --labels candidate_02 design001 --cells-per-group 6 --workers 8
```

출력 `<run>/disturbance/_validation/`: `summary.json`(비율 중앙값, 차이, 순위 상관), `rows.csv`, `zero.csv`, `validation.png`.
읽는 법: 순위 상관이 1에 가까우면 그 설정을 바꿔도 설계 순위가 같다는 뜻입니다. 지금까지의 결과는 timestep 수렴, 속도 ±5 %, hold ≈ 0.85 × ramp, 순위 상관 0.98 이상입니다.

#### `batch_disturbance.py` — 설계 전부 한 번에

`--xml-dir`의 설계마다 workspace(`eval_design_rb5.py`) → 상태 저장(`disturb_collect.py`) → 외란(`disturb_force.py`)을 순서대로 돌립니다.
`--workers`개 설계를 동시에 돌리고, 한 설계가 실패해도 나머지는 계속합니다(로그 `disturbance/_logs/<label>.log`). 끝난 단계는 건너뛰므로 중단 후 같은 명령으로 이어집니다.

```powershell
python batch_disturbance.py --xml-dir ..\xml --workers 8
python batch_disturbance.py --xml-dir ..\xml --pattern "design00[2-4].xml" --workers 3   # 몇 개만
python batch_disturbance.py --force-args "--slip-mm 20"                                  # disturb_force 옵션 전달
python batch_disturbance.py --only-table --rank-level 1                                  # 표만 다시
```

출력 `<run>/disturbance/`:

| 파일 | 내용 |
|---|---|
| `all_designs.csv` | 설계당 한 줄: 설계값, 강성·preload, J, J_stable, graspable, 성공 cell, 외력 0 대조 제외 수, robust_area, F_min 중앙값, level별 pass/cond/J_robust, env·tip 방향별 중앙값(ramp, hold) |
| `graspable_designs.csv` | 1단계 결과: 외력 0 대조까지 통과한 성공 cell이 있는 설계 |
| `summary.json`, `summary.csv` | 모든 설계의 force_summary 모음 |

#### `plot_disturbance.py` — 전체 그림

```powershell
python plot_disturbance.py --top-union 10 --map design160 design075
python plot_disturbance.py --labels design001 candidate_02 --pareto-y area
```

| 옵션 | 뜻 |
|---|---|
| `--labels` | 그릴 설계. 기본은 design001, candidate_02, robust_area 상위 3개 |
| `--top-union N` | robust_area 상위 N과 J_robust@1mg 상위 N의 합집합 |
| `--map` | F_min map을 그릴 설계 |
| `--pareto-y` | Pareto y축: `median`(F_min 중앙값, 기본), `area`, `cond` |
| `--value` | heatmap 값: `auto`(모두 hold가 있으면 hold), `ramp`, `hold` |

출력 `<run>/plots/disturbance/`:

| 파일 | 내용 |
|---|---|
| `pareto.png` | 모든 설계의 workspace J_stable vs 파지 품질, Pareto front |
| `directions.png` | 설계별 방향별 F 분포(env/tip box plot, 최대 6개 설계) |
| `direction_heatmap.png` | 설계 × 방향, env와 tip 블록의 중앙값 |
| `survival.png` | 모든 방향에서 F 이상을 버티는 cell 수 (env, tip) |
| `fmin_map_<label>.png` | map의 cell별 F_min (주황 테두리 tip, 진한 회색 외력 0 대조 실패) |
| `selected_designs.csv` | 고른 설계의 설계값과 지표 비교표 |

#### `compare_force.py` — 고른 설계 비교

F는 물체 크기와 잡는 위치에 크게 좌우되고 설계마다 잡는 cell이 다르므로, 비교 조건을 정해서 봅니다.

```powershell
$L = "design074 design078 design076 design075 design138 design033 design163 candidate_02".Split(" ")
python compare_force.py --labels $L --coverage          # 반지름별 env/tip cell 수, 모두가 잡는 r
python compare_force.py --labels $L --r 40 50           # 같은 반지름에서 방향별 F 중앙값
python compare_force.py --labels $L --r 40 --x 0        # 가운데 cell만 (+x, -x 대칭)
python compare_force.py --labels $L --r 40 --common     # 모든 설계가 잡는 (x, y) cell만
python compare_force.py --labels $L --peak              # 설계별 최고값 (표 + 그래프)
python compare_force.py --labels $L --plot-all          # 모든 cell의 방향별 F 그래프
python compare_force.py --peak-csv runs\rb5_cmaes_final\plots\disturbance\peak_all_hold.csv   # 저장된 CSV로 그래프만
```

| 모드 | 내용 | 출력 (`<run>/plots/disturbance/`) |
|---|---|---|
| `--coverage` | 설계 × 반지름의 env/tip 성공 cell 수와 모든 설계가 잡는 반지름 | 콘솔 |
| `--r R …` | 그 반지름의 cell에서 방향별 F 중앙값 (`--group` env / tip / all, `--x`, `--common`) | `compare_<tag>.csv`, `.png` |
| `--peak` | 설계별로 모든 성공 cell 중: 방향별 최댓값과 그 cell, F_min 최대 cell(모든 방향을 동시에 가장 잘 버티는 파지), 상위 k개 평균, 90번째 백분위. 방향별 승자. 상한에 걸린 값은 동률로 표시 | `peak_<group>_<value>.csv`, `peak_dirs_*.png`(설계별 막대, 값과 그 cell의 반지름), `peak_by_direction_*.png`(방향별 설계 비교) |
| `--plot-all` | 모든 성공 cell의 방향별 F와 F_min을 점으로 (env 파랑, tip 주황, 상한 도달 △) | `all_cells_dirs_*.png`, `all_cells_fmin_*.png` |

값은 모든 설계에 hold test가 있으면 hold, 아니면 ramp를 씁니다(`--value`로 지정 가능). 방향별 최댓값은 방향마다 다른 cell에서 나올 수 있습니다.
상한(10 m g)에 걸린 방향까지 순위를 내려면 그 설계들을 `disturb_force.py --max-mg 30 --refine`으로 다시 잽니다.

#### `demo_disturbance.py` — viewer 시연

설계 XML(또는 run의 설계 이름)과 방향을 주면 실험과 같은 설정으로 잡고 들어 올린 뒤, 그 방향으로 물체가 떨어질 때까지 밀면서 viewer에 힘 화살표와 현재 값을 표시합니다.

```powershell
python demo_disturbance.py --xml ..\xml\design075.xml --dir px
python demo_disturbance.py --xml ..\xml\design075.xml --dir nx --cell 60 0 35
python demo_disturbance.py --label candidate_02 --dir py --skip-lift
python demo_disturbance.py --xml ..\xml\design075.xml --dir px --headless --video demo_075_px.mp4 --record disturb
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--xml` / `--label` | | 설계 XML(stage0 또는 RB5 손 형식) / run의 설계 이름 |
| `--dir` | | 방향 (`px nx py ny pz nz` 또는 `--dir=-x`) |
| `--cell R X Y` | | cell 지정 [mm]. 없으면 실험에서 그 방향으로 가장 잘 버틴 cell, 결과가 없으면 workspace 최고 cell |
| `--skip-lift` | | 저장된 들어 올린 직후 상태에서 바로 시작 |
| `--rate`, `--cap-mg` | 1 N/s, 30 m g | 힘 증가율, 상한 (실험보다 높게 밀어 실제로 떨어뜨림) |
| `--drop-mm` | 30 | 이만큼 밀리거나 접촉을 0.1 s 잃으면 떨어진 것으로 봄 |
| `--after-s` | 1.5 | 떨어진 뒤 힘 없이 더 보여 주는 시간 |
| `--speed` | 1 | viewer 재생 배속 |
| `--camera` | `close` | 미는 동안 손 가까이 (`wide`: 전체) |
| `--headless`, `--video`, `--record` | | 창 없이, MP4 저장(`pip install imageio imageio-ffmpeg`), 전체/미는 구간만 |
| `--trace` | | 미는 구간을 CSV로 저장 |

콘솔에는 실험 기준에 처음 걸린 힘(`F_max`), 물체가 떨어진 힘, 그리고 실험 결과가 있으면 같은 cell·방향의 실험값(ramp, hold)이 나옵니다.
`-y`(palm 쪽)는 물체가 palm에 받쳐 대개 떨어지지 않고 상한에서 멈춥니다.

### 결과 폴더 (`runs/rb5_cmaes_final/`)

```
references/<label>*                  1단계 workspace map (설계마다)
disturbance/
├── all_designs.csv, graspable_designs.csv, summary.json, summary.csv
├── _validation/                     0단계 결과
├── _logs/<label>.log                batch 로그
└── <label>/
    ├── cells.csv, states/, hands/, meta.json                 1단계
    └── traces/, force_dirs.csv, force_cells.csv,
        force_meta.json, force_summary.json                   2단계
plots/disturbance/                   pareto, directions, direction_heatmap, survival, fmin_map_*,
                                     selected_designs.csv, compare_*, peak_*, all_cells_*
```

### 주의

- 상태 저장과 외란은 같은 MuJoCo 버전·같은 컴퓨터에서 돌립니다. 다른 환경에서는 경계 cell의 판정이 바뀔 수 있고, 장면 서명이 다르면 2단계가 멈춥니다.
- F는 원기둥 질량 중심에 가하는 순수한 힘입니다(토크 없음). 사람이 옆면을 미는 경우처럼 표면에 힘이 걸리면 결과가 다를 수 있습니다.
- 외력 0 대조 실패를 실패로 간주하는 규칙 때문에 `success`, `J_stable`이 workspace 실험의 값보다 작습니다. 이전 방식은 `--zero-fail keep`입니다.

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
