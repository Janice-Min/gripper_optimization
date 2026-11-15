# rb5_xml_pick — RB5에 텐던 그리퍼를 붙여 물체를 집고, 외란 실험 A·B·C를 하는 폴더

`design001.xml` 계열의 독립형 그리퍼 XML을 RB5-850e 손목에 붙여 책상 위 원기둥을 집고 들어 올립니다.
설계마다 잡을 수 있는 (r, y) cell을 RB5로 다시 집어 **들어 올린 직후 상태**를 `grasp_cells/`에 저장하고, 이 상태에서 외란 실험 A, B, C를 합니다.

| 실험 | 외란 | 묻는 것 |
|---|---|---|
| A | 물체에 천천히 커지는 힘 (ramp) | 방향마다 얼마나 큰 힘까지 놓치지 않는가 (F_max) |
| B | 물체에 50 ms 짧은 충격 (pulse) | 충격을 받은 뒤 10 s 동안 계속 잡고 있는가 |
| C | 들어 올린 뒤 팔을 흔듦 | (계획 단계, 아래 "실험 C" 참고) |

RB5 mesh와 로봇·환경 XML은 `assets/`, `sources/`에 들어 있습니다. 실제 RB5 통신·구동 코드는 없습니다.

---

## 1. 전체 흐름

```
설계 생성 (make_finger_designs.py)  →  designs/*.json
XML 변환 (generate_mjcf.py)          →  xml/design0001.xml …        ┐ 이 폴더에 복사해서 씀
2지 workspace 평가 (evaluate_grasp_workspace.py) → workspace_results/ ┘
        │
        ▼
RB5 pick (run_rb5_workspace.py → run_rb5_xml.py)
        → grasp_cells/<design>/<cell>/   들어 올린 직후 상태 (PASS cell만)
        │
        ├─ 실험 A  disturb_force_rb5.py   → results_A/force/
        ├─ 실험 B  disturb_impulse_rb5.py → results_B/impulse/
        └─ 실험 C  (계획)
```

설계 생성·XML 변환·2지 workspace 평가는 이 폴더 밖에서 하고, 결과인 `xml/`와 `workspace_results/`만 이 폴더로 가져옵니다.

---

## 2. 폴더와 파일

```
rb5_xml_pick/
├── run_rb5_xml.py              XML 하나를 RB5에 붙여 집고 들어 올리기 (장착, IK, 충돌 검사, 성공 판정)
├── run_rb5_workspace.py        잡을 수 있는 (r, y) cell마다 run_rb5_xml.py를 불러 집고, PASS cell 상태를 grasp_cells/에 저장
├── design001.xml               예제 그리퍼 XML
├── requirements.txt            실행에 필요한 패키지
├── requirements-video.txt      영상 저장용 (imageio, imageio-ffmpeg)
├── assets/rb5_850e/            RB5 mesh (visual_obj/, collision/), LICENSE
├── sources/                    로봇·환경 XML, Class_IR_LICENSE
├── results/                    run_rb5_xml.py 한 번 실행의 결과 (통합 장면, summary.json, trace.csv)
│
├── xml/                        설계 XML (design0001.xml … design0301.xml)
├── workspace_results/          2지 workspace 평가 결과 (graspable_points.csv, grasp_config.json …)
├── grasp_cells/                RB5 pick 결과 = 외란 실험의 출발 상태
│   ├── rb5_pick_results.csv    시도한 모든 cell의 PASS/FAIL과 이유
│   ├── rb5_pick_config.json    pick 실행 설정 (장력, 단계별 시간, 받침대 높이 …)
│   ├── <design>/<cell>/        PASS cell 하나 (예: design0001/r060.0_y015.0)
│   │   ├── scene.xml           RB5 + 손 + 원기둥 장면, keyframe lifted
│   │   ├── state.npz           들어 올린 직후의 MuJoCo integration state
│   │   ├── summary.json        cell 정보, 물체 위치, 상승량, 링크 접촉력, 판정
│   │   └── log.txt             실행 로그
│   ├── _failed/                실패 cell (--save-failed를 줬을 때만)
│   └── _tmp/                   실행 중 쓰는 임시 폴더
│
├── disturb_impulse_rb5.py      실험 B: 펄스 외란 시뮬레이션, 판정, 표
├── plot_impulse_rb5.py         실험 B 그림 (levels 분포, 방향 heatmap, 설계 산점도, (r, y) 지도, 시간 응답)
├── rank_impulse_designs.py     실험 B 설계 순위 (bootstrap 순위 안정성, Pareto), 그림 2장
├── results_B/impulse/          실험 B 결과
│
├── disturb_force_rb5.py        실험 A: ramp 외력 시뮬레이션, 판정, 표, hold test
├── rank_force_designs.py       실험 A 설계 순위, 그림, 실험 B와 비교
├── results_A/force/            실험 A 결과
│
└── view_pick_disturb.py        cell 하나를 집어 올리고 hold 뒤 펄스를 주는 viewer
```

`disturb_force_rb5.py`, `view_pick_disturb.py`는 `disturb_impulse_rb5.py`의 함수(장면 읽기, 손 좌표계, 접촉력)를 가져다 쓰므로 같은 폴더에 있어야 합니다.

---

## 3. 공통 조건 (RB5 pick, 실험 A·B)

**장면.** RB5-850e 손목 flange에 100 mm 어댑터로 palm을 붙입니다. 손가락은 palm −x 방향으로 뻗습니다(`--forward-angle 180`). 원기둥은 세계 좌표 (0.60, 0) m, 높이 80 mm 받침대 위에 둡니다.

**물체와 장력.** 원기둥 높이 38 mm, 질량 0.3 kg(mg = 2.94 N), 텐던 장력 10 N(`grasp_config.json`과 같음). 외란 실험에서 힘 크기는 mg 단위로 씁니다.

**cell 좌표.** workspace 평가와 같습니다. r = 원기둥 반지름, y = palm 앞면(x = −1 mm)에서 원기둥 표면까지 거리, x = 좌우 오프셋(0 고정). palm 좌표의 원기둥 중심은 `(−0.001 − y − r, x, 0)` m입니다.

**손 좌표계** (외란 방향)

| 방향 | 이름 | 의미 |
|---|---|---|
| ±x | `px`, `nx` | jaw 방향. +x는 upper 손가락 쪽(palm +y) |
| +y / −y | `py` / `ny` | 손 밖으로 빼내기(손가락이 뻗는 방향) / palm 쪽으로 밀기 |
| ±z | `pz`, `nz` | 원기둥 축 = palm z. 이 장착에서는 +z가 위쪽 |

대각선 12방향은 `pxpy`, `nxpz` …(`--dirs edges`), 전체 18방향은 `--dirs all`입니다.

**접촉 형태.** 손가락 링크 4개가 모두 0.1 N 이상으로 닿으면 env(enveloping), 양쪽 손가락이 닿지만 4개 모두는 아니면 tip입니다.

**외란 실험의 출발 상태 S0.** `state.npz`를 `mj_setState`로 복원하고, 외력 없이 0.5 s 둡니다(저장 상태는 상승이 끝난 직후라 팔이 아직 조금 움직임). 마지막 0.1 s 링크 접촉력 평균으로 시작 접촉 형태를 정합니다. 팔은 저장된 관절 명령을 유지하고, 텐던 장력과 중력은 그대로입니다. A와 B는 같은 S0에서 시작하므로 같은 cell의 결과를 직접 비교할 수 있습니다.

**실패 판정** (A·B 공통, 손 기준)

- slip: 물체 변위 ≥ 5 mm
- rotation: 물체 회전 ≥ 5° (`--rot-mode tilt`이면 원기둥 축 기울기만)
- lost: 한쪽 손가락 접촉력 합 < 0.1 N이 20 ms 이상
- collision: 물체가 손 밖의 물체(팔 링크, 받침대 등)와 0.05 N 이상으로 닿음

**외력 0 대조.** 같은 S0를 외력 없이 약 10 s 둡니다. 여기서 실패한 cell은 외란이 없어도 못 버티므로 `zero_fail`로 두고 지표에서 뺍니다.

---

## 4. 단계 0 — RB5 pick과 grasp_cells

### 4.1 `run_rb5_xml.py`: XML 하나 집기

```powershell
python -m pip install -r requirements.txt
python run_rb5_xml.py --xml design001.xml
python run_rb5_xml.py --xml "C:\Users\User\Desktop\BEAR lab\design002.xml"   # 공백 있는 경로는 따옴표
python run_rb5_xml.py --xml design001.xml --headless
python run_rb5_xml.py --xml design001.xml --video pick_lift.mp4              # requirements-video.txt 필요
```

동작: settle 0.6 s → 하강 3.0 s → 접근 2.5 s → 정지 0.5 s → 텐던 장력 증가 1.5 s → 파지 유지 1.0 s → 150 mm 상승 3.0 s → 2 s 유지.
성공 조건은 마지막 1초 동안 **목표 상승량의 80% 이상**, **양쪽 손가락 접촉**, **받침대 접촉 해제**를 모두 유지하는 것입니다. 4개 링크 모두 접촉(`four_link_contact`)은 따로 기록하고, 성공 조건에는 넣지 않습니다.
종료 시 `PASS`(종료코드 0) 또는 `FAIL`(1)과 원인을 출력하고, `results/`에 `rb5_attached_scene.xml`, `summary.json`, `trace.csv`를 씁니다.
옵션과 XML 호환 범위는 부록 A에 있습니다. macOS에서 viewer를 쓸 때는 `mjpython`으로 실행합니다.

### 4.2 `run_rb5_workspace.py`: cell마다 집고 상태 저장

`workspace_results/graspable_points.csv`의 cell마다 원기둥 중심을 `--grasp-xy`로 넘겨 `run_rb5_xml.py`로 집습니다(자동 위치 조정은 끔). PASS한 cell만 `grasp_cells/<design>/<cell>/`에 저장하고, 시도한 모든 cell은 `rb5_pick_results.csv`에 기록합니다.

```powershell
python run_rb5_workspace.py --dry-run
python run_rb5_workspace.py --designs design0001 --viewer
python run_rb5_workspace.py --workers 6
python run_rb5_workspace.py --workers 6 --resume
python run_rb5_workspace.py --show grasp_cells/design0001/r055.0_y020.0
```

- 받침대: 손목 링크가 손바닥 평면 위아래로 약 5 cm 나와 있어서, 38 mm 원기둥을 책상에 바로 두면(`--support-height 0`) 손목이 책상과 부딪힙니다.
- `scene.xml`에는 mesh의 절대 경로가 들어 있습니다. 폴더를 옮기면 외란 스크립트에 `--mesh-dir <rb5_xml_pick>\assets\rb5_850e\meshes`를 주면 됩니다.

### 4.3 결과

| 항목 | 값 |
|---|---|
| 설계 | 301개 (그중 PASS cell이 있는 설계 296개) |
| 시도한 cell (2지 workspace에서 잡을 수 있던 cell) | 4442 |
| RB5 pick PASS | 2545 (57.3 %) |
| FAIL 원인 | 상승 부족·접촉 상실·받침대 접촉 1707, 양쪽 접촉 없음 117, 열린 손이 원기둥과 겹침 60, 팔 경로 충돌 11, 의도치 않은 충돌 2 |

두 평가는 조건이 다릅니다. 2지 workspace 평가는 중력 없이 손을 닫고, RB5 pick은 중력이 있는 상태에서 받침대 위 물체를 잡아 팔이 가속하며 들어 올립니다.

---

## 5. 실험 A — 천천히 커지는 외력에 대한 유지 한계 F_max

on_rb5의 실험 A(`A/disturb_force.py`)와 같은 정의를 이 폴더의 `grasp_cells/`에 적용합니다.

### 절차

1. S0에서 시작해 물체 질량중심에 `xfrc_applied`로 **F(t) = rate·t**를 손 좌표 방향으로 가합니다. 기본 rate는 1 mg/s(= 2.94 N/s), 상한은 10 mg입니다. 방향당 최대 10 s입니다.
2. 공통 실패 판정 중 처음 걸린 순간의 F가 **F_max**입니다. 상한까지 버티면 F_max = 10 mg, censored로 표시합니다. 중력은 켜 두므로 F_max는 중력에 더해 견디는 외력입니다.
3. 물체가 30 mm 밀리거나 접촉을 0.1 s 잃으면 그 방향을 멈춥니다. 그래서 판정 기준을 slip 30 mm, lost 0.1 s까지는 `--rescore`로 시뮬레이션 없이 바꿀 수 있습니다.
4. 외력 0 대조 10 s. 대조가 ramp보다 먼저 실패한 방향은 `creep_limited`로 표시합니다.
5. ramp 동안 링크 접촉력의 0.1 s 이동평균으로 접촉 형태를 계속 판정해, F_max 시점의 형태와 처음 형태에서 바뀐 순간의 힘을 기록합니다.
6. (`--refine`) **hold test**: 일정한 힘을 0.5 s 동안 부드럽게 올린 뒤 1 s 유지했을 때 버티는 최대 힘을 bisection으로 찾습니다(허용오차 0.02 mg).

rate를 mg/s로 정한 이유: on_rb5 A의 1 N/s는 0.1 kg 물체에서 약 1 mg/s였습니다. 지금 물체는 0.3 kg이므로 같은 시간 척도가 되도록 1 mg/s로 맞췄습니다.

### 실행

```powershell
python disturb_force_rb5.py --dry-run
python disturb_force_rb5.py --workers 6                                     # 전체 (끊기면 같은 명령으로 이어서)
python disturb_force_rb5.py --rescore --slip-mm 30                          # 판정 기준만 바꿔 다시 판정
python disturb_force_rb5.py --designs design0023 design0111 --refine        # hold test 추가
python disturb_force_rb5.py --rescore --metric hold --designs design0023 design0111
python disturb_force_rb5.py --tag edges --designs design0023 design0111 --dirs all   # 대각선 포함 18방향은 따로
python rank_force_designs.py                                                # 순위표, 그림, 실험 B와 비교
```

### 지표

| 지표 | 정의 |
|---|---|
| F_min, weakest_dir | cell의 방향별 F 중 최솟값과 그 방향 |
| robust_area | Σ_성공 cell min(F_min, 10 mg). 설계 순위 (workspace 크기 × 강건성) |
| F_min_median_mg | 성공 cell의 F_min 중앙값. workspace 크기와 분리한 파지 품질 |
| pass@k, cond@k, J_robust@k | 모든 방향에서 F ≥ k mg인 cell 수 / 성공 cell 중 비율 / env_pass + 0.2·tip_pass (k = 0.25 0.5 0.75 1 1.5 2 3 5) |
| F_<dir>_median_env/tip_mg | 방향별 F 중앙값 (env, tip 따로) |
| F_min_hold_median_mg | hold test를 한 설계의 F_min 중앙값 |

### 결과 파일

```
results_A/force/                    (--tag edges 이면 force_edges/)
├── force_meta.json                 시뮬레이션 설정, 마지막 판정 기준
├── force_summary.json              합계, robust_area 상위 10, F_min 중앙값 상위 10 (성공 cell ≥ 5)
├── all_designs.csv                 설계당 한 줄 (위 지표)
├── direction_stats.csv             설계 × env/tip × 방향: 중앙값, 사분위, censored, creep_limited, 값 목록
├── A_design_ranking.csv            robust_area 순위, bootstrap 순위 안정성 (rank_force_designs.py)
├── A_fig_directions.png            방향별 F_max 분포 (env | tip)
├── A_fig_design_ranking.png        상위 32개 robust_area + 성공 cell 수 vs F_min 중앙값
├── A_vs_B.csv, A_vs_B.png          같은 cell·방향에서 (B의 펄스 크기 k − A의 F_max)별 B 유지 비율, 설계 순위 상관
└── <design>/
    ├── traces/<cell>.npz           방향별 ramp 기록 (10 ms 간격), 외력 0 대조, hold test
    ├── force_dirs.csv              cell × 방향 한 줄: F_max, 원인, censored, 접촉 형태 변화, hold 값
    └── force_cells.csv             cell 한 줄: 방향별 F, F_min, 약한 방향, 외력 0 대조, pass@k
```

### 결과

(전체 실행 결과를 정리해 여기에 넣습니다.)

---

## 6. 실험 B — 짧은 충격 후 파지 유지

### 절차

1. S0에서 시작해 물체 질량중심에 `xfrc_applied`로 **F = k·mg를 50 ms** 동안 줍니다(사각 펄스, `--shape halfsine` 가능). k = 0.5, 1, 2, 3 mg, 손 좌표 6방향. 방향은 S0의 손 좌표계로 정해 펄스 동안 고정합니다.
2. 펄스 시작부터 펄스 후 10 s까지 공통 실패 판정 중 하나도 일어나지 않으면 **held**입니다.
3. 실패는 시작 시점으로 나눕니다: **pulse**(펄스 중), **transient**(펄스 후 0.5 s 안), **delayed**(그 뒤).
4. 물체가 30 mm 밀리거나 접촉을 0.1 s 잃으면 시행을 멈춥니다(`--rescore` 범위).
5. cell마다 외력 0 대조 1회 + 6방향 × 4크기 = 25회입니다.

### 실행

```powershell
python disturb_impulse_rb5.py --dry-run
python disturb_impulse_rb5.py --workers 6                                       # 전체 (끊기면 같은 명령으로 이어서)
python disturb_impulse_rb5.py --rescore --slip-mode residual                    # 펄스 후 0.5 s부터의 변위만 셈
python disturb_impulse_rb5.py --mags 0.25 0.75 1.5 --workers 6                  # 크기 추가: 새 시행만 돌림
python disturb_impulse_rb5.py --pulse-ms 20 --mags 1 2 3 5 --tag p20 --workers 6  # 설정이 다른 실험은 따로
python plot_impulse_rb5.py
python rank_impulse_designs.py
```

### 지표

| 지표 | 정의 |
|---|---|
| level (cell × 방향) | k 이하의 펄스를 **모두** 버틴 가장 큰 k. 가장 작은 펄스에서 실패하면 0 |
| level_min, weakest_dir | 방향별 level 중 최솟값과 그 방향 |
| pass@k, cond@k | 모든 방향에서 level ≥ k인 성공 cell 수 / 성공 cell 중 비율 |
| J_impulse@k | env_pass@k + 0.2 · tip_pass@k |
| impact_area | Σ_성공 cell level_min [mg]. 설계 순위 (workspace 크기 × 강건성) |
| type_changed | 버틴 시행 중 끝의 접촉 형태가 시작과 다른 수 |

### 결과 파일

```
results_B/impulse/                  (--tag p20 이면 impulse_p20/)
├── impulse_meta.json               시뮬레이션 설정, 마지막 판정 기준
├── impulse_summary.json            합계, impact_area 상위 10
├── all_designs.csv                 설계당 한 줄: 성공·zero_fail cell, impact_area, pass/cond/J_impulse@k,
│                                    실패 구간·원인 수, 방향 × 크기별 held 비율
├── direction_stats.csv             설계 × env/tip × 방향 × 크기: 시행 수, held 수
├── B_design_ranking.csv            impact_area 순위, bootstrap 순위 안정성, Pareto (rank_impulse_designs.py)
├── B_fig_direction_survival.png    방향별 유지 비율 vs 펄스 크기 (env | tip)
├── B_fig_design_ranking.png        상위 32개 impact_area + 성공 cell 수 vs 평균 level_min
├── plots/                          plot_impulse_rb5.py 그림
└── <design>/
    ├── traces/<cell>.npz           시행마다 시간 기록 (펄스 후 0.55 s까지 5 ms, 그 뒤 0.1 s 간격)
    ├── impulse_trials.csv          cell × 방향 × 크기 한 줄: held, 원인, 구간, 실패 시각, 변위, 회전, 끝 접촉 형태
    └── impulse_cells.csv           cell 한 줄: 방향별 level, level_min, weakest_dir, 외력 0 대조, pass@k
```

### 결과 (기본 설정: 50 ms 사각 펄스, 0.5 / 1 / 2 / 3 mg, 6방향)

**cell 수.** RB5 pick PASS 2545 cell 중 499개(19.6 %)가 외력 0 대조에서 10 s 안에 실패했습니다. 남은 성공 cell은 2046개(env 783, tip 1263)이고, 시행은 61,080회입니다. 성공 cell이 1개 이상인 설계는 273개입니다.

**접촉 형태별 level_min** (모든 방향에서 버틴 최대 펄스)

| | 0 | 0.5 mg | 1 mg | 2 mg | 3 mg | ≥ 0.5 mg | ≥ 1 mg |
|---|---|---|---|---|---|---|---|
| env (783) | 39.0 % | 33.7 % | 24.3 % | 2.9 % | 0.1 % | 61 % | 27 % |
| tip (1263) | 62.2 % | 35.5 % | 2.4 % | 0 | 0 | 38 % | 2.4 % |

**방향별 1 mg 펄스 유지 비율** (`B_fig_direction_survival.png`)

| 방향 | 위 (+z) | palm 쪽 (−y) | +x | −x | 빼내기 (+y) | 아래 (−z) |
|---|---|---|---|---|---|---|
| env | 100 % | 91 % | 45 % | 47 % | 37 % | 32 % |
| tip | 100 % | 92 % | 19 % | 20 % | 4.5 % | 8 % |

- tip의 가장 약한 방향은 빼내기(+y)와 아래(−z), env는 아래(−z)와 빼내기(+y)입니다. 위(+z)와 palm 쪽(−y)은 두 형태 모두 강합니다.
- 실패 시작 시점: 펄스 중 77 %, 펄스 후 0.5 s 안 21 %, 그 뒤 1.6 %. 원인: slip 76 %, 회전 24 %, 접촉 상실·충돌 각 0.2 % 미만.
- 버틴 시행 중 216건은 끝의 접촉 형태가 시작과 달랐습니다.

**설계 순위** (`B_design_ranking.csv`, `B_fig_design_ranking.png`)

| 순위 | 설계 | impact_area | 성공 cell (env) | 평균 level_min [mg] |
|---|---|---|---|---|
| 1 | design0023 | 6.5 | 16 (2) | 0.41 |
| 2 | design0111 | 6.0 | 10 (8) | 0.60 |
| 3 | design0008 | 5.5 | 10 (8) | 0.55 |
| 4 | design0043 | 5.5 | 7 (2) | 0.79 |
| 5–18 | 0139, 0205, 0219, 0136, 0053, 0009, 0117, 0124, 0247, 0213, 0033, 0060, 0116, 0081 | 5.0 | 3–20 | 0.25–1.67 |

- 5~18위 14개, 19~32위 14개가 각각 같은 impact_area입니다. 펄스 크기가 0.5 mg 간격이라 점수가 0.5 단위로만 갈립니다.
- cell bootstrap에서 top 10에 들 확률이 50 %를 넘는 설계는 design0023(64 %), design0111(51 %)뿐입니다. 순위를 더 가르려면 `--mags 0.25 0.75 1.5`를 추가로 돌립니다.
- 상위 18개 설계의 성공 cell 중 env 비율 중앙값은 0.82, 나머지 설계는 0.12입니다.
- 성공 cell 수와 평균 level_min의 Spearman 상관은 −0.31입니다. 둘 다 좋은 쪽 경계(Pareto front): design0124, 0219, 0136, 0139, 0043, 0111, 0117, 0023, 0116, 0301.

---

## 7. 실험 C — 들어 올린 뒤 팔 흔들기

물체에 직접 힘을 주는 A·B와 달리, 팔(손목)을 움직여 그 관성으로 생기는 외란을 버티는지 봅니다.

(코드, 절차, 결과를 정리해 여기에 넣습니다.)

---

## 8. viewer — `view_pick_disturb.py`

cell 하나를 RB5로 집어 들어 올린 뒤, `--hold-s`(기본 5 s)가 지나면 물체에 펄스를 주고 그 뒤 10 s를 보여 줍니다.

```powershell
python view_pick_disturb.py design0001/r060.0_y015.0                       # 집기부터, +y 1 mg
python view_pick_disturb.py design0001/r060.0_y015.0 --dir px --mag 2
python view_pick_disturb.py design0001/r060.0_y015.0 --from saved          # 저장 상태에서 시작 (실험 B와 같은 출발점)
python view_pick_disturb.py design0001/r060.0_y015.0 --video pick_px2.mp4
python view_pick_disturb.py design0001/r060.0_y015.0 --headless            # 판정만 출력
```

- `--from pick`(기본)은 장면의 `task_home` 자세에서 run_rb5_xml.py와 같은 단계와 시간으로 다시 집습니다. 팔은 palm 목표 경로를 IK로 따라가고, 열림 보조 PD와 장력 ramp는 `rb5_pick_config.json` 값을 씁니다. run_rb5_xml.py를 부르는 것이 아니라 같은 절차를 다시 구현한 것이라, 접촉이 아슬아슬한 cell(특히 tip)은 저장 상태와 조금 다를 수 있습니다.
- `--from saved --hold-s 0.5`이면 실험 B의 시행과 같습니다.
- 화면 왼쪽 위에 단계, 링크 접촉력, 손 기준 미끄러짐·회전, 판정이 나오고, 펄스 중에는 주황색 화살표로 힘 방향을 그립니다. 펄스 앞뒤는 `--slowmo 0.1` 배속으로 보여 줍니다. Space로 일시정지합니다.

---

## 부록 A. `run_rb5_xml.py` 옵션과 XML 호환 범위

### 옵션

`run_rb5_xml.py`의 `if __name__ == '__main__':` 아래에 모든 `parser.add_argument(...)`가 있습니다. 원하는 옵션의 `default=`만 바꾸거나 실행 명령에 옵션을 추가합니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--xml` | 같은 폴더의 design001.xml | 입력 XML |
| `--tension` | 20 | 실제 텐던 장력, N; 다중 텐던이면 각각 같은 장력 |
| `--lift` | 0.15 | 상승 거리, m |
| `--radius` | None | 원기둥 반지름, m; None이면 XML 값 |
| `--object-height` | None | 원기둥 전체 높이, m; None이면 XML 값 |
| `--mass` | None | 물체 질량, kg; None이면 XML 값 |
| `--grasp-xy` | None | palm 좌표계의 물체 중심 x/y, m; 입력하면 자동 위치 조정 해제 |
| `--auto-grasp` | True | 초기 겹침을 피하도록 파지 위치를 손가락 진행 방향으로 자동 이동 |
| `--object-xy` | 0.60 0 | 세계 좌표계에서 물체를 놓을 위치, m |
| `--support-height` | 0.08 | 받침대 높이, m |
| `--mount-length` | 0.10 | 손목 flange부터 palm까지 어댑터 길이, m |
| `--approach-distance` | 0.12 | 물체 뒤쪽에서 접근하는 거리, m |
| `--ramp-time` | 1.5 | 장력 증가 시간, s |
| `--lift-time` | 3.0 | 상승 시간, s |
| `--hold-time` | 2.0 | 상승 후 유지 시간, s |
| `--speed` | 1.0 | viewer 재생 속도 |
| `--camera-distance` | 1.45 | 카메라 거리 |
| `--headless` | False | viewer 생략 |
| `--exit-after-task` | False | 동작 후 창을 자동으로 닫음 |
| `--output` | results | 통합 XML과 결과를 저장할 폴더 |

`run_rb5_workspace.py`는 장력을 `grasp_config.json` 값(10 N)으로 넘깁니다.

### XML에서 가져오는 내용

- palm 아래 손가락 형상, 길이, 관절 위치와 각도 제한, 관절 stiffness / damping / springref / armature
- tendon의 site / wrapping geom / sidesite / gear
- `open` keyframe의 관절각 (없으면 관절 springref)
- 원기둥의 반지름, 높이, 질량, 마찰과 손–물체 상대 위치

손가락 link·관절 이름은 코드에 고정되어 있지 않습니다. 두 finger branch의 collidable geom으로 접촉력을 집계합니다. 통합 장면에서만 입력 XML 이름 앞에 `hand__`를 붙이고, 원본 XML은 수정하지 않습니다.

### 호환 범위 (design001 형식)

1. `palm`이라는 고정 body와 그 아래 두 articulated finger branch
2. 이름이 있는 hinge joint와 이름이 있는 접촉 geom
3. `object_body`에 직접 속한 cylinder geom 한 개와 freejoint 한 개
4. 원기둥 축이 palm z와 평행
5. 이름과 0이 아닌 gear를 가진 tendon motor (여러 개면 같은 장력)
6. object weld 등 equality constraint 없음
7. default / contact / tendon이 참조하는 대상이 손·물체 안에 있음

body 이름만 다르면 `--palm-body`, `--object-body`로 지정합니다. 손가락이 palm −x로 뻗으면 `--forward-angle 180`(+x는 0, +y는 90)입니다.

### 파지 보조와 보정

- 원본 손–물체 위치에서 겹치면 자동 탐색으로 파지 위치를 손가락 진행 방향으로 옮깁니다(design001: x = −75 mm에서 약 4.1 mm 겹쳐 x = −90 mm로 이동). `--no-auto-grasp`이면 원본 위치 그대로입니다.
- 접근 중 손가락이 관성으로 닫히지 않도록 관절 PD 열림 보조를 씁니다. 장력을 주기 시작하면 0.3 s 동안 부드럽게 해제하고, 파지·상승 중에는 0입니다(`--open-kp`, `--open-kd`, `--open-torque-limit`).
- 장력은 `ctrl = −T / gear`로 바꿉니다(gear = −0.5, T = 20 N이면 ctrl = 40). 통합 XML에서 ctrlrange를 넓혀 지정한 장력이 그대로 들어가게 합니다.
- 손목은 position actuator로 움직이고 RB5 링크의 gravity compensation을 유지합니다. 물체를 손에 붙이는 weld, 물체 위치 덮어쓰기, 위쪽 외력은 쓰지 않습니다.

---

## 부록 B. 모델 출처와 라이선스

RB5 및 실험실 모델은 앞서 사용한 `rb5_tendon_pick_lift` 패키지의 모델을 포함했습니다. RB5 kinematics, inertia, mesh는 Rainbow Robotics `rbpodo_ros2`의 `rb5_850e` 기반입니다.

- https://github.com/RainbowRobotics/rbpodo_ros2
- RB5 라이선스: `assets/rb5_850e/LICENSE`
- 환경 소스 라이선스: `sources/Class_IR_LICENSE`
