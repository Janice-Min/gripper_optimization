# gripper_optimization

텐던 구동 2지 그리퍼의 설계를 MuJoCo에서 평가하고, RB5-850e에 장착해 외란에 대한 파지 유지 성능을 비교하는 코드입니다.
이 README는 `disturbance_object/`의 설계 생성, workspace 평가, 외란 실험 A·B·C와 통합 비교를 설명합니다.

- **Workspace**: 어떤 크기·위치의 원기둥을 잡고 들어 올릴 수 있는가
- **실험 A — 힘 ramp**: 물체에 천천히 커지는 힘을 가할 때 방향별 유지 한계 `F_max`는 얼마인가
- **실험 B — 펄스**: 짧은 충격 후에도 물체를 계속 잡고 있는가
- **실험 C — 팔 흔들기**: RB5를 가속·감속할 때 물체를 계속 잡고 있는가
- **통합 비교**: A·B·C를 모두 견디는 설계와 파지 위치는 무엇인가

## 주요 폴더

| 경로 | 역할 |
|---|---|
| [`disturbance_object/finger_samples/`](disturbance_object/finger_samples) | 형상 변수 4개 샘플링, 설계 JSON·MJCF 생성, 손 단독 workspace 평가 |
| [`disturbance_object/rb5_xml_pick/`](disturbance_object/rb5_xml_pick) | RB5 pick-and-lift 재검증, 파지 상태 저장, 외란 A/B/C, 순위·그림·통합 비교·viewer |

현재 새 실험 설계는 `design0001` ~ `design0301`입니다. 최상위 `xml/`의 `design001` ~ `design204`와는 별개입니다.
workspace의 `r`은 원기둥 반지름, `x`는 좌우 오프셋, `y`는 palm 면에서 원기둥 **표면**까지의 거리입니다.
표와 cell 이름의 길이는 mm, 설계 JSON·시뮬레이션 XML의 길이는 m를 사용합니다.

---

## disturbance_object/ — 설계 샘플링과 외란 실험 A·B·C

실행 흐름은 **설계 JSON 생성 → MJCF 생성 → 손 단독 workspace 평가 → RB5 pick-and-lift 재검증·상태 저장 → 외란 A/B/C → 설계 비교**입니다.
이미 저장된 결과를 분석할 때는 앞의 시뮬레이션을 다시 돌릴 필요가 없습니다.

### finger_samples/ — 설계와 손 단독 workspace

아래 경로는 `disturbance_object/finger_samples/` 기준입니다.

| 파일·폴더 | 설명 |
|---|---|
| `make_finger_designs.py` | `L_total`, `L_p/L_total`, `G`, `th_open_p`의 4개 형상 변수를 maximin LHS(기본 300개) 또는 전요인 grid로 샘플링. 기준 설계 `design0001`을 추가하고 설계 JSON·인덱스·설정 저장 |
| `generate_mjcf.py` | 설계 JSON의 링크 길이·간격·열림 각도·강성·preload를 `gripper_stage0.xml`에 반영해 MJCF 생성. `--design_input`으로 JSON 하나 또는 폴더 지정 |
| `evaluate_grasp_workspace.py` | 반지름 `r`과 palm–물체 표면 거리 `y`를 바꾸며 손 단독 파지 평가. 초기 겹침·도달 범위를 검사하고, 중력 OFF에서 0.3 s 장력 ramp 후 중력 ON으로 유지. 양쪽 접촉 비율·전체 이동·수직 slip으로 성공 판정 |
| `gripper_stage0.xml` | 형상·스프링·텐던·물체를 정의하는 MJCF 템플릿 |
| `designs/designNNNN.json` | 설계값 9개: `L_p`, `L_d`, `G` [m], `th_open_p`, `th_open_d` [deg], `k_p`, `k_d` [N·m/rad], `tau0_p`, `tau0_d` [N·m] |
| `designs/designs_index.csv`, `designs/DOE_configuration.json` | 전체 설계 변수표, 샘플링 방법·범위·seed·고정 변수·단위 |
| `xml/designNNNN.xml` | 생성된 301개 손 MJCF. 최상위 `xml/designNNN.xml`의 204개 설계와 별개 |
| `workspace_results/grasp_map.csv` | 모든 설계·cell의 성공/실패, 원인, 접촉 비율·변위·slip·파지 형태 |
| `workspace_results/graspable_points.csv` | 성공한 cell만 모은 표. 다음 단계 `run_rb5_workspace.py`의 입력 |
| `workspace_results/grasp_summary.csv`, `grasp_config.json` | 설계별 성공 cell 수·파지 가능한 r/y 범위와 평가 설정 |

현재 샘플의 범위는 총 길이 100–150 mm, 근위 링크 비율 0.3–0.7, 간격 `G` 50–100 mm, 근위 열림 각도 10–45°입니다.
원위 열림 각도는 0°로 고정하고 강성·preload도 고정합니다. `design0001`은 이 조건의 기준 설계이므로 원본 stage0의 원위 열림 각도와는 다를 수 있습니다.
기본 workspace는 `r = 10:5:100 mm`, `y = 0:5:140 mm`, `x = 0`, 장력 10 N, XML의 물체 질량 0.3 kg·높이 38 mm를 사용합니다.

### rb5_xml_pick/ — 실행 코드

아래 경로는 `disturbance_object/rb5_xml_pick/` 기준입니다. A·C·viewer가 B의 공통 함수를 import하므로 관련 파일을 같은 폴더에 둡니다.

| 파일 | 역할·주요 출력 |
|---|---|
| `run_rb5_xml.py` | XML 하나를 RB5 손목에 장착하고 IK 접근 → 텐던 파지 → 150 mm 상승 → 유지. 최상위 `rb5_xml_pick/run_rb5_xml.py`와 같은 코드이며, workspace 실행기가 이 모듈을 재사용 |
| `run_rb5_workspace.py` | `xml/`와 `workspace_results/graspable_points.csv`를 읽어 cell마다 RB5로 재검증. 최종 유지 판정이 PASS인 cell의 **상승 직후** 상태를 `grasp_cells/`에 저장. `--resume`, `--dry-run`, `--viewer`, `--show` 지원 |
| `collect_cell_xmls.py` | 성공 cell의 `scene.xml`을 `grasp_cells_xml/design0001_r50y05.xml`처럼 모으고, `<custom>`에 원래 cell ID와 summary를 포함. `index.csv` 생성, pick 결과·설정도 복사. `--check`로 `lifted` key 로딩 확인 |
| `disturb_force_rb5.py` | A: 외력 ramp로 방향별 `F_max`, cell의 `F_min`, `robust_area` 계산. `--refine`으로 일정한 힘의 hold test, `--rescore`로 재판정 |
| `disturb_impulse_rb5.py` | B: 방향·크기별 펄스와 외력 0 대조를 실행해 held·level·`impact_area` 계산. A·C가 쓰는 cell 검색/로딩, mesh 경로 복구, 손 좌표계, 접촉력·trace 처리도 제공 |
| `disturb_shake_rb5.py` | C: 손 자세를 고정한 채 RB5를 왕복 이동시켜 가속·감속. IK와 속도·역동역학 feedforward를 사용하고 파지 유지 및 실제 팔 추종 성능 평가. `shake_area`, `arm_limited`, `infeasible` 등을 기록 |
| `rank_force_designs.py` | A의 `robust_area` 순위·cell bootstrap 순위 안정성·방향 분포·Pareto 분석. `A_design_ranking.csv`와 그림, B 결과가 있으면 `A_vs_B.csv/.png` |
| `rank_impulse_designs.py` | B의 `impact_area` 순위·bootstrap·Pareto 분석. `B_design_ranking.csv`와 순위·방향별 생존 그림 |
| `plot_impulse_rb5.py` | B 전용 그림. 설계별 `(r, y)` 지도(`--map`), cell 시간 응답(`--trace`), 방향·크기별 held 비율. 출력 `results_B/impulse[_tag]/plots/` |
| `plot_disturb_results.py` | A/B/C의 `all_designs.csv`를 자동 구분해 순위·workspace/품질 Pareto·생존 곡선·방향 heatmap·실패 원인·설계 변수 그림. 출력 `figs_disturb/A/`, `B/`, `C/` |
| `compare_abc.py` | 세 실험의 설계 점수를 정규화해 `ABC_score`, Pareto layer, 공통 상위 설계 계산. cell 표가 있으면 같은 cell끼리 매칭해 `joint_area`, `joint_pass@k`, `joint_J@k`도 계산. 출력 `figs_disturb/ABC/` |
| `view_cell.py` | 성공 cell XML의 `lifted` key를 바로 여는 간단한 viewer. 두 번째 위치 인자로 다른 key 지정 가능 |
| `view_pick_disturb.py` | cell 하나의 집기부터 펄스 외란까지 재생. `--from saved`는 저장 상태부터 시작하며, `--video`로 영상 저장 |
| `view_shake_rb5.py` | C와 같은 계획·시뮬레이션·판정 함수를 사용해 팔 흔들기 재생. 명령/실측 가속도 화살표, 물체 slip·회전·접촉력 표시. 영상·시간 응답 그림·CSV 저장 지원 |
| `gripper_disturbance_experiment_B.md` | 새 계열 실험 B의 목적·초기값·측정량 설명 |
| `rb5_xml_pick` (확장자 없는 문서) | cell XML 묶음 방식의 사용법과 기존 `state.npz` 방식과의 비교 검증 기록 |

### 입력·상태·결과 폴더

| 경로 | 내용 |
|---|---|
| `xml/`, `workspace_results/` | `finger_samples/`에서 생성한 설계 XML과 workspace 결과의 사본. 새로 생성한 입력은 `--xml-dir`, `--workspace-dir`로 직접 지정 가능 |
| `grasp_cells/<design>/<cell>/` | `scene.xml`, `state.npz`, `summary.json`. XML에는 상승 직후 자세인 `<key name="lifted">`가 들어 있고, NPZ는 MuJoCo integration state를 저장 |
| `grasp_cells/rb5_pick_results.csv`, `rb5_pick_config.json` | 시도한 전체 cell의 RB5 PASS/FAIL과 원인, pick-and-lift 설정 |
| `grasp_cells_xml/` | `lifted` key와 cell summary가 포함된 XML 묶음, `index.csv`, 복사한 pick 결과·설정. XML 하나가 cell 하나 |
| `results_A/force[_tag]/` | 전체 `all_designs.csv`, `direction_stats.csv`, `force_meta.json`, `force_summary.json`. 설계별 `traces/*.npz`, `force_dirs.csv`, `force_cells.csv` |
| `results_B/impulse[_tag]/` | 전체 `all_designs.csv`, `direction_stats.csv`, `impulse_meta.json`, `impulse_summary.json`. 설계별 `traces/*.npz`, `impulse_trials.csv`, `impulse_cells.csv` |
| `results_C/shake[_tag]/` | 전체 `all_designs.csv`, `direction_stats.csv`, `shake_meta.json`, `shake_summary.json`. 설계별 `traces/*.npz`, `shake_trials.csv`, `shake_cells.csv` |
| `figs_disturb/A/`, `B/`, `C/` | 실험별 순위 CSV와 그림 |
| `figs_disturb/ABC/` | `ABC_ranking.csv`, `ABC_summary.txt`, cell 매칭 시 `ABC_joint_cells.csv`, 실험 간 비교·통합 지도 그림 |
| `results/`, `validation/` | 단일 XML pick-and-lift의 결과와 입력/실패 사례 검증 자료 |
| `sources/`, `assets/rb5_850e/` | RB5·작업 장면 XML, mesh와 라이선스 |
| `folder_list.csv` | 폴더 목록을 저장한 자료 |

A/B/C는 기본적으로 `grasp_cells_xml/`을 먼저 찾고, 없으면 `grasp_cells/`을 읽습니다. `--cells-dir`로 사용할 상태 묶음을 명시할 수 있습니다.
XML 묶음은 `lifted` key, 폴더 묶음은 `state.npz`를 복원하고, 외력 없이 0.5 s 안정화한 상태 S0에서 각 시행을 시작합니다.
상태를 다시 수집한 경우 `collect_cell_xmls.py`도 다시 실행하거나 `--cells-dir grasp_cells`를 지정해 최신 상태를 사용하세요.

### 외란 조건과 지표

아래는 **새 결과 폴더에서 실행할 때의 CLI 기본값**입니다. 저장된 결과의 조건은 `force_meta.json`, `impulse_meta.json`, `shake_meta.json`에 남습니다.

| 실험 | 입력·기본 조건 | 주요 지표 |
|---|---|---|
| A | 물체 질량중심에 1 mg/s ramp, 상한 10 mg. `--refine`: 0.5 s 상승 + 1 s 일정 힘 유지 | 방향별 `F_max`, 가장 약한 방향의 `F_min`, `robust_area` |
| B | 물체 질량중심에 50 ms 직사각형 펄스, 크기 0.5/1/2/3 mg, 펄스 후 10 s 관찰 | 시행별 held, 방향별 level, cell별 `level_min`, `impact_area` |
| C | 손 가속도 0.5/1 g, 왕복 `+a → -a → +a`, 단계 기준 시간 100 ms·전환 ramp 20 ms, 운동 후 10 s 관찰 | held·`level_min`, `shake_area`, 팔 추종·포화·충돌 지표 |

`mg`는 밀리그램이 아니라 물체 무게 `m × g`입니다. 0.3 kg 물체에서 1 mg = 2.943 N이며, C의 1 g는 손 가속도 9.81 m/s²를 뜻합니다.
현재 저장된 B 결과는 기본 크기 외에 0.25/0.75/1.5 mg도 포함합니다(`impulse_meta.json` 참고).

- 방향은 손 좌표계의 `px/nx`(jaw 폭), `py/ny`(손 바깥/palm 쪽), `pz/nz`(원기둥 축 방향)입니다. `--dirs axes`는 6방향, `edges`는 두 축 대각선 12방향, `all`은 이 둘을 합친 18방향입니다.
- C의 방향 이름은 **손이 처음 움직이는 방향**입니다. 물체의 관성력은 손 가속도와 반대이며, 왕복 한 번에 양방향 하중이 걸립니다.
- 기본 실패 기준은 손 기준 slip 5 mm, 회전 5°, 한쪽 손가락 접촉력 0.1 N 미만이 20 ms 지속, 손 밖 물체와의 접촉 0.05 N입니다. 외란 0 대조 실패 cell은 `zero_fail`로 제외합니다.
- C는 계획 IK·충돌 검사 실패를 `infeasible`, 실제 가속도·위치·자세 추종/포화 등의 한계를 `arm_limited`로 구분하므로 이를 파지 실패와 함께 해석하지 않습니다.
- `pass@k`는 모든 평가 방향에서 k를 버틴 cell 수, `cond@k`는 성공 cell 중 통과 비율입니다. `J` 계열 지표는 env + 0.2 × tip으로 집계합니다. B·C의 level은 더 작은 시험 크기도 모두 통과한 최대 크기입니다.
- `robust_area`, `impact_area`, `shake_area`는 성공 cell의 최소 유지 level을 더한 값입니다. workspace 크기의 영향이 있으므로 cell당 평균·중앙값, 공통 cell 비교도 함께 봅니다. 크기·방향·판정 조건이 같은 결과끼리 비교하세요.

`compare_abc.py`의 기본 설계 점수는 area, 정규화는 각 실험 최댓값으로 나누기, 결합은 세 정규화 점수의 최솟값입니다.
기본 `--rank-by auto`는 cell 표가 있으면 `joint_area`, 없으면 `ABC_score`로 정렬합니다.
`joint_level`은 같은 cell의 A 유지 힘[mg], B 펄스 level[mg], C 가속도 level[g]의 최솟값을 사용한 통합 지표입니다.
설계 요약표만 있는 경우의 `joint_ub@k`는 공통 통과 cell 수의 **상한**이고, cell을 직접 매칭한 `joint_pass@k`와 다릅니다.

### 실행 예시

명령은 표시한 폴더에서 실행합니다. 새 실험의 기본 의존성은 `mujoco numpy pandas matplotlib`이고, 영상 저장에는 `imageio imageio-ffmpeg`가 추가로 필요합니다.
저장 상태를 재사용할 때는 cell의 `summary.json`에 기록된 `source_mujoco_version`과 실행 환경을 맞춥니다.

**저장된 결과의 그림·순위·통합 비교** (`disturbance_object/rb5_xml_pick/`에서):

```powershell
python -m pip install mujoco numpy pandas matplotlib
python plot_disturb_results.py results_A/force results_B/impulse results_C/shake --designs-index ../finger_samples/designs/designs_index.csv
python rank_force_designs.py
python rank_impulse_designs.py
python plot_impulse_rb5.py --map design0001
python compare_abc.py results_A/force results_B/impulse results_C/shake --designs-index ../finger_samples/designs/designs_index.csv
```

**저장된 cell로 빠르게 시험** (같은 폴더, `smoke` tag로 결과 분리):

```powershell
python disturb_force_rb5.py --designs design0001 --limit 3 --workers 2 --tag smoke
python disturb_impulse_rb5.py --designs design0001 --limit 3 --workers 2 --tag smoke
python disturb_shake_rb5.py --designs design0001 --limit 3 --workers 2 --tag smoke
python view_cell.py grasp_cells_xml/design0001_r50y05.xml
python view_pick_disturb.py design0001/r050.0_y005.0 --from saved --dir py --mag 1
python view_shake_rb5.py design0001/r050.0_y005.0 --dir py --mag 1
```

전체 cell은 `--designs`와 `--limit`를 빼고 실행합니다. A/B/C는 완료된 trace/시행을 재사용하므로 중단 후 같은 명령으로 이어집니다.
판정만 바꾸려면 예를 들어 `python disturb_impulse_rb5.py --tag smoke --rescore --slip-mode residual`을 사용합니다.
펄스 길이·힘 증가율 등 시뮬레이션 조건을 바꿀 때는 새 `--tag` 또는 `--out`을 사용합니다.
`--clean`은 해당 `results_X` 상위 폴더 안의 기존 결과 전체를 삭제하므로 이어서 실행할 때는 넣지 않습니다.

**새 설계부터 생성** (`disturbance_object/finger_samples/`에서 시작, 새 폴더 사용):

```powershell
python make_finger_designs.py --n 300 --out-dir designs_new
python generate_mjcf.py --design_input designs_new --output xml_new
python evaluate_grasp_workspace.py --xml-dir xml_new --out-dir workspace_results_new --workers 6 --resume
cd ../rb5_xml_pick
python run_rb5_workspace.py --xml-dir ../finger_samples/xml_new --workspace-dir ../finger_samples/workspace_results_new --out-dir grasp_cells_new --workers 6 --resume
python collect_cell_xmls.py grasp_cells_new --out grasp_cells_xml_new --check
python disturb_force_rb5.py --cells-dir grasp_cells_xml_new --out results_A_new --designs-index ../finger_samples/designs_new/designs_index.csv --workers 6
python disturb_impulse_rb5.py --cells-dir grasp_cells_xml_new --out results_B_new --designs-index ../finger_samples/designs_new/designs_index.csv --workers 6
python disturb_shake_rb5.py --cells-dir grasp_cells_xml_new --out results_C_new --designs-index ../finger_samples/designs_new/designs_index.csv --workers 6
python compare_abc.py results_A_new/force results_B_new/impulse results_C_new/shake --out figs_disturb/ABC_new
```

`collect_cell_xmls.py`는 지정한 출력 폴더의 기존 XML을 정리한 뒤 다시 만듭니다.
저장된 장면의 mesh 경로가 다른 컴퓨터의 절대 경로라면 A/B/C의 `--mesh-dir assets/rb5_850e/meshes`로 현재 경로를 지정할 수 있습니다.
`--allow-missing-collision`은 충돌 모델을 생략하는 확인용 옵션이므로 정식 비교 실험에는 사용하지 않습니다.

---

## 다른 실행 패키지

| 문서 | 내용 |
|---|---|
| [`on_rb5/README.md`](on_rb5/README.md) | 기존 RB5 workspace·CMA-ES 최적화·외란 A/B의 폴더별 코드와 실행 방법 |
| [`rb5_xml_pick/README_KO.md`](rb5_xml_pick/README_KO.md) | XML 하나를 RB5에 장착해 집고 들어 올리는 단독 실행 패키지 |

RB5 kinematics·mesh는 Rainbow Robotics [`rbpodo_ros2`](https://github.com/RainbowRobotics/rbpodo_ros2)의 `rb5_850e` 기반입니다.
라이선스는 각 실행 폴더의 `assets/rb5_850e/LICENSE`, `sources/Class_IR_LICENSE`를 참고하세요.
