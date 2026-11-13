# 외란 실험 A — 천천히 커지는 외력에 대한 파지 유지 한계 F_max

on_rb5의 workspace 실험은 **얼마나 넓은 범위를 잡을 수 있는가**를 쟀다.
실험 A는 **잡은 물체를 얼마나 세게 밀거나 당겨도 놓치지 않는가**를 잰다.
대상은 `../xml`의 설계 204개 전부이고, workspace 실험과 같은 RB5 장면, 같은 격자(792 cell), 같은 판정 규칙을 쓴다.

## 1. 절차

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 0. 방법 검증 | `A/disturb_validate.py` | 일부 cell에서 ramp 속도, timestep, hold test, 외력 0 대조를 비교해 F_max가 시뮬레이션 설정이 아니라 파지의 성질인지 확인 |
| 1. 파지 가능한 설계 찾기 | `common/eval_design_rb5.py` → `grasp/disturb_collect.py` | 설계마다 workspace map(792 cell)을 평가하고, 성공 cell(env, tip)과 hold_mixed cell을 다시 돌려 **'lift hold'가 끝난 순간의 전체 상태**(`mj_getState`, `mjSTATE_INTEGRATION`)를 저장. 외력 0 대조(2단계)까지 통과한 성공 cell이 하나라도 있으면 graspable |
| 2. 외란 | `A/disturb_force.py` | 저장한 상태를 복원해 6방향으로 외력을 천천히 키우며 방향별 F_max 측정, 외력 0 대조 |
| 정리 | `A/batch_disturbance.py`, `A/plot_disturbance.py` | 204개를 한 번에 돌리고 표·그림 생성 |

`batch_disturbance.py`가 1, 2단계를 설계마다 순서대로 돌린다. 중단된 경우 같은 명령을 다시 실행하면 이어서 진행한다.

## 2. 정의

**좌표계 (손 기준, workspace map과 같음)**

| 방향 | 의미 | 버티는 원리 |
|---|---|---|
| ±x | 손가락이 닫히는 방향(jaw 폭). +x는 R 손가락 쪽 | 실 하나가 양쪽 손가락을 지나므로, 물체가 한쪽으로 밀리면 반대 손가락이 따라온다. 주로 관절 스프링이 버틴다 |
| +y / −y | 손 밖으로 빼내기 / palm 쪽으로 밀기 | +y: env는 distal link가 걸어 주고, tip은 마찰만으로 버틴다 |
| +z / −z | 원기둥 축 = 손가락 평면 밖, 위 / 아래 | 마찰과 torsional friction(condim 4) |

**하중.** 물체 질량 중심에 `xfrc_applied`로 F(t) = rate·t를 가한다(기본 1 N/s, 상한 10 mg). 팔은 마지막 관절 명령을 유지하고, 텐던은 **장력 고정**(모터 = 힘 제어, 20 N)이다. 중력은 켜 두므로 F_max는 **중력에 더해 견디는 외력**이다(m = 0.1 kg, mg = 0.981 N). F는 mg 단위로 보고한다.

**실패 판정.** 아래 중 하나가 처음 일어난 sample의 F가 F_max다.
- slip: 손(grasp site) 기준 물체 변위 ≥ 5 mm (`hold_displacement_mm`)
- rotation: 손 기준 물체 회전 ≥ 5° (`hold_rotation_deg`)
- lost: 한쪽 손가락의 접촉력 합 < 0.1 N 상태가 20 ms 이상 지속

판정은 world가 아니라 **손 기준**이다. 팔이 같이 밀린 양은 `hand_mm` 열에 따로 기록한다.
상한까지 버티면 F_max = 상한이고 censored로 표시한다(실제 값은 더 크다).
ramp는 물체가 30 mm 밀리거나 접촉을 0.1 s 잃을 때까지 계속하므로, 저장된 trace로 판정 기준만 바꿔 `--rescore`할 수 있다.

**외력 0 대조 (creep).** 같은 상태를 외력 없이 ramp 전체 시간(10 s) 동안 둔다. MuJoCo soft contact는 마찰 한도 안에서도 천천히 미끄러진다. **외력 없이도 판정에 걸리는 성공 cell은 실패로 간주한다**(그룹 `zero_fail`). 이 cell은 성공 cell 수, robust_area, pass level, J_stable에서 빠지고, `fmin_map`에서는 진한 회색으로 표시된다. env와 tip 모두에 같은 규칙을 적용한다. `--zero-fail keep`으로 바꾸면 표시만 하고 성공 cell로 남긴다(`--rescore` 가능). 대조가 ramp보다 먼저 실패한 방향은 `creep_limited`로 표시한다.

**접촉 형태 변화.** ramp 동안 link 접촉력의 0.1 s 이동평균으로 env / tip / mixed / none을 계속 판정해, F_max 시점의 형태와 처음 형태가 바뀐 힘(예: tip → env)을 기록한다.

**hold test (`--refine`).** 일정한 힘을 0.5 s 동안 부드럽게 올린 뒤 1 s 유지했을 때 위 판정에 걸리지 않는 **최대 일정 힘**을 bisection으로 찾는다(허용오차 0.02 mg). ramp 속도와 무관한 정의지만 계산량이 ramp의 약 5배라, 검증 단계에서 ramp와 순위가 같은지 확인하는 데 쓴다.

## 3. 지표

| 지표 | 정의 | 용도 |
|---|---|---|
| 방향별 F (env / tip 따로) | `direction_stats`: 그룹·방향별 중앙값, 사분위, censored·creep 개수 | 질문 그대로의 답: 각 grasp 형태가 어느 방향에 약한가 |
| F_min | cell의 6방향 중 최솟값(가장 약한 방향) | cell 단위 강건성 |
| robust_area | Σ_성공 cell min(F_min, 상한) [mg] | 설계 순위. 못 잡는 cell은 0이므로 **204개 설계를 같은 792 cell 위에서** 비교한다. 생존곡선 아래 면적이고, workspace 크기에 비례해 커진다 |
| F_min_median | 성공 cell의 F_min 중앙값 | workspace 크기와 무관한 파지 품질 |
| pass@k mg, J_robust@k | 모든 방향에서 F ≥ k·mg인 cell 수, 그 cell로 계산한 J | 특정 요구 하중이 있을 때 |
| conditional robustness | 통과 cell / 성공 cell | workspace 크기와 분리한 파지 품질 |
| J_stable | 외력 0 대조를 통과한 성공 cell로 계산한 workspace J (J_rerun은 규칙 적용 전) | 1단계 J를 "외력 없이도 유지되는 파지"로 고친 값. `pareto.png`의 x축 |

최종 그림은 workspace J_stable과 파지 품질(기본 F_min_median, `--pareto-y area|cond`로 변경)의 산점도와 Pareto front다(`pareto.png`). workspace가 최대인 설계가 외란에도 강한지, 아니면 둘 사이에 trade-off가 있는지를 보여 준다. robust_area를 y축으로 쓰면 workspace 크기 효과가 섞여서 trade-off가 가려진다.

## 4. 실행

```powershell
cd on_rb5        # 모든 명령은 on_rb5/에서. --run 기본값은 grasp/results/rb5_cmaes_final
# 0) 방법 검증: 이미 상태가 저장된 설계로 (design001, candidate_02)
python A/disturb_validate.py --labels candidate_02 design001 --workers 8
# 1+2) 204개 전부 (workspace -> 상태 저장 -> 외란)
python A/batch_disturbance.py --xml-dir ..\xml --workers 8
# 정리 그림
python A/plot_disturbance.py --map candidate_02
# 상위 설계 정밀 평가: 두 순위(robust_area, J_robust@1mg) 상위 N의 합집합에 hold test 추가
python A/disturb_force.py --labels <설계들> --refine --workers 8
python A/batch_disturbance.py --only-table
python A/plot_disturbance.py --top-union 10 --map <관심 설계>
# 판정 기준만 바꿔 다시 판정 (시뮬레이션 없음), 예: 미끄러져도 30 mm 안이면 버틴 것으로
python A/disturb_force.py --rescore --slip-mm 30
python A/batch_disturbance.py --only-table
```

## 5. 산출물

```
grasp/results/rb5_cmaes_final/
├── references/<label>*                    1단계 workspace map (설계마다)
└── grasp_states/<label>/
    ├── cells.csv, states/*.npz            1단계: 다시 돌린 cell과 저장된 상태 (A·B·C가 공통으로 씀)
    └── hands/, meta.json

A/results/rb5_cmaes_final/
├── all_designs.csv                        설계당 한 줄: 설계값, J, graspable, robust_area, 방향별 env/tip 중앙값, pass/J_robust
├── graspable_designs.csv                  1단계 결과: 성공 cell이 있는 설계
├── summary.json / summary.csv
├── _validation/                           0단계 결과 (summary.json, rows.csv, validation.png)
├── _logs/                                 batch_disturbance.py 설계별 로그
├── <label>/
│   ├── traces/*.npz                       2단계: 방향별 trace, 외력 0 대조, hold test
│   ├── force_dirs.csv                     cell × 방향 한 줄
│   ├── force_cells.csv                    cell 한 줄 (F_min, 약한 방향, 외력 0 대조)
│   └── force_summary.json
└── plots/                                 pareto, directions, survival, fmin_map_<label>,
                                           direction_heatmap (설계 × 방향, env|tip), selected_designs.csv,
                                           compare_*, peak_*, all_cells_* (compare_force.py)
```

## 6. 같이 고친 것: stage0 XML 변환 검사

`batch_disturbance.py`의 1단계(`eval_design_rb5.py`)가 design002~004에서
"joint angles under a 0.3 N tendon pull differ by up to 1.67 deg"로 멈췄다. 204개 중 168개가 같은 이유로 멈췄다.
원인은 설계 차이가 아니라 검사 방법이었다. preload가 있는 관절은 0.3 N에서 open stop에 눌린 채로 있는데, 눌린 깊이가 joint limit의 solref에 따라 다르다(stage0 XML 기본값 0.02, RB5 손 템플릿 0.002).
각도로 비교하는 방식은 다른 이유로도 성립하지 않는다. 무른 손가락은 닫힘 리밋에 닿는데 리밋 값이 두 모델에서 다르고(1.4/1.5 rad, 1.92 rad), break-away 장력 근처의 preload 관절은 쌍안정이다.
그래서 검사를 **손가락 운동 범위 전체(5×5 자세)에서 텐던 moment arm 비교**로 바꿨다. 강성과 preload는 원래 값 비교로 확인하므로, moment arm까지 같으면 정적 거동이 같다.
204개 모두 차이 2×10⁻⁷ mm 안으로 통과하고, 일부러 wrap 반지름을 1 mm 바꾼 손은 잡아낸다.

## 7. 0단계 검증 결과 (4개 설계, 22 cell × 6방향)

`disturb_validate.py --labels candidate_02 design001 design002 design003 --cells-per-group 3`, 기준 = 1 N/s ramp.

| 비교 | 비율 중앙값 (변형/기준) | \|ΔF\|/F 중앙값 | 90 % | 순위 상관 (방향) | 순위 상관 (cell F_min) |
|---|---|---|---|---|---|
| timestep ½ | 1.003 | 0.8 % | 8 % | 0.996 | 0.985 |
| 0.5 N/s | 0.946 | 6.8 % | 29 % | 0.994 | 0.988 |
| 2 N/s | 1.070 | 8.1 % | 44 % | 0.992 | 0.985 |
| hold test | 0.830 | 17 % | 53 % | 0.991 | 0.982 |

- timestep은 수렴했다.
- ramp 값은 속도에 약하게 의존한다(속도를 절반으로 줄이면 약 5 % 낮아진다). 마찰 creep이 시간에 따라 쌓이기 때문이다.
- hold test(일정 힘 1 s 유지)는 ramp보다 약 17 % 낮다(env 14 %, tip 20 %). 하지만 **순위 상관이 0.98 이상**이다.
- 따라서 204개 순위는 1 N/s ramp로 매기고, **절대값을 말할 때는 "ramp 값의 약 0.83배가 일정 하중 유지 한계"** 라고 보정하거나, 상위 설계만 `--refine`으로 다시 잰다.
- 외력 0 대조: 표본에서는 env 0/5, tip 3/12가 10 s 안에 판정에 걸렸다. 4개 설계 전체로 보면 tip cell 106개 중 30개(28 %)이고, 대부분 초당 0.5~1° 정도 도는 회전 creep으로 2~9 s 사이에 5°에 도달했다. env는 68개 중 1개다. 모든 경우 ramp가 creep보다 먼저 실패했으므로(`creep_limited` 0), F_max 값 자체는 힘에 의한 실패다. 다만 tip 파지의 약 4분의 1은 외력이 없어도 10 s를 버티지 못한다.

Windows(MuJoCo 3.3.7)에서 `--labels candidate_02 design001`(17 cell)로 다시 돌린 결과도 같았다: timestep ½ 비율 1.000, 0.5/2 N/s 0.950/1.055, hold 0.859, 순위 상관 모두 0.978 이상. 외력 0 대조에서는 env 0/5, tip 4/8이 실패했다.

예비 결과 (1 N/s ramp, 기본 판정, 외력 0 대조 실패 = 실패):

| 설계 | workspace J → J_stable | 성공 cell (env/tip), 대조 실패로 제외 | 전체 F_min 중앙값 | 모든 방향 ≥ 1 mg cell | robust_area |
|---|---|---|---|---|---|
| candidate_02 | 69.5 → 67.3 | 76 (65/11), 제외 7 (env 1, tip 6) | 0.53 mg | 0 | 41.4 |
| design003 | 9.0 → 6.8 | 34 (0/34), 제외 11 (tip) | 0.68 mg | 7 | 24.9 |
| design002 | 7.2 → 5.8 | 25 (1/24), 제외 7 (tip) | 0.42 mg | 2 | 14.6 |
| design001 | 3.6 → 2.4 | 8 (1/7), 제외 6 (tip) | 0.90 mg | 3 | 9.3 |

규칙을 적용하면 tip 성공 cell의 약 28 %가 빠진다(env는 68개 중 1개). 약하게 잡혀서 저절로 도는 tip 파지가 빠지므로, 남은 cell의 F_min 중앙값은 오히려 올라간다(design001 0.61 → 0.90 mg).
workspace 최적 설계(candidate_02)는 잡을 수 있는 cell은 가장 많지만, **모든 방향에서 1 mg를 버티는 cell은 하나도 없다.**
env 파지의 약한 방향은 거의 항상 ±x(jaw 방향)다. 실 하나가 양쪽 손가락을 지나서, 한쪽으로 밀면 반대쪽 손가락이 따라오기 때문이다.
+z(위로)는 중력 때문에 모든 그룹에서 가장 강하다.

## 8. 정해야 할 것 (기본값으로 돌아가며, 대부분 `--rescore`로 시뮬레이션 없이 바꿀 수 있다)

| 항목 | 기본값 | 대안 | 바꾸는 방법 |
|---|---|---|---|
| 회전 판정 | 전체 회전 ≥ 5° (workspace 'lift hold'와 같은 기준) | 원기둥 축 기울기만(축 둘레 spin은 허용): robust_area +1~6 %, pass@1 mg 7 → 10 (design003) | `--rescore --rot-mode tilt` |
| tip → env 전환 | 5 mm 넘게 미끄러지면 전환 중이라도 실패 | "결국 놓치지 않았다" 기준 | `--rescore --slip-mm 30` |
| 설계 순위 지표 | robust_area (workspace 크기 × 강건성) | F_min_median (크기와 분리한 품질), J_robust@k mg (요구 하중 k가 정해져 있을 때) | `all_designs.csv`의 열, `batch_disturbance.py --only-table --rank-level k` |
| 절대값 | 1 N/s ramp | 상위 설계만 hold test (순위는 ramp 유지, hold 값은 `F_hold_*`, `pass_hold_*` 열과 heatmap에) | `disturb_force.py --labels ... --refine` |
| 외력 0 대조에서 실패한 cell | **실패로 간주 (결정됨)**, env·tip 모두 | 표시만 하고 성공으로 유지 | `--rescore --zero-fail keep` |
| 상한 | 10 mg (이전 5 mg; design001·002의 env cell 일부 방향은 10 mg에서도 censored) | – | `--max-mg` |

## 9. 계산량

이 환경 기준으로 설계 1개당 workspace(792 cell) 약 12분, 상태 저장 + 외란 약 1~3분이 걸린다(코어 1개).
204개 × 약 13분 = 약 44 core-hour이므로 `--workers 8`이면 약 5~6시간이다. 대부분은 1단계 workspace 시간이다.
`--refine`을 전체에 켜면 외란 단계가 약 5배가 된다.

## 10. 고른 8개 설계의 18방향 결과 (6축 + 모서리 대각선 12)

design074, 078, 076, 075, 138, 033, 163, candidate_02에 `A/disturb_force.py --dirs edges --append`로 대각선 12방향(`±x±y`, `±x±z`, `±y±z`, 성분 같은 크기)을 더했다. 값은 1 N/s ramp, 성공 cell(env + tip, 외력 0 대조 실패 제외) 중 **방향마다 최댓값**이다(방향마다 다른 cell일 수 있음). 대각선에는 hold test가 없으므로 ramp 값이고, 일정 하중 유지 한계는 약 0.85배로 본다.

| 설계 | 6축 최약 | 18방향 최약 | 6축 최약보다 약한 대각선 |
|---|---|---|---|
| design075 | +y 3.93 | +y 3.93 | 없음 |
| design078 | +y 3.24 | +y 3.24 | 없음 |
| design074 | +y 3.34 | +y 3.34 | 없음 |
| design076 | +y 3.16 | +y 3.16 | 없음 |
| design138 | +y 2.18 | +y 2.18 | 없음 |
| design033 | +y 1.71 | −x+y 1.60 | +x−z, −x+y, −x−z |
| design163 | +y 1.48 | −x−z 1.23 | −x−z |
| candidate_02 | +y 1.58 | −x−z 1.08 | −x+y, −x−z |

(단위 m g.) θ_open,p = 17° 계열(074/075/076/078)은 대각선을 더해도 가장 약한 방향이 +y(빼내기) 그대로이고, `+y`가 들어간 대각선(±x+y 3.9~4.8, +y−z 3.5~4.8)이 그다음으로 약하다. 반면 design033, 163, candidate_02는 대각선에서 6축보다 약한 방향이 나와, 6축만으로는 최약 방향을 과대평가한다. `−y+z`는 8개 모두 거의 상한(≥ 10, candidate_02 9.0)이다.
그림: `plot_force_arrows.py --peak-csv .../peak_all_ramp.csv` (설계별 원기둥 + 화살표, 설계 × 방향 heatmap).
(`--dirs edges --append`와 `plot_force_arrows.py`는 GitHub 사본(36159177)에는 없고 로컬 코드에만 있다. 폴더 재구성 후에는 `A/`로 옮겨야 한다.)