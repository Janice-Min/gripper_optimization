# 외란 실험 B — 짧은 충격(펄스)에 대한 파지 유지

실험 A는 **천천히 커지는 힘**을 얼마나 버티는지(F_max)를 쟀다.
실험 B는 **짧은 충격을 받은 뒤에도 놓치지 않고 계속 잡고 있는지**를 잰다.
대상은 실험 A에서 graspable로 나온 설계(`A/results/rb5_cmaes_final/graspable_designs.csv`)이고, A와 같은 파지 상태(`grasp/results/rb5_cmaes_final/grasp_states/`)를 그대로 쓴다. 장면, 격자, 판정 규칙도 A와 같다.

## 1. 절차

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 0. 준비 | `grasp/disturb_collect.py`, 실험 A(`A/disturb_force.py`) | 파지 상태(`grasp_states/<label>/states/*.npz`)와 A의 성공 cell, 방향별 F_max가 이미 있어야 한다 |
| 1. 충격 | `B/disturb_impulse.py` | 상태를 복원해 방향 × 크기마다 50 ms 펄스를 주고 10 s 유지되는지 판정. 외력 0 대조 포함 |
| 정리 | `B/disturb_impulse.py --rescore`, `B/plot_impulse.py` | 판정 기준만 바꿔 다시 판정, 표·그림 |

설계 전체를 한 번의 명령으로 돌린다. cell마다 결과 파일을 하나씩 쓰므로, 중단돼도 같은 명령을 다시 실행하면 끝난 cell은 건너뛴다.

## 2. 정의

**대상 cell.** 기본(`--cells success`)은 A의 성공 cell 중 A의 외력 0 대조를 통과한 것(`force_cells.csv`의 group이 env 또는 tip)이다. `--cells rerun`이면 다시 돌린 성공 cell 전부를 쓴다.

**하중.** 물체 질량 중심에 `xfrc_applied`로
F(t) = k·mg (0 ≤ t < T), 그 뒤 0
을 가한다. 기본값은 T = 50 ms, k = 0.5, 1, 2, 3 mg, 6방향(손 좌표계, A와 같음)이다. `--shape halfsine`이면 최댓값이 k·mg인 반주기 사인이고, 충격량은 사각형의 2/π다.
팔은 마지막 관절 명령을 유지하고, 텐던은 장력 고정(20 N), 중력은 켜 둔다(A와 같은 조건).
충격량은 k·mg·T이다. 예를 들어 1 mg, 50 ms면 0.049 N·s이고, 잡혀 있지 않은 물체라면 0.49 m/s로 움직인다.

**성공 판정.** 펄스 시작부터 펄스가 끝난 뒤 10 s(`--after-s`)까지 아래 중 하나도 일어나지 않으면 **held**다. 기준은 A와 같다.
- slip: 손 기준 물체 변위 ≥ 5 mm. `--slip-mode residual`이면 펄스 후 0.5 s(`--settle-s`)부터의 변위만 센다(충격 순간에 휘었다가 돌아오는 변위는 허용)
- rotation: 손 기준 물체 회전 ≥ 5° (`--rot-mode tilt`면 원기둥 축 기울기만)
- lost: 한쪽 손가락 접촉력 합 < 0.1 N이 20 ms 이상
- unintended collision이나 불안정으로 끝난 시행도 held가 아니다

실패는 시작 시점으로 나눈다: **pulse**(펄스 중), **transient**(펄스 후 0.5 s 안), **delayed**(그 뒤. 충격으로 약해진 파지가 creep으로 빠짐).

**외력 0 대조.** 같은 상태를 외력 없이 T + 10 s 동안 둔다. 여기서 판정에 걸리는 cell은 충격이 없어도 못 버티므로 **실패로 간주**하고(그룹 `zero_fail`) 아래 지표에서 뺀다. A의 대조는 9.8 s였고 B는 10.05 s라서 몇 개가 더 빠질 수 있다(파일럿에서는 219개 중 1개). `--zero-fail keep`이면 표시만 한다.

## 3. 지표

| 지표 | 정의 | 용도 |
|---|---|---|
| level (cell × 방향) | k 이하의 펄스를 **모두** 버틴 가장 큰 k. 가장 작은 펄스에서 실패하면 0 | 방향별 충격 한계 |
| level_min (cell) | 6방향 level 중 최솟값, `weakest_dir` | cell 단위 강건성 |
| pass@k | 모든 방향에서 level ≥ k인 성공 cell 수 | 요구 충격이 정해져 있을 때 |
| cond@k | pass@k / 성공 cell | workspace 크기와 분리한 품질 |
| J_impulse@k | env_pass + w_tip·tip_pass + λ·Σm_F | A의 J_robust와 같은 척도 |
| impact_area | Σ_성공 cell level_min [mg] | A의 robust_area에 대응. 설계 순위 |
| held_above_A / failed_within_A | A의 ramp F_max < k인데 버틴 시행 수 / F_max ≥ k인데 실패한 시행 수 | 충격과 정적 하중의 차이 |
| type_changed | 버틴 시행 중 끝의 접촉 형태가 처음과 다른 것(예: tip → env) | 충격이 파지 형태를 바꾸는지 |

## 4. 실행

```powershell
cd on_rb5        # 모든 명령은 on_rb5/에서
# 빠른 시험 (설계 1개, cell 3개, 2방향)
python B/disturb_impulse.py --labels design075 --limit 3 --dirs py nx --workers 2
# graspable 설계 전부 (중단되면 같은 명령으로 이어서)
python B/disturb_impulse.py --workers 8
# 판정 기준만 바꿔 다시 판정 (시뮬레이션 없음)
python B/disturb_impulse.py --rescore --slip-mode residual
python B/disturb_impulse.py --rescore --rot-mode tilt
# 다른 설정은 --tag로 따로 저장 (예: 20 ms 펄스)
python B/disturb_impulse.py --labels design074 design075 candidate_02 --pulse-ms 20 --mags 1 2 3 5 --tag p20 --workers 8
# 그림
python B/plot_impulse.py
python B/plot_impulse.py --labels design074 design075 design076 design078 candidate_02 design001 --k 2
python B/plot_impulse.py --trace design075 r40_x0_y15
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--labels` | `graspable` | A의 graspable 설계 전부, 또는 설계 이름 |
| `--pulse-ms`, `--shape` | 50, `rect` | 펄스 길이, 모양(`halfsine`) |
| `--mags` | 0.5 1 2 3 | 펄스 크기 [mg] |
| `--dirs` | `axes` | `axes`(6방향), `edges`(대각선 12), `all`(18), 또는 `px nx pxpy …` |
| `--after-s` | 10 | 펄스 후 유지 시간 [s] |
| `--settle-s` | 0.5 | transient 구간 길이, residual slip 시작 |
| `--stop-mm`, `--stop-lost-s` | 30 mm, 0.1 s | 물체가 확실히 빠지면 시행을 멈추는 기준(재판정 가능 범위) |
| `--monotonic` | 꺼짐 | 한 방향에서 실패하면 더 큰 펄스는 건너뜀. 파일럿에서 계산량은 7 %만 줄었다 |
| `--slip-mm`, `--slip-mode`, `--rot-deg`, `--rot-mode`, `--grace-s` | config | 실패 기준 (`--rescore`에도 사용) |
| `--zero-fail` | `fail` | 외력 0 대조 실패 cell 처리 |
| `--tag` | 없음 | 결과 폴더 `B/results/<run>/impulse_<tag>/`. 설정이 다른 결과가 섞이지 않게 막고, 덮어쓰려면 `--overwrite` |

**trace 저장 방식.** 펄스 시작부터 T + settle(0.55 s)까지는 sample(5 ms)마다, 그 뒤에는 0.1 s마다 한 줄씩 저장한다. 각 줄에는 직전 줄 이후의 변위·회전·기울기 최댓값과 가장 긴 접촉 상실 시간이 함께 들어 있다. 그래서 `--rescore`의 held 여부는 어떤 기준으로 바꿔도 정확하고, 실패 시각은 0.55 s 이후에만 0.1 s 단위로 근사된다.

## 5. 산출물

```
B/results/rb5_cmaes_final/impulse/        (--tag p20 이면 impulse_p20/)
├── all_designs.csv                    설계당 한 줄: 설계값, A 지표(J_stable, robust_area, F_min 중앙값),
│                                       B 지표(success, impact_area, pass/cond/J_impulse@k, 방향·크기별 held 비율 env/tip,
│                                       실패 구간 수)
├── <label>/
│   ├── traces/<cell>.npz              시행마다 '<dir>_<k>mg_trace'(예: py_2mg_trace), '_ended', 외력 0 대조 'z0_*'
│   ├── impulse_trials.csv             cell × 방향 × 크기 한 줄: held, cause, phase, t_fail, 최대/잔류 변위, 회전,
│   │                                   손 이동, 끝 접촉 형태, A의 F_max(ramp, hold)
│   ├── impulse_cells.csv              cell 한 줄: 방향별 level, level_min, pass@k, 외력 0 대조
│   ├── impulse_meta.json              설정, 판정 기준
│   └── impulse_summary.json
└── plots/
    impulse_heatmap.png  impulse_vs_ramp.png  impulse_pareto.png  impulse_vs_A.png  impulse_trace_<label>_<cell>.png
```

## 6. 초기값과 파일럿 결과

**왜 이 값으로 시작하나.**
- **T = 50 ms**: 원래 계획값이다. 잡혀 있지 않은 물체는 1 mg 펄스 하나로 12 mm 움직이므로(d = k·g·T²/2), 파지가 버티지 못하면 펄스가 끝나기 전에 5 mm 기준을 넘는다. 반면 A의 ramp가 1~3 mg에 도달하는 데는 1~3 s가 걸리므로 시간 척도가 A와 다르다.
- **k = 0.5, 1, 2, 3 mg**: A의 F_max 분포를 덮는다. 가장 약한 방향의 F_min 중앙값은 설계마다 0.2~2.6 mg이고, 강한 env 방향은 3~10 mg이다. 원래 계획(1·2·3 mg)에 0.5 mg을 더했다. tip이 많은 설계는 1 mg에서 거의 다 실패해서, 0.5 mg이 없으면 이 설계들 사이의 차이가 보이지 않는다(파일럿에서 candidate_02의 pass@0.5 = 63, pass@1 = 6).
- **10 s 유지**: 요청값. 외력 0 대조도 같은 길이로 맞췄다.

**파일럿** (9개 설계 = 18방향 비교에서 고른 8개 + design001, 성공 cell 219개, 6방향 × 4크기, 50 ms 사각 펄스).

| 설계 | 성공 cell (env/tip) | 대조 실패 | pass@0.5 | @1 | @2 | @3 | impact_area | level_min 중앙값 | A robust_area | A F_min 중앙값 |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate_02 | 76 (65/11) | 0 | 63 | 6 | 0 | 0 | 34.5 | 0.5 | 41.4 | 0.53 |
| design163 | 40 (36/4) | 0 | 35 | 15 | 0 | 0 | 25.0 | 0.5 | 26.8 | 0.67 |
| design076 | 12 (10/2) | 1 | 10 | 8 | 8 | 2 | 19.0 | 2.0 | 19.0 | 1.77 |
| design075 | 16 (14/2) | 0 | 12 | 9 | 6 | 2 | 18.5 | 1.0 | 22.2 | 1.28 |
| design078 | 11 (8/3) | 0 | 9 | 8 | 8 | 2 | 18.5 | 2.0 | 18.2 | 1.81 |
| design033 | 27 (16/11) | 0 | 18 | 13 | 0 | 0 | 15.5 | 0.5 | 23.0 | 0.82 |
| design074 | 6 (6/0) | 0 | 6 | 6 | 5 | 4 | 15.0 | 3.0 | 16.4 | 2.61 |
| design138 | 22 (15/7) | 0 | 20 | 10 | 0 | 0 | 15.0 | 0.5 | 21.0 | 0.98 |
| design001 | 8 (1/7) | 0 | 5 | 3 | 1 | 1 | 6.0 | 0.5 | 9.3 | 0.90 |

- θ_open,p = 17° 계열(074/075/076/078)만 2 mg 펄스를 모든 방향에서 버티는 cell이 있다. 나머지는 pass@2 = 0이다(design001은 env cell 1개만 통과).
- **실패는 거의 펄스 중에 시작한다**: pulse 84 %, transient 13 %, delayed 3 %(대부분 회전 creep). 물체는 펄스가 지속되는 동안에만 미끄러지고, 펄스가 끝나면 새 위치에서 다시 붙잡힌다(`impulse_trace_design075_r40_x0_y15.png`).
- **50 ms 펄스의 결과는 A의 ramp F_max로 거의 예측된다.** 펄스 크기 k가 같은 cell·방향의 F_A보다 작으면 90~99 % 버티고, k − F_A가 0~0.2 mg면 56 %, 0.2~0.4 mg면 13 %, 0.4 mg 이상이면 3 % 이하다. 50 % 지점은 F_A + 약 0.2 mg이다(A의 hold test 값 기준으로는 약 +0.3 mg). 설계 순위도 비슷하다(Spearman: robust_area ↔ impact_area 0.78, F_min 중앙값 ↔ level_min 중앙값 0.87).
- 같은 계산을 design075에서 **T = 20 ms**로 하면 F_A보다 0.5~1 mg 큰 펄스도 55 %, 1~2 mg 큰 펄스도 23 % 버틴다. 짧은 펄스일수록 결과를 정하는 것이 힘이 아니라 충격량이 된다.
- `--slip-mode residual`은 거의 차이가 없다. slip 실패 1119건 중 49건만 settle 후 5 mm 아래로 돌아왔다. 휘었다가 돌아오는 경우보다 새 자세로 미끄러져 멈추는 경우가 대부분이다.
- 버틴 시행 중 74건은 접촉 형태가 바뀌었다(대부분 tip → env, 65건).
- 크기에 대해 단조가 아닌 cell(작은 펄스에서 실패하고 큰 펄스에서 버팀)은 2개였다.

## 7. 계산량

이 환경 기준으로 시행 1회(10 s)는 평균 0.73 s, 끝까지 버틴 시행은 약 1 s 걸린다. 성공 cell 하나는 25회(대조 1 + 6방향 × 4크기)로 약 18 s다.
graspable 205개 설계의 성공 cell은 5264개이므로 약 27 core-hour, `--workers 8`이면 약 3~4시간이다.
trace는 cell당 약 0.3~0.4 MB로 전체 1.5~2 GB 정도다(실험 A의 1.7 GB와 비슷). 줄이려면 `--coarse-s 0.2`를 쓴다.

## 8. 정해야 할 것

| 항목 | 기본값 | 대안 | 바꾸는 방법 |
|---|---|---|---|
| 펄스 길이 | 50 ms (계획값) | 10~20 ms: 충격량에 지배되는 영역. 50 ms는 A와 거의 같은 정보를 준다 | `--pulse-ms 20 --tag p20` (`--mags`도 더 크게, 예: 1 2 3 5) |
| 펄스 크기 | 0.5 1 2 3 mg | 계획대로 1 2 3 mg | `--mags 1 2 3` |
| 펄스 모양 | 사각 | 반주기 사인(실제 충돌에 가까움, 충격량 2/π배) | `--shape halfsine` |
| slip 판정 | 언제든 5 mm (A와 같음) | 펄스 후 0.5 s부터의 변위만 | `--rescore --slip-mode residual` |
| 회전 판정 | 전체 회전 5° | 축 기울기만 | `--rescore --rot-mode tilt` |
| 설계 순위 지표 | impact_area | J_impulse@k, cond@k | `B/results/<run>/impulse/all_designs.csv`의 열 |
