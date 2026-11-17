# 최적 PID · LQR · PPO 모델 (스펙 + 최소화 기준)

세 제어기를 **같은 기준, 같은 설계 상태**로 각각 최적화한 뒤, 최적화에 쓰지 않은 상태에서 서로 비교했어요.

## 0. 폴더 배치

이 폴더(예: `optimal_v2/`)를 레포 폴더 바로 아래에 두면 돼요.

```
InversePendulum/
├── cartpole_common.py, cartpole_pid.py, ...   # 기존 레포 파일 (수정하지 않음, import 만 함)
└── optimal_v2/
    ├── optimal_common.py, optimal_pid_lqr.py, optimal_ppo.py, optimal_compare.py
    ├── models/    # 최적 모델
    └── results/optimal/
```

- 기존 레포 파일(`cartpole_common.py` 등)은 같은 폴더에서 먼저 찾고, 없으면 상위 폴더에서 찾아요. 그래서 레포 폴더에 바로 풀어도 동작해요.
- `models/`, `results/` 경로는 `optimal_*.py`가 있는 폴더 기준이에요. 어느 폴더에서 실행해도 같은 위치에 저장돼요.

## 1. "최적 모델"의 정의

| 순위 | 내용 | 기준 |
|---|---|---|
| 1. 필수 (스펙) | ±5°~±30° 설계 상태에서 **전부 성공** | `evaluate_metrics.judge()` — 10 s 동안 \|x\|≤2.4 m (각도 실패 한계 없음), 마지막 2 s 동안 \|θ\|≤3°, \|θ̇\|≤0.2 rad/s, \|x\|≤0.2 m |
| 2. 최소화 | **평균 Settling Time** | `settling_time(use_x=True)` — θ, θ̇, **x 모두** 기준 안에 들어가 끝까지 유지되는 첫 시각 |
| 3. 상한 | 평균 Control Effort ≤ **115 N²·s** | 세 제어기에 같은 힘 예산 적용 (보상 가중치를 그대로 Q, R로 쓴 LQR의 effort 114.8을 기준으로 정함) |

목적함수 (작을수록 좋음, 단위 s):

```
J = mean(Ts)  +  10 × (실패한 설계 상태 수)  +  50 × max(0, mean(effort)/115 − 1)
```

- 실패한 상태의 Ts는 10 s로 계산해요.
- 실패 하나가 +10 s라서, 스펙을 지키는 해는 항상 어기는 해보다 J가 작아요. 즉 스펙이 1순위예요.

**설계 상태 148개** (`optimal_common.design_states()`, seed 2026)

- 각도마다 nominal `[0, 0, θ0, 0]` 1개와 x, ẋ, θ̇ ~ U(±0.6) 무작위 상태를 넣었어요.
- 무작위 상태는 실패 경계가 있는 큰 각도에 많이 배치했어요. 부호마다 5°·10°·15°에 4개, 20°에 8개, 25°에 16개, 30°에 32개예요.

## 2. 제어기별 최적화 방법

세 제어기 모두 같은 J와 같은 설계 상태를 써요.

| 제어기 | 최적화 변수 | 방법 |
|---|---|---|
| PID | 게인 6개 (kp·kd는 log 스케일, ki는 0 허용) | Differential Evolution 150세대 → Nelder-Mead |
| LQR | Q 대각 4개 (R = 1.6e-6 고정, Q/R 비율만 K에 영향) | Differential Evolution → Nelder-Mead |
| PPO | 정책 신경망 (하이퍼파라미터는 `cartpole_ppo.py`와 같음) | 보상 재설계 + 6개 run. 20k 스텝마다 체크포인트를 저장하고, **모든 체크포인트 중 J 최소** 선택 |

PPO 보상:

```
r = 1 − 0.5·1[기준 밖] − (θ² + 0.1x² + 0.01θ̇² + 0.01ẋ²) − W_U·(F/F_MAX)²     (|x| > 2.4 m 실패 시 −10, 종료)
```

- **`1[기준 밖]`:** 에피소드 합이 기준 밖에 머문 시간이라 Settling Time에 비례해요.
- **벌점 0.5 (< 1):** 기준 밖이어도 버티는 스텝의 보상이 양수예요. 그래서 "넘어지지 않기"가 계속 1순위예요.
- **W_U:** 0.03, 0.1, 0.3을 학습했어요.
- **학습 초기각:** U(±32°)에서 뽑되, 에피소드의 50%는 |θ0| ∈ [24°, 32°]에서 시작해요.

## 3. 결과

**테스트 조건:** 설계 상태와 다른 600개 상태예요. `angle_compare.py`의 반복 실행과 같아요(θ0 = ±5~±30°, 각 50회, MC_SEED=0).

| | PID | LQR | PPO |
|---|---|---|---|
| 성공률 | 100 % | 100 % | 100 % |
| 평균 Settling (θ, θ̇, x) | 1.26 s | 1.29 s | **1.22 s** |
| p95 / 최대 Settling | 1.78 / 1.86 s | 1.92 / 2.00 s | 1.86 / 2.04 s |
| 평균 Settling (θ, θ̇만) | 1.11 s | 1.10 s | **1.00 s** |
| 평균 Control Effort | 75.4 N²·s | 66.3 N²·s | **60.1 N²·s** |
| 버틴 최대 초기각 (nominal) | **±40°** | **±40°** | ±39° |
| 채터링 Σ\|ΔF\| | 168 N ⚠ | **37 N** | 39 N |
| Angle RMSE | 3.19° | **3.08°** | 3.21° |

**초기각별 결과** (`test_by_angle.csv`, `metrics_vs_angle.png`)

| \|θ0\| | Settling (x 포함) PID / LQR / PPO | Effort PID / LQR / PPO |
|---|---|---|
| 5° | 0.88 / 0.84 / **0.79** s | 34 / 15 / **11** |
| 30° | **1.62** / 1.74 / 1.65 s | 148 / 144 / **134** |

- PPO는 대부분의 각도에서 가장 빠르고 힘도 가장 적게 써요.
- ±30°에서는 x 포함 settling 기준으로 PID가 가장 빨라요.
- PID는 작은 각도에서도 힘을 많이 써요(5°에서 34 N²·s, LQR의 2배 이상).

**최적 파라미터**

- **PID** (`models/pid_gains_opt.json`)
  - `PID_THETA = dict(kp=399.475, ki=10.841, kd=76.068)`
  - `PID_X = dict(kp=65.706, ki=16.240, kd=94.768)`
- **LQR** (`models/lqr_weights_opt.json`)
  - `LQR_Q = [1.837, 0.5814, 0.0001769, 0.2357]`, `LQR_R = 1.6e-06`
  - 결과 게인 **K = [−45.08, −54.32, −262.87, −42.63]** (F = −K s)
  - Q 값 자체는 해석하기 어려우니 비교·보고는 K로 하세요.
- **PPO** (`models/ppo_opt.zip`)
  - run `wu0.1_hard0.5_2M_seed1`의 1,720,000 스텝 체크포인트

**그림** (`results/optimal/`)

- `summary_bars.png`: 6개 지표 비교
- `metrics_vs_angle.png`: |θ0|에 따른 Settling Time과 Effort
- `tradeoff.png`: Settling vs Effort. PID·LQR은 effort 상한 60/85/115/160에서 각각 최적화한 곡선도 함께 그렸어요.
- `timeseries_theta30.png`: θ0 = +30°에서의 θ, x, F 시간응답

## 4. 해석할 때 주의할 점

1. **Settling 정의에 x가 들어가 있어요.**
   - 최적화 기준이 "카트까지 중앙에 돌아와야 안정화"예요.
   - θ만 보는 settling(`angle_compare.py`의 기본값 `SETTLING_USE_X = False`)으로 보고할 때는 표의 "θ, θ̇만" 행을 쓰세요.
   - θ만 빠르게 잡는 걸 원하면 `optimal_common.SETTLING_USE_X = False`로 바꿔 다시 최적화하면 돼요.
2. **PID는 채터링이 커요.**
   - θ0 = 30° 그림의 0.15~0.35 s 구간에서 힘이 −20 ↔ +10 N으로 빠르게 진동해요(Σ|ΔF| 168, LQR·PPO의 약 4배).
   - 기준에 없는 항목이라 최적화가 미분 게인을 키운 결과예요.
   - 실제 하드웨어에 쓸 계획이면 Σ|ΔF| 상한을 기준에 추가하는 걸 권해요.
   - 또 θ0 = 30°에서 PID의 x가 −0.18 m 근처까지 천천히 흘러가요. 기준 0.2 m 안이지만 여유가 작아요.
3. **설계 상태가 적으면 스펙이 테스트에서 깨져요.**
   - 처음에는 설계 상태 60개(각도마다 4개)로 했어요.
   - 그때는 최적화가 스펙 경계까지 밀어붙여서, 설계 상태에 없던 ±30° 구석(x ≈ 0.6 m, ẋ > 0.4 m/s가 기울어진 쪽과 같은 방향)에서 테스트 실패가 났어요.
   - 148개로 늘린 뒤에는 세 모델 모두 600/600이에요.
4. **PPO는 seed와 설정에 따라 편차가 커요.**
   - run별 최적 J가 1.43~1.98이에요.
   - 선택된 run도 1.72M 스텝에서는 0/148 실패였지만, 마지막 2M 스텝 모델은 11/148 실패예요. 그래서 마지막 모델이 아니라 체크포인트 선택이 결과를 좌우해요.
   - PID·LQR은 DE seed를 바꿔도 같은 해로 수렴했어요(PID J 1.4314 / 1.4315, LQR seed 4개 모두 1.508).
5. **effort 상한을 더 키워도 거의 빨라지지 않아요.**
   - 설계 상태 effort가 약 105~110을 넘으면 Settling이 더 줄지 않아요(`budget_sweep.csv`, PID 1.433 s / LQR 1.508 s에서 멈춤).
   - F_MAX = 25 N 포화가 물리적 한계예요.
6. **기타**
   - 노이즈·외란이 없는 조건이에요.
   - PPO는 x에 약 0.04 m의 정상상태 오프셋이 남아요(기준 0.2 m 안).

## 5. 파일과 실행 순서

| 파일 | 역할 |
|---|---|
| `optimal_common.py` | 기준(스펙·상한·J), 설계 상태, 배치 시뮬레이터(원본 `simulate()`와 결과 동일, 약 15배 빠름), 폴더 설정 |
| `optimal_pid_lqr.py` | PID·LQR 최적화, effort 상한 sweep |
| `optimal_ppo.py` | 보상 재설계, PPO 학습(run 목록 `RUNS`), 체크포인트 선택 |
| `optimal_compare.py` | 최적 PID·LQR·PPO를 테스트 조건에서 비교 → `results/optimal/` |

```bash
python optimal_compare.py         # 약 1~2분: 저장된 최적 모델 3개 비교, CSV·그래프 생성
python optimal_pid_lqr.py check   # 몇 초: 배치 시뮬레이터가 원본 simulate()와 같은지 확인
python optimal_pid_lqr.py all     # 약 7분: PID·LQR 최적화를 다시 돌림 (models/*_opt.json 덮어씀)
python optimal_pid_lqr.py sweep   # 약 20분: effort 상한별 최적화 (tradeoff.png 곡선용)
python optimal_ppo.py grid        # 2코어 기준 약 40분: PPO 6개 run 학습 + 선택 (models/ppo_opt.zip 덮어씀)
```

인자 없이 `python optimal_pid_lqr.py`를 실행하면 `all`과 같아요.

**PPO 학습 기록에 대해**

- `models/ppo_opt_runs/*.log`와 `history.json`은 학습 중에 남긴 기록이에요.
- `wu0.3` run 2개는 effort 상한을 115로 정하기 전에 시작해서, 기록된 J에 상한 80이 적용돼 있어요.
- 최종 선택(`select`)은 모든 체크포인트를 현재 기준(상한 115)으로 다시 평가했어요.
- 체크포인트 파일(약 70 MB)은 zip에서 뺐어요.
