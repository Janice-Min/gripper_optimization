# Inverse Pendulum (Cart-Pole) — PID / LQR / PPO 비교

Cart-Pole 역진자를 **PID, LQR, PPO(강화학습)** 세 제어기로 균형 제어하고, 같은 시뮬레이터·같은 조건에서 평가지표로 성능을 비교하는 프로젝트입니다.

- 상태: `s = [x, ẋ, θ, θ̇]`, `θ = 0` 이 수직(Upright)
- 목표: 카트를 중앙(`x = 0`)에 두고 막대를 수직으로 유지
- 공통 조건: `dt = 0.02 s`, `F_MAX = 25 N`, 10초 시뮬레이션, `|x| > 2.4 m` 이면 실패 (각도 실패 한계 없음)

## 폴더 구조

```
InversePendulum/
├── cartpole_common.py     # 공통 설정 + 시뮬레이터 (모든 파일이 import)
├── cartpole_pid.py        # PID 제어기 (+ 게인 자동튜닝)
├── cartpole_lqr.py        # LQR 제어기 (+ Q/R 튜닝)
├── cartpole_ppo.py        # PPO 제어기 (학습 / 평가)
├── simulator.py           # 애니메이션 시각화 & 제어기 비교
├── evaluate_metrics.py    # 평가지표 계산 (초기각별 비교)
├── angle_compare.py       # ±5°~±30° 초기각별 비교 + 성공률(반복 실행)
├── angle_sweep.py         # 초기각을 1°씩 키우며 제어기별 한계 각도 탐색
├── models/                # 튜닝·학습된 제어기 파라미터
└── results/               # 실험 결과 (CSV, 그래프)
```

## 코드 설명

| 파일 | 역할 | 실행 |
|---|---|---|
| `cartpole_common.py` | 물리 파라미터, 입력 제한, 시뮬레이션 시간, 외란/노이즈, 초기상태 분포, 보상(성능 목표) 등 **공통 설정**과 RK4 기반 `CartPoleSim`, `simulate()`. 여기서 바꾸면 세 제어기에 똑같이 적용됨 | (직접 실행 X) |
| `cartpole_pid.py` | 각도 PID + 카트 위치 PID 를 더해 힘을 출력. Nelder-Mead 로 게인 자동튜닝 가능 | `python cartpole_pid.py tune / eval / all` |
| `cartpole_lqr.py` | Upright 에서 수치 선형화 → 이산 LQR 게인 `K` 계산, `F = -K s`. Q/R 을 성능 목표에서 바로 쓰거나(`objective`) 탐색(`search`) | `python cartpole_lqr.py tune / eval / all` |
| `cartpole_ppo.py` | Gymnasium 환경 + stable-baselines3 PPO 로 학습한 정책을 제어기로 사용 | `python cartpole_ppo.py train / eval / all` |
| `simulator.py` | 세 제어기를 불러와 같은 초기조건에서 애니메이션으로 비교, 또는 실험 세트 일괄 비교 | `python simulator.py` / `python simulator.py animate theta_deg=20` / `python simulator.py compare` |
| `evaluate_metrics.py` | 초기각(기본 ±5~20°)별로 세 제어기를 실행해 평가지표·시간응답·trade-off 그래프 생성. `judge()`, `compute_pdf_metrics()` 는 다른 스크립트에서 재사용 | `python evaluate_metrics.py` / `python evaluate_metrics.py 5 10 -5 -10` |
| `angle_compare.py` | ±5°~±30°(5° 간격) 초기각별 비교. 기준 실행으로 지표·overshoot 를, 무작위 초기상태 반복 실행(50회)으로 **성공률**을 계산 | `python angle_compare.py` |
| `angle_sweep.py` | ±21° 부터 1°씩 키우며 각 제어기가 **처음 실패하는 각도**와 실패 원인 탐색 | `python angle_sweep.py` |

> 각 파일 상단의 `# 0. ... 설정 — 여기만 바꾸면 됩니다` 블록에서 실험 조건을 바꿀 수 있습니다.

## 평가지표

| 지표 | 정의 | 좋은 방향 |
|---|---|---|
| Angle RMSE [deg] | `sqrt(mean θ²)` | ↓ |
| Settling Time [s] | `\|θ\| ≤ 3°`, `\|θ̇\| ≤ 0.2 rad/s` (+ `\|x\| ≤ 0.2 m`) 에 들어가 끝까지 유지되는 첫 시각 | ↓ |
| Success Rate [%] | 10초 동안 `\|x\| ≤ 2.4 m`, 마지막 2초 안정 조건 만족 | ↑ |
| Control Effort [N²·s] | `Σ F² Δt` | ↓ |
| Position RMSE [m] | `sqrt(mean x²)` | ↓ |

## `models/`

| 파일 | 내용 |
|---|---|
| `pid_gains.json` | 튜닝된 PID 게인 (`PID_THETA`, `PID_X`) |
| `lqr_weights.json` | LQR 가중치 `Q`, `R` |
| `ppo_cartpole_balance.zip` | 학습된 PPO 모델 (stable-baselines3) |

## `results/`

| 폴더 | 생성 스크립트 | 주요 내용 |
|---|---|---|
| `metrics_eval/` | `evaluate_metrics.py` | 초기각별 지표 표(`table_by_angle.csv`), 성공/실패 표, 시간응답(`timeseries_*.png`), 지표 막대그래프, `tradeoff.png` |
| `angle_compare/` | `angle_compare.py` | 지표·overshoot 표(`table_*.png`), 초기각에 따른 지표 변화(`metrics_vs_angle.png`), 반복 실행 결과(`mc_trials.csv`), 각도별 시간응답(`timeseries/`) |
| `angle_sweep/` | `angle_sweep.py` | 성공/실패 지도(`sweep_map.png`), 한계 각도 요약(`sweep_summary.csv`), 실패 직전·직후 시간응답(`sweep_fail_cases.png`) |

각 폴더의 `raw/` 에는 실행 1회의 원시 시계열 `t, x, ẋ, θ, θ̇, F` 가 `theta0_<각도>_<제어기>.csv` 로 저장됩니다.

## 빠른 시작

```bash
pip install numpy scipy matplotlib gymnasium stable-baselines3

python simulator.py              # 세 제어기 애니메이션 비교 (기본 θ0 = 10°)
python evaluate_metrics.py       # 평가지표 비교 → results/metrics_eval/
python angle_compare.py          # 초기각별 비교 → results/angle_compare/
python angle_sweep.py            # 한계 각도 탐색 → results/angle_sweep/
```

PPO 모델이 없으면 `python cartpole_ppo.py train` 으로 먼저 학습하세요 (없으면 PPO 는 자동으로 건너뜀).
