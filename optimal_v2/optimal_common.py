"""
Cart-Pole 역진자 — 세 제어기 공통 "최적 모델" 기준 (스펙 + 최소화)
================================================================
PID / LQR / PPO 를 "같은 기준, 같은 설계 상태"로 최적화하기 위한 공통 파일입니다.
(optimal_pid_lqr.py, optimal_ppo.py, optimal_compare.py 가 import)

  최적 모델의 정의
    1) 필수 (스펙)   : |θ0| ≤ 30° 설계 상태(148개)에서 전부 성공
                       → evaluate_metrics.judge() 기준: 10 s 동안 |x| ≤ 2.4 m (각도 실패 한계 없음), 마지막 2 s 동안
                         |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s, |x| ≤ 0.2 m 유지
    2) 최소화         : 평균 Settling Time  (θ, θ̇, x 모두 만족 = evaluate_metrics.settling_time(use_x=True))
    3) 상한           : 평균 Control Effort Σ F² Δt ≤ EFFORT_BUDGET

  목적함수 (작을수록 좋음, 단위 s)
    J = mean(Ts)  +  W_FAIL × (실패한 설계 상태 수)  +  W_EFFORT × max(0, mean(effort) / EFFORT_BUDGET − 1)
      - 실패한 상태의 Ts 는 SIM_TIME(10 s) 으로 계산
      - W_FAIL = 10 s → 실패가 하나라도 있으면 성공한 어떤 해보다 J 가 커짐 (스펙이 1순위)

  기존 파일(cartpole_common / evaluate_metrics / 각 제어기 파일)은 수정하지 않고 import 만 합니다.

  폴더 배치: optimal_*.py 는 레포 폴더에 두거나, 레포 바로 아래 하위 폴더(예: optimal_v2/)에 둬도 됩니다.
    - cartpole_common.py 등 기존 파일은 같은 폴더 → 없으면 상위 폴더에서 찾음
    - models/, results/ 경로는 optimal_*.py 가 있는 폴더 기준
"""

# ================================================================
# 0. 최적화 기준 설정 — 여기만 바꾸면 됩니다
# ================================================================
SPEC_ANGLE_MAGS_DEG = [5, 10, 15, 20, 25, 30]   # 스펙 초기각 |θ0| [deg] → ±θ0 모두 (총 12개)
DESIGN_RANDOM_PER_ANGLE = {     # 초기각마다 x, ẋ, θ̇ 를 무작위로 바꾼 설계 상태 개수 (+ nominal 1개), 부호마다
    5: 4, 10: 4, 15: 4, 20: 8, 25: 16, 30: 32,
}                               #   → 실패 경계가 있는 큰 각도에 많이 배치 (총 2 × (6 + 68) = 148개)
                                #   * 처음엔 각도마다 4개(총 60개)로 했더니, 최적화가 스펙 경계까지 밀어붙여
                                #     설계 상태에 없던 ±30° 구석(x≈0.6, ẋ>0.4 같은 방향)에서 테스트 실패가 나왔음
DESIGN_OTHER_RANGES = dict(     # 무작위 설계 상태의 나머지 성분 범위 (angle_compare 의 반복 실행과 같은 분포)
    x=(-0.6, 0.6),
    x_dot=(-0.6, 0.6),
    theta_dot=(-0.6, 0.6),
)
DESIGN_SEED = 2026              # 설계 상태 seed (평가용 angle_compare 의 MC_SEED=0 과 다른 값 → 설계/평가 분리)

SETTLING_USE_X = True           # Settling 기준에 |x| ≤ 0.2 m 포함 (Cart 가 중앙으로 돌아와야 "안정화")
EFFORT_BUDGET = 115.0           # 설계 상태 평균 Control Effort 상한 [N²·s]
                                #   = 기존 LQR(보상 가중치를 그대로 Q, R로 사용)의 설계 상태 평균 effort 114.8 을 반올림
                                #   → "기존 LQR 만큼의 힘으로 얼마나 빨리 안정화할 수 있나"
                                #   (설계 상태가 큰 각도 위주라 angle_compare 평균 effort 보다 큰 값)
W_FAIL = 10.0                   # 실패 1개당 벌점 [s]
W_EFFORT = 50.0                 # effort 상한 초과 비율 1 당 벌점 [s] (10 % 초과 → +5 s, 사실상 하드 제약)

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys

# ---------------- 폴더 설정 (레포 폴더 / 하위 폴더 어디서 실행해도 동작) ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if os.path.exists(os.path.join(_d, "cartpole_common.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)                  # 기존 파일(cartpole_common.py 등)을 import 할 수 있게
        break
else:
    raise ImportError(f"cartpole_common.py 를 찾지 못했습니다: {HERE} 또는 상위 폴더에 기존 레포 파일이 있어야 합니다.")
os.chdir(HERE)                                      # models/, results/ 는 이 파일이 있는 폴더 기준

import numpy as np
import cartpole_common as C
import evaluate_metrics as E

NAN = float("nan")
N_STEPS = int(round(C.SIM_TIME / C.DT))
HOLD_STEPS = int(round(E.CRIT["hold_time"] / C.DT))


# ---------------- 설계 상태 ----------------
def design_states():
    """스펙 초기각마다 nominal [0, 0, θ0, 0] 1개 + 나머지 성분 무작위 DESIGN_RANDOM_PER_ANGLE 개"""
    rng = np.random.default_rng(DESIGN_SEED)
    R = DESIGN_OTHER_RANGES
    out = []
    for mag in SPEC_ANGLE_MAGS_DEG:
        for sgn in (+1, -1):
            th = C.deg2rad(sgn * mag)
            out.append([0.0, 0.0, th, 0.0])
            for _ in range(DESIGN_RANDOM_PER_ANGLE[mag]):
                out.append([rng.uniform(*R["x"]), rng.uniform(*R["x_dot"]), th, rng.uniform(*R["theta_dot"])])
    return np.array(out, dtype=np.float64)


# ---------------- 배치 시뮬레이터 (최적화용, 여러 에피소드 동시 실행) ----------------
# cartpole_common.CartPoleSim / simulate() 와 같은 식·같은 순서로 계산합니다 (노이즈·외란 없음).
# 결과가 원본과 같은지는 optimal_pid_lqr.py check 로 확인할 수 있습니다.
_SIM = C.CartPoleSim()


def _deriv(S, F):
    p, lc, I = _SIM.p, _SIM.lc, _SIM.I
    M, m, g = p["M"], p["m"], p["g"]
    xd, th, thd = S[:, 1], S[:, 2], S[:, 3]
    c, sn = np.cos(th), np.sin(th)
    a11, a12, a22 = M + m, m * lc * c, I + m * lc ** 2
    r1 = F - p["b_cart"] * xd + m * lc * thd ** 2 * sn
    r2 = m * g * lc * sn - p["b_pivot"] * thd
    det = a11 * a22 - a12 ** 2
    xdd = (a22 * r1 - a12 * r2) / det
    thdd = (a11 * r2 - a12 * r1) / det
    return np.stack([xd, xdd, thd, thdd], axis=1)


def _step(S, F):
    h = C.DT / C.PHYSICS_SUBSTEPS
    F = F[:, None] if F.ndim == 1 else F
    F = F[:, 0]
    for _ in range(C.PHYSICS_SUBSTEPS):
        k1 = _deriv(S, F)
        k2 = _deriv(S + 0.5 * h * k1, F)
        k3 = _deriv(S + 0.5 * h * k2, F)
        k4 = _deriv(S + h * k3, F)
        S = S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    S[:, 2] = C.wrap_angle(S[:, 2])
    return S


def _stable(S):
    m = (np.abs(np.degrees(S[:, 2])) <= E.CRIT["theta_deg"]) & (np.abs(S[:, 3]) <= E.CRIT["theta_dot"])
    if SETTLING_USE_X:
        m &= np.abs(S[:, 0]) <= E.CRIT["x"]
    return m


def batch_run(policy, S0):
    """policy.reset(N), policy.act(S) -> F (N,) 를 N 개 에피소드에 동시에 적용.
    반환: 에피소드별 success, settling_time_s (실패/미수렴 = NaN), control_effort, failed"""
    S = np.array(S0, dtype=np.float64).copy()
    N = len(S)
    policy.reset(N)
    alive = np.ones(N, bool)
    effort = np.zeros(N)
    last_bad = np.where(_stable(S), -1, 0)          # 마지막으로 기준을 벗어난 시점 index (-1 = 없음)
    for k in range(N_STEPS):
        F = np.clip(policy.act(S), -C.F_MAX, C.F_MAX)
        F = np.where(alive, F, 0.0)
        S_new = _step(S, F)
        S = np.where(alive[:, None], S_new, S)       # 실패한 에피소드는 정지 (원본은 break)
        effort += np.where(alive, F ** 2 * C.DT, 0.0)
        failed_now = alive & (np.abs(S[:, 0]) > C.X_MAX)   # = C.is_failed (각도 실패 한계 없음)
        alive &= ~failed_now
        last_bad = np.where(alive & ~_stable(S), k + 1, last_bad)
    k_settle = last_bad + 1
    ok = alive & ((N_STEPS - k_settle) >= HOLD_STEPS)
    ts = np.where(ok, k_settle * C.DT, NAN)
    return dict(success=ok, settling_time_s=ts, control_effort=effort, failed=~alive)


def objective(res, budget=None):
    """batch_run 결과 → (J, 요약 dict)"""
    budget = EFFORT_BUDGET if budget is None else budget
    ok = res["success"]
    ts = np.where(ok, res["settling_time_s"], C.SIM_TIME)
    n_fail = int(np.sum(~ok))
    mean_eff = float(np.mean(res["control_effort"]))
    J = float(np.mean(ts) + W_FAIL * n_fail + W_EFFORT * max(0.0, mean_eff / budget - 1.0))
    return J, dict(J=J, n_fail=n_fail, n=len(ok), mean_ts=float(np.mean(ts)),
                   max_ts=float(np.max(ts)), mean_effort=mean_eff, feasible=(n_fail == 0 and mean_eff <= budget * 1.0001))


def fmt(summary):
    s = summary
    return (f"J={s['J']:.4f}  실패 {s['n_fail']}/{s['n']}  평균 Ts={s['mean_ts']:.3f}s  "
            f"최대 Ts={s['max_ts']:.2f}s  평균 effort={s['mean_effort']:.1f}  {'OK' if s['feasible'] else '스펙 위반'}")


# ---------------- 배치 제어기 ----------------
class BatchPID:
    """cartpole_pid.PIDController 와 같은 식 (measured 미분, anti-windup). gains: (6,) 또는 (N, 6)
    순서 [kp_θ, ki_θ, kd_θ, kp_x, ki_x, kd_x]"""

    def __init__(self, gains, i_limit=None):
        import cartpole_pid as P
        self.G = np.atleast_2d(np.asarray(gains, dtype=np.float64))
        self.il = P.INTEGRAL_LIMIT if i_limit is None else i_limit

    def reset(self, N):
        self.I_th = np.zeros(N)
        self.I_x = np.zeros(N)

    def act(self, S):
        x, xd, th, thd = S.T
        G = self.G
        self.I_th = np.clip(self.I_th + th * C.DT, -self.il["theta"], self.il["theta"])
        self.I_x = np.clip(self.I_x + x * C.DT, -self.il["x"], self.il["x"])
        return (G[:, 0] * th + G[:, 1] * self.I_th + G[:, 2] * thd
                + G[:, 3] * x + G[:, 4] * self.I_x + G[:, 5] * xd)


class BatchLinear:
    """F = -K s.  K: (4,) 또는 (N, 4)"""

    def __init__(self, K):
        self.K = np.atleast_2d(np.asarray(K, dtype=np.float64))

    def reset(self, N):
        pass

    def act(self, S):
        return -np.sum(self.K * S, axis=1)


class BatchPPO:
    """stable-baselines3 PPO 정책을 여러 상태에 한 번에 적용 (cartpole_ppo.make_obs 와 같은 관측)"""

    def __init__(self, model):
        self.model = model

    def reset(self, N):
        pass

    def act(self, S):
        obs = np.stack([S[:, 0], S[:, 1], np.sin(S[:, 2]), np.cos(S[:, 2]), S[:, 3]], axis=1).astype(np.float32)
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float64).reshape(-1) * C.F_MAX


def evaluate(policy, states=None, budget=None):
    states = design_states() if states is None else states
    return objective(batch_run(policy, states), budget)
