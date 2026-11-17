"""
Cart-Pole 역진자 비교 실험 — 공통 설정 & 공통 시뮬레이터
================================================================
PID / LQR / PPO 세 제어기 파일이 모두 이 파일을 import 합니다.
→ 물리 파라미터, 초기조건, 실험 세트, 제어 주기, 입력 제한은 "여기서 한 번만" 바꾸면
  세 제어기에 똑같이 적용됩니다. (제어기별 설정은 각 제어기 파일의 앞부분)

  - 상태: s = [x, x_dot, theta, theta_dot],  theta = 0 이 Upright (rad)
  - 목표: x = 0 m, theta = 0° (중앙에서 Upright 유지)
"""

# ================================================================
# 0. 공통 실험 설정 — 세 제어기에 동일하게 적용
# ================================================================

# ---------- 물리 파라미터 ----------
PHYS = dict(
    M=1.0,                      # Cart 질량 [kg]
    m=0.2,                      # Pendulum 질량 [kg]
    l=0.4,                      # Pendulum 길이 [m]
    pole_model="point",         # "point": 끝단 질점 (lc=l, I=0)
                                # "rod"  : 균일 막대 (lc=l/2, I=m*l^2/12)
    g=9.81,                     # 중력가속도 [m/s^2]
    b_pivot=0.10,               # Pivot 점성 마찰 [N·m·s/rad]
    b_cart=0.0,                 # Cart 점성 마찰 [N·s/m]
)
F_MAX = 25.0                    # 최대 제어입력 [N] (모든 제어기 출력이 여기서 포화)
X_MAX = 2.4                     # Cart 이동 한계 [m] (넘으면 실패 — 도중 종료 조건은 이것 하나)
                                # 각도에는 실패 한계가 없음: 진자가 넘어가도 10 s 끝까지 시뮬레이션하고,
                                # 마지막 2 s 안정 조건(evaluate_metrics.CRIT)으로만 성공/실패를 판정

# ---------- 시뮬레이션 ----------
SIM_TIME = 10.0                 # 실험 1회 길이 [s]
DT = 0.02                       # 제어 주기 [s] — 세 제어기 동일
PHYSICS_SUBSTEPS = 1            # 제어 주기 1번 안에서 RK4를 몇 번 나눠 적분할지 (2 → 물리 0.01 s)

# ---------- 외란 / 측정 노이즈 (평가용, 기본 실험은 0) ----------
DISTURBANCE = dict(
    enabled=False,
    force=5.0,                  # Cart에 가하는 펄스 외란 크기 [N]
    t_start=3.0,                # 외란 시작 시각 [s]
    duration=0.1,               # 지속 시간 [s]
)
NOISE_STD = [0.0, 0.0, 0.0, 0.0]  # 측정 노이즈 표준편차 [x, x_dot, theta(rad), theta_dot]
SEED = 0

# ---------- 평가 초기상태 & 실험 세트 ----------
# 각 실험은 BASE_S0에서 지정한 항목만 바꿉니다 (각도는 degree로 입력).
BASE_S0 = dict(x=0.0, x_dot=0.0, theta_deg=10.0, theta_dot=0.0)

EXPERIMENTS = {
    "Baseline":     [dict()],
    "A_theta0":     [dict(theta_deg=v) for v in [-20, -15, -10, -5, 5, 10, 15, 20]],
    "B_x0":         [dict(x=v) for v in [-0.5, -0.25, 0.0, 0.25, 0.5]],
    "C_theta_dot0": [dict(theta_dot=v) for v in [-0.5, -0.2, 0.0, 0.2, 0.5]],
    "D_x_dot0":     [dict(x_dot=v) for v in [-0.5, 0.0, 0.5]],
}
RUN_EXPERIMENTS = ["Baseline", "A_theta0", "B_x0", "C_theta_dot0", "D_x_dot0"]

SETTLE_TOL_DEG = 1.0            # |theta|가 이 값 이내로 "계속" 머무르기 시작한 시각 = Settling time

# ---------- 설계 조건 (PPO 학습 & PID 자동튜닝이 공유) ----------
# 데이터로 제어기를 설계할 때 쓰는 초기상태 분포. 평가 조건을 포함하도록 설정.
DESIGN_INIT_RANGES = dict(
    x=(-0.6, 0.6),
    x_dot=(-0.6, 0.6),
    theta_deg=(-25.0, 25.0),
    theta_dot=(-0.6, 0.6),
)
DESIGN_NOISE_STD = [0.0, 0.0, 0.0, 0.0]  # 설계 중 관측 노이즈 (강건성 설계 시 사용)
DESIGN_N_SAMPLES = 16           # PID·LQR 튜닝 목적함수에 쓰는 설계용 초기상태 개수 (둘이 같은 상태 집합 사용)

# 공통 성능 목표 = PPO의 보상 = PID 자동튜닝의 목적함수
#   매 스텝 r = alive - cost,   cost = w_theta*θ² + w_x*x² + w_theta_dot*θ̇² + w_x_dot*ẋ² + w_u*(F/F_MAX)²
#   실패(|x| > X_MAX)하면 그 스텝 r = -fail_penalty 후 종료
#   주의: 각도로는 종료되지 않으므로, 넘어진 상태(θ ≈ 180° → θ² ≈ 9.9)로 남는 스텝마다 r ≈ -9 가 쌓임.
#         PPO 를 다시 학습하면 "넘어진 뒤 레일 밖으로 나가 -fail_penalty 로 끝내기"가 보상상 이득이 될 수 있음
OBJECTIVE = dict(
    alive=1.0,
    w_theta=1.0,
    w_x=0.1,
    w_theta_dot=0.01,
    w_x_dot=0.01,
    w_u=0.001,
    fail_penalty=10.0,
)

RESULT_ROOT = "results"

# ================================================================
# 이하 구현부
# ================================================================
import os
import csv
import numpy as np


def deg2rad(d):
    return d * np.pi / 180.0


def wrap_angle(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


def make_s0(overrides):
    """BASE_S0에 overrides를 덮어써 초기상태 벡터 [x, x_dot, theta(rad), theta_dot] 생성"""
    d = dict(BASE_S0)
    d.update(overrides)
    return np.array([d["x"], d["x_dot"], deg2rad(d["theta_deg"]), d["theta_dot"]], dtype=np.float64)


def condition_label(overrides):
    return ", ".join(f"{k}={v}" for k, v in overrides.items()) or "base"


def sample_design_state(rng):
    """설계용 초기상태 1개를 DESIGN_INIT_RANGES에서 균일 샘플링"""
    r = DESIGN_INIT_RANGES
    return np.array([rng.uniform(*r["x"]), rng.uniform(*r["x_dot"]),
                     deg2rad(rng.uniform(*r["theta_deg"])), rng.uniform(*r["theta_dot"])])


def design_states():
    """PID·LQR 튜닝에 공통으로 쓰는 고정된 설계용 초기상태 집합 (SEED로 재현)"""
    rng = np.random.default_rng(SEED)
    return [sample_design_state(rng) for _ in range(DESIGN_N_SAMPLES)]


def mean_design_return(controller, states=None):
    """설계용 초기상태들에서의 평균 episode return (튜닝 목적함수, PPO 학습 조건과 동일: 외란 없음)"""
    states = design_states() if states is None else states
    total = 0.0
    for s0 in states:
        log = simulate(controller, s0, np.random.default_rng(SEED),
                       noise_std=DESIGN_NOISE_STD, use_disturbance=False)
        total += episode_return(log)
    return total / len(states)


def disturbance_force(t):
    if not DISTURBANCE["enabled"]:
        return 0.0
    t0 = DISTURBANCE["t_start"]
    return DISTURBANCE["force"] if t0 <= t < t0 + DISTURBANCE["duration"] else 0.0


# ---------------- 공통 시뮬레이터 ----------------
class CartPoleSim:
    """비선형 Cart-Pole 동역학 + RK4 적분. 세 제어기 모두 이 클래스를 사용."""

    def __init__(self, phys=None, dt=None, substeps=None):
        self.p = dict(PHYS if phys is None else phys)
        self.dt = DT if dt is None else dt
        self.substeps = PHYSICS_SUBSTEPS if substeps is None else substeps
        m, l = self.p["m"], self.p["l"]
        if self.p["pole_model"] == "point":
            self.lc, self.I = l, 0.0
        elif self.p["pole_model"] == "rod":
            self.lc, self.I = l / 2.0, m * l ** 2 / 12.0
        else:
            raise ValueError("pole_model은 'point' 또는 'rod'")

    def deriv(self, s, F):
        x, xd, th, thd = s
        M, m, g = self.p["M"], self.p["m"], self.p["g"]
        lc, I = self.lc, self.I
        c, sn = np.cos(th), np.sin(th)
        # 질량행렬 * [x_dd, th_dd] = rhs   (라그랑주 방정식)
        a11, a12, a22 = M + m, m * lc * c, I + m * lc ** 2
        r1 = F - self.p["b_cart"] * xd + m * lc * thd ** 2 * sn
        r2 = m * g * lc * sn - self.p["b_pivot"] * thd
        det = a11 * a22 - a12 ** 2
        xdd = (a22 * r1 - a12 * r2) / det
        thdd = (a11 * r2 - a12 * r1) / det
        return np.array([xd, xdd, thd, thdd])

    def step(self, s, F):
        """제어 주기 dt 동안 입력 F를 유지(ZOH)하며 RK4 적분"""
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.deriv(s, F)
            k2 = self.deriv(s + 0.5 * h * k1, F)
            k3 = self.deriv(s + 0.5 * h * k2, F)
            k4 = self.deriv(s + h * k3, F)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s[2] = wrap_angle(s[2])
        return s


def is_failed(s):
    return abs(s[0]) > X_MAX


def step_reward(s_next, F, failed):
    """공통 성능 목표의 한 스텝 값 (PPO 보상과 PID 튜닝에서 같은 함수 사용)"""
    if failed:
        return -OBJECTIVE["fail_penalty"]
    x, xd, th, thd = s_next
    o = OBJECTIVE
    cost = (o["w_theta"] * th ** 2 + o["w_x"] * x ** 2 + o["w_theta_dot"] * thd ** 2
            + o["w_x_dot"] * xd ** 2 + o["w_u"] * (F / F_MAX) ** 2)
    return o["alive"] - cost


# ---------------- 공통 실험 루프 ----------------
# 모든 제어기는 reset()과 act(state) -> F[N] 만 구현하면 이 함수로 평가됩니다.
def simulate(controller, s0, rng, noise_std=None, use_disturbance=True):
    sim = CartPoleSim()
    N = int(round(SIM_TIME / DT))
    noise = np.array(NOISE_STD if noise_std is None else noise_std, dtype=np.float64)
    controller.reset()                                # 적분값 등 내부 상태 초기화

    s = s0.copy()
    S, U = [s.copy()], []
    failed = False
    for k in range(N):
        t = k * DT
        meas = s + rng.normal(0.0, 1.0, 4) * noise     # 제어기는 측정값만 봄
        F = float(np.clip(controller.act(meas), -F_MAX, F_MAX))
        Fd = disturbance_force(t) if use_disturbance else 0.0
        s = sim.step(s, F + Fd)
        S.append(s.copy()); U.append(F)
        if is_failed(s):
            failed = True
            break
    S, U = np.array(S), np.array(U)
    return dict(t=np.arange(len(S)) * DT, s=S, u=U, failed=failed)


def episode_return(log):
    """simulate 결과의 누적 보상 (PPO 에피소드 return과 같은 정의, 할인 없음)"""
    n = len(log["u"])
    total = 0.0
    for k in range(n):
        last_fail = log["failed"] and k == n - 1
        total += step_reward(log["s"][k + 1], log["u"][k], last_fail)
    return total


def compute_metrics(log):
    th = np.degrees(log["s"][:, 2])
    u = log["u"]
    outside = np.where(np.abs(th) > SETTLE_TOL_DEG)[0]
    if log["failed"]:
        t_settle = float("nan")                      # 실패 → 정의 안 됨
    elif len(outside) == 0:
        t_settle = 0.0                               # 처음부터 허용범위 안
    elif outside[-1] + 1 < len(th):
        t_settle = float(log["t"][outside[-1] + 1])  # 마지막으로 벗어난 직후 시각
    else:
        t_settle = float("nan")                      # 끝까지 수렴 못함
    return dict(
        success=not log["failed"],
        angle_rmse_deg=float(np.sqrt(np.mean(th ** 2))),
        settling_time_s=t_settle,
        control_effort=float(np.sum(u ** 2) * DT),   # ∫F² dt [N²·s]
        mean_abs_force=float(np.mean(np.abs(u))),
        max_abs_x=float(np.max(np.abs(log["s"][:, 0]))),
        saturation_ratio=float(np.mean(np.abs(u) >= 0.99 * F_MAX)),
        episode_return=float(episode_return(log)),
    )


def run_experiments(controllers):
    """RUN_EXPERIMENTS의 모든 초기조건을 모든 제어기로 실행"""
    rows, logs = [], {}
    for exp in RUN_EXPERIMENTS:
        for ov in EXPERIMENTS[exp]:
            s0 = make_s0(ov)
            label = condition_label(ov)
            for ctrl in controllers:
                rng = np.random.default_rng(SEED)     # 모든 제어기·조건에 같은 노이즈 시퀀스
                log = simulate(ctrl, s0, rng)
                rows.append(dict(experiment=exp, condition=label, controller=ctrl.name,
                                 x0=s0[0], x_dot0=s0[1], theta0_deg=np.degrees(s0[2]),
                                 theta_dot0=s0[3], **compute_metrics(log)))
                logs[(exp, label, ctrl.name)] = log
    return rows, logs


# ---------------- 결과 저장 / 그래프 ----------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_grid(logs, controllers, out_dir):
    """실험 세트별: 행 = theta/x/F, 열 = 제어기, 색 = 초기조건"""
    plt = _plt()
    for exp in RUN_EXPERIMENTS:
        conds = [condition_label(ov) for ov in EXPERIMENTS[exp]]
        nc = len(controllers)
        fig, axes = plt.subplots(3, nc, figsize=(5.5 * nc, 8), sharex=True, squeeze=False)
        cmap = plt.get_cmap("viridis", max(len(conds), 2))
        for ci, ctrl in enumerate(controllers):
            for j, label in enumerate(conds):
                log = logs[(exp, label, ctrl.name)]
                c = cmap(j)
                axes[0, ci].plot(log["t"], np.degrees(log["s"][:, 2]), color=c, label=label)
                axes[1, ci].plot(log["t"], log["s"][:, 0], color=c)
                axes[2, ci].step(log["t"][:-1], log["u"], where="post", color=c)
            axes[0, ci].set_title(f"{exp} — {ctrl.name}")
            axes[0, ci].axhspan(-SETTLE_TOL_DEG, SETTLE_TOL_DEG, color="gray", alpha=0.15)
            axes[0, ci].set_ylabel("theta [deg]")
            axes[1, ci].set_ylabel("x [m]")
            axes[2, ci].set_ylabel("F [N]")
            axes[2, ci].set_xlabel("time [s]")
            for r in range(3):
                axes[r, ci].grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{exp}.png"), dpi=130)
        plt.close(fig)


def plot_overlay(logs, controllers, exp, out_dir):
    """같은 초기조건에서 제어기들을 한 그래프에 겹쳐 그림 (논문 그림용)"""
    plt = _plt()
    for ov in EXPERIMENTS[exp]:
        label = condition_label(ov)
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        for ctrl in controllers:
            log = logs[(exp, label, ctrl.name)]
            axes[0].plot(log["t"], np.degrees(log["s"][:, 2]), label=ctrl.name)
            axes[1].plot(log["t"], log["s"][:, 0])
            axes[2].step(log["t"][:-1], log["u"], where="post")
        axes[0].set_title(f"{exp} ({label})")
        axes[0].axhspan(-SETTLE_TOL_DEG, SETTLE_TOL_DEG, color="gray", alpha=0.15)
        axes[0].set_ylabel("theta [deg]"); axes[1].set_ylabel("x [m]"); axes[2].set_ylabel("F [N]")
        axes[2].set_xlabel("time [s]")
        axes[0].legend()
        for a in axes:
            a.grid(alpha=0.3)
        fig.tight_layout()
        safe = label.replace(", ", "_").replace("=", "")
        fig.savefig(os.path.join(out_dir, f"overlay_{exp}_{safe}.png"), dpi=130)
        plt.close(fig)


def plot_metric_summary(rows, controllers, out_dir):
    """초기조건 변화에 따른 지표 변화 (가로축 = 바꾼 초기조건 값, 선 = 제어기)"""
    plt = _plt()
    metrics = [("angle_rmse_deg", "Angle RMSE [deg]"),
               ("settling_time_s", "Settling time [s]"),
               ("control_effort", "Control effort [N²·s]")]
    for exp in RUN_EXPERIMENTS:
        ovs = EXPERIMENTS[exp]
        if len(ovs) < 2:
            continue
        key = list(ovs[0].keys())[0] if len(ovs[0]) == 1 else None
        xs = [ov[key] for ov in ovs] if key else list(range(len(ovs)))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ctrl in controllers:
            vals = {m: [] for m, _ in metrics}
            for ov in ovs:
                r = next(r for r in rows if r["experiment"] == exp
                         and r["condition"] == condition_label(ov) and r["controller"] == ctrl.name)
                for m, _ in metrics:
                    vals[m].append(r[m] if r["success"] else np.nan)  # 실패는 빈칸
            for a, (m, title) in zip(axes, metrics):
                a.plot(xs, vals[m], marker="o", label=ctrl.name)
        for a, (m, title) in zip(axes, metrics):
            a.set_title(title); a.set_xlabel(key or "condition"); a.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"{exp}  (failed conditions are not plotted)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"summary_{exp}.png"), dpi=130)
        plt.close(fig)


def save_csv(rows, out_dir):
    path = os.path.join(out_dir, "metrics.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path


def print_table(rows):
    print(f"\n{'experiment':<13}{'condition':<20}{'ctrl':<6}{'ok':<4}"
          f"{'RMSE[deg]':>10}{'Ts[s]':>8}{'Effort':>10}{'max|x|':>8}{'sat':>6}{'return':>9}")
    for r in rows:
        print(f"{r['experiment']:<13}{r['condition']:<20}{r['controller']:<6}"
              f"{'Y' if r['success'] else 'N':<4}{r['angle_rmse_deg']:>10.3f}"
              f"{r['settling_time_s']:>8.2f}{r['control_effort']:>10.1f}"
              f"{r['max_abs_x']:>8.3f}{r['saturation_ratio']:>6.2f}{r['episode_return']:>9.1f}")


def evaluate_and_save(controllers, out_subdir, overlay_experiments=()):
    """실험 실행 → CSV + 그래프 저장 + 표 출력"""
    out_dir = os.path.join(RESULT_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    rows, logs = run_experiments(controllers)
    csv_path = save_csv(rows, out_dir)
    plot_grid(logs, controllers, out_dir)
    if len(controllers) > 1:
        plot_metric_summary(rows, controllers, out_dir)
        for exp in overlay_experiments:
            plot_overlay(logs, controllers, exp, out_dir)
    print_table(rows)
    print(f"\n[평가 완료] {csv_path} 및 {out_dir}/*.png 저장")
    return rows, logs
