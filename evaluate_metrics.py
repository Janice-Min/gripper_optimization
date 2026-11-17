"""
Cart-Pole 역진자 — 평가지표 계산 & 제어기 비교 (PID / LQR / PPO)
================================================================
평가지표.pdf 의 정의대로 각 제어기의 10초 실행 결과를 평가하고, 초기각별로 비교합니다.
기존 파일(cartpole_common / cartpole_pid / cartpole_lqr / cartpole_ppo / simulator)은
수정하지 않고 import 만 합니다.

  - 제어기  : simulator.load_controllers() → 각 제어기 파일의 build_controller() (각 파일 설정 그대로)
  - 실행    : cartpole_common.simulate()   (평가·애니메이션과 같은 시뮬레이터 / dt / F_MAX / 노이즈 / 외란)
  - 초기상태: cartpole_common.BASE_S0 에서 theta_deg 만 바꿈 → s0 = [x, x_dot, θ, θ_dot] = [0, 0, θ0, 0]
  - 지표    : 이 파일에서 PDF 정의대로 계산 (cartpole_common.compute_metrics 의 1° 기준과는 별개)

실행 (다른 파일들과 같은 폴더에서):
  python evaluate_metrics.py                     # 아래 ANGLES_DEG 의 초기각들
  python evaluate_metrics.py 5 10 15 20 -5 -10 -15 -20  # 초기각을 명령줄에서 지정

  * PPO는 먼저 'python cartpole_ppo.py train' 으로 모델을 만들어야 합니다 (없으면 자동으로 건너뜀).

결과 (results/metrics_eval/):
  raw/theta0_+10deg_PID.csv   실행 1회의 원시 데이터 t, x, x_dot, θ, θ_dot, u   (PDF 2절)
  metrics_long.csv            실행 1회 = 1행, 모든 지표 원값 (직접 분석용)
  table_by_angle.csv          PDF 5절 형식 비교표 (초기각별 Metric × 제어기)
  success_table.csv           초기각 × 제어기 SUCCESS / FAIL(원인) + 성공률
  summary_by_controller.csv   제어기별 성공률 + 공통 성공 초기각에서의 평균 지표
  timeseries_theta0_*.png     초기각별 θ / x / F 시간응답 (제어기 겹쳐 그림, PDF 4절)
  timeseries_overview.png     모든 초기각을 한 장에 (+ _zoom: 처음 ZOOM_TIME 초 확대)
  metrics_primary.png         핵심 지표 (성공 여부, RMSE, Settling, Cart RMSE, Effort) 초기각별 막대그래프
  metrics_secondary.png       보조 지표 (Peak, Overshoot, Max |x|, Peak F, F_RMS, 포화, 입력 변화량, return)
  tradeoff.png                Control effort vs Angle RMSE / Settling time (성능–비용 trade-off)
"""

# ================================================================
# 0. 평가 설정 — 여기만 바꾸면 됩니다
# ================================================================
ANGLES_DEG = [5, 10, 15, 20]           # 비교할 초기 각도 [deg]  (나머지는 BASE_S0: x=0, x_dot=0, theta_dot=0)
CONTROLLERS = ["PID", "LQR", "PPO"]    # 비교할 제어기 (모델이 없는 제어기는 자동으로 건너뜀)

# 안정화 판정 기준 (평가지표.pdf ② Settling Time, ⑤ Stabilization Success)
CRIT = dict(
    theta_deg=3.0,       # |θ|  ≤ 3°
    theta_dot=0.2,       # |θ̇| ≤ 0.2 rad/s
    x=0.2,               # |x|  ≤ 0.2 m
    hold_time=2.0,       # 마지막 2초 동안 연속으로 만족해야 함
)                        # Cart 이동 한계(±2.4 m)는 cartpole_common.X_MAX 를 그대로 사용

ZOOM_TIME = 3.0                        # 응답이 빠르게 끝나므로 0 ~ ZOOM_TIME 초 확대 그래프도 저장 (None이면 안 함)
OUT_SUBDIR = "metrics_eval"            # 결과 폴더: results/metrics_eval/
SAVE_RAW = True                        # 실행마다 원시 데이터 CSV 저장
DPI = 140

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import csv
import numpy as np
import cartpole_common as C
from simulator import load_controllers, COLORS

NAN = float("nan")

# (key, 표·그래프 이름, 숫자 형식) — 표와 그래프에 나오는 순서
PRIMARY = [      # PDF 1차·2차 평가 지표
    ("angle_rmse_deg",        "Angle RMSE (deg) ↓",            "{:.3f}"),
    ("settling_time_s",       "Settling Time (s) ↓",           "{:.2f}"),
    ("settling_time_theta_s", "Settling Time, θ only (s) ↓",   "{:.2f}"),
    ("cart_rmse_m",           "Cart Position RMSE (m) ↓",      "{:.4f}"),
    ("control_effort",        "Control Effort (N²·s) ↓",       "{:.1f}"),
]
SECONDARY = [    # 그래프 해석용 보조 지표
    ("peak_angle_deg",        "Peak Angle (deg)",              "{:.2f}"),
    ("overshoot_deg",         "Overshoot, opposite side (deg)", "{:.2f}"),
    ("max_cart_disp_m",       "Max Cart Displacement (m)",     "{:.3f}"),
    ("peak_force_N",          "Peak Control Force (N)",        "{:.2f}"),
    ("u_rms_N",               "Control Force RMS (N)",         "{:.2f}"),
    ("saturation_pct",        "Saturated Time (%)",            "{:.1f}"),
    ("force_variation_N",     "Force Variation Σ|ΔF| (N)",     "{:.1f}"),
    ("episode_return",        "Design Objective Return ↑",      "{:.1f}"),
]

WHY_SHORT = dict(x_limit=f"|x|>{C.X_MAX:g}m",
                 theta="θ", theta_dot="θ̇", x="x")


# ---------------- 지표 계산 (평가지표.pdf 3절) ----------------
def _hold_steps():
    return int(round(CRIT["hold_time"] / C.DT))


def stable_mask(s, use_x=True):
    """각 시점에서 안정화 기준을 만족하는지 (True/False 배열)"""
    m = (np.abs(np.degrees(s[:, 2])) <= CRIT["theta_deg"]) & (np.abs(s[:, 3]) <= CRIT["theta_dot"])
    if use_x:
        m &= np.abs(s[:, 0]) <= CRIT["x"]
    return m


def settling_time(log, use_x=True):
    """② 기준을 만족하기 시작해 실험 끝(10 s)까지 계속 유지되는 최초 시각.
    유지 구간이 hold_time(2 s)보다 짧거나 도중에 실패하면 NaN (= Not Settled)"""
    if log["failed"]:
        return NAN
    m = stable_mask(log["s"], use_x)
    if not m[-1]:
        return NAN
    bad = np.where(~m)[0]
    k = 0 if len(bad) == 0 else int(bad[-1]) + 1
    if (len(m) - 1 - k) < _hold_steps():
        return NAN
    return float(log["t"][k])


def judge(log):
    """⑤ 성공 판정 → (성공 여부, 실패 원인 목록)"""
    s = log["s"]
    if log["failed"]:          # simulate()가 도중에 종료: |x| > X_MAX (각도 실패 한계는 없음)
        return False, ["x_limit"]
    why = []
    if np.max(np.abs(s[:, 0])) > C.X_MAX:                       # 10초 동안 이동 한계 위반
        why.append("x_limit")
    w = s[-(_hold_steps() + 1):]                                # 마지막 2초 (8.00 ~ 10.00 s)
    if np.max(np.abs(np.degrees(w[:, 2]))) > CRIT["theta_deg"]:
        why.append("theta")
    if np.max(np.abs(w[:, 3])) > CRIT["theta_dot"]:
        why.append("theta_dot")
    if np.max(np.abs(w[:, 0])) > CRIT["x"]:
        why.append("x")
    return len(why) == 0, why


def compute_pdf_metrics(log, theta0_deg):
    s, u = log["s"], log["u"]
    th = np.degrees(s[:, 2])
    x = s[:, 0]
    T = len(u) * C.DT                                  # 실제 실행 시간 (실패하면 10 s보다 짧음)
    ok, why = judge(log)
    Ju = float(np.sum(u ** 2) * C.DT)                  # ④ J_u ≈ Σ_{k=0}^{N-1} u_k² Δt  [N²·s]
    sgn = np.sign(theta0_deg)
    overshoot = float(max(0.0, np.max(-sgn * th))) if sgn != 0 else float(np.max(np.abs(th)))
    return dict(
        success=ok,
        terminated=bool(log["failed"]),
        t_end=float(log["t"][-1]),
        why=why,
        angle_rmse_deg=float(np.sqrt(np.mean(th[1:] ** 2))),     # ① k = 1..N (제어 후 상태들)
        settling_time_s=settling_time(log, use_x=True),          # ② θ, θ̇, x 모두 만족
        settling_time_theta_s=settling_time(log, use_x=False),   # ② θ, θ̇ 만 (Cart 복귀 제외, 참고용)
        cart_rmse_m=float(np.sqrt(np.mean(x[1:] ** 2))),         # ③
        control_effort=Ju,                                       # ④
        u_rms_N=float(np.sqrt(Ju / T)) if T > 0 else NAN,        # ④ u_RMS = sqrt(J_u / T)
        peak_angle_deg=float(np.max(np.abs(th))),
        overshoot_deg=overshoot,                                 # 수직을 지나 반대쪽으로 넘어간 최대 각
        max_cart_disp_m=float(np.max(np.abs(x))),
        peak_force_N=float(np.max(np.abs(u))) if len(u) else 0.0,
        saturation_pct=float(np.mean(np.abs(u) >= 0.99 * C.F_MAX) * 100) if len(u) else 0.0,
        force_variation_N=float(np.sum(np.abs(np.diff(u)))),     # 입력이 얼마나 자주·크게 바뀌는가 (채터링)
        episode_return=float(C.episode_return(log)),             # 세 제어기 공통 설계 목적함수 값
    )


def result_text(r, short=False):
    """SUCCESS / FAIL(원인) 문자열"""
    if r["success"]:
        return "SUCCESS"
    if r["terminated"]:
        return f"FAIL({WHY_SHORT[r['why'][0]]} @{r['t_end']:.2f}s)"
    return "FAIL(" + ",".join(WHY_SHORT[w] for w in r["why"]) + ")" if short else \
           "FAIL(last 2s: " + ", ".join(WHY_SHORT[w] for w in r["why"]) + ")"


# ---------------- 실행 ----------------
def run_all(controllers, angles):
    rows, logs = [], {}
    for a in angles:
        s0 = C.make_s0(dict(theta_deg=a))
        for ctrl in controllers:
            rng = np.random.default_rng(C.SEED)       # simulator.py / evaluate_and_save 와 같은 방식
            log = C.simulate(ctrl, s0, rng)
            rows.append(dict(theta0_deg=a, controller=ctrl.name, **compute_pdf_metrics(log, a)))
            logs[(a, ctrl.name)] = log
    return rows, logs


def row_of(rows, a, name):
    return next(r for r in rows if r["theta0_deg"] == a and r["controller"] == name)


def angle_tag(a):
    return f"theta0_{a:+g}deg"


# ---------------- 저장 ----------------
def save_raw(logs, out_dir):
    d = os.path.join(out_dir, "raw")
    os.makedirs(d, exist_ok=True)
    for (a, name), log in logs.items():
        S, U, t = log["s"], log["u"], log["t"]
        with open(os.path.join(d, f"{angle_tag(a)}_{name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "x_m", "x_dot_mps", "theta_rad", "theta_deg", "theta_dot_radps", "u_N"])
            for k in range(len(S)):
                u = f"{U[k]:.6f}" if k < len(U) else ""     # u_k는 [t_k, t_k+1) 동안의 힘 → 마지막 상태엔 없음
                w.writerow([f"{t[k]:.2f}", f"{S[k, 0]:.6f}", f"{S[k, 1]:.6f}", f"{S[k, 2]:.6f}",
                            f"{np.degrees(S[k, 2]):.4f}", f"{S[k, 3]:.6f}", u])


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _cell(r, key, fmt):
    """표 칸: 도중 종료된 실행은 지표를 비교하지 않음 (PDF: 실패 데이터로 RMSE 등을 비교하면 왜곡)"""
    if r["terminated"]:
        return "—"
    v = r[key]
    if not np.isfinite(v):
        return "Not Settled" if key.startswith("settling") else "—"
    return fmt.format(v)


def build_tables(rows, angles, names):
    by_angle = []
    for a in angles:
        rs = [row_of(rows, a, n) for n in names]
        by_angle.append([a, "Stabilization Success"] + [result_text(r) for r in rs])
        for key, label, fmt in PRIMARY + SECONDARY:
            by_angle.append([a, label] + [_cell(r, key, fmt) for r in rs])

    success = [[f"{a:+g}"] + [result_text(row_of(rows, a, n)) for n in names] for a in angles]
    rates = {n: 100.0 * np.mean([row_of(rows, a, n)["success"] for a in angles]) for n in names}
    success.append(["Success Rate (%)"] + [f"{rates[n]:.0f}" for n in names])

    # 제어기별 평균: "모든 제어기가 성공한 초기각"에서만 평균 → 어려운 조건에서 실패한 제어기가
    # 쉬운 조건만으로 평균 내져 좋아 보이는 생존 편향(survivor bias) 방지
    common = [a for a in angles if all(row_of(rows, a, n)["success"] for n in names)]
    summary = []
    for n in names:
        line = [n, f"{rates[n]:.0f}", " ".join(f"{a:+g}" for a in common)]
        for key, _, fmt in PRIMARY:
            vals = [row_of(rows, a, n)[key] for a in common]
            line.append(fmt.format(np.mean(vals)) if vals else "—")
        summary.append(line)
    summary_header = ["controller", "success_rate_%", "common_success_angles_deg"] + \
                     [f"mean_{k}" for k, _, _ in PRIMARY]
    return by_angle, success, summary, summary_header, common


def save_all_csv(rows, tables, names, out_dir):
    by_angle, success, summary, summary_header, _ = tables
    keys = ["theta0_deg", "controller", "success", "result", "terminated", "t_end"] + \
           [k for k, _, _ in PRIMARY + SECONDARY]
    _write_csv(os.path.join(out_dir, "metrics_long.csv"), keys,
               [[result_text(r) if k == "result" else r[k] for k in keys] for r in rows])
    _write_csv(os.path.join(out_dir, "table_by_angle.csv"), ["theta0_deg", "metric"] + names, by_angle)
    _write_csv(os.path.join(out_dir, "success_table.csv"), ["theta0_deg"] + names, success)
    _write_csv(os.path.join(out_dir, "summary_by_controller.csv"), summary_header, summary)


def print_tables(tables, angles, names):
    by_angle, success, summary, _, common = tables
    w0, w = 34, 26
    for a in angles:
        print(f"\n━━ 초기각 θ0 = {a:+g}°   (s0 = [0, 0, {a:+g}°, 0]) " + "━" * 30)
        print(f"{'Metric':<{w0}}" + "".join(f"{n:>{w}}" for n in names))
        for line in (l for l in by_angle if l[0] == a):
            print(f"{line[1]:<{w0}}" + "".join(f"{c:>{w}}" for c in line[2:]))

    print("\n━━ 성공 여부 (초기각 × 제어기) " + "━" * 40)
    print(f"{'theta0':<{w0}}" + "".join(f"{n:>{w}}" for n in names))
    for line in success:
        print(f"{line[0]:<{w0}}" + "".join(f"{c:>{w}}" for c in line[1:]))

    print("\n━━ 제어기별 평균 (모든 제어기가 성공한 공통 초기각만 사용) " + "━" * 14)
    if not common:
        print("  공통으로 성공한 초기각이 없어 평균을 내지 않습니다. 초기각별 표를 보세요.")
    else:
        print(f"  공통 성공 초기각: {', '.join(f'{a:+g}°' for a in common)}")
        print(f"{'':<{w0}}" + "".join(f"{n:>{w}}" for n in names))
        for i, (key, label, _) in enumerate(PRIMARY):
            print(f"{label:<{w0}}" + "".join(f"{s[3 + i]:>{w}}" for s in summary))


# ---------------- 그래프 ----------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_timeseries(axes, logs, rows, a, names, legend_fs=8):
    ax_th, ax_x, ax_u = axes
    hold0 = C.SIM_TIME - CRIT["hold_time"]
    for ax in axes:
        ax.axvspan(hold0, C.SIM_TIME, color="gold", alpha=0.12, lw=0)        # 성공 판정 구간 (마지막 2초)
        ax.grid(alpha=0.3)
    ax_th.axhspan(-CRIT["theta_deg"], CRIT["theta_deg"], color="gray", alpha=0.15, lw=0)
    ax_x.axhspan(-CRIT["x"], CRIT["x"], color="gray", alpha=0.15, lw=0)
    for v in (-C.F_MAX, C.F_MAX):
        ax_u.axhline(v, color="red", ls=":", lw=0.8)
    for n in names:
        log, r, col = logs[(a, n)], row_of(rows, a, n), COLORS.get(n)
        ts = r["settling_time_s"]
        label = f"{n}: {result_text(r, short=True)}" + (f", Ts={ts:.2f}s" if np.isfinite(ts) else "")
        ax_th.plot(log["t"], np.degrees(log["s"][:, 2]), color=col, lw=1.4, label=label)
        ax_x.plot(log["t"], log["s"][:, 0], color=col, lw=1.4)
        if len(log["u"]):
            ax_u.step(log["t"][:-1], log["u"], where="post", color=col, lw=1.0)
        if np.isfinite(ts):
            for ax in axes:
                ax.axvline(ts, color=col, ls="--", lw=0.8, alpha=0.6)
        if r["terminated"]:
            ax_th.plot(log["t"][-1], np.degrees(log["s"][-1, 2]), "x", color=col, ms=10, mew=2)
            ax_x.plot(log["t"][-1], log["s"][-1, 0], "x", color=col, ms=10, mew=2)
    ax_th.set_ylabel("θ [deg]")
    ax_x.set_ylabel("x [m]")
    ax_u.set_ylabel("F [N]")
    ax_u.set_xlabel("time [s]")
    ax_th.set_xlim(0, C.SIM_TIME)
    ax_th.legend(fontsize=legend_fs, loc="upper right")


def plot_timeseries(logs, rows, angles, names, out_dir):
    plt = _plt()
    note = (f"gray band: |θ|≤{CRIT['theta_deg']:g}°, |x|≤{CRIT['x']:g} m   yellow: last {CRIT['hold_time']:g} s "
            f"(success window)   dashed: settling time   red dotted: ±F_MAX")
    for a in angles:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
        _draw_timeseries(axes, logs, rows, a, names)
        axes[0].set_title(f"θ0 = {a:+g}°   (s0 = [0, 0, {a:+g}°, 0])")
        fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="dimgray")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(os.path.join(out_dir, f"timeseries_{angle_tag(a)}.png"), dpi=DPI)
        plt.close(fig)

    n = len(angles)
    for t_max, fname in [(C.SIM_TIME, "timeseries_overview.png"), (ZOOM_TIME, "timeseries_overview_zoom.png")]:
        if not t_max:
            continue
        fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 8.5), sharex=True, squeeze=False)
        for j, a in enumerate(angles):
            _draw_timeseries(axes[:, j], logs, rows, a, names, legend_fs=6.5)
            axes[0, j].set_title(f"θ0 = {a:+g}°" + (f"   (0–{t_max:g} s)" if t_max < C.SIM_TIME else ""))
            axes[0, j].set_xlim(0, t_max)
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="dimgray")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(os.path.join(out_dir, fname), dpi=DPI)
        plt.close(fig)


def _bar_panel(ax, rows, angles, names, key, title, fmt):
    nc = len(names)
    width = 0.8 / nc
    n_bars = len(angles) * nc
    annotate = n_bars <= 24
    rot = 0 if n_bars <= 12 else 90                           # 막대가 많으면 값 글씨를 세로로
    for i, n in enumerate(names):
        for j, a in enumerate(angles):
            r = row_of(rows, a, n)
            xp = j + (i - (nc - 1) / 2) * width
            v = r[key]
            if r["terminated"]:                               # 도중 종료 → 막대 없이 FAIL 표시
                ax.text(xp, 0, "FAIL", rotation=90, ha="center", va="bottom", fontsize=7,
                        color="red", fontweight="bold")
                continue
            if not np.isfinite(v):                            # Settling 정의 안 됨
                ax.text(xp, 0, "N/S", rotation=90, ha="center", va="bottom", fontsize=7, color=COLORS.get(n))
                continue
            ok = r["success"]
            ax.bar(xp, v, width * 0.92, color=COLORS.get(n), alpha=1.0 if ok else 0.45,
                   hatch=None if ok else "///", edgecolor="none" if ok else "k", lw=0.5)
            if annotate:
                ax.text(xp, v, " " + fmt.format(v) if rot else fmt.format(v), ha="center", va="bottom",
                        fontsize=6.5, rotation=rot)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:+g}°" for a in angles])
    ax.set_xlabel("initial angle θ0")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.15)


def _success_panel(ax, rows, angles, names):
    from matplotlib.colors import ListedColormap
    M = np.array([[1.0 if row_of(rows, a, n)["success"] else 0.0 for a in angles] for n in names])
    ax.imshow(M, cmap=ListedColormap(["#f6c1c1", "#bfe6bf"]), vmin=0, vmax=1, aspect="auto")
    for i, n in enumerate(names):
        for j, a in enumerate(angles):
            r = row_of(rows, a, n)
            ax.text(j, i, result_text(r, short=True).replace("(", "\n("), ha="center", va="center", fontsize=7)
    rates = [100.0 * np.mean(M[i]) for i in range(len(names))]
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:+g}°" for a in angles])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\n{rt:.0f}%" for n, rt in zip(names, rates)])
    ax.set_xlabel("initial angle θ0")
    ax.set_title("Stabilization Success  (success rate)", fontsize=10)


def _legend_handles(names):
    from matplotlib.patches import Patch
    h = [Patch(color=COLORS.get(n), label=n) for n in names]
    h.append(Patch(facecolor="lightgray", hatch="///", edgecolor="k", label="ran 10 s but FAIL (criteria)"))
    h.append(Patch(color="white", label="FAIL = terminated (no bar),  N/S = Not Settled"))
    return h


def plot_metrics(rows, angles, names, out_dir):
    plt = _plt()
    # 1) 핵심 지표
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    _success_panel(axes[0], rows, angles, names)
    for ax, (key, title, fmt) in zip(axes[1:], PRIMARY):
        _bar_panel(ax, rows, angles, names, key, title, fmt)
    fig.legend(handles=_legend_handles(names), loc="lower center", ncol=len(names) + 2, fontsize=9)
    fig.suptitle("Evaluation metrics by initial angle — primary", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "metrics_primary.png"), dpi=DPI)
    plt.close(fig)

    # 2) 보조 지표
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (key, title, fmt) in zip(axes.ravel(), SECONDARY):
        _bar_panel(ax, rows, angles, names, key, title, fmt)
    fig.legend(handles=_legend_handles(names), loc="lower center", ncol=len(names) + 2, fontsize=9)
    fig.suptitle("Evaluation metrics by initial angle — secondary (for interpretation)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "metrics_secondary.png"), dpi=DPI)
    plt.close(fig)


def plot_tradeoff(rows, angles, names, out_dir):
    """성능–비용 trade-off: 같은 초기각에서 왼쪽 아래일수록 '적은 입력으로 정확/빠르게'"""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (key, ylabel) in zip(axes, [("angle_rmse_deg", "Angle RMSE [deg]"),
                                         ("settling_time_s", "Settling time [s]")]):
        for n in names:
            pts = [(r["control_effort"], r[key], r["theta0_deg"], r["success"])
                   for r in (row_of(rows, a, n) for a in angles)
                   if not r["terminated"] and np.isfinite(r[key])]
            if not pts:
                continue
            col = COLORS.get(n)
            for sgn in (1, -1):                               # +각도끼리, -각도끼리 |θ0| 순서로 연결
                seg = sorted((p for p in pts if np.sign(p[2]) == sgn), key=lambda p: abs(p[2]))
                ax.plot([p[0] for p in seg], [p[1] for p in seg], color=col, lw=0.8, alpha=0.5)
            for ex, ey, a, ok in pts:
                ax.plot(ex, ey, "o", ms=8, color=col, mfc=col if ok else "white", mew=1.5)
                ax.annotate(f"{a:+g}°", (ex, ey), textcoords="offset points", xytext=(5, 4), fontsize=8)
            ax.plot([], [], "o", color=col, label=n)
        ax.set_xlabel("Control effort ∫F² dt [N²·s]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_title("Accuracy vs. control cost", fontsize=10)
    axes[1].set_title("Speed vs. control cost  (Not Settled omitted)", fontsize=10)
    fig.suptitle("Performance–cost trade-off   (filled = SUCCESS, hollow = FAIL by criteria; "
                 "terminated runs omitted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tradeoff.png"), dpi=DPI)
    plt.close(fig)


# ---------------- main ----------------
def main():
    angles = [float(v) for v in sys.argv[1:]] or list(ANGLES_DEG)
    out_dir = os.path.join(C.RESULT_ROOT, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    controllers = load_controllers(CONTROLLERS)
    names = [c.name for c in controllers]
    print(f"\n[평가 조건] 제어기 {names}, 초기각 {[f'{a:+g}°' for a in angles]}, "
          f"{C.SIM_TIME:g} s, dt={C.DT:g} s, F_MAX=±{C.F_MAX:g} N")
    print(f"  노이즈 {C.NOISE_STD}, 외란 {'ON' if C.DISTURBANCE['enabled'] else 'OFF'}   "
          f"성공 기준: 마지막 {CRIT['hold_time']:g}초 |θ|≤{CRIT['theta_deg']:g}°, "
          f"|θ̇|≤{CRIT['theta_dot']:g} rad/s, |x|≤{CRIT['x']:g} m, 10초 동안 |x|≤{C.X_MAX:g} m")

    rows, logs = run_all(controllers, angles)
    tables = build_tables(rows, angles, names)

    if SAVE_RAW:
        save_raw(logs, out_dir)
    save_all_csv(rows, tables, names, out_dir)
    print_tables(tables, angles, names)

    plot_timeseries(logs, rows, angles, names, out_dir)
    plot_metrics(rows, angles, names, out_dir)
    plot_tradeoff(rows, angles, names, out_dir)
    print(f"\n[평가 완료] 결과 저장 위치: {out_dir}/  (CSV 4개 + raw/ + 그래프)")
    return rows, logs


if __name__ == "__main__":
    main()
