"""
Cart-Pole 역진자 — 초기각별 제어기 비교 (PID / LQR / PPO,  ±5° ~ ±30°, 5° 간격)
================================================================
초기각 θ0 = ±5°, ±10°, ..., ±30° 에서 세 제어기를 같은 조건으로 실행하고
시간응답(θ, x, F), overshoot, 5개 평가지표를 각도·제어기별 표와 그래프로 정리합니다.

기존 파일은 수정하지 않고 import 만 합니다.
  - 제어기   : simulator.load_controllers() → 각 제어기 파일의 build_controller()
               (최적화한 값 그대로: PID_THETA / PID_X,  LQR_Q / LQR_R,  models/ppo_cartpole_balance.zip)
  - 실행     : cartpole_common.simulate()  (같은 시뮬레이터 / dt / F_MAX / 외란·노이즈 설정)
  - 판정·지표: evaluate_metrics.py 의 judge(), compute_pdf_metrics() 재사용 (같은 정의·같은 기준)

두 종류의 실행
  ① 기준 실행 (각도·제어기마다 1회): s0 = [x, ẋ, θ, θ̇] = [0, 0, θ0, 0]
     → 시간응답, overshoot, Angle RMSE, Settling Time, Control Effort, Position RMSE
  ② 반복 실행 (각도·제어기마다 MC_TRIALS 회): θ0 는 고정, 나머지 초기상태 x, ẋ, θ̇ 를 무작위로
     → Success Rate = 성공 횟수 / 시행 횟수
       (기준 실행은 결정론적이라 한 각도의 성공률이 0 % 아니면 100 % 이므로 반복 실행으로 계산)
     · 무작위 범위 기본값 = cartpole_common.DESIGN_INIT_RANGES (PPO 학습·PID/LQR 튜닝에 쓴 분포)
     · 시행 i 의 무작위 초기상태는 모든 제어기·모든 각도에서 같음 (공정 비교)
     · −θ0 에서는 +θ0 의 초기상태를 좌우 반전해서 사용 → PID·LQR 은 ± 결과가 정확히 같고,
       ± 차이는 제어기 자체의 비대칭(PPO)에서만 생김

평가지표
  1. Angle RMSE [deg]       ↓  sqrt(mean θ_k²), k = 1..N
  2. Settling Time [s]      ↓  |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s 에 들어가 끝까지(최소 2 s) 유지되는 첫 시각
                               (SETTLING_USE_X = True 면 |x| ≤ 0.2 m 도 포함 = evaluate_metrics 기본 방식)
  3. Success Rate [%]       ↑  반복 실행 중 evaluate_metrics.judge() 성공 비율
                               (10 s 동안 |x| ≤ 2.4 m  그리고  마지막 2 s |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s, |x| ≤ 0.2 m)
  4. Control Effort [N²·s]  ↓  Σ F_k² Δt
  5. Position RMSE [m]      ↓  sqrt(mean x_k²), k = 1..N

Overshoot (기준 실행, θ0 의 부호 방향을 "+" 로 봄)
  θ overshoot [deg] : 수직(0°)을 지나 반대쪽으로 넘어간 최대 각도   (θ overshoot [%] = |θ0| 대비)
  x overshoot [m]   : 카트가 θ0 쪽으로 나갔다가(x peak) 중앙을 지나 반대쪽으로 넘어간 최대 거리
  F overshoot [N]   : 처음 미는 방향과 반대 방향(제동)으로 낸 최대 힘   (F peak = 처음 방향 최대 힘)

실행 (다른 파일들과 같은 폴더에서):
  python angle_compare.py

  * PPO는 먼저 'python cartpole_ppo.py train' 으로 모델을 만들어야 합니다 (없으면 자동으로 건너뜀).

결과 (results/angle_compare/):
  metrics_long.csv            각도 × 제어기 = 1행, 5개 지표 + overshoot + 성공률 95% 신뢰구간 (분석용)
  table_by_angle.csv          각도별 [지표 × 제어기] 표
  summary_by_controller.csv   제어기별 전체 성공률 + 공통 성공 각도에서의 평균 지표
  mc_trials.csv               반복 실행 1회 = 1행 (무작위 초기상태, 성공 여부, 지표)
  raw/theta0_+5deg_PID.csv    기준 실행 원시 데이터 t, x, ẋ, θ, θ̇, F
  table_metrics.png           5개 지표 표 (각도 × 제어기, 각 행의 최고값 강조)
  table_overshoot.png         overshoot 표
  metrics_vs_angle.png        5개 지표의 초기각에 따른 변화 (+ 제어기별 전체 성공률)
  overshoot_vs_angle.png      θ / x / F 의 peak 와 overshoot 의 초기각에 따른 변화
  timeseries/theta0_+05deg.png   초기각별 θ / x / F 시간응답 (세 제어기 겹쳐 그림, overshoot 지점 표시)
  timeseries_overview_theta.png  모든 초기각의 θ(t) 한 장에  (_x, _F 도 같은 형식)
  timeseries_by_controller.png   제어기별로 모든 초기각의 θ / x / F (색 = 초기각)
"""

# ================================================================
# 0. 비교 설정 — 여기만 바꾸면 됩니다
# ================================================================
ANGLE_MAGS_DEG = [5, 10, 15, 20, 25, 30]   # |θ0| [deg] → +θ0, −θ0 모두 실행 (총 12개 각도)
CONTROLLERS = ["PID", "LQR", "PPO"]        # 모델이 없는 제어기는 자동으로 건너뜀

SETTLING_USE_X = False          # Settling Time 기준: False = θ, θ̇ 만 / True = θ, θ̇, x (evaluate_metrics 기본)

# ---------- Success Rate (반복 실행) ----------
MC_TRIALS = 50                  # 각도·제어기당 시행 횟수
MC_INIT_RANGES = None           # 무작위로 뽑을 나머지 초기상태 범위 (균일분포)
                                #   None → cartpole_common.DESIGN_INIT_RANGES 의 x, x_dot, theta_dot (±0.6)
                                #   예: dict(x=(-0.3, 0.3), x_dot=(-0.3, 0.3), theta_dot=(-0.3, 0.3))
MC_NOISE_STD = None             # 측정 노이즈 표준편차 [x, x_dot, θ(rad), θ_dot]. None → cartpole_common.NOISE_STD (기본 0)
                                #   주의: 노이즈를 켜면 "마지막 2 s |θ̇| ≤ 0.2 rad/s" 기준 때문에 넘어지지 않아도
                                #   FAIL 이 될 수 있음 (θ 노이즈 0.2° 만으로도 PID·LQR 이 θ̇ 기준을 넘음)
MC_MIRROR = True                # −θ0 에서 +θ0 의 무작위 초기상태를 좌우 반전해서 사용
MC_SEED = 0

# ---------- 그래프 ----------
ZOOM_TIME = 2.5                 # 초기각별 시간응답 그림의 오른쪽 확대 구간 0 ~ ZOOM_TIME [s]
OVERVIEW_TIME = dict(theta=4.0, x=10.0, F=2.0)   # overview / by_controller 그림의 가로축 0 ~ 이 시간 [s]
OUT_SUBDIR = "angle_compare"    # 결과 폴더: results/angle_compare/
SAVE_RAW = True                 # 기준 실행의 원시 데이터 CSV 저장 → raw/
DPI = 140

# ================================================================
# 이하 구현부
# ================================================================
import os
import csv
import numpy as np
import cartpole_common as C
import evaluate_metrics as E                   # 성공 판정·지표 계산을 그대로 재사용
from simulator import load_controllers

NAN = float("nan")

# 제어기 색: 색각이상(적록·청황)에서도 서로 구분되는지 검증한 3색 (기존 그림과 같은 계열: 파랑/주황/초록)
COLORS = {"PID": "#2a78d6", "LQR": "#eb6834", "PPO": "#1baf7a"}
MARKERS = {"PID": "o", "LQR": "s", "PPO": "^"}
LINESTYLES = {"PID": "-", "LQR": "--", "PPO": "-."}
INK, INK2, GRID, BAND = "#1f1f1e", "#52514e", "#dddcd8", "#ecebe7"

# (key, 이름, 숫자 형식, 좋은 방향)
METRICS = [
    ("angle_rmse_deg",   "Angle RMSE [deg]",      "{:.3f}", "min"),
    ("settling_time_s",  "Settling Time [s]",     "{:.2f}", "min"),
    ("success_rate_pct", "Success Rate [%]",      "{:.0f}", "max"),
    ("control_effort",   "Control Effort [N²·s]", "{:.1f}", "min"),
    ("position_rmse_m",  "Position RMSE [m]",     "{:.4f}", "min"),
]
OVERSHOOT = [
    ("theta_overshoot_deg", "θ overshoot [deg]",       "{:.2f}", "min"),
    ("theta_overshoot_pct", "θ overshoot [% of |θ0|]", "{:.1f}", "min"),
    ("x_peak_m",            "x peak [m]",              "{:.3f}", "min"),
    ("x_overshoot_m",       "x overshoot [m]",         "{:.3f}", "min"),
    ("F_overshoot_N",       "F overshoot [N]",         "{:.1f}", "min"),
]
PEAKS = [("theta_peak_deg", "θ peak [deg]", "{:.2f}", "min"),
         ("F_peak_N",       "F peak [N]",   "{:.1f}", "min")]
ALL_KEYS = METRICS + OVERSHOOT + PEAKS
LABEL = {k: lab for k, lab, _, _ in ALL_KEYS}


def angle_list():
    mags = sorted({abs(int(m)) for m in ANGLE_MAGS_DEG})
    return [-m for m in reversed(mags)] + mags          # −30 … −5, +5 … +30


def row_of(rows, a, name):
    return next(r for r in rows if r["theta0_deg"] == a and r["controller"] == name)


def settling_desc():
    s = f"|θ|≤{E.CRIT['theta_deg']:g}°, |θ̇|≤{E.CRIT['theta_dot']:g} rad/s"
    return s + (f", |x|≤{E.CRIT['x']:g} m" if SETTLING_USE_X else "")


# ---------------- 지표 ----------------
def peak_and_overshoot(sig, sgn, t):
    """θ0 방향(+sgn) 최대값, 반대 방향 최대값(overshoot), overshoot 시각"""
    if len(sig) == 0:
        return 0.0, 0.0, NAN
    rev = -sgn * sig
    k = int(np.argmax(rev))
    os_ = float(max(0.0, rev[k]))
    return float(max(0.0, np.max(sgn * sig))), os_, (float(t[k]) if os_ > 0 else NAN)


def nominal_metrics(log, a):
    m = E.compute_pdf_metrics(log, a)
    sgn = float(np.sign(a))
    t, th, x, u = log["t"], np.degrees(log["s"][:, 2]), log["s"][:, 0], log["u"]
    th_pk, th_os, th_t = peak_and_overshoot(th, sgn, t)
    x_pk, x_os, x_t = peak_and_overshoot(x, sgn, t)
    F_pk, F_os, F_t = peak_and_overshoot(u, sgn, t[:-1])
    return dict(
        nominal_success=m["success"], nominal_result=E.result_text(m), terminated=m["terminated"],
        angle_rmse_deg=m["angle_rmse_deg"],
        settling_time_s=m["settling_time_s"] if SETTLING_USE_X else m["settling_time_theta_s"],
        control_effort=m["control_effort"],
        position_rmse_m=m["cart_rmse_m"],
        u_rms_N=m["u_rms_N"],
        theta_peak_deg=th_pk, theta_overshoot_deg=th_os, theta_overshoot_pct=100.0 * th_os / abs(a),
        x_peak_m=x_pk, x_overshoot_m=x_os,
        F_peak_N=F_pk, F_overshoot_N=F_os,
        saturation_pct=m["saturation_pct"],
        _t_os=dict(theta=th_t, x=x_t, F=F_t),           # 그래프 표시용 (CSV에는 저장 안 함)
        _os=dict(theta=th_os, x=x_os, F=F_os), _sgn=sgn,
    )


def wilson_ci(k, n, z=1.96):
    """성공률의 95 % 신뢰구간 (Wilson) [%]"""
    if n == 0:
        return NAN, NAN
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# ---------------- 실행 ----------------
def run_nominal(controllers, angles):
    rows, logs = [], {}
    for a in angles:
        s0 = C.make_s0(dict(theta_deg=a))               # BASE_S0 에서 θ0 만 바꿈 → [0, 0, θ0, 0]
        for ctrl in controllers:
            log = C.simulate(ctrl, s0, np.random.default_rng(C.SEED))
            rows.append(dict(theta0_deg=a, controller=ctrl.name, **nominal_metrics(log, a)))
            logs[(a, ctrl.name)] = log
    return rows, logs


def mc_ranges():
    R = C.DESIGN_INIT_RANGES if MC_INIT_RANGES is None else MC_INIT_RANGES
    return {k: tuple(R[k]) for k in ("x", "x_dot", "theta_dot")}


def mc_initial_states(a):
    R = mc_ranges()
    out = []
    for i in range(MC_TRIALS):
        rng = np.random.default_rng([MC_SEED, i])       # 시행 i → 모든 각도·제어기에서 같은 값
        d = {k: float(rng.uniform(*R[k])) for k in ("x", "x_dot", "theta_dot")}
        if MC_MIRROR and a < 0:
            d = {k: -v for k, v in d.items()}
        d["theta_deg"] = a
        out.append(d)
    return out


def run_mc(controllers, angles):
    noise = C.NOISE_STD if MC_NOISE_STD is None else MC_NOISE_STD
    trials = []
    names = [c.name for c in controllers]
    print(f"\n[반복 실행] 각도·제어기당 {MC_TRIALS}회  (θ0 고정, x·ẋ·θ̇ ~ U{mc_ranges()}, 노이즈 {list(noise)})")
    print(f"{'θ0':>6}  " + "".join(f"{n:>16}" for n in names))
    for a in angles:
        states = mc_initial_states(a)
        cells = []
        for ctrl in controllers:
            k = 0
            for i, d in enumerate(states):
                log = C.simulate(ctrl, C.make_s0(d), np.random.default_rng([MC_SEED, i, 1]), noise_std=noise)
                m = E.compute_pdf_metrics(log, a)
                k += m["success"]
                trials.append(dict(
                    theta0_deg=a, controller=ctrl.name, trial=i,
                    x0_m=d["x"], x_dot0_mps=d["x_dot"], theta_dot0_radps=d["theta_dot"],
                    success=m["success"], result=E.result_text(m), why=m["why"], terminated=m["terminated"],
                    angle_rmse_deg=m["angle_rmse_deg"],
                    settling_time_s=m["settling_time_s"] if SETTLING_USE_X else m["settling_time_theta_s"],
                    control_effort=m["control_effort"], position_rmse_m=m["cart_rmse_m"]))
            cells.append(f"{100 * k / MC_TRIALS:.0f}% ({k}/{MC_TRIALS})")
        print(f"{a:>+5g}°  " + "".join(f"{c:>16}" for c in cells))
    return trials


def merge_success(rows, trials):
    for r in rows:
        ts = [t for t in trials if t["theta0_deg"] == r["theta0_deg"] and t["controller"] == r["controller"]]
        k, n = sum(t["success"] for t in ts), len(ts)
        lo, hi = wilson_ci(k, n)
        reasons = {}
        for t in ts:
            if not t["success"]:                        # 도중 종료면 종료 원인 1개, 아니면 마지막 2 s 에 넘은 기준들
                key = f"{E.WHY_SHORT[t['why'][0]]} (terminated)" if t["terminated"] else \
                      "last 2s: " + "/".join(E.WHY_SHORT[w] for w in t["why"])
                reasons[key] = reasons.get(key, 0) + 1
        r.update(success_rate_pct=100.0 * k / n if n else NAN, success_n=k, trials_n=n,
                 success_ci95_lo=lo, success_ci95_hi=hi,
                 mc_fail_reasons=", ".join(f"{w} ×{c}" for w, c in reasons.items()))


# ---------------- 표 ----------------
def value_of(r, key):
    """비교에 쓸 값 (도중 종료된 기준 실행은 지표를 비교하지 않음 → NaN)"""
    if key == "success_rate_pct":
        return r[key]
    return NAN if r["terminated"] else r[key]


def cell(r, key, fmt, short=False):
    v = value_of(r, key)
    if np.isfinite(v):
        return fmt.format(v)
    if r["terminated"] and key != "success_rate_pct":
        return "FAIL" if short else "— (FAIL)"
    return ("N/S" if short else "Not Settled") if key == "settling_time_s" else "—"


def best_index(rs, key, fmt, better):
    """표시 자릿수 기준으로 가장 좋은 제어기들 (모두 같으면 강조하지 않음)"""
    vals = []
    for r in rs:
        v = value_of(r, key)
        if not np.isfinite(v):
            v = np.inf if better == "min" else -np.inf
        else:
            v = float(fmt.format(v))
        vals.append(v)
    target = min(vals) if better == "min" else max(vals)
    if not np.isfinite(target):
        return set()
    best = {i for i, v in enumerate(vals) if v == target}
    return set() if len(best) == len(vals) else best


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def common_success_angles(rows, angles, names):
    """모든 제어기의 기준 실행이 성공한 각도 (평균은 여기서만 → 생존 편향 방지)"""
    return [a for a in angles if all(row_of(rows, a, n)["nominal_success"] for n in names)]


def controller_summary(rows, trials, angles, names):
    common = common_success_angles(rows, angles, names)
    out = []
    for n in names:
        ts = [t for t in trials if t["controller"] == n]
        k, N = sum(t["success"] for t in ts), len(ts)
        lo, hi = wilson_ci(k, N)
        line = dict(controller=n, overall_success_rate_pct=100.0 * k / N if N else NAN,
                    overall_success_ci95_lo=lo, overall_success_ci95_hi=hi, success_n=k, trials_n=N,
                    nominal_success=f"{sum(row_of(rows, a, n)['nominal_success'] for a in angles)}/{len(angles)}",
                    common_success_angles_deg=" ".join(f"{a:+g}" for a in common))
        for key in ("angle_rmse_deg", "settling_time_s", "control_effort", "position_rmse_m",
                    "theta_overshoot_deg", "x_overshoot_m", "F_overshoot_N"):
            vals = [row_of(rows, a, n)[key] for a in common]
            line[f"mean_{key}"] = float(np.mean(vals)) if vals else NAN
        out.append(line)
    return out, common


def save_tables(rows, trials, summary, angles, names, out_dir):
    long_keys = ["theta0_deg", "controller",
                 "success_rate_pct", "success_n", "trials_n", "success_ci95_lo", "success_ci95_hi", "mc_fail_reasons",
                 "nominal_result", "angle_rmse_deg", "settling_time_s", "control_effort", "position_rmse_m",
                 "u_rms_N", "theta_peak_deg", "theta_overshoot_deg", "theta_overshoot_pct",
                 "x_peak_m", "x_overshoot_m", "F_peak_N", "F_overshoot_N", "saturation_pct"]
    _write_csv(os.path.join(out_dir, "metrics_long.csv"), long_keys,
               [[r[k] for k in long_keys] for r in rows])

    by_angle = []
    for a in angles:
        rs = [row_of(rows, a, n) for n in names]
        by_angle.append([a, "Nominal run (s0=[0,0,θ0,0])"] + [r["nominal_result"] for r in rs])
        for i, (key, label, fmt, better) in enumerate(METRICS):
            arrow = " ↓" if better == "min" else " ↑"
            by_angle.append([a, f"{i + 1}. {label}{arrow}"] + [cell(r, key, fmt) for r in rs])
        for key, label, fmt, _ in OVERSHOOT + PEAKS:
            by_angle.append([a, label] + [cell(r, key, fmt) for r in rs])
    _write_csv(os.path.join(out_dir, "table_by_angle.csv"), ["theta0_deg", "metric"] + names, by_angle)

    _write_csv(os.path.join(out_dir, "summary_by_controller.csv"), list(summary[0].keys()),
               [list(s.values()) for s in summary])

    tk = ["theta0_deg", "controller", "trial", "x0_m", "x_dot0_mps", "theta_dot0_radps", "success", "result",
          "angle_rmse_deg", "settling_time_s", "control_effort", "position_rmse_m"]
    _write_csv(os.path.join(out_dir, "mc_trials.csv"), tk, [[t[k] for k in tk] for t in trials])


def print_tables(rows, summary, common, angles, names):
    w0, w = 26, 8
    head = f"{'':<{w0}}" + "".join(f"{a:>+{w - 1}g}°" for a in angles)
    for title, blocks in (("평가지표", METRICS), ("Overshoot (기준 실행)", OVERSHOOT[:1] + OVERSHOOT[3:])):
        print(f"\n━━ {title}   (* = 그 각도에서 가장 좋은 값) " + "━" * 40)
        for key, label, fmt, better in blocks:
            arrow = "↓" if better == "min" else "↑"
            print(f"\n{label} {arrow}")
            print(head)
            best = {a: best_index([row_of(rows, a, n) for n in names], key, fmt, better) for a in angles}
            for i, n in enumerate(names):
                cells = []
                for a in angles:
                    c = cell(row_of(rows, a, n), key, fmt, short=True) + ("*" if i in best[a] else " ")
                    cells.append(f"{c:>{w}}")
                print(f"  {n:<{w0 - 2}}" + "".join(cells))

    print("\n━━ 제어기별 요약 " + "━" * 64)
    print(f"  평균은 모든 제어기의 기준 실행이 성공한 각도에서만: {' '.join(f'{a:+g}°' for a in common) or '없음'}")
    print(f"  {'':<6}{'Success (전체)':>22}{'RMSE[deg]':>11}{'Ts[s]':>8}{'Effort':>9}{'PosRMSE':>9}{'θ OS[deg]':>11}")
    for s in summary:
        sr = f"{s['overall_success_rate_pct']:.1f}% [{s['overall_success_ci95_lo']:.0f}–{s['overall_success_ci95_hi']:.0f}]"
        print(f"  {s['controller']:<6}{sr:>22}{s['mean_angle_rmse_deg']:>11.3f}{s['mean_settling_time_s']:>8.2f}"
              f"{s['mean_control_effort']:>9.1f}{s['mean_position_rmse_m']:>9.4f}{s['mean_theta_overshoot_deg']:>11.2f}")


# ---------------- 그래프 공통 ----------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9a9995", "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "grid.color": GRID, "grid.linewidth": 0.6,
    })
    return plt


def _c(name):
    return COLORS.get(name, "#6b6a66")


def _angle_axis(ax, angles):
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{a:+g}°" for a in angles], fontsize=7.5)
    ax.axvline(0, color="#c9c8c4", lw=0.8)
    ax.set_xlabel("initial angle θ0")
    ax.grid(True)


def _angle_line(ax, angles, vals, name, dodge=0.0, err=None):
    """−θ0 쪽과 +θ0 쪽을 따로 이음 (θ0 = 0 은 실행하지 않으므로 가운데를 잇지 않음)"""
    col = _c(name)
    first = True
    for side in (-1, 1):
        idx = [i for i, a in enumerate(angles) if np.sign(a) == side]
        if not idx:
            continue
        xs = np.array([angles[i] for i in idx], float) + dodge
        ys = np.array([vals[i] for i in idx], float)
        if err is not None:
            lo = np.array([err[0][i] for i in idx], float)
            hi = np.array([err[1][i] for i in idx], float)
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none", ecolor=col, elinewidth=1.0,
                        capsize=2.5, alpha=0.8, zorder=2)
        ax.plot(xs, ys, color=col, ls=LINESTYLES.get(name, "-"), lw=1.8, marker=MARKERS.get(name, "o"),
                ms=6.5, mec="white", mew=0.9, label=name if first else None, zorder=3)
        first = False


def _mark_missing(ax, rows, angles, names, key):
    """값이 없는 점(도중 종료 / Not Settled)을 그래프 아래쪽에 글자로 표시"""
    for i, n in enumerate(names):
        for a in angles:
            r = row_of(rows, a, n)
            if not np.isfinite(value_of(r, key)):
                txt = "FAIL" if r["terminated"] else "N/S"
                ax.text(a, 0.03 + 0.07 * i, f"{n} {txt}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=6.5, color=INK2)


def _signal(log, sig):
    t = log["t"]
    if sig == "theta":
        return t, np.degrees(log["s"][:, 2])
    if sig == "x":
        return t, log["s"][:, 0]
    return t[:-1], log["u"]


SIG_LABEL = dict(theta="θ [deg]", x="x [m]", F="F [N]")
SIG_UNIT = dict(theta="°", x=" m", F=" N")
SIG_FMT = dict(theta="{:.2f}", x="{:.3f}", F="{:.1f}")


def _draw_signal(ax, log, r, name, sig, t_max, lw=1.6):
    t, y = _signal(log, sig)
    kw = dict(color=_c(name), ls=LINESTYLES.get(name, "-"), lw=lw)
    if sig == "F":
        if len(y):
            ax.step(t, y, where="post", **kw)
    else:
        ax.plot(t, y, **kw)
    t_os = r["_t_os"][sig]
    if np.isfinite(t_os) and t_os <= t_max:                      # overshoot 지점
        ax.plot(t_os, -r["_sgn"] * r["_os"][sig], marker=MARKERS.get(name, "o"), ms=7.5,
                color=_c(name), mec="white", mew=1.0, zorder=5, ls="none")
    if r["terminated"]:
        ax.plot(t[-1] if len(t) else 0, y[-1] if len(y) else 0, "x", color=INK, ms=9, mew=2, zorder=6)


def _sig_background(ax, sig):
    if sig == "theta":
        ax.axhspan(-E.CRIT["theta_deg"], E.CRIT["theta_deg"], color=BAND, lw=0, zorder=0)
    if sig == "x" and SETTLING_USE_X:
        ax.axhspan(-E.CRIT["x"], E.CRIT["x"], color=BAND, lw=0, zorder=0)
    if sig == "F":
        for v in (-C.F_MAX, C.F_MAX):
            ax.axhline(v, color="#a9a8a4", ls=":", lw=0.9)
    ax.axhline(0, color="#c9c8c4", lw=0.7, zorder=0)
    ax.grid(True)


# ---------------- 그래프: 지표 ----------------
def plot_metrics_vs_angle(rows, summary, angles, names, out_dir):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    nc = len(names)
    for ax, (key, label, fmt, better) in zip(axes, METRICS):
        if key == "success_rate_pct":
            for i, n in enumerate(names):
                rs = [row_of(rows, a, n) for a in angles]
                _angle_line(ax, angles, [r[key] for r in rs], n, dodge=(i - (nc - 1) / 2) * 0.7,
                            err=([r["success_ci95_lo"] for r in rs], [r["success_ci95_hi"] for r in rs]))
            lowest = min(r["success_ci95_lo"] for r in rows)        # 아래쪽 빈 공간을 줄여 차이가 잘 보이게
            ax.set_ylim(max(0.0, np.floor((lowest - 5) / 10) * 10), 103)
        else:
            for n in names:
                _angle_line(ax, angles, [value_of(row_of(rows, a, n), key) for a in angles], n)
            _mark_missing(ax, rows, angles, names, key)
            ax.set_ylim(bottom=0)
        ax.set_title(f"{label}  {'↓ lower is better' if better == 'min' else '↑ higher is better'}")
        _angle_axis(ax, angles)
    axes[0].legend(loc="upper center")

    # 6번째 칸: 제어기별 전체 성공률 (모든 각도 × 모든 시행)
    ax = axes[5]
    for i, s in enumerate(summary):
        v, lo, hi = s["overall_success_rate_pct"], s["overall_success_ci95_lo"], s["overall_success_ci95_hi"]
        ax.barh(i, v, height=0.55, color=_c(s["controller"]))
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        inside = v > 40
        ax.text(2 if inside else v + 2, i, f"{v:.1f}%  ({s['success_n']}/{s['trials_n']})", va="center",
                ha="left", fontsize=8.5, color="white" if inside else INK, fontweight="bold")
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([s["controller"] for s in summary])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("success rate [%]  (bar = 95 % CI)")
    ax.set_title("Overall success rate — all angles × trials")
    ax.grid(True, axis="x")

    R = mc_ranges()
    note = (f"Success Rate: {MC_TRIALS} trials per angle, θ0 fixed, x, ẋ, dθ/dt ~ uniform "
            f"x{R['x']}, ẋ{R['x_dot']}, dθ/dt{R['theta_dot']}"
            f"{', mirrored for −θ0' if MC_MIRROR else ''}; error bars = 95 % Wilson CI.   "
            f"Other metrics: nominal run s0 = [0, 0, θ0, 0].   "
            f"Settling: |θ| ≤ {E.CRIT['theta_deg']:g}°, |dθ/dt| ≤ {E.CRIT['theta_dot']:g} rad/s"
            f"{', |x| ≤ %g m' % E.CRIT['x'] if SETTLING_USE_X else ''} held to the end (≥ {E.CRIT['hold_time']:g} s).")
    fig.text(0.5, 0.008, note, ha="center", fontsize=7.5, color=INK2, wrap=True)
    fig.suptitle("Controller comparison by initial angle θ0", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "metrics_vs_angle.png"), dpi=DPI)
    plt.close(fig)


def plot_overshoot_vs_angle(rows, angles, names, out_dir):
    plt = _plt()
    # θ peak(= θ0) 와 F peak(= F_MAX 포화) 는 모든 경우에 거의 같아 그래프에서 빼고 CSV 에만 저장
    panels = [[("theta_overshoot_deg", "θ overshoot (past upright, opposite side) [deg]"),
               ("x_overshoot_m", "x overshoot (past center, opposite side) [m]")],
              [("F_overshoot_N", "F overshoot (reverse / braking force) [N]"),
               ("x_peak_m", "x peak (max cart travel toward θ0 side) [m]")]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for i, row in enumerate(panels):
        for j, (key, title) in enumerate(row):
            ax = axes[i, j]
            for n in names:
                _angle_line(ax, angles, [value_of(row_of(rows, a, n), key) for a in angles], n)
            _mark_missing(ax, rows, angles, names, key)
            if key.startswith("F_"):
                ax.axhline(C.F_MAX, color="#a9a8a4", ls=":", lw=0.9)
                ax.text(angles[0], C.F_MAX, " F_MAX", va="bottom", fontsize=7, color=INK2)
            ax.set_ylim(bottom=0)
            ax.set_title(title)
            _angle_axis(ax, angles)
    axes[0, 0].legend(loc="upper center")
    fig.suptitle("Peak and overshoot by initial angle θ0  (nominal run s0 = [0, 0, θ0, 0]; "
                 "\"+\" direction = sign of θ0)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "overshoot_vs_angle.png"), dpi=DPI)
    plt.close(fig)


def plot_table(rows, angles, names, blocks, title, note, path):
    """각도 × [지표별 제어기] 표 그림. 각 행에서 가장 좋은 값은 굵게 + 배경 강조"""
    plt = _plt()
    from matplotlib.patches import Rectangle
    nb, nc = len(blocks), len(names)
    ncol, nrow = 1 + nb * nc, 2 + len(angles)
    fig, ax = plt.subplots(figsize=(0.66 * ncol + 0.4, 0.31 * nrow + 1.1))
    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow, 0)
    ax.axis("off")
    ax.text(0.5, 1.5, "θ0", ha="center", va="center", fontsize=8.5, fontweight="bold")
    for b, (key, label, fmt, better) in enumerate(blocks):
        x0 = 1 + b * nc
        arrow = " ↓" if better == "min" else " ↑"
        ax.text(x0 + nc / 2, 0.5, label + arrow, ha="center", va="center", fontsize=8.5, fontweight="bold")
        for j, n in enumerate(names):
            ax.text(x0 + j + 0.5, 1.5, n, ha="center", va="center", fontsize=8, color=INK2)
        ax.plot([x0, x0], [0.1, nrow], color="#cfcec9", lw=0.8)
    ax.plot([0, ncol], [2, 2], color=INK2, lw=0.8)
    for i, a in enumerate(angles):
        y0 = 2 + i
        if i % 2:
            ax.add_patch(Rectangle((0, y0), ncol, 1, color="#f5f4f1", lw=0, zorder=0))
        if i == len(angles) // 2 and angles[i - 1] < 0 < a:                  # −θ0 / +θ0 경계
            ax.plot([0, ncol], [y0, y0], color="#bdbcb8", lw=0.8)
        ax.text(0.5, y0 + 0.5, f"{a:+g}°", ha="center", va="center", fontsize=8)
        rs = [row_of(rows, a, n) for n in names]
        for b, (key, label, fmt, better) in enumerate(blocks):
            x0 = 1 + b * nc
            best = best_index(rs, key, fmt, better)
            for j, r in enumerate(rs):
                if j in best:
                    ax.add_patch(Rectangle((x0 + j + 0.05, y0 + 0.1), 0.9, 0.8, color="#d6e5f8", lw=0, zorder=1))
                ax.text(x0 + j + 0.5, y0 + 0.5, cell(r, key, fmt, short=True), ha="center", va="center",
                        fontsize=8, fontweight="bold" if j in best else "normal", zorder=2)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    fig.text(0.01, 0.01, note, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------- 그래프: 시간응답 ----------------
def _os_summary(rows, a, names, sig):
    unit, fmt = SIG_UNIT[sig], SIG_FMT[sig]
    return ",  ".join(f"{n} {fmt.format(row_of(rows, a, n)['_os'][sig])}{unit}" for n in names)


def plot_timeseries_angle(logs, rows, a, names, path):
    plt = _plt()
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col", gridspec_kw=dict(width_ratios=[1.7, 1]))
    sig_names = ("theta", "x", "F")
    os_title = dict(theta="θ overshoot", x="x overshoot", F="F overshoot (reverse)")
    for col, t_max in enumerate((C.SIM_TIME, ZOOM_TIME)):
        for row, sig in enumerate(sig_names):
            ax = axes[row, col]
            _sig_background(ax, sig)
            for n in names:
                r = row_of(rows, a, n)
                _draw_signal(ax, logs[(a, n)], r, n, sig, t_max)
                ts = r["settling_time_s"]
                if sig == "theta" and np.isfinite(ts) and ts <= t_max:
                    ax.axvline(ts, color=_c(n), ls=":", lw=1.1)
            ax.set_xlim(0, t_max)
            ax.set_ylabel(SIG_LABEL[sig])
            if col == 0:
                ax.set_title(f"{os_title[sig]}:  {_os_summary(rows, a, names, sig)}", loc="left", fontsize=9)
            elif row == 0:
                ax.set_title(f"zoom 0–{t_max:g} s", fontsize=9)
        axes[2, col].set_xlabel("time [s]")
    handles = []
    for n in names:
        r = row_of(rows, a, n)
        ts = r["settling_time_s"]
        lab = (f"{n}: {r['nominal_result']}" + (f", Ts = {ts:.2f} s" if np.isfinite(ts) else ", Not Settled")
               + f", success rate {r['success_rate_pct']:.0f}%")
        handles.append(axes[0, 0].plot([], [], color=_c(n), ls=LINESTYLES.get(n, "-"), lw=1.8,
                                        marker=MARKERS.get(n, "o"), mec="white", label=lab)[0])
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(f"θ0 = {a:+g}°   (nominal run s0 = [0, 0, {a:+g}°, 0])", fontweight="bold")
    fig.text(0.5, 0.005, f"gray band: |θ| ≤ {E.CRIT['theta_deg']:g}°   ·   dotted vertical: settling time "
             f"({settling_desc().replace('θ̇', 'dθ/dt')})   ·   marker: overshoot point (opposite side of θ0)   ·   "
             f"dotted horizontal: ±F_MAX", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_overview(logs, rows, angles, names, sig, out_dir):
    """한 신호(θ / x / F)를 모든 초기각에 대해 한 장에: 행 = +θ0 / −θ0, 열 = |θ0|"""
    plt = _plt()
    mags = sorted({abs(a) for a in angles})
    signs = [s for s in (1, -1) if any(np.sign(a) == s for a in angles)]
    t_max = OVERVIEW_TIME[sig]
    fig, axes = plt.subplots(len(signs), len(mags), figsize=(3.0 * len(mags), 2.9 * len(signs) + 0.9),
                             sharex=True, squeeze=False)
    for i, sg in enumerate(signs):
        for j, m in enumerate(mags):
            ax, a = axes[i, j], sg * m
            if a not in angles:
                ax.axis("off")
                continue
            _sig_background(ax, sig)
            for n in names:
                _draw_signal(ax, logs[(a, n)], row_of(rows, a, n), n, sig, t_max, lw=1.4)
            ax.set_xlim(0, t_max)
            ax.set_title(f"θ0 = {a:+g}°", fontsize=9)
            if j == 0:
                ax.set_ylabel(SIG_LABEL[sig])
            if i == len(signs) - 1:
                ax.set_xlabel("time [s]")
    handles = [axes[0, 0].plot([], [], color=_c(n), ls=LINESTYLES.get(n, "-"), lw=1.8,
                               marker=MARKERS.get(n, "o"), mec="white", label=n)[0] for n in names]
    fig.legend(handles=handles, loc="upper right", ncol=len(names), fontsize=9)
    fig.suptitle(f"{SIG_LABEL[sig]} vs time, all initial angles (0–{t_max:g} s)   marker = overshoot point",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, f"timeseries_overview_{sig}.png"), dpi=DPI)
    plt.close(fig)


def _angle_colors(angles):
    """−θ0 = 파랑 계열, +θ0 = 빨강 계열, |θ0| 가 클수록 진하게 (발산형 색)"""
    from matplotlib.colors import LinearSegmentedColormap
    neg = LinearSegmentedColormap.from_list("neg", ["#9ec5f4", "#2a78d6", "#0d366b"])
    pos = LinearSegmentedColormap.from_list("pos", ["#f4b1ab", "#e34948", "#6e1a1a"])
    mags = sorted({abs(a) for a in angles})
    f = (lambda m: 0.5) if len(mags) == 1 else (lambda m: mags.index(m) / (len(mags) - 1))
    return {a: (neg if a < 0 else pos)(f(abs(a))) for a in angles}


def plot_by_controller(logs, angles, names, out_dir):
    plt = _plt()
    cols = _angle_colors(angles)
    fig, axes = plt.subplots(3, len(names), figsize=(5.2 * len(names) + 1.4, 8.8), squeeze=False)
    for j, n in enumerate(names):
        for i, sig in enumerate(("theta", "x", "F")):
            ax = axes[i, j]
            _sig_background(ax, sig)
            for a in angles:
                t, y = _signal(logs[(a, n)], sig)
                if sig == "F":
                    ax.step(t, y, where="post", color=cols[a], lw=1.1)
                else:
                    ax.plot(t, y, color=cols[a], lw=1.3)
            ax.set_xlim(0, OVERVIEW_TIME[sig])
            if j == 0:
                ax.set_ylabel(SIG_LABEL[sig])
            ax.set_xlabel("time [s]")
        axes[0, j].set_title(n, fontweight="bold")
    handles = [axes[0, 0].plot([], [], color=cols[a], lw=2, label=f"θ0 = {a:+g}°")[0] for a in angles]
    fig.legend(handles=handles, loc="center right", fontsize=8)
    fig.suptitle("Time response by controller — color = initial angle (blue: −θ0, red: +θ0, darker = larger)",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.2 / (5.2 * len(names) + 1.4), 0.96))
    fig.savefig(os.path.join(out_dir, "timeseries_by_controller.png"), dpi=DPI)
    plt.close(fig)


# ---------------- main ----------------
def main():
    angles = angle_list()
    out_dir = os.path.join(C.RESULT_ROOT, OUT_SUBDIR)
    ts_dir = os.path.join(out_dir, "timeseries")
    os.makedirs(ts_dir, exist_ok=True)

    controllers = load_controllers(CONTROLLERS)
    names = [c.name for c in controllers]
    print(f"\n[비교 조건] 제어기 {names}, 초기각 {[f'{a:+g}°' for a in angles]}")
    print(f"  {C.SIM_TIME:g} s, dt = {C.DT:g} s, F_MAX = ±{C.F_MAX:g} N, 노이즈 {C.NOISE_STD}, "
          f"외란 {'ON' if C.DISTURBANCE['enabled'] else 'OFF'}")
    print(f"  Settling: {settling_desc()} 을 끝까지(≥{E.CRIT['hold_time']:g} s) 유지하기 시작한 시각")
    print(f"  성공 기준: 10 s 동안 |x|≤{C.X_MAX:g} m, 마지막 {E.CRIT['hold_time']:g} s "
          f"|θ|≤{E.CRIT['theta_deg']:g}°, |θ̇|≤{E.CRIT['theta_dot']:g} rad/s, |x|≤{E.CRIT['x']:g} m")

    rows, logs = run_nominal(controllers, angles)
    trials = run_mc(controllers, angles)
    merge_success(rows, trials)
    summary, common = controller_summary(rows, trials, angles, names)

    save_tables(rows, trials, summary, angles, names, out_dir)
    if SAVE_RAW:
        E.save_raw(logs, out_dir)
    print_tables(rows, summary, common, angles, names)

    print("\n그래프 저장 중...")
    R = mc_ranges()
    plot_table(rows, angles, names, METRICS, "Evaluation metrics by initial angle and controller",
               f"Success Rate: {MC_TRIALS} trials per angle (θ0 fixed, x, ẋ, dθ/dt random in "
               f"x{R['x']}, ẋ{R['x_dot']}, dθ/dt{R['theta_dot']}).  Other metrics: nominal run s0 = [0, 0, θ0, 0].  "
               f"Bold + shaded = best in the row (ties not marked).  N/S = Not Settled.",
               os.path.join(out_dir, "table_metrics.png"))
    plot_table(rows, angles, names, OVERSHOOT, "Overshoot by initial angle and controller (nominal run)",
               "θ overshoot: max angle past upright on the opposite side of θ0.  x peak: max cart travel toward θ0.  "
               "x overshoot: max travel past center on the opposite side.  F overshoot: max reverse (braking) force.",
               os.path.join(out_dir, "table_overshoot.png"))
    plot_metrics_vs_angle(rows, summary, angles, names, out_dir)
    plot_overshoot_vs_angle(rows, angles, names, out_dir)
    for a in angles:
        plot_timeseries_angle(logs, rows, a, names, os.path.join(ts_dir, f"theta0_{a:+03d}deg.png"))
    for sig in ("theta", "x", "F"):
        plot_overview(logs, rows, angles, names, sig, out_dir)
    plot_by_controller(logs, angles, names, out_dir)
    print(f"\n[완료] 결과 저장 위치: {out_dir}/  (CSV 4개 + raw/ + timeseries/ + 그래프)")
    return rows, trials, summary


if __name__ == "__main__":
    main()
