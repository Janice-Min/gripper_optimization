"""
Cart-Pole 역진자 — 초기각 한계 탐색 (PID / LQR / PPO)
================================================================
초기각을 ±START_DEG 부터 1°씩 키우며 (±21°, ±22°, ...) 세 제어기를 10초씩 실행하고,
처음 실패하는 각도를 찾습니다. 실패한 제어기는 그 각도 이후로 더 평가하지 않습니다.

  - 제어기  : 각 제어기 파일의 build_controller() → 파일에 설정된 최적화 파라미터 그대로
              (PID: PID_THETA / PID_X,  LQR: LQR_Q / LQR_R,  PPO: models/ppo_cartpole_balance.zip)
  - 실행    : cartpole_common.simulate()  (같은 시뮬레이터 / dt / F_MAX, 초기상태 [0, 0, θ0, 0])
  - 성공 판정: evaluate_metrics.py 와 같은 기준 (평가지표.pdf ⑤)
              10초 동안 |x| ≤ 2.4 m  그리고  마지막 2초 동안 |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s, |x| ≤ 0.2 m
              → 도중에 레일을 벗어나거나(|x|>2.4 m), 10초 안에 안정화되지 않으면 FAIL
                (각도 실패 한계는 없음: 진자가 넘어가도 끝까지 실행하고 마지막 2초로 판정)

필요 파일: cartpole_common.py, cartpole_pid.py, cartpole_lqr.py, cartpole_ppo.py, simulator.py, evaluate_metrics.py

실행:
  python angle_sweep.py

결과 (results/angle_sweep/):
  sweep_results.csv     실행 1회 = 1행 (초기각, 제어기, 성공 여부, 실패 원인, 지표)
  sweep_summary.csv     제어기·방향별 마지막 성공 각도 / 첫 실패 각도 / 실패 원인
  sweep_map.png         초기각 × 제어기 성공(초록) / 실패(빨강) / 평가 안 함(회색) 지도
  sweep_trend.png       한계에 가까워질수록 지표가 어떻게 나빠지는지 (|θ0|에 따른 지표 변화)
  sweep_fail_cases.png  제어기별 "마지막 성공 각도" vs "첫 실패 각도" 시간응답 (어떻게 실패하는지)
"""

# ================================================================
# 0. 탐색 설정 — 여기만 바꾸면 됩니다
# ================================================================
START_DEG = 21                 # 시작 각도 [deg]  (±21°부터)
MAX_DEG = 45                   # 마지막 각도 [deg] (각도 실패 한계가 없어져서 더 크게 잡아도 됨)
STEP_DEG = 1                   # 증가 간격 [deg]
CONTROLLERS = ["PID", "LQR", "PPO"]

SEPARATE_SIGNS = False         # False: 같은 크기에서 +θ, −θ를 모두 돌려보고 하나라도 실패하면 그 제어기 종료
                               # True : +쪽과 −쪽을 따로 추적 (한쪽이 실패해도 반대쪽은 계속 → 방향별 한계)
                               #        PID·LQR은 선형이라 양쪽이 같지만, PPO는 비대칭일 수 있음

FAIL_PLOT_TIME = 4.0           # sweep_fail_cases.png 가로축 0 ~ 이 시간[s] (레일 이탈로 일찍 끝난 실패만 있을 때 확대)
                               # 실패 원인이 "10초 안에 안정화 못함"이면 자동으로 10초 전체를 보여줌
OUT_SUBDIR = "angle_sweep"     # 결과 폴더: results/angle_sweep/
SAVE_RAW = False               # True면 실행마다 원시 데이터 CSV 저장 (raw/)
DPI = 140

# ================================================================
# 이하 구현부
# ================================================================
import os
import csv
import numpy as np
import cartpole_common as C
import evaluate_metrics as E                     # 성공 판정·지표 계산을 그대로 재사용
from simulator import load_controllers, COLORS

SIGNS = (+1, -1)
SIGN_TXT = {+1: "+", -1: "−"}
METRIC_LABEL = {k: lab for k, lab, _ in E.PRIMARY + E.SECONDARY}


# ---------------- 탐색 ----------------
def run_one(ctrl, a):
    s0 = C.make_s0(dict(theta_deg=a))
    log = C.simulate(ctrl, s0, np.random.default_rng(C.SEED))
    return log, E.compute_pdf_metrics(log, a)


def sweep(controllers):
    rows, logs = [], {}
    active = {(c.name, sg): True for c in controllers for sg in SIGNS}
    names = [c.name for c in controllers]
    print(f"\n{'|θ0|':>6}  " + "".join(f"{n:^22}" for n in names))

    mag = START_DEG
    while mag <= MAX_DEG and any(active.values()):
        cells = []
        for ctrl in controllers:
            todo = [sg for sg in SIGNS if active[(ctrl.name, sg)]]
            failed = []
            parts = []
            for sg in todo:
                a = sg * mag
                log, m = run_one(ctrl, a)
                rows.append(dict(abs_deg=mag, sign=SIGN_TXT[sg], theta0_deg=a, controller=ctrl.name, **m))
                logs[(a, ctrl.name)] = log
                parts.append(f"{SIGN_TXT[sg]}{'S' if m['success'] else 'F'}")
                if not m["success"]:
                    failed.append(sg)
            # 실패 처리: 그 제어기(또는 그 방향)는 다음 각도부터 평가하지 않음
            for sg in (failed if SEPARATE_SIGNS else (SIGNS if failed else ())):
                active[(ctrl.name, sg)] = False
            cells.append(" ".join(parts) if parts else "·")
        print(f"{mag:>5}°  " + "".join(f"{c:^22}" for c in cells))
        mag += STEP_DEG
    return rows, logs


# ---------------- 요약 ----------------
def summarize(rows, names):
    """제어기·방향별: 마지막 성공 각도, 첫 실패 각도와 원인"""
    out = []
    for n in names:
        for sg in SIGNS:
            rs = sorted((r for r in rows if r["controller"] == n and r["sign"] == SIGN_TXT[sg]),
                        key=lambda r: r["abs_deg"])
            ok = [r for r in rs if r["success"]]
            bad = [r for r in rs if not r["success"]]
            last_ok = ok[-1]["theta0_deg"] if ok else None
            if bad:
                f = bad[0]
                status = "FAIL 발견"
                fail_deg, reason = f["theta0_deg"], E.result_text(f)
                detail = (f"peak F {f['peak_force_N']:.1f} N, 포화 {f['saturation_pct']:.0f}%, "
                          f"max|x| {f['max_cart_disp_m']:.2f} m")
            else:
                fail_deg, reason, detail = None, "", ""
                tested_max = rs[-1]["abs_deg"] if rs else None
                status = (f"{MAX_DEG}°까지 모두 성공" if tested_max == MAX_DEG
                          else "반대 방향 실패로 중단")
            out.append(dict(controller=n, direction=SIGN_TXT[sg],
                            last_success_deg=last_ok, first_fail_deg=fail_deg,
                            status=status, fail_reason=reason, fail_detail=detail))
    return out


def _deg(v):
    return "—" if v is None else f"{v:+g}°"


def print_summary(summary, names):
    print("\n━━ 초기각 한계 요약 " + "━" * 60)
    print(f"{'제어기':<6}{'방향':^6}{'마지막 성공':>12}{'첫 실패':>10}   결과")
    for s in summary:
        res = f"{s['fail_reason']}  [{s['fail_detail']}]" if s["first_fail_deg"] is not None else s["status"]
        print(f"{s['controller']:<8}{s['direction']:^6}{_deg(s['last_success_deg']):>12}"
              f"{_deg(s['first_fail_deg']):>10}   {res}")

    print("\n━━ 한 줄 요약 " + "━" * 66)
    for n in names:
        ss = [s for s in summary if s["controller"] == n]
        fails = [s for s in ss if s["first_fail_deg"] is not None]
        if not fails:
            print(f"  {n}: ±{START_DEG}° ~ ±{MAX_DEG}° 모두 성공 (이 범위에서는 한계를 찾지 못함)")
            continue
        both_ok = min(abs(s["last_success_deg"]) if s["last_success_deg"] is not None else START_DEG - STEP_DEG
                      for s in ss)
        first = min(fails, key=lambda s: abs(s["first_fail_deg"]))
        lim = f"±{both_ok:g}°까지 양방향 성공" if both_ok >= START_DEG else f"±{START_DEG}°부터 실패"
        print(f"  {n}: {lim}, 첫 실패 {', '.join(_deg(s['first_fail_deg']) for s in fails)} → {first['fail_reason']}")


# ---------------- 저장 ----------------
def save_csv(rows, summary, out_dir):
    keys = ["abs_deg", "sign", "theta0_deg", "controller", "success", "result", "terminated", "t_end"] + \
           [k for k, _, _ in E.PRIMARY + E.SECONDARY]
    with open(os.path.join(out_dir, "sweep_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([E.result_text(r) if k == "result" else r[k] for k in keys])
    with open(os.path.join(out_dir, "sweep_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


# ---------------- 그래프 ----------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _short_reason(r):
    if r["terminated"]:
        return f"x>{C.X_MAX:g}"
    return ",".join(E.WHY_SHORT[w] for w in r["why"])


def plot_map(rows, names, out_dir):
    plt = _plt()
    from matplotlib.colors import ListedColormap
    mags = sorted({r["abs_deg"] for r in rows})
    keys = [(n, sg) for n in names for sg in SIGNS]
    M = np.full((len(keys), len(mags)), np.nan)              # 1 성공, 0 실패, NaN 평가 안 함
    for i, (n, sg) in enumerate(keys):
        for j, mg in enumerate(mags):
            r = next((r for r in rows if r["controller"] == n and r["theta0_deg"] == sg * mg), None)
            if r is not None:
                M[i, j] = 1.0 if r["success"] else 0.0
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(mags) + 2.5), 0.6 * len(keys) + 2))
    ax.imshow(np.nan_to_num(M, nan=0.5), cmap=ListedColormap(["#f6c1c1", "#eeeeee", "#bfe6bf"]),
              vmin=0, vmax=1, aspect="auto")
    for i, (n, sg) in enumerate(keys):
        for j, mg in enumerate(mags):
            if M[i, j] == 0.0:
                r = next(r for r in rows if r["controller"] == n and r["theta0_deg"] == sg * mg)
                ax.text(j, i, "F\n" + _short_reason(r), ha="center", va="center", fontsize=6.5,
                        color="darkred", fontweight="bold")
            elif M[i, j] == 1.0:
                ax.text(j, i, "S", ha="center", va="center", fontsize=7, color="darkgreen")
    ax.set_xticks(range(len(mags)))
    ax.set_xticklabels([f"{m:g}°" for m in mags], fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{n}  {SIGN_TXT[sg]}θ0" for n, sg in keys])
    for i in range(2, len(keys), 2):
        ax.axhline(i - 0.5, color="k", lw=1)
    ax.set_xlabel("|initial angle θ0|")
    rule = "per direction" if SEPARATE_SIGNS else "a controller stops after its first failure (either sign)"
    ax.set_title(f"Initial-angle limit sweep  —  green: SUCCESS, red: FAIL, gray: not evaluated  ({rule})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_map.png"), dpi=DPI)
    plt.close(fig)


def plot_trend(rows, names, out_dir):
    """성공한 실행의 지표를 |θ0|에 따라 그림. 실패 각도는 세로 점선 + FAIL 표시"""
    plt = _plt()
    keys = ["settling_time_s", "overshoot_deg", "max_cart_disp_m",
            "saturation_pct", "control_effort", "angle_rmse_deg"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, key in zip(axes.ravel(), keys):
        for n in names:
            col = COLORS.get(n)
            for sg, ls in ((+1, "-"), (-1, "--")):
                rs = sorted((r for r in rows if r["controller"] == n and r["sign"] == SIGN_TXT[sg]),
                            key=lambda r: r["abs_deg"])
                ok = [r for r in rs if r["success"] and np.isfinite(r[key])]
                if ok:
                    ax.plot([r["abs_deg"] for r in ok], [r[key] for r in ok], ls, marker="o", ms=3.5,
                            color=col, lw=1.3, label=f"{n} {SIGN_TXT[sg]}θ0")
            fails = {}                                        # 실패 각도 → 실패한 방향들 (같은 각도면 한 줄로)
            for r in rows:
                if r["controller"] == n and not r["success"]:
                    fails.setdefault(r["abs_deg"], []).append(r["sign"])
            for mg, sgs in fails.items():
                ax.axvline(mg, color=col, ls=":", lw=1.2)
                ax.text(mg, 1.0, f" {n} FAIL ({''.join(sorted(sgs))}θ0)", color=col, fontsize=6.5,
                        rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())
        ax.set_title(METRIC_LABEL.get(key, key), fontsize=10)
        ax.set_xlabel("|initial angle θ0| [deg]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("How metrics degrade toward the limit  (successful runs only; dotted line = first failure, "
                 "solid = +θ0, dashed = −θ0)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "sweep_trend.png"), dpi=DPI)
    plt.close(fig)


def plot_fail_cases(rows, logs, summary, names, out_dir):
    """제어기별: 마지막 성공 각도(실선) vs 첫 실패 각도(점선)의 θ / x / F"""
    plt = _plt()
    fig, axes = plt.subplots(3, len(names), figsize=(5.2 * len(names), 8.5), sharex=True, squeeze=False)
    fail_rows = [r for r in rows if not r["success"]]
    t_view = C.SIM_TIME if (not FAIL_PLOT_TIME or any(not r["terminated"] for r in fail_rows)) \
        else min(FAIL_PLOT_TIME, C.SIM_TIME)
    for j, n in enumerate(names):
        col = COLORS.get(n)
        ss = [s for s in summary if s["controller"] == n]
        fails = [s for s in ss if s["first_fail_deg"] is not None]
        if fails:                                              # 처음 실패한 방향의 직전 성공 각도와 비교
            f = min(fails, key=lambda s: abs(s["first_fail_deg"]))
            a_fail = f["first_fail_deg"]
            a_ok = a_fail - np.sign(a_fail) * STEP_DEG
            cases = [(a_ok, "-", 1.0), (a_fail, "--", 1.0)]
        else:
            last = max((s["last_success_deg"] for s in ss if s["last_success_deg"] is not None), key=abs)
            cases = [(last, "-", 1.0)]
        ax_th, ax_x, ax_u = axes[:, j]
        hold0 = C.SIM_TIME - E.CRIT["hold_time"]
        for ax in axes[:, j]:
            ax.axvspan(hold0, C.SIM_TIME, color="gold", alpha=0.12, lw=0)
            ax.grid(alpha=0.3)
        ax_th.axhspan(-E.CRIT["theta_deg"], E.CRIT["theta_deg"], color="gray", alpha=0.15, lw=0)
        ax_x.axhspan(-E.CRIT["x"], E.CRIT["x"], color="gray", alpha=0.15, lw=0)
        for v in (-C.F_MAX, C.F_MAX):
            ax_u.axhline(v, color="red", ls=":", lw=0.8)
        for a, ls, _ in cases:
            log = logs.get((a, n))
            if log is None:                                    # 시작 각도에서 바로 실패한 경우 등
                continue
            r = next(r for r in rows if r["controller"] == n and r["theta0_deg"] == a)
            lab = f"θ0={a:+g}°: {E.result_text(r, short=True)}"
            ax_th.plot(log["t"], np.degrees(log["s"][:, 2]), ls, color=col, lw=1.4, label=lab)
            ax_x.plot(log["t"], log["s"][:, 0], ls, color=col, lw=1.4)
            if len(log["u"]):
                ax_u.step(log["t"][:-1], log["u"], ls, where="post", color=col, lw=1.0)
            if r["terminated"]:
                ax_th.plot(log["t"][-1], np.degrees(log["s"][-1, 2]), "x", color="k", ms=10, mew=2)
                ax_x.plot(log["t"][-1], log["s"][-1, 0], "x", color="k", ms=10, mew=2)
        ax_th.set_title(n + ("" if fails else f"  (no failure up to ±{MAX_DEG}°)"),
                        color=col, fontweight="bold")
        ax_th.legend(fontsize=8, loc="upper right")
        ax_th.set_xlim(0, t_view)
        ax_u.set_xlabel("time [s]")
        if j == 0:
            ax_th.set_ylabel("θ [deg]")
            ax_x.set_ylabel("x [m]")
            ax_u.set_ylabel("F [N]")
    fig.suptitle(f"Last success (solid) vs first failure (dashed), 0–{t_view:g} s   gray band: success criteria, "
                 "yellow: last 2 s, ×: terminated", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "sweep_fail_cases.png"), dpi=DPI)
    plt.close(fig)


# ---------------- main ----------------
def main():
    out_dir = os.path.join(C.RESULT_ROOT, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    controllers = load_controllers(CONTROLLERS)
    names = [c.name for c in controllers]
    print(f"\n[초기각 한계 탐색] 제어기 {names},  ±{START_DEG}° → ±{MAX_DEG}° ({STEP_DEG}° 간격),  "
          f"{C.SIM_TIME:g} s, F_MAX=±{C.F_MAX:g} N")
    print(f"  성공 기준: 10초 동안 |x|≤{C.X_MAX:g} m, 마지막 {E.CRIT['hold_time']:g}초 |θ|≤{E.CRIT['theta_deg']:g}°, "
          f"|θ̇|≤{E.CRIT['theta_dot']:g} rad/s, |x|≤{E.CRIT['x']:g} m   "
          f"({'방향별 추적' if SEPARATE_SIGNS else '첫 실패 시 제어기 종료'})")
    print("  표기: +S/−S 성공, +F/−F 실패, · 평가 안 함")

    rows, logs = sweep(controllers)
    summary = summarize(rows, names)
    print_summary(summary, names)

    save_csv(rows, summary, out_dir)
    if SAVE_RAW:
        E.save_raw(logs, out_dir)
    plot_map(rows, names, out_dir)
    plot_trend(rows, names, out_dir)
    plot_fail_cases(rows, logs, summary, names, out_dir)
    print(f"\n[완료] 결과 저장 위치: {out_dir}/")
    return rows, summary


if __name__ == "__main__":
    main()
