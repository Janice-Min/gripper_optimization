"""
Cart-Pole 역진자 — PPO "최적 모델" 찾기
================================================================
optimal_common.py 의 기준(스펙 + 최소화)에 맞춰 PPO 를 다시 학습하고, PID·LQR 과 같은 목적함수 J 로
가장 좋은 모델을 고릅니다. (cartpole_ppo.py 의 기존 모델·설정은 건드리지 않음)

  보상 (기존 보상에 Settling 항 추가)
    r = 1 (생존)  −  W_BAND · 1[안정화 기준 밖]  −  (θ² + 0.1x² + 0.01θ̇² + 0.01ẋ²)  −  W_U · (F/F_MAX)²
    실패(|x| > 2.4 m)하면 −FAIL_PENALTY 후 종료 (각도로는 종료하지 않음 — cartpole_common.OBJECTIVE 주석 참고)
      - 1[안정화 기준 밖]: |θ| ≤ 3°, |θ̇| ≤ 0.2 rad/s, |x| ≤ 0.2 m 중 하나라도 벗어나면 1
        → 에피소드 합 ≈ 기준 밖에 있던 시간 ∝ Settling Time.  W_BAND < 1 이라 기준 밖이어도
          버티는 스텝의 보상은 양수 → "넘어지지 않기"가 여전히 1순위
      - θ², x² 등 2차 항: 학습 신호를 부드럽게 해주는 보조 항 (기존 보상과 같은 가중치)
      - W_U: effort 상한을 맞추기 위한 가중치. 값마다 따로 학습해 상한을 지키는 모델 중에서 고름

  학습 초기상태: θ0 ~ U(±TRAIN_THETA_DEG), 단 HARD_FRAC 비율의 에피소드는 |θ0| ~ U(HARD_THETA_DEG) 로 시작
                나머지 성분은 optimal_common.DESIGN_OTHER_RANGES (스펙 ±30° 를 덮도록)
  모델 선택   : 학습 중 EVAL_FREQ 마다 설계 상태 148개에서 J 계산, 체크포인트 저장 → select 에서 모든 run·체크포인트 중 J 최소 선택
                (PID·LQR 이 같은 설계 상태에서 J 를 최소화한 것과 같은 절차)

실행:
  python optimal_ppo.py grid                     # RUNS 중 아직 없는 run 학습 (N_PARALLEL 개씩 동시에) 후 select
  python optimal_ppo.py train 0.1 0 0.5 1000000  # W_U=0.1, seed=0, HARD_FRAC=0.5, 1M 스텝 하나만 학습
  python optimal_ppo.py select           # 학습된 run 들 중 J 최소 → models/ppo_opt.zip
"""

# ================================================================
# 0. PPO 최적화 설정 — 여기만 바꾸면 됩니다
# ================================================================
REWARD = dict(
    alive=1.0,
    w_band=0.5,                 # 안정화 기준 밖 1 스텝당 벌점
    w_theta=1.0, w_x=0.1, w_theta_dot=0.01, w_x_dot=0.01,   # 보조 2차 항 (기존 OBJECTIVE 와 같음)
    fail_penalty=10.0,
)
RUNS = [                        # 학습할 설정 (W_U, seed, HARD_FRAC, 총 스텝). grid 는 아직 없는 run 만 학습
    # 참고: 처음엔 초기각 균일분포(HARD_FRAC=0)로 학습했는데, 학습 후반에 ±25~30° 에서 넘어지는 정책으로
    #       수렴해서 큰 초기각 오버샘플링(HARD_FRAC=0.5)으로 바꿈
    (0.3, 0, 0.5, 1_000_000), (0.3, 1, 0.5, 1_000_000),
    # effort 가 상한(80)보다 한참 작게 나와 W_U 를 낮추고 학습을 길게
    (0.1, 0, 0.5, 2_000_000), (0.1, 1, 0.5, 2_000_000),
    (0.03, 0, 0.5, 2_000_000), (0.03, 1, 0.5, 2_000_000),
]
TRAIN_THETA_DEG = 32.0          # 학습 초기각 범위 ±[deg] (스펙 30° 보다 약간 넓게)
HARD_THETA_DEG = (24.0, 32.0)   # HARD_FRAC 비율의 에피소드는 |θ0| 를 이 범위에서 시작
                                #   → 균일분포로는 드문 ±25~30° 상황을 자주 겪게 함
N_ENVS = 4
EVAL_FREQ = 20_000              # 이 스텝마다 설계 상태 J 로 체크포인트 평가
N_PARALLEL = 2                  # grid 모드에서 동시에 돌리는 학습 개수 (CPU 코어 수 이하)

RUN_DIR = "models/ppo_opt_runs"
BEST_PATH = "models/ppo_opt"    # .zip 자동으로 붙음
SUMMARY_PATH = "models/ppo_opt.json"

# ================================================================
# 이하 구현부
# ================================================================
import os
import sys
import json
import glob
import time
import subprocess
import numpy as np
import optimal_common as O          # 먼저 import → 기존 파일 경로 설정
import cartpole_common as C
import cartpole_ppo as PP

PPO_KW = dict(PP.PPO_KW)        # 하이퍼파라미터는 기존 cartpole_ppo.py 와 같게


def run_name(w_u, seed, hard_frac, steps):
    return f"wu{w_u:g}_hard{hard_frac:g}_{steps / 1e6:g}M_seed{seed}"


def make_env_class(w_u, hard_frac):
    import gymnasium as gym
    from gymnasium import spaces
    R = REWARD

    class CartPoleSpecEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.sim = C.CartPoleSim()
            self.n_steps = int(round(C.SIM_TIME / C.DT))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            r, rng = O.DESIGN_OTHER_RANGES, self.np_random
            if rng.uniform() < hard_frac:
                th = rng.choice([-1.0, 1.0]) * rng.uniform(*HARD_THETA_DEG)
            else:
                th = rng.uniform(-TRAIN_THETA_DEG, TRAIN_THETA_DEG)
            self.s = np.array([rng.uniform(*r["x"]), rng.uniform(*r["x_dot"]),
                               C.deg2rad(th), rng.uniform(*r["theta_dot"])])
            self.k = 0
            return PP.make_obs(self.s), {}

        def step(self, action):
            F = float(np.clip(action[0], -1.0, 1.0)) * C.F_MAX
            self.s = self.sim.step(self.s, F)
            self.k += 1
            failed = bool(C.is_failed(self.s))
            if failed:
                reward = -R["fail_penalty"]
            else:
                x, xd, th, thd = self.s
                out = not bool(O._stable(self.s[None, :])[0])
                reward = (R["alive"] - R["w_band"] * out
                          - (R["w_theta"] * th ** 2 + R["w_x"] * x ** 2 + R["w_theta_dot"] * thd ** 2
                             + R["w_x_dot"] * xd ** 2) - w_u * (F / C.F_MAX) ** 2)
            return PP.make_obs(self.s), float(reward), failed, self.k >= self.n_steps, {}

    return CartPoleSpecEnv


def train(w_u, seed, hard_frac, steps):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback

    name = run_name(w_u, seed, hard_frac, steps)
    out_dir = os.path.join(RUN_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    states = O.design_states()
    hist = []

    class SelectCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = np.inf
            self.next_eval = EVAL_FREQ

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.next_eval += EVAL_FREQ
                J, s = O.evaluate(O.BatchPPO(self.model), states)
                hist.append(dict(step=int(self.num_timesteps), **s))
                self.model.save(os.path.join(out_dir, "ckpt", f"{self.num_timesteps:08d}"))   # select 에서 재평가
                tag = ""
                if J < self.best:
                    self.best = J
                    self.model.save(os.path.join(out_dir, "best"))
                    tag = "  ← best"
                print(f"[{name}] {self.num_timesteps:>8d} steps  {O.fmt(s)}{tag}", flush=True)
                with open(os.path.join(out_dir, "history.json"), "w") as f:
                    json.dump(hist, f, indent=1, ensure_ascii=False)
            return True

    venv = make_vec_env(make_env_class(w_u, hard_frac), n_envs=N_ENVS, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **PPO_KW)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=SelectCallback())
    print(f"[{name}] 학습 완료 ({time.time() - t0:.0f}s)", flush=True)


def grid():
    jobs = [r for r in RUNS if not os.path.isdir(os.path.join(RUN_DIR, run_name(*r), "ckpt"))]
    print(f"학습할 run {len(jobs)}개 (이미 있는 run 은 건너뜀)")
    os.makedirs(RUN_DIR, exist_ok=True)
    running = []
    while jobs or running:
        while jobs and len(running) < N_PARALLEL:
            r = jobs.pop(0)
            log = open(os.path.join(RUN_DIR, run_name(*r) + ".log"), "w")
            running.append(subprocess.Popen([sys.executable, os.path.join(O.HERE, "optimal_ppo.py"), "train", *map(str, r)],
                                            stdout=log, stderr=subprocess.STDOUT))
        time.sleep(5)
        running = [p for p in running if p.poll() is None]
    select()


def select():
    """모든 run 의 모든 체크포인트를 현재 설계 상태로 다시 평가해 J 최소인 모델 선택"""
    from stable_baselines3 import PPO
    import shutil
    states = O.design_states()
    rows, best = [], None
    for run_dir in sorted(glob.glob(os.path.join(RUN_DIR, "*", ""))):
        name = os.path.basename(os.path.normpath(run_dir))
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt", "*.zip")))
        if not ckpts:
            continue
        run_best = None
        for path in ckpts:
            J, s = O.evaluate(O.BatchPPO(PPO.load(path, device="cpu")), states)
            if run_best is None or J < run_best["J"]:
                run_best = dict(run=name, path=path, step=int(os.path.basename(path)[:-4]), **s)
        rows.append(run_best)
        print(f"[{name}] 체크포인트 {len(ckpts)}개 중 최적 (step {run_best['step']})  {O.fmt(run_best)}")
        if best is None or run_best["J"] < best["J"]:
            best = run_best
    if best is None:
        sys.exit("학습된 run 이 없습니다.")
    shutil.copy(best["path"], BEST_PATH + ".zip")
    with open(SUMMARY_PATH, "w") as f:
        json.dump(dict(best=best, runs=rows, reward=REWARD, run_settings="wu<W_U>_hard<HARD_FRAC>_<steps>M_seed<seed>",
                       train_theta_deg=TRAIN_THETA_DEG, hard_theta_deg=HARD_THETA_DEG, eval_freq=EVAL_FREQ, ppo_kw=PPO_KW,
                       design_states=len(states), effort_budget=O.EFFORT_BUDGET), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[PPO 최적] {best['run']} (step {best['step']})  {O.fmt(best)}")
    print(f"  저장: {BEST_PATH}.zip, {SUMMARY_PATH}")


class PPOOptController(PP.PPOController):
    """원본 평가 파이프라인(C.simulate, angle_compare 등)에서 쓰는 최적 PPO 제어기"""
    def __init__(self, path=BEST_PATH):
        super().__init__(path)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "train":
        train(float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    elif mode == "grid":
        grid()
    elif mode == "select":
        select()
