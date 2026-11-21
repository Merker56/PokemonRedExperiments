"""Where does training time go? Run from the baselines folder (venv active,
training stopped so nothing else is using the CPU/GPU):

    python bench_speed.py              # 12 envs
    python bench_speed.py --envs 14

It measures, one stage at a time:
  1. one emulator on its own (no env code)
  2. one full environment
  3. N environments in parallel through SubprocVecEnv, random actions (no model)
  4. the same plus the model choosing actions on the GPU (what training does,
     minus the learning update)
The stage where steps/sec drops sharply is the bottleneck.
"""
import argparse
import time
from pathlib import Path

import numpy as np


def env_config(rank, n_envs):
    return {
        'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 24, 'init_state': '../Bulbasaur.state', 'max_steps': 20480,
        'print_rewards': False, 'save_video': False, 'fast_video': True,
        'session_path': Path('runs/bench'), 'gb_path': '../PokemonRed.gb',
        'symbol_file': '../pokered.sym', 'debug': False, 'reward_scale': 2,
        'explore_weight': 0.05, 'disable_wild_encounters': ['MT_MOON_1F', 'MT_MOON_B1F', 'MT_MOON_B2F'],
        'log_agent_id': 999, 'rank': rank, 'update_freq': 512, 'print_freq': 100,
        'save_debug_jpegs': False, 'frame_stacks': 1, 'render_resize_mode': 'stride',
        'archive_dir': None, 'n_envs': n_envs,
    }


def make(rank, n_envs):
    def _init():
        from red_gym_env_v2 import RedGymEnv
        return RedGymEnv(env_config(rank, n_envs))
    return _init


def rate(n_steps, seconds):
    return f"{n_steps / seconds:,.0f} steps/s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1500, help="steps per env per test")
    args = ap.parse_args()
    Path("runs/bench").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    import pyboy
    try:
        from importlib.metadata import version
        print(f"pyboy {version('pyboy')}")
    except Exception:
        pass

    # 1 + 2: single process
    env = make(0, 1)()
    env.reset()
    t = time.time()
    for _ in range(args.steps * 24):
        env.pyboy.tick(1, False)
    print(f"1. raw emulator, frames not drawn:    {rate(args.steps, time.time() - t)}  (one core)")
    t = time.time()
    for _ in range(args.steps):
        env.step(int(rng.integers(0, 7)))
    print(f"2. one full environment:              {rate(args.steps, time.time() - t)}  (one core)")
    fast_path = "tick(self.act_freq - 1, False)" in open("red_gym_env_v2.py").read()
    print(f"   (frame-skip code present: {fast_path})")
    env.close()

    # 3: parallel, no model
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([make(i, args.envs) for i in range(args.envs)], start_method="fork")
    venv.reset()
    n = args.steps // 3
    t = time.time()
    for _ in range(n):
        venv.step(rng.integers(0, 7, args.envs))
    print(f"3. {args.envs} envs in parallel, no model:    {rate(n * args.envs, time.time() - t)}")

    # 4: parallel + model inference on GPU
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(4)
    model = PPO("MultiInputPolicy", venv, n_steps=1024, batch_size=512, n_epochs=1, device="cuda", verbose=0)
    obs = venv.reset()
    t = time.time()
    for _ in range(n):
        actions, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = venv.step(actions)
    print(f"4. {args.envs} envs + model picking actions: {rate(n * args.envs, time.time() - t)}")

    # 5: a short real learn() including the update, for the number SB3 calls "fps"
    t = time.time()
    model.learn(total_timesteps=1024 * args.envs * 2)
    print(f"5. real training (2 rollouts + updates): {rate(1024 * args.envs * 2, time.time() - t)}")
    venv.close()


if __name__ == "__main__":
    main()
