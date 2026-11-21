import argparse
from pathlib import Path
import uuid
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from tensorboard_callback import TensorboardCallback
from heartbeat_callback import HeartbeatCallback
from checkin_report import CheckinReportCallback


def find_latest_checkpoint(runs_dir):
    """Newest poke_model_*_steps.zip across all sessions, or None."""
    ckpts = list(Path(runs_dir).glob("session_*/poke_model_*_steps.zip"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env_conf['rank'] = rank
        base_env = RedGymEnv(env_conf)
        should_stream = env_conf.get("enable_stream", False)
        if should_stream:
            env = StreamWrapper(
                base_env,
                stream_metadata={
                    "user": "JoetheAIGuy",
                    "env_id": rank,
                    "color": "#000080",
                    "extra": "",
                    "upload_interval": env_conf.get("stream_upload_interval", 120),
                    "ws_timeout_s": env_conf.get("stream_timeout_s", 0.5),
                    "include_extra": not env_conf.get("headless", True),
                },
            )
        else:
            env = base_env
        env.rank = rank
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pokemon Red PPO agent")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the newest checkpoint in runs/ (the watchdog uses this after a restart)")
    parser.add_argument("--checkpoint", default=None,
                        help="continue from this checkpoint .zip instead of the newest one")
    parser.add_argument("--num-envs", type=int, default=12,
                        help="parallel game environments (12 suits a 16-thread CPU with 24 GB for WSL)")
    parser.add_argument("--stream", action="store_true",
                        help="quality mode: broadcast to the live map and save debug images (slower)")
    parser.add_argument("--base-agents", type=int, default=4,
                        help="agents that always start from the base state; the rest rotate through milestone saves")
    parser.add_argument("--ladder-window", type=int, default=5,
                        help="rotating agents only use the newest N milestones")
    parser.add_argument("--no-ladder", action="store_true",
                        help="everyone starts from the base state (no milestone saves)")
    parser.add_argument("--checkin-minutes", type=float, default=60,
                        help="write a check-in report (runs/checkins/) this often")
    parser.add_argument("--hours", type=float, default=0,
                        help="stop after roughly this many hours of training (0 = run until stopped)")
    args = parser.parse_args()

    mode_speed = not args.stream  # True: throughput mode, False: quality/streaming mode
    ep_length = 1024 * 20
    
    # Define a top-level directory for all training runs
    base_output_dir = Path('runs')
    init_state = '../Bulbasaur.state'  # the base start state (intro already done)
    archive_dir = None if args.no_ladder else str(base_output_dir.resolve() / "archive")
    base_output_dir.mkdir(exist_ok=True)

    # Create a unique session ID and the full path for this run
    sess_id = str(uuid.uuid4())[:8]
    sess_path = base_output_dir / f'session_{sess_id}'

    # The CheckpointCallback saves the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length, # Save every full episode per-agent
        save_path=sess_path,
        name_prefix="poke_model"
    )

    # The TensorboardCallback handles logging rewards and summaries
    tensorboard_callback = TensorboardCallback(log_dir=sess_path, log_freq=1024)

    # Heartbeat: lets the watchdog spot a stalled run and feeds the morning summary
    heartbeat_callback = HeartbeatCallback(
        heartbeat_path=base_output_dir / "heartbeat.jsonl",
        session_id=sess_id,
        every_n_calls=2048,  # roughly every 30 seconds
        stop_after_hours=args.hours,
    )

    # Check-in reports to look through with the kids: runs/checkins/index.html
    checkin_callback = CheckinReportCallback(
        out_dir=base_output_dir / "checkins",
        session_id=sess_id,
        every_minutes=args.checkin_minutes,
        archive_dir=archive_dir,
        base_state=Path(init_state).name,
    )

    # Combine them in a list
    callbacks = [checkpoint_callback, tensorboard_callback, heartbeat_callback, checkin_callback]

    base_env_config = {
        'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 24, 'init_state': init_state, 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'symbol_file': '../pokered.sym', 'debug': False, 'reward_scale': 2, 'explore_weight': 0.05,
        'disable_wild_encounters': ['MT_MOON_1F', 'MT_MOON_B1F', 'MT_MOON_B2F'],
        'log_agent_id': 0,
        'stream_timeout_s': 0.5,
        'reset_stagger_s': 0.08,
        'gc_every_n_resets': 4,
        'reset_state_mode': 'disk',
        'release_action_tick': -1,
        'trace_step_phases': False,
        'trace_step_file': False,  # heavy: writes+flushes every emulator frame; the heartbeat replaces it
        'trace_tick_every': 1,
        'trace_flush_every': 1,
        'trace_first_n_steps': 300,
        'trace_from_reset': 2,
        'hang_dump_enabled': False,
        'hang_dump_timeout_s': 45,
        'frame_stacks': 3,
        'render_resize_mode': 'mean',
    }
    if mode_speed:
        mode_overrides = {
            'num_envs': args.num_envs,
            'print_freq': 100,
            'update_freq': 512,
            'save_debug_jpegs': False,
            'enable_stream': False,
            'stream_upload_interval': 512,
            'frame_stacks': 1,
            'render_resize_mode': 'stride',
        }
    else:
        mode_overrides = {
            'num_envs': args.num_envs,
            'print_freq': 50,
            'update_freq': 512,
            'save_debug_jpegs': True,
            'enable_stream': True,
            'stream_upload_interval': 512,
            'frame_stacks': 3,
            'render_resize_mode': 'mean',
        }
    env_config = {**base_env_config, **mode_overrides}
    # Batch stream uploads on the same cadence as heavier env updates.
    env_config['stream_upload_interval'] = env_config['update_freq']
    # Milestone saves + start-state ladder (milestones.py). Lives in runs/archive so it
    # carries across sessions and watchdog restarts.
    env_config['archive_dir'] = archive_dir
    env_config['n_envs'] = env_config['num_envs']
    env_config['ladder_n_base'] = args.base_agents
    env_config['ladder_window'] = args.ladder_window

    print(f"Using mode_speed={mode_speed}")
    print(env_config)

    num_cpu = env_config.get("num_envs", 16)
    print(f"Using {num_cpu} parallel environments")
    # fork: workers share the already-imported libraries instead of re-importing
    # torch etc. in every process (less memory, faster start). Safe because CUDA
    # isn't initialised until the model is created below.
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)], start_method="fork")

    # The main process only needs a few CPU threads (the GPU does the training);
    # leaving PyTorch's default (one per CPU thread) makes it fight the emulators.
    torch.set_num_threads(4)

    # Fresh start by default. --resume picks up the newest checkpoint (used by the
    # watchdog after a restart); --checkpoint picks a specific one.
    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif args.resume:
        checkpoint = find_latest_checkpoint(base_output_dir)

    train_steps_batch = ep_length // 20

    model = None
    if checkpoint is not None and checkpoint.exists():
        print(f"\nloading checkpoint {checkpoint}")
        try:
            model = PPO.load(str(checkpoint), env=env, device="cuda")
        except Exception as e:
            # e.g. a checkpoint from before USE CUT was added (7 actions, now 8)
            print(f"\nCould not load {checkpoint} ({e}); starting fresh instead.")
            model = None
            checkpoint = None
    if model is not None:
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.02,
                    tensorboard_log=sess_path, device="cuda")

    print(model.policy)

    # Effectively "until stopped": Ctrl+C, --hours, or the watchdog ends the run.
    total_timesteps = (ep_length) * num_cpu * 10_000

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="PPO",
                    reset_num_timesteps=checkpoint is None)
    finally:
        env.close()
