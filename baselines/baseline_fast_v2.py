from os.path import exists
from pathlib import Path
import uuid
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from tensorboard_callback import TensorboardCallback

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
    mode_speed = True  # True: throughput mode, False: quality/streaming mode
    ep_length = 1024 * 20
    
    # Define a top-level directory for all training runs
    base_output_dir = Path('runs')
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

    # Combine them in a list
    callbacks = [checkpoint_callback, tensorboard_callback]

    base_env_config = {
        'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 24, 'init_state': '../Bulbasaur.state', 'max_steps': ep_length,
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
        'trace_step_file': True,
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
            'num_envs': 8,
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
            'num_envs': 8,
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

    print(f"Using mode_speed={mode_speed}")
    print(env_config)

    num_cpu = env_config.get("num_envs", 16)
    print(f"Using {num_cpu} parallel environments")
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    file_name = "runs/session_1221bf4c/poke_model_6635520_steps"  # Put a checkpoint here you want to start from

    train_steps_batch = ep_length // 20

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.02,
                    tensorboard_log=sess_path, device="cuda")

    print(model.policy)

    model.learn(total_timesteps=(ep_length) * num_cpu * 50, callback=callbacks, tb_log_name="PPO")
