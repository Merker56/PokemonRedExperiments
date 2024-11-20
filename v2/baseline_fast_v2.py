from os.path import exists
from pathlib import Path
import uuid
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from tensorboard_callback import TensorboardCallback
import time
import wandb

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf), 
            stream_metadata = { # All of this is part is optional
                "user": "JoetheAIGuy", # choose your own username
                "env_id": rank, # environment identifier
                "color": "#000080", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
        env.rank = rank
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":

    use_wandb_logging = True
    ep_length = 2048 * 200 # Increasing to 8000+ surpasses RAM with 24 CPU
    sess_id = str(uuid.uuid4())[:8] #str(uuid.uuid4())[:8] // 
    sess_path = Path(f'session_{sess_id}')

    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False, ## Set headless false to display the environments
                'action_freq': 24, 'init_state': '../Bulbasaur.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 1, 'explore_weight': 0.03
            }

    class WandbCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(WandbCallback, self).__init__(verbose)
            self.start_time = None

        def _on_training_start(self) -> None:
            # Record the start time at the beginning of training
            self.start_time = time.time()

        def _on_step(self) -> bool:
            # Log metrics from all environments every 100 steps
            if self.n_calls % 100 == 0:  # Log every 100 steps
                # Calculate the steps per second
                elapsed_time = time.time() - self.start_time
                steps_per_second = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0

                # Collect reward information from all environments
                rewards_list = self.training_env.get_attr("get_game_state_reward")

                # Iterate over each environment and aggregate its reward metrics
                reward_data = {"total_timesteps": self.num_timesteps, "steps_per_second": steps_per_second}
                for i, rewards in enumerate(rewards_list):
                    if callable(rewards):
                        rewards = rewards()
                    
                    # Collect rewards data
                    reward_data.update({
                        f'env_{i}_event_reward': rewards['event'],
                        f'env_{i}_level_reward': rewards['level'],
                        f'env_{i}_badge_reward': rewards['badge'],
                        f'env_{i}_explore_reward': rewards['explore'],
                        f'env_{i}_captured_reward': rewards['captured'],
                    })
                
                # Perform logging from the main process only
                wandb.log(**{f'environment/{k}': v for k, v in reward_data.items()})

            return True
    
    print(env_config)
    
    num_cpu = 16  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    if use_wandb_logging:
        import wandb
        #from wandb.integration.sb3 import WandbCallback ## remove when new system works
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            entity="GamingReinforcementLearning",
            project="StableBaselinePokemonRed",
            id=sess_id,
            name=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())
    #env_checker.check_env(env)

    # put a checkpoint here you want to start from
    file_name = "session_7edf7249/poke_40304640_steps" # <- a cerulean checkpoint

    train_steps_batch = ep_length // 200
    
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01,\
                    tensorboard_log=sess_path, device = "cuda")
    
    print(model.policy)

    model.learn(total_timesteps=(ep_length)*num_cpu*100, callback=CallbackList(callbacks), tb_log_name="poke_ppo")

    if use_wandb_logging:
        run.finish()
