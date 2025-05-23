import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce
import wandb

def merge_dicts(dicts):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir, log_freq = 100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.log_freq = log_freq

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:
        
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos, distributions = merge_dicts(all_final_infos)
            # TODO log distributions, and total return
            for key, val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)

            for key, distrib in distributions.items():
                self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                self.logger.record(f"env_stats_max/{key}", max(distrib))
                
            #images = self.training_env.get_attr("recent_screens")
            #images_row = rearrange(np.array(images), "(r f) h w c -> (r c h) (f w)", r=2)
            #self.logger.record("trajectory/image", Image(images_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

            map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=2)
            self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
            merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

        if self.n_calls % 100 == 0:  # Log every 50 steps
            try:
                # Get rewards from all environments in the vector
                run_state_scores = self.training_env.get_attr("progress_reward")

                if not run_state_scores or len(run_state_scores) == 0:
                    raise ValueError("progress_reward attribute returned None or an empty list.")

                # Flatten progress rewards into a list of dicts (one per agent)
                all_agent_rewards = [stats for stats in run_state_scores if stats]

                if len(all_agent_rewards) == 0:
                    raise ValueError("No valid progress rewards found in agents.")

                # Compute mean values across all agents
                mean_rewards = {key: np.mean([agent.get(key, 0) for agent in all_agent_rewards])
                                for key in all_agent_rewards[0]}

                # Log rewards per agent
                reward_logs = {}
                for agent_idx, rewards in enumerate(all_agent_rewards):
                    for key, value in rewards.items():
                        reward_logs[f"agent_{agent_idx}/rewards/{key}"] = value  # Log each reward per agent

               # Get party info from all environments
                run_pokemon_info = self.training_env.get_attr("party_info")
                all_agent_party = [party for party in run_pokemon_info if party]
                
                # Create a single table for all agents' Pokémon
                columns = ["agent_id", "pokemon_id", "pokemon_name", "level", "move1", "move2", "move3", "move4"]
                pokemon_table = wandb.Table(columns=columns)
                
                # Aggregate party stats for all agents
                party_stats = {}
                
                # Add each agent's Pokémon to the table
                for agent_idx, party_dict in enumerate(all_agent_party):
                    # Track stats for this agent
                    active_pokemon_count = 0
                    total_level = 0
                    
                    # Add each Pokémon's data as a row
                    for pokemon_name, pokemon_data in party_dict.items():
                        # Skip empty slots (pokemon with ID 0)
                        if pokemon_name == 0 or pokemon_data[0] == 0:
                            continue
                            
                        pokemon_id = pokemon_data[0]
                        level = pokemon_data[1]
                        move1 = pokemon_data[2]
                        move2 = pokemon_data[3]
                        move3 = pokemon_data[4]
                        move4 = pokemon_data[5]
                        
                        # Add row to table
                        pokemon_table.add_data(agent_idx, pokemon_name, pokemon_id, level, move1, move2, move3, move4)
                        
                        # Update stats
                        active_pokemon_count += 1
                        total_level += level
                    
                    # Calculate and store summary stats
                    avg_level = total_level / max(active_pokemon_count, 1)
                    party_stats[f"agent_{agent_idx}/party_stats/pokemon_count"] = active_pokemon_count
                    party_stats[f"agent_{agent_idx}/party_stats/avg_level"] = avg_level
                
                # Log the single table for all agents
                wandb.log({"pokemon_party_table": pokemon_table}, step=self.num_timesteps)
                
                # Log party stats
                party_stats["global_step"] = self.num_timesteps
                wandb.log(party_stats)

                # Log mean rewards across agents
                for key, value in mean_rewards.items():
                    reward_logs[f"rewards_mean/{key}"] = value

                # Log total mean reward and global step
                reward_logs["rewards_mean/total"] = sum(mean_rewards.values())
                reward_logs["global_step"] = self.num_timesteps

                # Log to wandb
                wandb.log(reward_logs)

            except Exception as e:
                print(f"Error logging to wandb: {e}")
                import traceback
                traceback.print_exc()  # Print full error traceback for debugging

        return True
    
    def _on_training_end(self):
        if self.writer:
            self.writer.close()

