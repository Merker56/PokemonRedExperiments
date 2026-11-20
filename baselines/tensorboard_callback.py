import os
import json
from collections import defaultdict
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange, reduce

class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.log_freq = log_freq
        # Define the frequency for our long-term summary log
        self.summary_log_freq = 204_800 # Approx 200k steps, or one full episode length

        # Memory-efficient accumulators for the long-term summary
        self.reward_accumulator = defaultdict(float)
        self.stats_accumulator = {}
        self.accumulation_steps = 0
        
        # Memory usage tracking for debugging
        self.max_agents_tracked = 0

    def _on_training_start(self):
        if self.writer is None:
            # Create a sub-directory for histograms to keep the main log clean
            hist_log_dir = os.path.join(self.log_dir, 'histograms')
            os.makedirs(hist_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=hist_log_dir)

    def _on_step(self) -> bool:
        
        # === FREQUENT LOGGING (Every `log_freq` steps) ===
        if self.n_calls % self.log_freq == 0:
            # --- Log instantaneous rewards ---
            current_rewards = self.training_env.get_attr("progress_reward")
            if current_rewards:
                # Calculate mean rewards across all agents for this step
                mean_rewards = {key: np.mean([agent.get(key, 0) for agent in current_rewards])
                                for key in current_rewards[0]}
                for key, value in mean_rewards.items():
                    self.logger.record(f"rewards_mean/{key}", value)
                self.logger.record("rewards_mean/total", sum(mean_rewards.values()))

                # --- Update long-term reward accumulators ---
                for agent_rewards in current_rewards:
                    for key, value in agent_rewards.items():
                        self.reward_accumulator[key] += value
                self.accumulation_steps += len(current_rewards)

            # --- Get the latest agent stats (overwrite mode for memory efficiency) ---
            latest_stats = self.training_env.get_attr("agent_stats")
            if latest_stats:
                # Clear existing stats before adding new ones to prevent accumulation
                self.stats_accumulator.clear()
                
                # Store only the most recent stats, overwriting everything
                for agent_idx, stats_list in enumerate(latest_stats):
                    if stats_list:  # Check if list is not empty
                        # Only keep the latest stat, don't accumulate
                        self.stats_accumulator[agent_idx] = stats_list[0]
                
                # Track max agents for memory monitoring
                self.max_agents_tracked = max(self.max_agents_tracked, len(self.stats_accumulator))

        # === SUMMARY LOGGING (Every `summary_log_freq` steps) ===
        if self.n_calls > 0 and self.n_calls % self.summary_log_freq == 0:
            
            # --- Log accumulated reward summary ---
            if self.accumulation_steps > 0:
                print(f"\n--- Logging Summary at step {self.n_calls} ---")
                print(f"Max agents tracked: {self.max_agents_tracked}")
                
                # Calculate and log the mean reward over the entire cycle
                total_accumulated_reward = 0
                for key, total_value in self.reward_accumulator.items():
                    mean_value = total_value / self.accumulation_steps
                    self.logger.record(f"summary/reward_mean_{key}", mean_value)
                    total_accumulated_reward += mean_value
                self.logger.record("summary/reward_mean_total", total_accumulated_reward)
            
            # --- Log accumulated stats summary ---
            if self.stats_accumulator:
                # Process the snapshot of final stats from all agents
                all_final_stats = list(self.stats_accumulator.values())
                
                # Calculate and log the mean of key stats over the cycle
                mean_badge = np.mean([s.get('badge', 0) for s in all_final_stats])
                mean_level_sum = np.mean([s.get('levels_sum', 0) for s in all_final_stats])
                mean_map_progress = np.mean([s.get('max_map_progress', 0) for s in all_final_stats])
                
                self.logger.record("summary/mean_badge_count", mean_badge)
                self.logger.record("summary/mean_level_sum", mean_level_sum)
                self.logger.record("summary/mean_max_map_progress", mean_map_progress)

                # Create numpy arrays for histograms and immediately use them
                badge_array = np.array([s.get('badge', 0) for s in all_final_stats])
                level_array = np.array([s.get('levels_sum', 0) for s in all_final_stats])
                progress_array = np.array([s.get('max_map_progress', 0) for s in all_final_stats])
                
                # Log distributions for key metrics to the histogram writer
                self.writer.add_histogram("summary_dist/badge_count", badge_array, self.n_calls)
                self.writer.add_histogram("summary_dist/level_sum", level_array, self.n_calls)
                self.writer.add_histogram("summary_dist/max_map_progress", progress_array, self.n_calls)
                
                # Explicitly delete arrays to help garbage collection
                del badge_array, level_array, progress_array, all_final_stats

            # --- Log exploration map ---
            try:
                explore_map = np.array(self.training_env.get_attr("explore_map"))
                # Combine maps from all envs by taking the max value for each pixel
                if explore_map.ndim == 3:  # Should be (num_envs, height, width)
                    map_sum = reduce(explore_map, "f h w -> h w", "max")
                    self.logger.record("trajectory/explore_map", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))
                    # Explicitly delete to help garbage collection
                    del explore_map, map_sum
            except Exception as e:
                print(f"Warning: Could not log exploration map: {e}")

            # --- CRITICAL: Reset accumulators for the next cycle ---
            print(f"--- Summary complete. Resetting accumulators for next cycle. ---")
            self.reward_accumulator.clear()
            self.stats_accumulator.clear()
            self.accumulation_steps = 0
            self.max_agents_tracked = 0  # Reset tracking counter

        return True

    def _on_training_end(self):
        if self.writer:
            self.writer.close()
            
        # Final cleanup
        self.reward_accumulator.clear()
        self.stats_accumulator.clear()
        self.accumulation_steps = 0
        self.max_agents_tracked = 0