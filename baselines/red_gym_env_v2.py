import uuid
import json
from pathlib import Path
import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
#from pyboy.logger import log_level
import mediapy as media
from einops import repeat
from collections import deque 

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE
from map_lookup import MapIds
from tilesets import Tilesets
from field_moves import FieldMoves
from generate_stream_overlay import update_agent_overlay, prepare_overlay_data

event_flags_start = 0xD747
event_flags_end = 0xD7F6 # 0xD761 # 0xD886 temporarily lower event flag range for obs input
museum_ticket = (0xD754, 0)

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3
        self.agent_id = config["rank"]
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        self.has_used_cut = 0

        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }
        if isinstance(config["disable_wild_encounters"], bool): 
            self.disable_wild_encounters = config["disable_wild_encounters"]
            self.setup_disable_wild_encounters_maps = set([])
        elif isinstance(config["disable_wild_encounters"], list):
            self.disable_wild_encounters = len(config["disable_wild_encounters"]) > 0
            self.disable_wild_encounters_maps = {
                MapIds[item].name for item in config["disable_wild_encounters"] ## Need to find MapIds
            }
        else:
            raise ValueError("Disable wild enounters must be a boolean or a list of MapIds")
        self.infinite_money = config["infinite_money"]
        self.recent_coords = deque([0] * 10, maxlen=10)  # Initialize with 10 zeros & Track last 50 positions
        self.local_area_hashes = set()
        self.progress_reward = []

        self.last_position = [0, 0, 0] 
        self.last_action = []
        self.directional_reward = 0
        self.frontier_reward = 0
        self.path_progress = set()
        self.last_coord = []
        self.party_info = {} #Store pokemon id, moves, levels, and pp
        self.pokemon_levels = {} #Store pokemon id and max level, updating each iteration

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        with open("map_data.json") as f:
            map_data = json.load(f)
        self.map_data = map_data

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks),
                "recent_path": spaces.Box(low=0, high=1e8, shape=(10,), dtype=np.int32),  # Last 10 area hashes
            }
        )

        head = "headless" if config["headless"] else "SDL2"

        #log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            symbol_file=config["symbol_file"],
            debugging=False,
            disable_input=False,
            window_type=head
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options={}):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros( self.output_shape, dtype=np.uint8)
        
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.last_power = 0
        self.total_healing_rew = 0
        self.total_healing_pp_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.start_menu_count = 0
        self.start_stats_count = 0
        self.start_pokemenu_count = 0
        self.start_itemmenu_count = 0
        self.stuck_penalty = 0
        self.novelty_reward = 0
        self.last_new_step = 0
        self.last_action = []
        self.directional_reward = 0
        self.frontier_reward = 0
        self.path_progress = set()
        self.last_coord = []
        self.has_used_cut = 0

        self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        self.pokemon_levels = {} #Store pokemon id and max level, updating each iteration
        self.party_info = {} #Store pokemon id, moves, levels, and pp
        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray()[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):
        
        screen = self.render()

        self.update_recent_screens(screen)
        
        # normalize to approx 0-1
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        recent_path = np.array(self.recent_coords, dtype=np.uint8)  # Last 10 areas

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum), ## May need to come back and change this to the same as the level reward
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "recent_path": np.array(recent_path, dtype=np.int32),
        }

        return observation

    def step(self, action):

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_explore_map()

        self.update_heal_reward()

        self.party_size = self.read_m(0xD163)

        self.check_in_menus()

        self.cut_if_next()

        # Compute new reward
        new_reward = self.update_reward()

        self.update_seen_coords()

        if (
            self.disable_wild_encounters
            and self.read_m(0xD35E) in [59, 60, 61] ##Shutting off fights in Mt Moon
        ):
            self.pyboy.set_memory_value(0xD887, 0X00)
        if (
            self.infinite_money
            and self.step_count % 50 == 0
        ):
            self.pyboy.set_memory_value(0xD347, 0x01)
        if self.step_count % 100 == 0:
            self.update_10_pokeballs() # Always have 10 pokeballs    
            self.force_learn_HMs() # Force learn HMs
            if self.headless == False:
                print("Updating overlay data...")
                prepare_overlay_data(self, location = self.get_current_location(), badges = self.get_badges_status()) # Update overlay data

        self.last_health = self.read_hp_fraction()

        self.last_power = self.get_low_power_moves()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        #if step_limit_reached:
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                # release button
                self.pyboy.send_input(self.release_actions[action])
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ] ## TODO: Update this to the same as level rewards
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(self.pokemon_levels.values()),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "pokemon_seen": self.get_pokedex_seen(),
                "pokemon_caught": self.get_pokedex_caught(),
                "start_menu": self.start_menu_count,
                "stats_menu": self.start_stats_count,
                "pokemon_menu": self.start_pokemenu_count,
                "item_menu": self.start_itemmenu_count,
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad*4, self.coords_pad*4), 
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:,:,0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:,:,0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    def get_game_coords(self):
        ### D35E is the map location in map_lookup
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def get_local_area_hash(self, x, y):
        # Cluster coordinates into 8x8 blocks
        grid_size = 4  # Changed from 8 to 4 for finer granularity
        return hash((
            self.read_m(0xD35E),  # Current map ID
            (x // grid_size), 
            (y // grid_size)
        ))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        self.seen_coords[coord_string] = self.step_count

    def update_directional_persistence(self, action):
        """
        Encourages the agent to maintain movement in a consistent direction.
        - Keeps accumulating reward if moving the same way.
        - Slowly decreases if direction changes unnecessarily.
        - Resets if the agent is stuck.
        """
        if self.in_battle():
            return  # No directional reward in battles or menus

        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"

        # If the agent didn't move, decay the reward
        if coord_string == self.last_coord:
            self.directional_reward = max(self.directional_reward - 10, 0)  # Gradually reduce reward if stuck
            return

        # If moving in the same direction, keep accumulating reward
        if action == self.last_action:
            self.directional_reward += 0.1  # Builds up over time
        else:
            # If forced to turn, apply a smaller decrease instead of resetting
            if self.is_forced_turn(coord_string):
                self.directional_reward = max(self.directional_reward - 5, 0)
            else:
                self.directional_reward = max(self.directional_reward - 10, 0)  # Reduce faster if unnecessary turn

        self.last_coord = coord_string
        self.last_action = action

    def find_location_by_id(self, map_data, location_id):
        if "regions" in map_data and isinstance(map_data["regions"], list):
            for region in map_data["regions"]:
                if region["id"] == str(location_id):
                    return region["name"]
        else:
            print(f"Invalid map data format or missing 'regions' key.")
        return None  # Return None if no matching id is found

    def get_current_location(self):
        map_x, map_y, map_n = self.get_game_coords()
        return self.find_location_by_id(self.map_data, map_n)

    def get_badges_status(self):
        return [self.read_bit(0xD356, i) for i in range(8)]

    def is_forced_turn(self, coord_string):
        """
        Determines if the agent is forced to turn because it's at a dead end.
        This occurs if the agent has only one viable movement option.
        """
        x_pos, y_pos, map_n = self.get_game_coords()

        # Check possible movement options
        possible_moves = [
            f"x:{x_pos+1} y:{y_pos} m:{map_n}",
            f"x:{x_pos-1} y:{y_pos} m:{map_n}",
            f"x:{x_pos} y:{y_pos+1} m:{map_n}",
            f"x:{x_pos} y:{y_pos-1} m:{map_n}",
        ]

        # Count how many moves are available
        open_paths = sum(1 for move in possible_moves if move not in self.seen_coords)

        return open_paths <= 1  # True if only one option remains

    def get_revisit_penalty(self):
        """
        Applies a small penalty when the agent re-enters an area too soon.
        - No penalty for first-time visits.
        - Penalty scales based on revisit frequency.
        - No penalty during battles.
        """
        if self.in_battle():
            return 0.0  # No penalty if the agent is in a battle
        
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"

        if map_n == 88 & self.read_bit(0xD7F2, 7) == 1:
            return 0.0 # No penalty in Bill's Lab until we get the SS Anne Ticket
        if map_n == 101 & self.read_bit(0xD803, 2) == 1:
            return 0.0 # No penalty in Captain's Quarters until we get the HM01
        if map_n == 92 & self.read_bit(0xD773, 1) == 1:
            return 0.0 # No penalty in Vermillion Gym until locks are opened

        # Ensure we are not incorrectly penalizing first-time visits
        if coord_string not in self.seen_coords:
            return 0.0  # No penalty for a truly new visit

        # Calculate time since last visit
        last_seen_step = self.seen_coords[coord_string]
        time_since_last_visit = self.step_count - last_seen_step

        if time_since_last_visit >= 500:
            return 0.0  # No penalty if enough time has passed

        # Scale penalty based on how soon the agent returns
        penalty = -0.01 * (time_since_last_visit)  # Starts at -5, reduces as more time passes
        # Want to invert this so penalty is larger the sooner it revisits the space.

        return max(penalty, -5)  # Ensure it doesn’t exceed -5
    
    def get_immediate_revisit_penalty(self):
        """
        Applies a small penalty when the agent re-enters an area too soon.
        - No penalty for first-time visits.
        - Penalty scales based on revisit frequency.
        - No penalty during battles.
        """
        if self.in_battle():
            return 0.0  # No penalty if the agent is in a battle

        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"

        # Ensure we are not incorrectly penalizing first-time visits
        if coord_string not in self.seen_coords:
            return 0.0  # No penalty for a truly new visit

        # Calculate time since last visit
        last_seen_step = self.seen_coords[coord_string]
        time_since_last_visit = self.step_count - last_seen_step

        if time_since_last_visit >= 100:
            return 0.0  # No penalty if enough time has passed
        if map_n == 88 & self.read_bit(0xD7F2, 7) == 1:
            return 0.0 # No penalty in Bill's Lab until we get the SS Anne Ticket
        if map_n == 101 & self.read_bit(0xD803, 2) == 1:
            return 0.0 # No penalty in Captain's Quarters until we get the HM01
        if map_n == 92 & self.read_bit(0xD773, 1) == 1:
            return 0.0 # No penalty in Vermillion Gym until locks are opened

        return -5.0


    def get_frontier_bonus(self):
        """
        Encourages movement toward unexplored tiles.
        - If the agent doesn't move and isn't in battle, the reward is 0.
        - The bonus is higher for moving into areas with many unexplored neighbors.
        """
        if self.in_battle():
            return self.frontier_reward  # No penalty for standing still in battle

        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"

        # If the agent didn't move, set the bonus to 0
        if coord_string == self.last_coord:
            return 0.0  

        # Check how many neighboring tiles are unexplored
        neighbors = [
            f"x:{x_pos+1} y:{y_pos} m:{map_n}",
            f"x:{x_pos-1} y:{y_pos} m:{map_n}",
            f"x:{x_pos} y:{y_pos+1} m:{map_n}",
            f"x:{x_pos} y:{y_pos-1} m:{map_n}",
        ]

        unexplored_neighbors = sum(1 for n in neighbors if n not in self.seen_coords)

        # Reward based on number of unexplored neighbors
        self.frontier_reward = 0.2 * unexplored_neighbors  # Slightly reduced to prevent over-rewarding

    def is_on_a_new_path(self, coord_string):
        """
        Determines if the agent is entering a new section of the map 
        (not just an unexplored tile but a new route).
        """
        return coord_string not in self.path_progress

    def in_battle(self):
        #Check if player is in any type of battle
        return (
            self.read_m(0xD057) == 0x0F or  # Battle menu active
            self.read_m(0xCFFB) == 0x01     # Wild battle flag
        )
    """
    def update_exploration_rewards(self):
        x, y, map_n = self.get_game_coords()
        area_hash = self.get_local_area_hash(x, y)
        freq_penalty = 0.0

        # Convert hash to positive integer
        positive_hash = abs(area_hash) % (10**8)  # 8-digit number

        if self.in_battle():
            freq_penalty = 1.0  # Pause exploration rewards during battles
        elif self.read_m(0xD803) == 0 and self.read_m(0xD057) == 0 and self.read_m(0xCF13) == 0:
        # Block rewards only in non-essential menus
            if self.read_m(0xFF8C) == 6:
                # Start Menu
                if self.read_m(0xCF94) == 0:
                    freq_penalty = 0.0
                # Stats Menu
                if self.read_m(0xCF94) == 1:
                    freq_penalty = 0.0
                # Pokemon Menu
                if self.read_m(0xCF94) == 2:
                    freq_penalty = 0.0
        else:
            # Recent path penalty calculation
            recent_count = list(self.recent_coords).count(positive_hash)
            freq_penalty = max(0, 1.0 - (recent_count * 0.3))  # Stronger penalty

        # First visit reward
        if positive_hash not in self.local_area_hashes:
            self.local_area_hashes.add(positive_hash)
            base_reward = 2.0
        else:
            base_reward = 0.0
        
        # Store in recent memory
        self.recent_coords.append(positive_hash)
        self.last_position = (x, y, map_n)
        
        # Rest of exploration calculation...        
        return base_reward + freq_penalty
"""
    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            #print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0]-self.coords_pad:c[0]+self.coords_pad,
                c[1]-self.coords_pad:c[1]+self.coords_pad
            ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:,:, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:,:, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

        """
        if done:
            self.all_runs.append(self.progress_reward)
            with open(
                self.s_path / Path(f"all_runs_{self.instance_id}.json"), "w"
            ) as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                compression="gzip",
                mode="a",
            )
        """

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)
    
    def set_m(self, addr, val):
        return self.pyboy.set_memory_value(addr, val)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end) 
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def store_pokemon_levels(self):
        ## Storing the level for each unique pokemon in the party throughout the game
        current_party_levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        current_party_names = [self.read_m(a) for a in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
        for i in range(6):
            if current_party_names[i] in self.pokemon_levels:
                self.pokemon_levels[current_party_names[i]] = max(current_party_levels[i], self.pokemon_levels[current_party_names[i]])
            else:
                self.pokemon_levels[current_party_names[i]] = current_party_levels[i]

    def update_pokemon_info(self):
        """
        Reads party Pokémon data from memory, calculates correct HP,
        and stores it along with other info in self.party_info.
        """
        self.party_info = {}

        try:
            # --- Read all data first ---

            # Current HP Bytes (Lower Address First = Most Significant Byte for Big Endian)
            party_current_hp_byte1 = [self.read_m(a) for a in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
            party_current_hp_byte2 = [self.read_m(a) for a in [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]]

            # Max HP Bytes (Using YOUR ORIGINAL addresses - REVERTED)
            # (Lower Address First = Most Significant Byte for Big Endian)
            party_max_hp_byte1 = [self.read_m(a) for a in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]] # YOUR ORIGINAL ADDRESS LIST 1
            party_max_hp_byte2 = [self.read_m(a) for a in [0xD18E, 0xD1BA, 0xD1E6, 0xD212, 0xD23E, 0xD26A]] # YOUR ORIGINAL ADDRESS LIST 2
            # --- IF the standard addresses ARE correct (D16E/F, D19A/B etc.), uncomment below and comment out the two lines above ---
            # party_max_hp_byte1 = [self.read_m(a) for a in [0xD16E, 0xD19A, 0xD1C6, 0xD1F2, 0xD21E, 0xD24A]] # Standard Address List 1
            # party_max_hp_byte2 = [self.read_m(a) for a in [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]] # Standard Address List 2
            # --- End Address Choice ---

            party_levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
            party_species_ids = [self.read_m(a) for a in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]]
            party_move1_ids = [self.read_m(a) for a in [0xD173, 0xD19F, 0xD1CB, 0xD1F7, 0xD223, 0xD24F]]
            party_move2_ids = [self.read_m(a) for a in [0xD174, 0xD1A0, 0xD1CC, 0xD1F8, 0xD224, 0xD250]]
            party_move3_ids = [self.read_m(a) for a in [0xD175, 0xD1A1, 0xD1CD, 0xD1F9, 0xD225, 0xD251]]
            party_move4_ids = [self.read_m(a) for a in [0xD176, 0xD1A2, 0xD1CE, 0xD1FA, 0xD226, 0xD252]]

            # --- Process data for each party slot ---
            for i in range(6):
                species_id = party_species_ids[i]
                if species_id == 0 or species_id == 0xFF:
                    continue # Skip empty slot

                # **FIX 2b: Combine HP bytes as Big Endian**
                # current_hp = (Most Significant Byte << 8) | Least Significant Byte
                # Most Significant Byte is usually at the LOWER memory address (_byte1)
                current_hp = (party_current_hp_byte1[i] << 8) | party_current_hp_byte2[i]
                max_hp = (party_max_hp_byte1[i] << 8) | party_max_hp_byte2[i]

                # Create the display string
                hp_string = f"{current_hp}/{max_hp}"

                # Get Pokemon Name from Species ID
                pokemon_name = self.get_pokemon_name(species_id)

                # Get Move Names from IDs
                # **FIX 1: Add '+ 1' back if your mapping function requires it**
                moves = [
                    self.get_move_names_map(party_move1_ids[i] + 1),
                    self.get_move_names_map(party_move2_ids[i] + 1),
                    self.get_move_names_map(party_move3_ids[i] + 1),
                    self.get_move_names_map(party_move4_ids[i] + 1)
                ]
                # --- If '+ 1' was NOT needed, remove it from the 4 lines above ---

                level = party_levels[i]
                party_key = f"pokemon_{i+1}"

                # Store data including integer HP values
                self.party_info[party_key] = (
                    pokemon_name,    # 0: Pokemon name (string)
                    level,           # 1: Level (int)
                    moves[0],        # 2: Move 1 name (string)
                    moves[1],        # 3: Move 2 name (string)
                    moves[2],        # 4: Move 3 name (string)
                    moves[3],        # 5: Move 4 name (string)
                    hp_string,       # 6: HP display string (string)
                    current_hp,      # 7: Current HP (int)
                    max_hp           # 8: Max HP (int)
                )

        except Exception as e:
            # Ensure party_info is at least an empty dict if error occurs
            self.party_info = {}

    def get_levels_reward(self): 
        ## Do this by taking the max for each pokemon encountered
        self.store_pokemon_levels()
        self.update_pokemon_info()
        level_sum = sum(self.pokemon_levels.values()) - 6
        if level_sum < 20:
            scaled = level_sum 
        elif level_sum < 40:
            scaled = 20 + ((level_sum - 20) / 2)
        else:
            scaled = 30 + ((level_sum - 40) / 4)
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    
    def update_10_pokeballs(self):
        if self.pyboy.get_memory_value(0xD31E):
            self.pyboy.set_memory_value(0xD31F, 10)
    
    def get_item_reward(self):
        ##Checks the bag slots and awards points for items like pokeballs, potions and revives
        self.item_reward = 0
        bag_qty = {0xD31E: 0xD31F,
                   0xD320: 0xD321,
                   0xD322: 0xD323,
                   0xD324: 0xD325,
                   0xD326: 0xD327,
                   0xD328: 0xD329,
                   0xD32A: 0xD32B,
                   0xD32C: 0xD32D,
                   0xD32E: 0xD32F,
                   0xD330: 0xD331,
                   0xD332: 0xD333,
                   0xD334: 0xD335,
                   0xD336: 0xD337,
                   0xD338: 0xD339,
                   0xD33A: 0xD33B,
                   0xD33C: 0xD33D,
                   0xD33E: 0xD33F,
                   0xD340: 0xD341,
                   0xD342: 0xD343,
                   0xD344: 0xD345}
        ## TODO: Add storage items as well - https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Items
        for k, v in bag_qty.items():
            if self.read_m(k) is not None:
                self.item_reward += min(10, self.bit_count(self.read_m(v)))
        return self.item_reward
    
    def get_pokedex_caught(self):
        ## Checks pokedex for pokemon caught and sums that number up to add to reward
        self.num_caught = 0
        for address in [0xD2F7, 0xD2F8, 0xD2F9, 0xD2FA, 0xD2FB, 0xD2FC, 0xD2FD, 0xD2FE, 0xD2FF, 0xD300, 0xD301, 0xD302, 0xD303, 0xD304, 0xD305, 0xD306, 0xD307, 0xD308, 0xD309]:
            self.num_caught += self.bit_count(self.read_m(address))
        return max(self.num_caught, 0)

    def get_pokedex_seen(self):
        ## Checks pokedex for pokemon seen and sums that number up to add to reward
        SEEN_POKE_ADDR = range(0xD30A, 0xD31D)
        num_seen = [self.read_m(a) for a in SEEN_POKE_ADDR]
        return max(sum([self.bit_count(n) for n in num_seen]), 0)

    def setup_disable_wild_encounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_register(
            bank,
            addr + 8,
            self.disable_wild_encounter_hook,
            None,
        )

    def disable_wild_encounter_hook(self, *args, **kwargs):
        if (
            self.disable_wild_encounters
            and self.read_m(0xD35E) not in self.disable_wild_encounters_maps
        ):
            self.pyboy.set_memory_value(0xD887, 0X00)

    def setup_enable_wild_ecounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_deregister(bank, addr)
    
    def check_in_menus(self):
        if self.read_m(0xD803) == 0 and self.read_m(0xD057) == 0 and self.read_m(0xCF13) == 0:
            if self.read_m(0xFF8C) == 6:
                # Start Menu
                if self.read_m(0xCF94) == 0:
                    self.start_menu_count = 1
                # Stats Menu
                if self.read_m(0xCF94) == 1:
                    self.start_stats_count = 1
                # Pokemon Menu
                if self.read_m(0xCF94) == 2:
                    self.start_pokemenu_count = 1
            # Item menu
            if self.read_m(0xCF94) == 3:
                self.start_itemmenu_count = 1
    
    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]
    
    def get_lead_pokemon_info(self):
        ## Create string to add to stream['extra'] with pokemon name and level
        self.extra_string = f"{self.get_pokemon_name(self.read_m(0xD164))}: {self.read_m(0xD18C)}" 
        return self.extra_string

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=True):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 10, ## Increased to 10x to encourage more trainer battles
            "level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew * 10,
            #"op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            "dead": self.reward_scale * self.died_count * -0.01,
            "badge": self.reward_scale * self.get_badges() * 100,
            "exp": self.reward_scale * self.explore_weight * len(self.seen_coords),
            #"exp_boost": self.update_exploration_rewards(),
            #"stuck": self.reward_scale * self.explore_weight * self.stuck_penalty,## Removing novelty for now + self.novelty_reward,
            #"items": self.reward_scale * self.get_item_reward(), ## Removed due to AI spam buying antidotes for score TODO: Add back only for potions and pokeballs
            "seen": self.reward_scale * self.get_pokedex_seen(),
            "cap": self.reward_scale * self.get_pokedex_caught() * 2,
            "menu": (self.start_menu_count + self.start_stats_count + self.start_pokemenu_count + self.start_itemmenu_count) * 0.01,
            # New Rewards
            #"frontier": self.frontier_reward,  # Encourages exploring new paths; must be before directional_reward;; Removed for now
            #"dir": self.directional_reward,  # Reward for maintaining movement
            "revisit": self.get_revisit_penalty(),  # Penalizes unnecessary backtracking,
            "ir": self.get_immediate_revisit_penalty(), #Penalizes immediate backtracking
            "move": self.reward_scale * self.move_rewards(), ## Reward for learning stronger moves
            "HMs": self.reward_scale * self.has_HMs(),
            "ucut": self.reward_scale * self.has_used_cut
        }

        return state_scores

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        x_pos, y_pos, map_n = self.get_game_coords()
        if self.read_m(0xD163) == self.party_size:
            # if health increased and party size did not change
            if cur_health > self.last_health:
                    if self.last_health > 0:
                        if map_n in [41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 182]:
                            # Make sure in pokemon center
                            heal_amount = cur_health - self.last_health
                            self.total_healing_rew += heal_amount
                    else:
                        self.died_count += 1
                        self.last_health = 1
                        if (
                            self.disable_wild_encounters
                        ):
                            self.pyboy.set_memory_value(0xD887, 0X0A)

    def read_hp_fraction(self):
        hp_sum = sum([
            
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)
    
    def has_HMs(self):
        HM_reward = 0
        if self.read_bit(0xD803, 0) == 1:
            HM_reward += 0.01
        if self.read_bit(0xD7E0, 6) == 1:
            HM_reward += 0.1
        if self.read_bit(0xD857, 0) == 1:
            HM_reward += 1
        if self.read_bit(0xD78E, 0) == 1:
            HM_reward += 10
        if self.read_bit(0xD7C2, 0) == 1:
            HM_reward += 100
        return HM_reward
    
    def move_rewards(self):
        # Memory addresses for all move slots across 6 Pokémon (4 moves each)
        move_addresses = [
            # First Pokémon moves
            0xD173, 0xD174, 0xD175, 0xD176,
            # Second Pokémon moves
            0xD19F, 0xD1A0, 0xD1A1, 0xD1A2,
            # Third Pokémon moves
            0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE,
            # Fourth Pokémon moves
            0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA,
            # Fifth Pokémon moves
            0xD223, 0xD224, 0xD225, 0xD226,
            # Sixth Pokémon moves
            0xD24F, 0xD250, 0xD251, 0xD252
        ]
        move_reward = 0
        # Check if any move slot contains Cut (either as value 15 or string "CUT")
        for address in move_addresses:
            move_value = self.read_m(address)
            if move_value in [15, 19, 57, 70]:
                move_reward += 100
            elif move_value in [63, 56, 72, 83, 87, 89, 90, 94, 136, 143]:
                move_reward += 1
            elif move_value in [12, 14, 17, 25, 26, 28, 58, 59, 60, 61, 62, 91, 92, 123, 124, 126, 127, 129, 158, 161]:
                move_reward += 0.5
            elif move_value in [4, 5, 7, 8, 9, 11, 24, 32, 33, 37, 41, 42, 44, 49, 51, 52, 53, 55, 65, 66, 67, 69, 71, 82, 85, 88, 93, 99, 101, 106, 119, 125, 128, 131, 134, 139, 140, 149, 152, 157, 162, 163]:
                move_reward += 0.2
            elif move_value in [33, 16, 1, 2, 3, 6, 10, 18, 20, 21, 22, 23, 29, 30, 34, 35, 36, 38, 40, 64, 73, 80, 84, 98, 121, 130, 132, 141, 145, 146, 154, 155]:
                move_reward += 0.1
        return move_reward
        
    def force_learn_HMs(self):
        if self.read_bit(0xD803, 0) == 1: # Has Obtained Cut HM
            if self.read_m(0xD164) in [153, 154, 9]: # Pokémon is Bulbasaur, Ivysaur, or Venusaur
                if self.read_m(0xD174) == 45: # Replace Growl with Cut
                    self.set_m(0xD174, 15)
            if self.read_m(0xD165) in [153, 154, 9]: # Pokémon is Bulbasaur, Ivysaur, or Venusaur
                if self.read_m(0xD1A0) == 45: # Replace Growl with Cut
                    self.set_m(0xD1A0, 15)
            if self.read_m(0xD166) in [153, 154, 9]: # Pokémon is Bulbasaur, Ivysaur, or Venusaur
                if self.read_m(0xD1CC) == 45: # Replace Growl with Cut
                    self.set_m(0xD1CC, 15)
            if self.read_m(0xD167) in [153, 154, 9]: # Pokémon is Bulbasaur, Ivysaur, or Venusaur
                if self.read_m(0xD1F8) == 45: # Replace Growl with Cut
                    self.set_m(0xD1F8, 15)
            if self.read_m(0xD168) in [153, 154, 9]: # Pokémon is Bulbasaur, Ivysaur, or Venusaur
                if self.read_m(0xD224) == 45: # Replace Growl with Cut
                    self.set_m(0xD224, 15)
            if self.read_m(0xD169) in [153, 154, 9]: # Pokémon is Bulbasaur, Ivysaur, or Venusaur
                if self.read_m(0xD250) == 45: # Replace Growl with Cut
                    self.set_m(0xD250, 15)

    def _execute_cut_sequence(self, pokemon_index):
        """
        Executes the button presses for CUT using the Pokemon at the given index (0-5).
        Returns True if sequence seems successful, False otherwise.
        """
        try:
            # Simple fixed ticks for menus often work better
            menu_tick_duration = 8
            action_tick_duration = 24 # Longer pause after actions

            # Open start menu
            self.pyboy.button("START")
            self.pyboy.tick(action_tick_duration)
            # TODO: Add check here: Is Start Menu open? (e.g., check wMenuJoypadPoll + wCurrentMenuItem)

            # Select Pokemon menu item (usually index 1)
            # Assumes 'Pokemon' is the second item (index 1)
            if self.read_m(self.pyboy.symbol_lookup("wCurrentMenuItem")[1]) != 0: # If not already on first item
                 self.pyboy.button("UP") # Go to top
                 self.pyboy.tick(menu_tick_duration)
            self.pyboy.button("DOWN") # Go to Pokemon (item 1)
            self.pyboy.tick(menu_tick_duration)
            # TODO: Verify wCurrentMenuItem is now 1

            self.pyboy.button("A")
            self.pyboy.tick(action_tick_duration)
            # TODO: Check if party screen is open (e.g., check wPartyMenuCursorBackup?)

            # Select the correct Pokemon
            # Move cursor down 'pokemon_index' times
            for _ in range(pokemon_index):
                 self.pyboy.button("DOWN")
                 self.pyboy.tick(menu_tick_duration)
            # TODO: Verify cursor is on the right Pokemon?

            # Select Pokemon ('A')
            self.pyboy.button("A")
            self.pyboy.tick(action_tick_duration)
            # TODO: Check if Pokemon submenu is open

            # Select CUT from submenu
            # This part is tricky - finding CUT reliably might still need looping
            # Or assume CUT is always the first field move? (Risky)
            # Using the wFieldMoves approach from original code:
            found_cut = False
            _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
            for _ in range(5): # Max 5 items: CUT, FLY, SURF, STR, FLASH
                 current_item_index = self.read_m(self.pyboy.symbol_lookup("wCurrentMenuItem")[1])
                 field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4] # Read current field moves
                 if current_item_index < len(field_moves) and field_moves[current_item_index] == FieldMoves.CUT.value:
                     found_cut = True
                     break
                 self.pyboy.button("DOWN")
                 self.pyboy.tick(menu_tick_duration)

            if not found_cut:
                # Press B multiple times to back out safely
                for _ in range(4):
                    self.pyboy.button("B")
                    self.pyboy.tick(action_tick_duration)
                return False

            # Use CUT ('A')
            self.pyboy.button("A")
            self.pyboy.tick(action_tick_duration * 2) # Wait longer for cut animation/text

            # Need to wait for the "used Cut!" message and field effect
            # This might require waiting until control is returned (e.g., wPlayerStandingTile != 0 then == 0 again)
            # Or just a longer fixed delay
            self.pyboy.tick(80) # Wait for cut animation/message box

            # Press A/B to clear message box if necessary
            self.pyboy.button("A")
            self.pyboy.tick(action_tick_duration)
            self.pyboy.button("B") # Just in case
            self.pyboy.tick(action_tick_duration)

            self.has_used_cut = getattr(self, 'has_used_cut', 0) + 1 # Increment counter safely
            return True

        except Exception as e:
            # Attempt to back out of menus
            try:
                 for _ in range(4):
                    self.pyboy.button("B")
                    self.pyboy.tick(action_tick_duration)
            except: pass # Ignore errors during bailout
            return False
        
    def _find_pokemon_with_cut_index(self):
        """
        Finds the party index (0-5) of the first Pokemon with Cut.
        Returns None if no Pokemon has Cut.
        (This is a robust alternative to menu looping)
        """
        try:
            MOVE_CUT_ID = 0x0F # Corresponds to 15
            # Simplified: Assumes party structure similar to update_pokemon_info
            # Adapt based on how you can quickly read party moves
            party_move_offsets = [0xD174, 0xD1A0, 0xD1CC, 0xD1F8, 0xD224, 0xD250] # Start of moves for each mon
            for i in range(6):
                species_addr = 0xD16B + i * 44 # Check species ID to see if slot is valid
                if self.read_m(species_addr) in [0, 0xFF]: continue # Skip empty

                move_addr = party_move_offsets[i]
                moves = [self.read_m(move_addr + j) for j in range(4)]
                if MOVE_CUT_ID in moves:
                    return i # Return index 0-5
            return None
        except Exception as e:
            return None

    def cut_if_next(self):
        # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
        # Check standing still
        try:
            # Checking if player has Cascade Badge, otherwise no need to check for cuttable objects
            WRAM_OBTAINED_BADGES = 0xD356
            CASCADE_BADGE_BIT_INDEX = 1 # Bit index for read_bit (0-7)
            has_cascade_badge = self.read_bit(WRAM_OBTAINED_BADGES, CASCADE_BADGE_BIT_INDEX)
            if not has_cascade_badge:
                # print("DEBUG: Missing Cascade Badge") # Optional
                return False
            
            WRAM_MAP_GROUP = 0xD35E
            WRAM_MAP_NUMBER = 0xD35F
            WRAM_TILE_MAP = 0xD4A0
            WRAM_PARTY_COUNT = 0xD163
            WRAM_PARTY_MON_START = 0xD16B
            PARTY_MON_STRUCT_SIZE = 44
            PARTY_MON_MOVES_OFFSET = 14 # Offset from start of Pokemon struct to Move 1
            MOVE_CUT_ID = 0x0F
            TILE_MAP_WIDTH = 20
            TILE_MAP_HEIGHT = 18
            CUT_BUSH_OVERWORLD = 0x3D
            CUT_TREE_GYM = 0x50
            # Check map ID (USE THE CORRECT LOGIC FROM POINT 1 ABOVE)
            current_map_number = self.read_m(WRAM_MAP_NUMBER)
            ERIKA_GYM_ID = 134
            OVERWORLD_CUT_MAP_IDS = set(range(0, 89))
            is_erika_gym = (current_map_number == ERIKA_GYM_ID)
            is_overworld = (current_map_number in OVERWORLD_CUT_MAP_IDS) # Check if current map is in the set

            if not (is_overworld or is_erika_gym):
                return False # Exit if not on a relevant map
            # 2. Check adjacent tiles
            tileMap_bytes_list = []
            start_addr = WRAM_TILE_MAP
            num_bytes = TILE_MAP_WIDTH * TILE_MAP_HEIGHT
            for i in range(num_bytes):
                byte_val = self.read_m(start_addr + i)
                tileMap_bytes_list.append(byte_val)

            # Reshape into the map
            try:
                tileMap = np.array(tileMap_bytes_list, dtype=np.uint8).reshape((TILE_MAP_HEIGHT, TILE_MAP_WIDTH))
            except ValueError as e:
                print(f"ERROR: Reshape failed. Bytes read: {len(tileMap_bytes_list)}. Error: {e}")
                return False
            # Player screen position (center)
            y, x = 8, 8 # Center of the 18x20 screen buffer
            direction_to_press = None
            target_tile = CUT_TREE_GYM if is_erika_gym else CUT_BUSH_OVERWORLD

            # Check UP (Tiles at y=6,7 ; x=8,9)
            if target_tile in tileMap[y - 2 : y, x : x + 2]:
                direction_to_press = "UP"
            # Check DOWN (Tiles at y=10,11 ; x=8,9)
            elif target_tile in tileMap[y + 2 : y + 4, x : x + 2]:
                direction_to_press = "DOWN"
            # Check LEFT (Tiles at y=8,9 ; x=6,7)
            elif target_tile in tileMap[y : y + 2, x - 2 : x]:
                direction_to_press = "LEFT"
            # Check RIGHT (Tiles at y=8,9 ; x=10,11)
            elif target_tile in tileMap[y : y + 2, x + 2 : x + 4]:
                direction_to_press = "RIGHT"
            if direction_to_press is None:
                # logging.debug("No cuttable object adjacent.") # Optional debug
                return False # No cuttable object found
            else:
                print("Found cut location!") # DEBUG
                self.has_used_cut = getattr(self, 'has_used_cut', 0) + 0.01 # Increment counter safely
            # 3. Face the object
            self.pyboy.button(direction_to_press)
            self.pyboy.tick(8) # Short tick to register facing direction

            # 4. Find Pokemon with Cut
            cut_pokemon_index = self._find_pokemon_with_cut_index()
            if cut_pokemon_index is None:
                print(f"Agent {self.agent_id}: No Pokemon with Cut found in party.") # DEBUG
                return False # Cannot proceed
            else:
                self.has_used_cut = getattr(self, 'has_used_cut', 0) + 0.1 # Increment counter safely

            # 5. Execute the sequence
            success = self._execute_cut_sequence(cut_pokemon_index)

            return success # Return True if sequence attempted/succeeded, False if failed

        except Exception as e:
            return e # Return False on any unexpected error

    def update_pp_heal_reward(self):
        cur_power = self.get_low_power_moves()
        # if health increased and party size did not change
        if cur_power > self.last_power and self.read_m(0xD163) == self.party_size:
            if self.last_power > 0:
                heal_pp_amount = cur_power - self.last_power
                self.total_healing_pp_rew += heal_pp_amount
    
    def get_low_power_moves(self):
        self.low_power_moves = 0
        for add in [0xD188, 0xD189, 0xD18A, 0xD18B, 0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7, 0xD1E0, 0xD1E1, 0xD1E2, 0xD1E3, \
                    0xD20C, 0xD20D, 0xD20E, 0xD20F, 0xD238, 0xD239, 0xD23A, 0xD23B, 0xD264, 0xD265, 0xD266, 0xD267]:
            #if self.bit_count(self.read_m(add)) < 10:
            #    self.low_power_moves += (10 - self.bit_count(self.read_m(add)))
            self.low_power_moves += self.read_hp(add)
        return self.low_power_moves

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
    
    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))
    
    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))
    
    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_map_location(self, map_idx):
        map_locations = {
            0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
            1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
            2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
            3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
            62: {"name": "Invaded house (Cerulean City)", "coordinates": np.array([290, 227])},
            63: {"name": "trade house (Cerulean City)", "coordinates": np.array([290, 212])},
            64: {"name": "Pokémon Center (Cerulean City)", "coordinates": np.array([290, 197])},
            65: {"name": "Pokémon Gym (Cerulean City)", "coordinates": np.array([290, 182])},
            66: {"name": "Bike Shop (Cerulean City)", "coordinates": np.array([290, 167])},
            67: {"name": "Poké Mart (Cerulean City)", "coordinates": np.array([290, 152])},
            35: {"name": "Route 24", "coordinates": np.array([250, 235])},
            36: {"name": "Route 25", "coordinates": np.array([270, 267])},
            12: {"name": "Route 1", "coordinates": np.array([70, 43])},
            13: {"name": "Route 2", "coordinates": np.array([70, 151])},
            14: {"name": "Route 3", "coordinates": np.array([100, 179])},
            15: {"name": "Route 4", "coordinates": np.array([150, 197])},
            33: {"name": "Route 22", "coordinates": np.array([20, 71])},
            37: {"name": "Red house first", "coordinates": np.array([61, 9])},
            38: {"name": "Red house second", "coordinates": np.array([61, 0])},
            39: {"name": "Blues house", "coordinates": np.array([91, 9])},
            40: {"name": "Oaks lab", "coordinates": np.array([91, 1])},
            41: {"name": "Pokémon Center (Viridian City)", "coordinates": np.array([100, 54])},
            42: {"name": "Poké Mart (Viridian City)", "coordinates": np.array([100, 62])},
            43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
            44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
            47: {"name": "Gate (Viridian City/Pewter City) (Route 2)", "coordinates": np.array([91,143])},
            49: {"name": "Gate (Route 2)", "coordinates": np.array([91,115])},
            50: {"name": "Gate (Route 2/Viridian Forest) (Route 2)", "coordinates": np.array([91,115])},
            51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
            52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
            53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
            54: {"name": "Pokémon Gym (Pewter City)", "coordinates": np.array([49, 176])},
            55: {"name": "House with disobedient Nidoran♂ (Pewter City)", "coordinates": np.array([51, 184])},
            56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
            57: {"name": "House with two Trainers (Pewter City)", "coordinates": np.array([51, 184])},
            58: {"name": "Pokémon Center (Pewter City)", "coordinates": np.array([45, 161])},
            59: {"name": "Mt. Moon (Route 3 entrance)", "coordinates": np.array([153, 234])},
            60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
            61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
            68: {"name": "Pokémon Center (Route 3)", "coordinates": np.array([135, 197])},
            193: {"name": "Badges check gate (Route 22)", "coordinates": np.array([0, 87])}, # TODO this coord is guessed, needs to be updated
            230: {"name": "Badge Man House (Cerulean City)", "coordinates": np.array([290, 137])}
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return {"name": "Unknown", "coordinates": np.array([80, 0])} # TODO once all maps are added this case won't be needed

    def get_pokemon_name(self, pokemon_id):
        pokemon_dict = {
            0: 'None',
            177: 'Squirtle',
            1: 'Rhydon',
            2: 'Kangaskhan',
            3: 'Nidoran♂',
            4: 'Clefairy',
            5: 'Spearow',
            6: 'Voltorb',
            7: 'Nidoking',
            8: 'Slowbro',
            9: 'Ivysaur', 
            10: 'Exeggutor',
            11: 'Lickitung',
            12: 'Exeggcute',
            13: 'Grimer',
            14: 'Gengar',
            15: 'Nidoran♀',
            16: 'Nidoqueen',
            17: 'Cubone',
            18: 'Rhyhorn',
            19: 'Lapras',
            20: 'Arcanine',
            21: 'Mew',
            22: 'Gyarados',
            23: 'Shellder',
            24: 'Tentacool',
            25: 'Gastly',
            26: 'Scyther',
            27: 'Staryu',
            28: 'Blastoise',
            29: 'Pinsir',
            30: 'Tengela',
            33: 'Growlithe',
            34: 'Onix',
            35: 'Fearow',
            36: 'Pidgey',
            37: 'Slowpoke',
            38: 'Kadabra',
            39: 'Graveler',
            40: 'Chansey',
            41: 'Machoke',
            42: 'Mr. Mime',
            43: 'Hitmonlee',
            44: 'Hitmonchan',
            45: 'Arbok',
            46: 'Parasect',
            47: 'Psyduck',
            48: 'Drowsee',
            49: 'Golem',
            51: 'Magmar',
            53: 'Electabuzz',
            54: 'Magneton',
            55: 'Koffing',
            57: 'Mankey',
            58: 'Seel',
            59: 'Diglett',
            60: 'Tauros',
            64: 'Farfetchd',
            65: 'Venonat',
            66: 'Dragonite',
            70: 'Duduo',
            71: 'Poliwag',
            72: 'Jynx',
            73: 'Moltres',
            74: 'Articuno',
            75: 'Zapdos',
            76: 'Ditto',
            77: 'Meowth',
            78: 'Krabby',
            82: 'Vulpix',
            83: 'Ninetales',
            84: 'Pikachu',
            85: 'Raichu',
            88: 'Dratini',
            89: 'Dragonair',
            90: 'Kabuto',
            91: 'Kabutops',
            92: 'Horsea',
            93: 'Seadra',
            96: 'Sandshrew',
            97: 'Sandslash',
            98: 'Omanyte',
            99: 'Omastar',
            100: 'Jigglypuff',
            101: 'Wigglytuff',
            102: 'Eevee',
            103: 'Flareon',
            104: 'Jolteon',
            105: 'Vaporeon',
            106: 'Machop',
            107: 'Zubat',
            108: 'Ekans',
            109: 'Paras',
            110: 'Poliwhirl',
            111: 'Poliwrath',
            112: 'Weedle',
            113: 'Kakuna',
            114: 'Beedrill',
            116: 'Dodrio',
            117: 'Primeape',
            118: 'Dugtrio',
            119: 'Venomoth',
            120: 'Dewgong',
            123: 'Caterpie',
            124: 'Metapod',
            125: 'Butterfree',
            131: 'Mewtwo',
            132: 'Snorlax',
            133: 'Magicarp',
            136: 'Muk',
            138: 'Kingler',
            139: 'Cloyster',
            140: 'Electrode',
            142: 'Clefable',
            143: 'Weezing',
            144: 'Persian',
            145: 'Marowak',
            147: 'Haunter',
            148: 'Abra',
            149: 'Alakazam',
            150: 'Pidgeotto',
            151: 'Pidgeot',
            152: 'Starmie',
            153: 'Bulbasaur',
            154: 'Venusaur',
            155: 'Tentacruel',
            157: 'Goldeen',
            158: 'Seaking',
            163: 'Ponyta',
            164: 'Rapidash',
            165: 'Rattata',
            166: 'Raticate',
            167: 'Nidorino',
            168: 'Nidorina',
            169: 'Geodude',
            170: 'Porygon',
            171: 'Aerodactyl',
            173: 'Magnemite',
            176: 'Charmander',
            178: 'Charmeleon',
            179: 'Wartortle',
            180: 'Charizard',
            185: 'Oddish',
            186: 'Gloom',
            187: 'Vileplume',
            188: 'Bellsprout',
            189: 'Weepinbell',
            190: 'Victreebel'
        }
        return pokemon_dict[pokemon_id]

    def get_move_names_map(self, move_id):
        """Returns a dictionary mapping byte values to Pokémon move names."""
        move_names_map = {
            0: "No move",
            1: "None",
            2: "Pound",
            3: "Karate Chop",
            4: "Double Slap",
            5: "Comet Punch",
            6: "Mega Punch",
            7: "Pay Day",
            8: "Fire Punch",
            9: "Ice Punch",
            10: "Thunder Punch",
            11: "Scratch",
            12: "Vise Grip",
            13: "Guillotine",
            14: "Razor Wind",
            15: "Swords Dance",
            16: "Cut",
            17: "Gust",
            18: "Wing Attack",
            19: "Whirlwind",
            20: "Fly",
            21: "Bind",
            22: "Slam",
            23: "Vine Whip",
            24: "Stomp",
            25: "Double Kick",
            26: "Mega Kick",
            27: "Jump Kick",
            28: "Rolling Kick",
            29: "Sand Attack",
            30: "Headbutt",
            31: "Horn Attack",
            32: "Fury Attack",
            33: "Horn Drill",
            34: "Tackle",
            35: "Body Slam",
            36: "Wrap",
            37: "Take Down",
            38: "Thrash",
            39: "Double-Edge",
            40: "Tail Whip",
            41: "Poison Sting",
            42: "Twineedle",
            43: "Pin Missile",
            44: "Leer",
            45: "Bite",
            46: "Growl",
            47: "Roar",
            48: "Sing",
            49: "Supersonic",
            50: "Sonic Boom",
            51: "Disable",
            52: "Acid",
            53: "Ember",
            54: "Flamethrower",
            55: "Mist",
            56: "Water Gun",
            57: "Hydro Pump",
            58: "Surf",
            59: "Ice Beam",
            60: "Blizzard",
            61: "Psybeam",
            62: "Bubble Beam",
            63: "Aurora Beam",
            64: "Hyper Beam",
            65: "Peck",
            66: "Drill Peck",
            67: "Submission",
            68: "Low Kick",
            69: "Counter",
            70: "Seismic Toss",
            71: "Strength",
            72: "Absorb",
            73: "Mega Drain",
            74: "Leech Seed",
            75: "Growth",
            76: "Razor Leaf",
            77: "Solar Beam",
            78: "Poison Powder",
            79: "Stun Spore",
            80: "Sleep Powder",
            81: "Petal Dance",
            82: "String Shot",
            83: "Dragon Rage",
            84: "Fire Spin",
            85: "Thunder Shock",
            86: "Thunderbolt",
            87: "Thunder Wave",
            88: "Thunder",
            89: "Rock Throw",
            90: "Earthquake",
            91: "Fissure",
            92: "Dig",
            93: "Toxic",
            94: "Confusion",
            95: "Psychic",
            96: "Hypnosis",
            97: "Meditate",
            98: "Agility",
            99: "Quick Attack",
            100: "Rage",
            101: "Teleport",
            102: "Night Shade",
            103: "Mimic",
            104: "Screech",
            105: "Double Team",
            106: "Recover",
            107: "Harden",
            108: "Minimize",
            109: "Smokescreen",
            110: "Confuse Ray",
            111: "Withdraw",
            112: "Defense Curl",
            113: "Barrier",
            114: "Light Screen",
            115: "Haze",
            116: "Reflect",
            117: "Focus Energy",
            118: "Bide",
            119: "Metronome",
            120: "Mirror Move",
            121: "Self-Destruct",
            122: "Egg Bomb",
            123: "Lick",
            124: "Smog",
            125: "Sludge",
            126: "Bone Club",
            127: "Fire Blast",
            128: "Waterfall",
            129: "Clamp",
            130: "Swift",
            131: "Skull Bash",
            132: "Spike Cannon",
            133: "Constrict",
            134: "Amnesia",
            135: "Kinesis",
            136: "Soft-Boiled",
            137: "High Jump Kick",
            138: "Glare",
            139: "Dream Eater",
            140: "Poison Gas",
            141: "Barrage",
            142: "Leech Life",
            143: "Lovely Kiss",
            144: "Sky Attack",
            145: "Transform",
            146: "Bubble",
            147: "Dizzy Punch",
            148: "Spore",
            149: "Flash",
            150: "Psywave",
            151: "Splash",
            152: "Acid Armor",
            153: "Crabhammer",
            154: "Explosion",
            155: "Fury Swipes",
            156: "Bonemerang",
            157: "Rest",
            158: "Rock Slide",
            159: "Hyper Fang",
            160: "Sharpen",
            161: "Conversion",
            162: "Tri Attack",
            163: "Super Fang",
            164: "Slash",
            165: "Substitute",
            166: "Struggle"
        }
        return move_names_map[move_id]