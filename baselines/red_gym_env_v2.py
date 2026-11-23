import uuid
import json
import gc
import io
import time
import faulthandler
import os
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
from milestones import Archive, Ladder, MilestoneTracker
import cut_action

# ============================================================================
# THE SCORE SHEET: what the agents get points for. Edit these together!
# Every value is multiplied by reward_scale (2) in training.
# "once" = paid one time per run; "each" = paid for every one.
# ============================================================================
REWARDS = {
    # --- April 2026 values (the setup that beat Brock by ~20M steps) ---
    "event":            10,    # each story event flag set
    "badge":           100,    # each gym badge: deliberately outsized
    "new_tile":       0.05,    # each new tile (same as explore_weight 0.05)
    "new_map":           2,    # each new map (town, route, building). Was April's escalating 20/30/50/75,
                               # which made maps ~70% of all points at Vermilion (Nov 23)
    # revisit: April's get_revisit_penalty + get_immediate_revisit_penalty
    "level":             1,    # team levels (see get_levels_reward)
    "caught":            2,    # each Pokemon caught
    "heal":             10,    # healing at a Pokemon Center
    "death":         -0.01,    # each time the team faints
    "menu":           0.01,    # opening each start-menu screen once
    # --- added since April ---
    "milestone":         0,    # per milestone reached (off: April's events/badges already cover it)
    "cut_got_hm01":     20,    # HM01 in the bag (once)
    "cut_first_tree":   50,    # first tree cut (once)
    "cut_new_tree":     10,    # each different tree cut
}
from tilesets import Tilesets
from field_moves import FieldMoves
from generate_stream_overlay import update_agent_overlay, prepare_overlay_data

# Near the top with other imports
from enum import Enum, auto

# You can place this Enum definition near the top of the RedGymEnv class definition
class CutState(Enum):
    IDLE = auto()                   # Not in the cut sequence
    FACE_BUSH = auto()              # Turn to face the bush
    OPEN_START_MENU = auto()        # Press START
    SELECT_POKEMON_MENU = auto()    # Navigate to 'POKéMON'
    OPEN_POKEMON_LIST = auto()      # Press 'A' to see party
    SELECT_POKEMON_WITH_CUT = auto()# Navigate to the correct Pokémon
    OPEN_POKEMON_SUBMENU = auto()   # Press 'A' on the Pokémon
    SELECT_CUT_MOVE = auto()        # Navigate to 'CUT' in the submenu
    EXECUTE_CUT = auto()            # Press 'A' to use CUT
    WAIT_FOR_ANIMATION = auto()     # Wait for the cut effect/text
    CLEANUP = auto()                # Finalize and return to IDLE
    FAIL_AND_BAIL = auto()          # Recover from an unexpected state

event_flags_start = 0xD747
event_flags_end = 0xD7F6 # This must be the MAX value for the obs space definition
event_flags_reward_end = 0xD887  # end of wEventFlags (exclusive): the REWARD counts every event flag;
                                 # event_flags_end above stays as the observation size
museum_ticket = (0xD754, 0)

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.log_agent_id = config.get("log_agent_id", 0)
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.print_freq = config.get("print_freq", 100)
        self.update_freq = config.get("update_freq", 100)
        self.save_debug_jpegs = config.get("save_debug_jpegs", True)
        self.render_resize_mode = config.get("render_resize_mode", "mean")
        self.frame_stacks = config.get("frame_stacks", 3)
        self.reset_state_mode = config.get("reset_state_mode", "disk")
        self.release_action_tick = config.get("release_action_tick", -1)
        self.trace_step_phases = config.get("trace_step_phases", False)
        self.trace_first_n_steps = config.get("trace_first_n_steps", 120)
        self.trace_from_reset = config.get("trace_from_reset", 2)
        self.trace_step_file = config.get("trace_step_file", False)
        self.trace_tick_every = config.get("trace_tick_every", 8)
        self.trace_flush_every = config.get("trace_flush_every", 1)
        self.hang_dump_enabled = config.get("hang_dump_enabled", False)
        self.hang_dump_timeout_s = config.get("hang_dump_timeout_s", 60)
        self.reset_stagger_s = config.get("reset_stagger_s", 0.0)
        self.gc_every_n_resets = config.get("gc_every_n_resets", 0)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)
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

        # Milestone saves + start-state ladder (see milestones.py)
        self.milestones = None
        self.start_key = "base"
        self.auto_teach_hms = config.get("auto_teach_hms", True)
        self._render_all_next = True
        self.trees_cut = set()
        self.cut_results = {}
        self.revisit_cost_total = 0.0
        self.last_tile_key = None
        if config.get("archive_dir"):
            archive = Archive(config["archive_dir"], Path(self.init_state).name)
            ladder = Ladder(archive, config["rank"], config.get("n_envs", 1),
                            n_base=config.get("ladder_n_base", 4),
                            window=config.get("ladder_window", 5))
            self.milestones = MilestoneTracker(self, archive, ladder)
        else:
            self.milestones = MilestoneTracker(self)  # milestone rewards without the save archive

        self.trace_path = self.s_path / Path(f"trace_agent_{config['rank']}.log")
        self._trace_file = None
        self._trace_write_count = 0
        if self.trace_step_file:
            self._trace_file = open(self.trace_path, "a", encoding="utf-8")
        self.hang_dump_path = self.s_path / Path(f"hang_agent_{config['rank']}.log")
        self._hang_dump_file = None
        self._step_trace_t0_ns = 0
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        self.reward_cache = {}
        self.cached_event_bits = np.zeros((event_flags_end - event_flags_start) * 8, dtype=np.int8)
        self.can_cut_badge = False
        self.can_cut_move = False
        self.menu_check_complete = False
        self.cut_state = CutState.IDLE
        self.cut_state_timer = 0
        self.cut_pokemon_party_idx = 0
        self.event_flags_end_short = 0xD761 # The short limit for early-game observation
        self.has_used_cut = 0
        self.maps_visited = set()
        self.last_map_checked = None  # Cache to avoid unnecessary reads

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
        self.infinite_money = False
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

        self.cut_state = CutState.IDLE
        self.cut_state_timer = 0
        self.cut_pokemon_party_idx = 0
        self.cut_direction_to_face = None # 

        # --- CRITICAL HARDCODED ADDRESSES (VERIFY THESE FOR POKEMON RED 1.0/1.1 ENGLISH) ---
        self.ADDR_WRAM_PLAYER_DIRECTION = 0xC109     # Current facing direction of the player
        self.ADDR_WRAM_OBTAINED_BADGES = 0xD356      # Bitfield for obtained badges
        self.CASCADE_BADGE_BIT_INDEX = 1             # Bit 1 (0-indexed) for Cascade Badge

        self.ADDR_WRAM_MENU_JOYPAD_POLL = 0xD057     # If non-zero, a menu is active that polls joypad
                                                    # Could also be wJoyIgnore (0xFFF0) or wMenuWatchMovingArrow (0xC2A9)
                                                    # but 0xD057 is common for active d-pad menus.

        self.ADDR_WRAM_MAP_ID = 0xD35E               # Current Map ID (e.g., Pallet Town, Route 1, Erika's Gym)
                                                    # Sometimes this is wMapGroup, and wMapNumber (0xD35F) is also needed.
                                                    # For simplicity, using one if your game version maps directly.
        self.ERIKA_GYM_MAP_ID_EXPECTED = 134         # Expected ID for Erika's Gym
        self.OVERWORLD_MAX_MAP_ID_EXPECTED = 88      # Maps 0-88 are generally overworld

        self.ADDR_WRAM_TILE_MAP_START = 0xD4A0       # Start of the 20x18 visible tile map in WRAM
        self.TILE_MAP_WIDTH = 20
        self.TILE_MAP_HEIGHT = 18
        self.BUSH_TILE_IDS_OVERWORLD = (46, 61)           # VERIFY this. (Previously discussed 0x2D / 45)
        self.BUSH_TILE_ID_GYM = 0x50                 # VERIFY this for Erika's Gym tree

        # Menu Cursor Addresses (These are the trickiest and can vary by menu context)
        # For the main START menu:
        self.ADDR_WRAM_START_MENU_CURSOR_POS = 0xC68B # Common for wCurrentMenuItem
        # For the POKéMON party list screen:
        self.ADDR_WRAM_PARTY_LIST_CURSOR_POS = 0xC68B # Often reused, but VERIFY
        # For the POKéMON's individual field move list (after selecting a Pokémon):
        self.ADDR_WRAM_FIELD_MOVE_LIST_CURSOR_POS = 0xC68B # Often reused, but VERIFY

        # Field Moves Data
        # This is where it gets complex. `wFieldMoves` isn't a simple array of HM IDs.
        # It's usually a list of POINTERS or specific data related to the *current Pokémon's* usable field moves.
        # We might need to find a more direct way if this is too complex without symbols.
        # A common approach is that after selecting a Pokemon, the game populates a temporary
        # RAM area (e.g., starting at 0xD0A2 as you had, or another like wListMenuID)
        # with the IDs of field moves that Pokemon can use.
        # The `wCurrentMenuItem` (or equivalent for that specific menu) would then index into THIS temporary list.
        self.ADDR_WRAM_CURRENT_POKEMON_FIELD_MOVES_LIST_START = 0xD0A2 # Example, VERIFY (wListMenuID or similar)
        self.FIELD_MOVE_ID_FOR_CUT = 0x01 # The ID representing CUT in that temporary list (often 1 if CUT is the first HM)
                                        # This is NOT the global move ID 0x0F.
                                        # It's how the menu system identifies "CUT" among field moves.

        # Global Move ID for CUT (used by _find_pokemon_with_cut_index)
        self.GLOBAL_MOVE_ID_CUT = 0x0F

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
        # Action 7 = USE CUT (runs the Cut menu sequence; the game decides if it works)
        self.USE_CUT = len(self.valid_actions)
        self.n_actions = len(self.valid_actions) + 1
        self.action_space = spaces.Discrete(self.n_actions)
        
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
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions) + 1] * self.frame_stacks),
                "recent_path": spaces.Box(low=0, high=1e8, shape=(10,), dtype=np.int32),  # Last 10 area hashes
            }
        )

        head = "null" if config["headless"] else "SDL2"

        #log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            symbols=config["symbol_file"],
            log_level='CRITICAL',
            sound_emulated = False,
            no_input=False,
            window=head
        )

        self.screen = self.pyboy.screen

        # Two start-menu options can stall PyBoy for minutes per frame: saving the
        # game and the Pokedex screen. (This was the April overnight "hang", and the
        # stall at episode end.) Training needs neither, so we patch the game in
        # memory: SAVE does nothing, and POKEDEX goes straight back to the menu.
        if config.get("block_stall_menus", True):
            try:
                bank, addr = self.pyboy.symbol_lookup("SaveSAV")
                self.pyboy.memory[bank, addr] = 0xC9                      # ret
                bank, addr = self.pyboy.symbol_lookup("StartMenu_Pokedex")
                _, back = self.pyboy.symbol_lookup("RedisplayStartMenu")
                self.pyboy.memory[bank, addr] = 0xC3                      # jp RedisplayStartMenu
                self.pyboy.memory[bank, addr + 1] = back & 0xFF
                self.pyboy.memory[bank, addr + 2] = back >> 8
                # The same Pokedex entry screen also pops up after catching a new
                # Pokemon; skip it there too.
                bank, addr = self.pyboy.symbol_lookup("ShowPokedexData")
                self.pyboy.memory[bank, addr] = 0xC9                      # ret
            except Exception as e:
                print(f"[env] could not patch the start menu: {e}")

        # Stall guard: if one step or reset ever takes longer than this (60 s; normal is under 1 s), exit this worker so
        # training stops with an error and the watchdog restarts from the last
        # checkpoint within a minute, instead of every agent waiting on one stuck
        # game all night. (faulthandler's timer runs outside Python, so it fires
        # even while PyBoy is stuck inside a frame.)
        self.step_timeout_s = config.get("step_timeout_s", 60)

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)
        else:
            # PyBoy 2.x defaults to real-time (1x) speed with a sleep-based frame
            # limiter, even with no window. Training never wants that: turn it off
            # explicitly so emulator speed never depends on timer/sleep behaviour.
            self.pyboy.set_emulation_speed(0)
        # Keep a cached copy of the initial savestate for faster, consistent resets.
        with open(self.init_state, "rb") as f:
            self._init_state_bytes = f.read()

        # Optional: dump stack traces if a worker thread stops making progress
        # (useful for pinpointing hard hangs in emulator tick paths).
        if self.hang_dump_enabled:
            self._hang_dump_file = open(self.hang_dump_path, "a", encoding="utf-8")
            faulthandler.enable(file=self._hang_dump_file, all_threads=True)
            faulthandler.dump_traceback_later(
                self.hang_dump_timeout_s,
                repeat=True,
                file=self._hang_dump_file,
            )

    def reset(self, seed=None, options={}):
        self._arm_stall_guard(at_reset=True)
        self.seed = seed
        next_reset_count = self.reset_count + 1
        if self.gc_every_n_resets > 0 and next_reset_count % self.gc_every_n_resets == 0:
            gc.collect()
        if self.reset_stagger_s > 0:
            time.sleep(self.reset_stagger_s * self.agent_id)
        # Event rewards count flags beyond the *base* start state. Measure that once
        # from the base state itself, even if this worker's first run starts from a
        # milestone save (otherwise base-state runs would earn no event reward until
        # they passed the milestone's event count).
        if not hasattr(self, 'base_event_flags'):
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
            self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_reward_end)
            ])

        start_path = None
        if self.milestones is not None:
            try:
                start_path = self.milestones.choose_start()
            except Exception as e:  # a bad archive must never stop training
                print(f"[milestone] agent {self.agent_id}: could not pick a start ({e}); using base state")
                start_path = None
        loaded = False
        if start_path:
            try:
                with open(start_path, "rb") as f:
                    self.pyboy.load_state(f)
                loaded = True
            except Exception as e:
                print(f"[milestone] agent {self.agent_id}: could not load {start_path} ({e}); using base state")
        if not loaded:
            if self.milestones is not None:
                self.milestones.start_key = "base"
                self.milestones.offset = 0
            if self.reset_state_mode == "memory":
                self.pyboy.load_state(io.BytesIO(self._init_state_bytes))
            else:
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
        self.start_key = self.milestones.start_key if self.milestones is not None else "base"
        self._render_all_next = True

        self.init_map_mem()

        self.agent_stats = []
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.explore_map_dim = GLOBAL_MAP_SHAPE

        if hasattr(self, 'recent_screens'):
            self.recent_screens.fill(0)  # Clear existing array
        else:
            self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        
        if hasattr(self, 'explore_map'):
            self.explore_map.fill(0)  # Clear existing array
        else:
            self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

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
        self.reward_cache = {}
        self.can_cut_badge = False
        self.can_cut_move = False
        self.menu_check_complete = False
        self.cut_state = CutState.IDLE
        self.cut_state_timer = 0
        self.cut_pokemon_party_idx = 0
        self.cut_direction_to_face = None
        self.has_used_cut = 0
        self.trees_cut = set()
        self.cut_results = {}
        self.revisit_cost_total = 0.0
        self.last_tile_key = None
        self.current_event_flags_set = {}
        self.maps_visited = set()
        self.last_map_checked = None  # Cache to avoid unnecessary reads

        self.pokemon_levels = {} #Store pokemon id and max level, updating each iteration
        self.party_info = {} #Store pokemon id, moves, levels, and pp
        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        if self.milestones is not None:
            self.milestones.after_load()
        self._refresh_event_bits_cache()
        # Fill the reward cache from the loaded game *before* taking the starting
        # total. Otherwise the first cache refresh (~100 steps in) would pay out
        # every level/move/event already in the save as if the agent just earned it,
        # a large free reward on every milestone start.
        self._update_reward_cache()
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count = next_reset_count
        # Calculate initial health and pass it to _get_obs
        initial_health = self.read_hp_fraction()
    
        obs = self._get_obs(initial_health)
        if self.step_timeout_s:
            faulthandler.cancel_dump_traceback_later()
        return obs, {}

    def init_map_mem(self):
        self.seen_coords = {}
        self.inherited_tiles = 0

    def apply_lineage(self, coords, maps):
        """Starting from a milestone save: mark everything its line of agents already
        explored as seen (long ago, so no revisit cost) and its maps as visited, so
        walking back over old ground earns nothing. Called before the starting
        reward total is taken, so this gives no reward either."""
        for x, y, m in coords.tolist():
            self.seen_coords[(x, y, m)] = -10**9
            gy, gx = local_to_global(y, x, m)
            if 0 <= gy < self.explore_map.shape[0] and 0 <= gx < self.explore_map.shape[1]:
                self.explore_map[gy, gx] = 255
        self.maps_visited.update(int(m) for m in maps.tolist())
        self.inherited_tiles = len(self.seen_coords)

    def full_screen(self):
        """Full-size current screen for check-in reports. (A separate name from
        render() so it still works through StreamWrapper, whose render() drops
        arguments.)"""
        return self.render(reduce_res=False)

    def render(self, reduce_res=True):
        game_pixels_render = self.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            if self.render_resize_mode == "stride":
                # Faster than local mean and usually sufficient for RL state.
                game_pixels_render = game_pixels_render[::2, ::2, :]
            else:
                game_pixels_render = (
                    downscale_local_mean(game_pixels_render, (2,2,1))
                ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self, cur_health):
        
        screen = self.render()

        self.update_recent_screens(screen)
        recent_path = np.fromiter(self.recent_coords, dtype=np.int32, count=10)
        badge_bits = ((self.read_m(0xD356) >> np.arange(7, -1, -1)) & 1).astype(np.int8)

        observation = {
            "screens": self.recent_screens,
            "health": np.array([cur_health]), # USE THE PASSED-IN VALUE
            "level": self.reward_cache.get('level_encoded', np.zeros(self.enc_freqs)), ## May need to come back and change this to the same as the level reward
            "badges": badge_bits,
            "events": self.cached_event_bits,
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "recent_path": recent_path,
        }

        return observation

    def _arm_stall_guard(self, at_reset=False):
        if self.step_timeout_s:
            if getattr(self, "_stall_log", None) is None:
                # Where a stuck worker was when the guard fired, for diagnosis
                stall_dir = self.s_path.parent / "stalls"
                stall_dir.mkdir(parents=True, exist_ok=True)
                self._stall_log = open(stall_dir / f"agent_{self.agent_id:02d}.txt", "a")
            if at_reset:  # one line per episode, so the file stays small
                self._stall_log.write(f"--- reset {self.reset_count + 1} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._stall_log.flush()
            faulthandler.dump_traceback_later(self.step_timeout_s, exit=True, file=self._stall_log)

    def step(self, action):
        # The stall guard only times this step's own work; it's switched off again
        # before returning, so time spent waiting for the trainer never counts.
        self._arm_stall_guard()
        self._step_trace_t0_ns = time.perf_counter_ns()
        self._trace_step_phase("step_begin")
        if self.save_video and self.step_count == 0:
            self.start_video()

        # Core state machine logic:
        # If the cut sequence is active, it takes over the emulator.
        # Otherwise, the agent's action is performed.
        if self.cut_state != CutState.IDLE:
            self._trace_step_phase("cut_state_tick_begin")
            self._tick_cut_state_machine()
            self._trace_step_phase("cut_state_tick_end")
        elif action == self.USE_CUT:
            result = cut_action.use_cut(self)
            self.cut_results[result] = self.cut_results.get(result, 0) + 1
        else:
            self._trace_step_phase("action_tick_begin")
            self.run_action_on_emulator(action)
            self._trace_step_phase("action_tick_end")

        self._trace_step_phase("post_emulator")
        self.update_recent_actions(action)
        self.update_explore_map()

        cur_health = self.read_hp_fraction()
        self.update_heal_reward(cur_health)

        self.party_size = self.read_m(0xD163)
        self.check_in_menus()
        new_reward = self.update_reward()
        self.update_seen_coords()
        if self.milestones is not None:
            try:
                self.milestones.step()
            except Exception as e:  # never let milestone bookkeeping stop training
                if self.step_count % 1000 == 0:
                    print(f"[milestone] agent {self.agent_id}: {e}")
        self._trace_step_phase("post_reward")

        if (self.disable_wild_encounters and self.read_m(0xD35E) in [59, 60, 61]):
            self.set_m(0xD887, 0X00)

        if self.update_freq > 0 and self.step_count > 0 and self.step_count % self.update_freq == 0:
            self._update_reward_cache()
            self._refresh_event_bits_cache()
            self.append_agent_stats(action, self.reward_cache)
            self.update_10_pokeballs()
            if self.headless == False:
                prepare_overlay_data(self, location=self.get_current_location(), badges=self.get_badges_status())

        self.last_health = cur_health
        self.update_map_progress()
        step_limit_reached = self.check_if_done()
        self._trace_step_phase("pre_obs")
        obs = self._get_obs(cur_health)
        self._trace_step_phase("post_obs")
        self.save_and_print_info(step_limit_reached, obs)

        if self.update_freq > 0 and self.step_count > 0 and self.step_count % self.update_freq == 0:
            self.current_event_flags_set.clear()
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            # This print can be noisy, so it's commented out
                            # print(f"could not find key: {key}")
                            pass

        self.step_count += 1
        self._trace_step_phase("step_end")
        if self.step_timeout_s:
            faulthandler.cancel_dump_traceback_later()
        return obs, new_reward, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
        """Press the button, run act_freq frames, release.

        Performance: only the last frame of each step is rendered to the screen
        buffer, because that's the only one the agent sees (the observation is
        taken after the step). Rendering every frame was more than half of the
        emulator's time. Emulation itself is identical either way.
        """
        self.pyboy.send_input(self.valid_actions[action])
        # Right after loading a save, PyBoy's tile cache is only rebuilt while
        # rendering, so render every frame of the first step (once per run).
        # A visible window (headless=False, e.g. run_pretrained_interactive.py) needs
        # every frame drawn so playback is smooth; so does a full-frame video.
        render_all = (not self.headless) or (self.save_video and not self.fast_video) or self._render_all_next
        self._render_all_next = False
        tracing = self.trace_step_phases or self.trace_step_file
        if render_all or tracing or self.release_action_tick >= 0:
            # Slow path: frame-by-frame (video, tracing, or a mid-step release)
            for i in range(self.act_freq):
                if tracing and i % self.trace_tick_every == 0:
                    self._trace_step_phase(f"emulator_tick_begin_{i}")
                if self.release_action_tick >= 0 and i == self.release_action_tick:
                    self.pyboy.send_input(self.release_actions[action])
                if self.save_video and not self.fast_video:
                    self.add_video_frame()
                self.pyboy.tick(1, render_all or i == self.act_freq - 1)
                if tracing and i % self.trace_tick_every == 0:
                    self._trace_step_phase(f"emulator_tick_end_{i}")
        else:
            # Fast path: one call for all frames but the last, then render the last
            if self.act_freq > 1:
                self.pyboy.tick(self.act_freq - 1, False)
            self.pyboy.tick(1, True)
        if self.release_action_tick < 0:
            self.pyboy.send_input(self.release_actions[action])
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def _trace_step_phase(self, phase):
        if self.reset_count < self.trace_from_reset:
            return
        if self.step_count > self.trace_first_n_steps:
            return
        now = time.time()
        dt_ms = 0.0
        if self._step_trace_t0_ns:
            dt_ms = (time.perf_counter_ns() - self._step_trace_t0_ns) / 1e6
        line = (
            f"[TRACE] ts={now:.3f} pid={os.getpid()} agent={self.agent_id} "
            f"reset={self.reset_count} step={self.step_count} dt_ms={dt_ms:.3f} phase={phase}"
        )
        if self.trace_step_phases and self.agent_id == self.log_agent_id:
            print(line, flush=True)
        if self.trace_step_file and self._trace_file is not None:
            self._trace_file.write(line + "\n")
            self._trace_write_count += 1
            if self.trace_flush_every <= 1 or self._trace_write_count % self.trace_flush_every == 0:
                self._trace_file.flush()

    def append_agent_stats(self, action, cached_rewards):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        # MODIFIED: Instead of appending, we now just store the single, most recent dictionary.
        # This prevents the list from growing and causing a memory leak.
        self.agent_stats = [{
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
            #"pokemon_seen": cached_rewards.get('pokedex_seen', 0),
            "pokemon_caught": cached_rewards.get('pokedex_caught', 0),
            "start_menu": self.start_menu_count,
            "stats_menu": self.start_stats_count,
            "pokemon_menu": self.start_pokemenu_count,
            "item_menu": self.start_itemmenu_count,
            "milestone": (self.milestones.highest + 1) if self.milestones is not None else 0,  # 1-based; 0 = none
            "trees_cut": len(self.trees_cut),
            "new_tiles": len(self.seen_coords) - getattr(self, "inherited_tiles", 0),
            "cut_tries": sum(self.cut_results.values()),
            "knows_cut": cut_action.party_slot_with_cut(self) is not None,
            "milestones_this_run": len(self.milestones.reached_this_run) if self.milestones is not None else 0,
        }]

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
        #coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        # Change self.seen_coords to use tuple keys instead of strings
        coord_key = (x_pos, y_pos, map_n)  # Faster than string formatting
        self.seen_coords[coord_key] = self.step_count

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
    
    def get_accumulating_map_reward(self):
        """
        Returns accumulating reward that grows with each new map discovered.
        Optimized: uses mathematical formula instead of loop.
        Efficient: only reads memory when map actually changes.
        """
        current_map = self.read_m(0xD35E)
        
        # Only process if map actually changed (avoid unnecessary work)
        if current_map != self.last_map_checked:
            self.last_map_checked = current_map
            
            # Add new map to visited set if not seen before
            if current_map not in self.maps_visited:
                self.maps_visited.add(current_map)
        
        # Flat reward per new map (REWARDS["new_map"]); the old escalating curve
        # (20/30/50/75) is kept below for reference.
        maps_discovered = len(self.maps_visited)
        total_reward = REWARDS["new_map"] * maps_discovered
        # if maps_discovered <= 5: total_reward = 20 * maps_discovered
        # elif maps_discovered <= 10: total_reward = 100 + (30 * (maps_discovered - 5))
        # elif maps_discovered <= 15: total_reward = 250 + (50 * (maps_discovered - 10))
        # else: total_reward = 500 + (75 * (maps_discovered - 15))

        return total_reward

    def _read_visible_tile_map(self):
        """
        Returns the currently visible 20x18 tile map from WRAM.

        This is used by field-move shaping rewards so we can tell when the
        agent is positioned next to a cuttable bush. If the tile map cannot be
        read for any reason, return None and let the caller fail gracefully.
        """
        try:
            _, wTileMap_addr = self.pyboy.symbol_lookup("wTileMap")
            tile_map_bytes = self.pyboy.memory[
                wTileMap_addr : wTileMap_addr + self.TILE_MAP_WIDTH * self.TILE_MAP_HEIGHT
            ]
            return np.array(tile_map_bytes, dtype=np.uint8).reshape(
                (self.TILE_MAP_HEIGHT, self.TILE_MAP_WIDTH)
            )
        except Exception:
            return None

    def is_adjacent_to_cuttable_bush(self):
        """
        Returns True when the player is standing next to a bush/tree that can
        be cleared with CUT in the current map context.

        This is part of the CUT reward loop:
        1. Get HM01
        2. Teach CUT to a party member
        3. Earn Cascade Badge so CUT is legal in the overworld
        4. Stand next to a cuttable bush
        5. Successfully clear the bush
        """
        if self.in_battle():
            return False

        tile_map = self._read_visible_tile_map()
        if tile_map is None:
            return False

        player_screen_y, player_screen_x = 9, 9
        map_id = self.read_m(self.ADDR_WRAM_MAP_ID)
        is_erika_gym = map_id == self.ERIKA_GYM_MAP_ID_EXPECTED
        expected_bush_ids = (self.BUSH_TILE_ID_GYM,) if is_erika_gym else self.BUSH_TILE_IDS_OVERWORLD

        adj_checks = [
            (player_screen_y - 1, player_screen_x),
            (player_screen_y + 1, player_screen_x),
            (player_screen_y, player_screen_x - 1),
            (player_screen_y, player_screen_x + 1),
        ]

        for check_y, check_x in adj_checks:
            if 0 <= check_y < self.TILE_MAP_HEIGHT and 0 <= check_x < self.TILE_MAP_WIDTH:
                if tile_map[check_y, check_x] in expected_bush_ids:
                    return True

        return False

    def get_cut_progress_reward(self):
        """
        One-time rewards for the Cut chain (values in REWARDS at the top of this file).

        These used to be "while true" rewards (e.g. +20 while standing next to a
        tree). Because the agent is paid the *change* in score each step, that meant
        +20 for walking up to a tree and -20 for walking away, which taught it to
        park next to trees. Everything here only ever goes up.
        """
        R = REWARDS
        reward = 0.0
        if cut_action.has_hm01(self):
            reward += R["cut_got_hm01"]
        if self.trees_cut:
            reward += R["cut_first_tree"]
        reward += R["cut_new_tree"] * len(self.trees_cut)
        return reward

    def get_badges_status(self):
        badge_byte = self.read_m(0xD356)  # Single read
        return [int(bit) for bit in f"{badge_byte:08b}"]

    def get_revisit_penalty(self):
        """
        Applies a small penalty when the agent re-enters an area too soon.
        - No penalty for first-time visits.
        - Penalty is strongest on immediate revisits and fades over time.
        - No penalty during battles.
        """
        if self.in_battle():
            return 0.0  # No penalty if the agent is in a battle
        
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_key = (x_pos, y_pos, map_n)  # Faster than string formatting

        if map_n == 88 and self.read_bit(0xD7F2, 7):
            return 0.0 # No penalty in Bill's Lab until we get the SS Anne Ticket
        if map_n == 101 and self.read_bit(0xD803, 2):
            return 0.0 # No penalty in Captain's Quarters until we get the HM01
        if map_n == 92 and self.read_bit(0xD773, 1):
            return 0.0 # No penalty in Vermillion Gym until locks are opened

        # Ensure we are not incorrectly penalizing first-time visits
        if coord_key not in self.seen_coords:
            return 0.0  # No penalty for a truly new visit

        # Calculate time since last visit
        last_seen_step = self.seen_coords[coord_key]
        time_since_last_visit = self.step_count - last_seen_step

        if time_since_last_visit >= 500:
            return 0.0  # No penalty if enough time has passed

        # Strongest penalty on quick revisits, then linearly taper to zero.
        penalty = -5.0 * (1 - (time_since_last_visit / 500))
        return max(penalty, -5.0)
    
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
        # Change self.seen_coords to use tuple keys instead of strings
        coord_key = (x_pos, y_pos, map_n)  # Faster than string formatting

        # Ensure we are not incorrectly penalizing first-time visits
        if coord_key not in self.seen_coords:
            return 0.0  # No penalty for a truly new visit

        # Calculate time since last visit
        last_seen_step = self.seen_coords[coord_key]
        time_since_last_visit = self.step_count - last_seen_step

        if time_since_last_visit >= 100:
            return 0.0  # No penalty if enough time has passed
        if map_n == 88 and self.read_bit(0xD7F2, 7):
            return 0.0 # No penalty in Bill's Lab until we get the SS Anne Ticket
        if map_n == 101 and self.read_bit(0xD803, 2):
            return 0.0 # No penalty in Captain's Quarters until we get the HM01
        if map_n == 92 and self.read_bit(0xD773, 1):
            return 0.0 # No penalty in Vermillion Gym until locks are opened

        return -5.0

    def in_battle(self):
        # Check most common case first, cache result briefly
        if hasattr(self, '_battle_cache') and self.step_count - self._battle_cache_step < 3:
            return self._battle_cache_result
        
        result = self.read_m(0xD057) == 0x0F or self.read_m(0xCFFB) == 0x01
        self._battle_cache_result = result
        self._battle_cache_step = self.step_count
        return result
   
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
        # 2x upscale (same result as einops.repeat, ~10x faster)
        return out.repeat(2, axis=0).repeat(2, axis=1)
    
    def update_recent_screens(self, cur_screen):
        # Newest screen first, then the older ones (same as np.roll + assign, cheaper)
        if self.recent_screens.shape[2] == 1:
            self.recent_screens = cur_screen[:, :, :1].copy()
        else:
            self.recent_screens = np.concatenate(
                (cur_screen[:, :, :1], self.recent_screens[:, :, :-1]), axis=2)

    def update_recent_actions(self, action):
        self.recent_actions = np.concatenate(
            (np.array([action], dtype=self.recent_actions.dtype), self.recent_actions[:-1]))

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        ## Renable if you want to save state on done
        # if done: 
        #    # Use self.s_path to save inside the session folder
        #    with open(self.s_path / f"cut_test_state_{self.agent_id}.state", "wb") as f:
        #        self.pyboy.save_state(f)
        #    print("DEBUG: Saved state to cut_test_state.state")
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        should_print_step = (
            self.print_rewards
            and self.agent_id == self.log_agent_id
            and self.print_freq > 0
            and self.step_count % self.print_freq == 0
        )
        if should_print_step:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(prog_string, flush=True)

        if self.save_debug_jpegs and self.update_freq > 0 and self.step_count > 0 and self.step_count % self.update_freq == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.print_rewards and self.agent_id == self.log_agent_id and done:
            print("episode finished", flush=True)
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
        return self.pyboy.memory[addr]
    
    def set_m(self, addr, val):
        self.pyboy.memory[addr] = val

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def _refresh_event_bits_cache(self):
        num_badges = self.get_badges()
        end_addr = self.event_flags_end_short if num_badges < 4 else event_flags_end
        bytes_array = np.array(
            [self.read_m(addr) for addr in range(event_flags_start, end_addr)],
            dtype=np.uint8,
        )
        bits = np.unpackbits(bytes_array, bitorder='little').astype(np.int8)
        self.cached_event_bits.fill(0)
        self.cached_event_bits[:len(bits)] = bits

    def read_event_bits(self):
        return self.cached_event_bits

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
        if self.read_m(0xD31E):
            self.set_m(0xD31F, 10)
    
    
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
            self.set_m(0xD887, 0X00)

    def setup_enable_wild_ecounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_deregister(bank, addr)
    
    def check_in_menus(self):
        if self.menu_check_complete:
            return
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

        total_menus_visited = self.start_menu_count + self.start_stats_count + self.start_pokemenu_count + self.start_itemmenu_count
        if total_menus_visited >= 4:
            self.menu_check_complete = True
    
    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]
    
    def get_lead_pokemon_info(self):
        """Label shown next to this agent on the live map, e.g.
        'Bulbasaur: 12 (+2) | 1 badge': lead Pokemon and level, how many others are
        in the team, and the badge count."""
        lead = f"{self.get_pokemon_name(self.read_m(0xD164))}: {self.read_m(0xD18C)}"
        others = max(min(self.read_m(0xD163), 6) - 1, 0)
        if others:
            lead += f" (+{others})"
        badges = self.get_badges()
        self.extra_string = f"{lead} | {badges} badge{'' if badges == 1 else 's'}"
        return self.extra_string

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_reward_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )
    
    def _update_reward_cache(self):
        # Lazily initialize base_event_flags once
        if not hasattr(self, 'base_event_flags'):
            self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_reward_end)
            ])

        # Update rewards that change infrequently
        #self.reward_cache['pokedex_seen'] = self.get_pokedex_seen()
        self.reward_cache['pokedex_caught'] = self.get_pokedex_caught()
        self.reward_cache['move_rewards'] = self.move_rewards()
        self.reward_cache['has_HMs'] = self.has_HMs()
        self.reward_cache['level'] = self.get_levels_reward()
        self.reward_cache['event'] = self.update_max_event_rew() # MOVED HERE

        # Also cache the raw level sum for the observation space
        level_sum = 0.02 * sum([self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]])
        self.reward_cache['level_encoded'] = self.fourier_encode(level_sum)

        # Bookkeeping: teach Cut once HM01 is in the bag (the agent still has to use it)
        if self.auto_teach_hms:
            cut_action.auto_teach_cut(self)

        # Update cutting capability flags
        self.can_cut_badge = self.read_bit(self.ADDR_WRAM_OBTAINED_BADGES, self.CASCADE_BADGE_BIT_INDEX)
        self.can_cut_move = self._find_pokemon_with_cut_index() is not None

    def get_game_state_reward(self, print_stats=True):
        # This function is now much faster, reading almost all values from the cache.
        
        # Use cached values for expensive calculations
        #pokedex_seen = self.reward_cache.get('pokedex_seen', 0)
        pokedex_caught = self.reward_cache.get('pokedex_caught', 0)
        move_rewards = self.reward_cache.get('move_rewards', 0)
        has_HMs = self.reward_cache.get('has_HMs', 0)
        level_reward = self.reward_cache.get('level', 0)
        event_reward = self.reward_cache.get('event', 0) # READ FROM CACHE

        R, k = REWARDS, self.reward_scale
        milestones_reached = len(self.milestones.reached_this_run) if self.milestones is not None else 0
        state_scores = {
            "event": k * event_reward * R["event"],
            "level": k * level_reward * R["level"],
            "heal": k * self.total_healing_rew * R["heal"],
            "dead": k * self.died_count * R["death"],
            "badge": k * self.get_badges() * R["badge"],
            "milestone": k * milestones_reached * R["milestone"],
            "exp": k * len(self.seen_coords) * R["new_tile"],
            "cap": k * pokedex_caught * R["caught"],
            "menu": (self.start_menu_count + self.start_stats_count + self.start_pokemenu_count + self.start_itemmenu_count) * R["menu"],
            # Revisit penalties removed (Nov 21): their +/-5 swings drowned out the new-tile
            # signal and taught agents to circle Pallet. get_revisit_penalty() and
            # get_immediate_revisit_penalty() are kept below in case we want them back.
            "maps": k * self.get_accumulating_map_reward(),
            "move": k * move_rewards,
            "HMs": k * has_HMs,
            "cut": k * self.get_cut_progress_reward(),
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

    def update_heal_reward(self, cur_health):
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
                            self.set_m(0xD887, 0X0A)

    def read_hp_fraction(self):
        # Only recalculate HP every few steps (HP changes are rare outside battle)
        if self.step_count % 5 == 0 or not hasattr(self, '_cached_hp'):
            hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
            max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
            max_hp_sum = max(max_hp_sum, 1)
            self._cached_hp = hp_sum / max_hp_sum
        return self._cached_hp

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
            if move_value in [19, 57, 70]:  # Fly, Surf, Strength (Cut has its own rewards now)
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

    def _find_pokemon_with_cut_index(self):
        """
        Finds the party index (0-5) of the first Pokemon that knows CUT.
        Returns the index if found, otherwise None.
        """
        # The global move ID for CUT is 15 (0x0F)
        for i in range(self.read_m(0xD163)): # Iterate through party size
            # Each Pokémon's data block is 44 bytes long in WRAM
            # Moves start at offset 8 within this block (wPartyMon1Moves, etc.)
            moves_start_addr = 0xD16B + (i * 44) + 8
            moves = [self.read_m(moves_start_addr + j) for j in range(4)]
            if self.GLOBAL_MOVE_ID_CUT in moves:
                return i
        return None

    def cut_if_next(self):
        """
        Optimized check for a cuttable bush. Uses cached capability flags to avoid
        expensive memory reads on every step.
        """
        # --- OPTIMIZATION: Fast exit using cached flags ---
        if not self.can_cut_badge or not self.can_cut_move or self.cut_state != CutState.IDLE:
            return

        # Check 1: Are we in a battle or a menu? (These are cheap checks)
        in_menu = self.read_m(self.ADDR_WRAM_MENU_JOYPAD_POLL) != 0
        if self.in_battle() or in_menu:
            return

        # Check 2: Does the Pokémon with CUT still exist? (Cheap re-check for robustness)
        pokemon_idx = self._find_pokemon_with_cut_index()
        if pokemon_idx is None:
            # This can happen if the pokemon was released/deposited.
            # The flag will be corrected in the next cache update.
            return

        # --- If all fast checks pass, proceed to the expensive tile map read ---
        tile_map = self._read_visible_tile_map()
        if tile_map is None:
            return

        player_screen_y, player_screen_x = 9, 9
        map_id = self.read_m(self.ADDR_WRAM_MAP_ID)
        is_erika_gym = (map_id == self.ERIKA_GYM_MAP_ID_EXPECTED)
        expected_bush_ids = (self.BUSH_TILE_ID_GYM,) if is_erika_gym else self.BUSH_TILE_IDS_OVERWORLD

        adj_checks = [
            {'direction': "Up",    'y': player_screen_y - 1, 'x': player_screen_x,     'button': WindowEvent.PRESS_ARROW_UP},
            {'direction': "Down",  'y': player_screen_y + 1, 'x': player_screen_x,     'button': WindowEvent.PRESS_ARROW_DOWN},
            {'direction': "Left",  'y': player_screen_y,     'x': player_screen_x - 1, 'button': WindowEvent.PRESS_ARROW_LEFT},
            {'direction': "Right", 'y': player_screen_y,     'x': player_screen_x + 1, 'button': WindowEvent.PRESS_ARROW_RIGHT},
        ]

        for check in adj_checks:
            check_y, check_x = check['y'], check['x']
            
            if not (0 <= check_y < self.TILE_MAP_HEIGHT and 0 <= check_x < self.TILE_MAP_WIDTH):
                continue
            
            tile_id = tile_map[check_y, check_x]

            if tile_id in expected_bush_ids:
                # All conditions met, start the state machine
                self.cut_pokemon_party_idx = pokemon_idx
                self.cut_direction_to_face = check['button']
                self.cut_state = CutState.FACE_BUSH
                return # Exit after finding the first bush
            
    def _tick_cut_state_machine(self):
        """
        Runs a single, self-contained step of the cut sequence state machine.
        It correctly looks up release actions from self.release_actions to avoid errors.
        """
        current_state_before_tick = self.cut_state

        button_to_press = None
        presses = 1

        if self.cut_state == CutState.FACE_BUSH:
            button_to_press = self.cut_direction_to_face
            self.cut_state = CutState.OPEN_START_MENU
        elif self.cut_state == CutState.OPEN_START_MENU:
            button_to_press = WindowEvent.PRESS_BUTTON_START
            self.cut_state = CutState.SELECT_POKEMON_MENU
        elif self.cut_state == CutState.SELECT_POKEMON_MENU:
            button_to_press = WindowEvent.PRESS_ARROW_DOWN
            self.cut_state = CutState.OPEN_POKEMON_LIST
        elif self.cut_state == CutState.OPEN_POKEMON_LIST:
            button_to_press = WindowEvent.PRESS_BUTTON_A
            self.cut_state = CutState.SELECT_POKEMON_WITH_CUT
        elif self.cut_state == CutState.SELECT_POKEMON_WITH_CUT:
            if self.cut_pokemon_party_idx > 0:
                button_to_press = WindowEvent.PRESS_ARROW_DOWN
                presses = self.cut_pokemon_party_idx
            self.cut_state = CutState.OPEN_POKEMON_SUBMENU
        elif self.cut_state == CutState.OPEN_POKEMON_SUBMENU:
            button_to_press = WindowEvent.PRESS_BUTTON_A
            self.cut_state = CutState.EXECUTE_CUT
        elif self.cut_state == CutState.EXECUTE_CUT:
            button_to_press = WindowEvent.PRESS_BUTTON_A
            self.cut_state = CutState.WAIT_FOR_ANIMATION
            self.cut_state_timer = 20
        elif self.cut_state == CutState.WAIT_FOR_ANIMATION:
            self.cut_state_timer -= 1
            if self.cut_state_timer <= 0:
                button_to_press = WindowEvent.PRESS_BUTTON_A
                self.cut_state = CutState.CLEANUP
        elif self.cut_state == CutState.CLEANUP:
            self.has_used_cut += 1
            print(f"INFO [CUT]: Sequence complete. Total cuts this session: {self.has_used_cut}")
            self.cut_state = CutState.IDLE
            self.cut_state_timer = 0
            self.cut_pokemon_party_idx = 0
            self.cut_direction_to_face = None

        # --- Action Execution ---
        if button_to_press is not None:
            # Find the index of the press action to get the corresponding release action
            action_index = self.valid_actions.index(button_to_press)
            release_action = self.release_actions[action_index]

            for _ in range(presses):
                self.pyboy.send_input(button_to_press)
                for i in range(self.act_freq):
                    if i == self.act_freq // 2:
                        self.pyboy.send_input(release_action)
                    if self.save_video and not self.fast_video:
                        self.add_video_frame()
                    self.pyboy.tick()
                if self.save_video and self.fast_video:
                    self.add_video_frame()
        else:
            # If no button is pressed (e.g., waiting), just tick the emulator
            for _ in range(self.act_freq):
                if self.save_video and not self.fast_video:
                    self.add_video_frame()
                self.pyboy.tick()
            if self.save_video and self.fast_video:
                self.add_video_frame()

        # if current_state_before_tick != self.cut_state:
        #     print(f"DEBUG [CUT_FSM_TICK]: State transition: {current_state_before_tick.name} -> {self.cut_state.name}")
    
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
        # Cache expensive computation if value hasn't changed much
        rounded_val = round(val, 2)  # Reduce precision for caching
        if not hasattr(self, '_fourier_cache'):
            self._fourier_cache = {}
        
        if rounded_val not in self._fourier_cache:
            self._fourier_cache[rounded_val] = np.sin(val * 2 ** np.arange(self.enc_freqs))
            # Limit cache size
            if len(self._fourier_cache) > 100:
                self._fourier_cache.clear()
        
        return self._fourier_cache[rounded_val]
    
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

    def close(self):
        if self._trace_file is not None:
            try:
                self._trace_file.flush()
                self._trace_file.close()
            except Exception:
                pass
            self._trace_file = None
        if self.hang_dump_enabled:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass
            if self._hang_dump_file is not None:
                try:
                    self._hang_dump_file.flush()
                    self._hang_dump_file.close()
                except Exception:
                    pass
                self._hang_dump_file = None
        try:
            self.pyboy.stop()
        except Exception:
            pass
