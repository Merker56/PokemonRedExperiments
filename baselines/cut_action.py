"""USE CUT as an action the agent chooses, plus auto-teaching HM01.

The split we agreed on:
  * The game handles bookkeeping: teaching HM01 once it's in the bag, and the
    menu button presses for Cut (Start -> Pokemon -> pick the one with Cut -> CUT).
  * The agent decides everything else: getting HM01, finding a tree, facing it,
    and choosing USE CUT at the right moment.

USE CUT runs the whole menu sequence inside one step, in whatever direction the
player is facing. The game itself decides whether it works, exactly like a
human player: no tree in front -> "There isn't anything to CUT!", no Cascade
Badge -> "A new BADGE is required". Success is detected by the tree tile
disappearing from the screen.
"""
from pyboy.utils import WindowEvent

CUT_MOVE_ID = 0x0F
CUT_PP = 30
ITEM_HM01 = 0xC4
PARTY_COUNT = 0xD163
PARTY_SPECIES = 0xD164          # wPartySpecies, one byte per slot
PARTY_MON_1 = 0xD16B            # wPartyMon1, each party mon is 44 bytes
PARTY_MON_SIZE = 44
MOVES_OFFSET = 0x08
PP_OFFSET = 0x1D
IS_IN_BATTLE = 0xD057
PLAYER_FACING = 0xC109
GOT_POKEDEX = (0xD74B, 5)

# Species (internal ids) that can learn Cut, early-game ones. Starters first.
CUT_COMPATIBLE = {
    0x99, 0x09, 0x9A,          # Bulbasaur, Ivysaur, Venusaur
    0xB0, 0xB2, 0xB4,          # Charmander, Charmeleon, Charizard
    0xB9, 0xBA, 0xBB,          # Oddish, Gloom, Vileplume
    0xBC, 0xBD, 0xBE,          # Bellsprout, Weepinbell, Victreebel
    0x6D, 0x2E,                # Paras, Parasect
    0x60, 0x61,                # Sandshrew, Sandslash
    0x40,                      # Farfetch'd
}
# Moves we'd rather lose than keep when Cut needs a slot (status-only moves)
LOW_VALUE_MOVES = [45, 39, 43, 81, 28, 48, 47]   # Growl, Tail Whip, Leer, String Shot, Sand-Attack, Supersonic, Sing


def _moves_addr(slot):
    return PARTY_MON_1 + slot * PARTY_MON_SIZE + MOVES_OFFSET


def party_slot_with_cut(env):
    for slot in range(min(env.read_m(PARTY_COUNT), 6)):
        if CUT_MOVE_ID in [env.read_m(_moves_addr(slot) + j) for j in range(4)]:
            return slot
    return None


def has_hm01(env):
    n = min(env.read_m(0xD31D), 20)
    return any(env.read_m(0xD31E + 2 * i) == ITEM_HM01 for i in range(n)) or env.read_bit(0xD803, 0)


def auto_teach_cut(env):
    """If HM01 is in the bag and nobody knows Cut, teach it to the first party
    member that can learn it. Returns True if a Pokemon was taught."""
    if env.read_m(IS_IN_BATTLE) != 0 or not has_hm01(env) or party_slot_with_cut(env) is not None:
        return False
    for slot in range(min(env.read_m(PARTY_COUNT), 6)):
        if env.read_m(PARTY_SPECIES + slot) not in CUT_COMPATIBLE:
            continue
        base = _moves_addr(slot)
        moves = [env.read_m(base + j) for j in range(4)]
        if 0 in moves:
            j = moves.index(0)
        else:
            j = next((moves.index(m) for m in LOW_VALUE_MOVES if m in moves), 3)
        env.set_m(base + j, CUT_MOVE_ID)
        env.set_m(PARTY_MON_1 + slot * PARTY_MON_SIZE + PP_OFFSET + j, CUT_PP)
        return True
    return False


# The player stands on screen tiles rows 8-9, cols 8-9; the 2x2 block in front:
FRONT_BLOCK = {0x0: (10, 8), 0x4: (6, 8), 0x8: (8, 6), 0xC: (8, 10)}  # facing: down, up, left, right


def front_block(env, facing):
    """Tile ids of the 2x2 block the player is facing (None if unreadable)."""
    tm = env._read_visible_tile_map()
    if tm is None or facing not in FRONT_BLOCK:
        return None
    r, c = FRONT_BLOCK[facing]
    return tuple(int(v) for v in tm[r:r + 2, c:c + 2].ravel())


def _press(env, button, release, frames=None, hold=8):
    frames = frames or env.act_freq
    env.pyboy.send_input(button)
    for i in range(frames):
        if i == hold:
            env.pyboy.send_input(release)
        env.pyboy.tick()


def _wait(env, frames):
    for _ in range(frames):
        env.pyboy.tick()


def use_cut(env):
    """Run Start -> Pokemon -> (Pokemon with Cut) -> CUT in the facing direction.
    Returns 'no_cut_pokemon', 'in_battle', 'no_effect' or 'cut'."""
    if env.read_m(IS_IN_BATTLE) != 0:
        _wait(env, env.act_freq)
        return "in_battle"
    slot = party_slot_with_cut(env)
    if slot is None:
        _wait(env, env.act_freq)
        return "no_cut_pokemon"

    E = WindowEvent
    # Close anything already open (start menu, a finished text box) first.
    for _ in range(2):
        _press(env, E.PRESS_BUTTON_B, E.RELEASE_BUTTON_B)
    _wait(env, 8)

    facing = env.read_m(PLAYER_FACING)
    x, y, map_n = env.get_game_coords()
    before = front_block(env, facing)

    # The start menu remembers its last position; put the cursor back on the top
    # item so the presses below always land on POKEMON.
    try:
        _, addr = env.pyboy.symbol_lookup("wBattleAndStartSavedMenuItem")
        env.set_m(addr, 0)
    except Exception:
        pass

    # Menus in Gen 1 draw slowly and ignore input while drawing, so each press
    # gets a generous number of frames.
    _press(env, E.PRESS_BUTTON_START, E.RELEASE_BUTTON_START, frames=60)
    if env.read_bit(*GOT_POKEDEX):  # POKEDEX is the first item once you have it
        _press(env, E.PRESS_ARROW_DOWN, E.RELEASE_ARROW_DOWN, frames=20)
    _press(env, E.PRESS_BUTTON_A, E.RELEASE_BUTTON_A, frames=90)          # open party list
    for _ in range(slot):
        _press(env, E.PRESS_ARROW_DOWN, E.RELEASE_ARROW_DOWN, frames=20)
    _press(env, E.PRESS_BUTTON_A, E.RELEASE_BUTTON_A, frames=40)          # that Pokemon's menu
    _press(env, E.PRESS_BUTTON_A, E.RELEASE_BUTTON_A, frames=40)          # CUT is the first entry
    _wait(env, 240)                                                        # animation / message
    for _ in range(4):                                         # close text and any menus
        _press(env, E.PRESS_BUTTON_B, E.RELEASE_BUTTON_B, frames=30)
    _wait(env, 30)

    after = front_block(env, facing)
    changed = before is not None and after is not None and before != after
    if changed and env.get_game_coords() == (x, y, map_n) and env.read_m(IS_IN_BATTLE) == 0:
        env.trees_cut.add((map_n, x, y, facing))
        return "cut"
    return "no_effect"
