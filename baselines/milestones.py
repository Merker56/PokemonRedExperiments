"""Milestones, the save archive, and the start-state ladder.

The agents make their own practice saves. The first time an agent reaches a
milestone in a run, the game is saved into runs/archive/<milestone>/. Each save
records its *total* steps since the base start state (steps to reach the save
it started from + steps taken this run), so saves of the same milestone made by
agents with different starting points can be compared fairly: fewer total
steps = a faster route.

The ladder decides where each agent starts a run:

* Agents 0..n_base-1 always start from the base state. They keep the early
  game in practice and keep looking for faster routes to early milestones.
* The other agents rotate. Run 1 everyone starts from the base state. From
  run 2 on, one rotating agent per run moves up to the newest milestone in the
  archive and stays there until its turn comes round again. The rest keep the
  milestone they were last given.
* Whenever an agent starts from a milestone it loads one of the two best
  (fewest total steps) saves for it, alternating between them. A faster route
  found by any agent is picked up automatically on the next run.
* Only the newest `window` milestones are used by rotating agents; older ones
  are retired from the rotation (base agents still cover them).

Each save also carries its *lineage*: every tile its line of agents has explored
and every map it has visited, all the way back to the base state. An agent that
starts from the save gets those preloaded, so walking back to old towns earns
nothing; only genuinely new ground pays. (Saves from before this existed are
ignored, and the ladder rebuilds itself.)

Everything lives in files, so the emulator processes never need to talk to
each other, and the ladder carries on after a watchdog restart.
"""
SAVE_FORMAT = 2   # 2 = saves carry lineage (explored tiles + visited maps)
import io
import json

import numpy as np
import os
import time
import uuid
from pathlib import Path

# --- memory addresses (pokered) ---------------------------------------------
CUR_MAP = 0xD35E
BADGES = 0xD356
NUM_BAG_ITEMS = 0xD31D
BAG_ITEMS = 0xD31E          # (item id, quantity) pairs
IS_IN_BATTLE = 0xD057       # 0 = not in battle

ITEM_OAKS_PARCEL = 0x46
ITEM_SS_TICKET = 0x3F
ITEM_HM01 = 0xC4
ITEM_SILPH_SCOPE = 0x48
ITEM_POKE_FLUTE = 0x49
ITEM_HM03_SURF = 0xC6
ITEM_HM04_STRENGTH = 0xC7

MT_MOON_MAPS = {0x3B, 0x3C, 0x3D}
SS_ANNE_MAPS = set(range(0x5F, 0x69))
ROCK_TUNNEL_MAPS = {0x52, 0xE8}


def bag_has(env, item_id):
    n = min(env.read_m(NUM_BAG_ITEMS), 20)
    return any(env.read_m(BAG_ITEMS + 2 * i) == item_id for i in range(n))


def on_map(env, *maps):
    return env.read_m(CUR_MAP) in maps


def badge(env, bit):
    return (env.read_m(BADGES) >> bit) & 1 == 1


def out_of_mt_moon(env):
    # Route 4 is one map on both sides of Mt. Moon. The west part (Pokemon Center
    # door x=11, cave entrance x=18) is reached from Route 3; the east part starts
    # at the cave's exit (x=24) and is reachable only through the cave (the ledges
    # east of it are one-way). The old check (Route 4 after visiting B2F) also fired
    # for agents that went down to B2F and walked back out the WEST entrance, so the
    # ladder handed out "through Mt. Moon" saves that were still west of it (Nov 24).
    return on_map(env, 0x0F) and env.read_m(0xD362) >= 24


# (key, label, check). Order = progress order; "newest" = highest index reached.
MILESTONES = [
    ("01_oaks_parcel",   "Got Oak's Parcel",            lambda e: bag_has(e, ITEM_OAKS_PARCEL) or e.read_bit(0xD74E, 0)),
    ("02_pokedex",       "Got the Pokédex",              lambda e: e.read_bit(0xD74B, 5)),
    ("03_viridian_forest", "Entered Viridian Forest",    lambda e: on_map(e, 0x33)),
    ("04_pewter",        "Reached Pewter City",          lambda e: on_map(e, 0x02)),
    ("05_boulder_badge", "Boulder Badge",                lambda e: badge(e, 0)),
    ("06_mt_moon",       "Entered Mt. Moon",             lambda e: e.read_m(CUR_MAP) in MT_MOON_MAPS),
    ("07_out_of_mt_moon", "Through Mt. Moon",            out_of_mt_moon),
    ("08_cerulean",      "Reached Cerulean City",        lambda e: on_map(e, 0x03)),
    ("09_cascade_badge", "Cascade Badge",                lambda e: badge(e, 1)),
    ("10_ss_ticket",     "Got the S.S. Ticket",          lambda e: bag_has(e, ITEM_SS_TICKET)),
    ("11_vermilion",     "Reached Vermilion City",       lambda e: on_map(e, 0x05)),
    ("12_ss_anne",       "Boarded the S.S. Anne",        lambda e: e.read_m(CUR_MAP) in SS_ANNE_MAPS),
    ("13_hm01",          "Got HM01 (Cut)",               lambda e: bag_has(e, ITEM_HM01) or e.read_bit(0xD803, 0)),
    ("14_vermilion_gym", "Cut the tree: in Vermilion Gym", lambda e: on_map(e, 0x5C)),
    ("15_thunder_badge", "Thunder Badge",                lambda e: badge(e, 2)),
    # --- the rest of the game (not every route has to go in this exact order) ---
    ("16_rock_tunnel",   "Entered Rock Tunnel",          lambda e: e.read_m(CUR_MAP) in ROCK_TUNNEL_MAPS),
    ("17_lavender",      "Reached Lavender Town",        lambda e: on_map(e, 0x04)),
    ("18_celadon",       "Reached Celadon City",         lambda e: on_map(e, 0x06)),
    ("19_rainbow_badge", "Rainbow Badge",                lambda e: badge(e, 3)),
    ("20_silph_scope",   "Got the Silph Scope",          lambda e: bag_has(e, ITEM_SILPH_SCOPE)),
    ("21_poke_flute",    "Got the Poké Flute",           lambda e: bag_has(e, ITEM_POKE_FLUTE)),
    ("22_fuchsia",       "Reached Fuchsia City",         lambda e: on_map(e, 0x07)),
    ("23_soul_badge",    "Soul Badge",                   lambda e: badge(e, 4)),
    ("24_hm03_surf",     "Got HM03 (Surf)",              lambda e: bag_has(e, ITEM_HM03_SURF)),
    ("25_hm04_strength", "Got HM04 (Strength)",          lambda e: bag_has(e, ITEM_HM04_STRENGTH)),
    ("26_saffron",       "Reached Saffron City",         lambda e: on_map(e, 0x0A)),
    ("27_marsh_badge",   "Marsh Badge",                  lambda e: badge(e, 5)),
    ("28_cinnabar",      "Reached Cinnabar Island",      lambda e: on_map(e, 0x08)),
    ("29_volcano_badge", "Volcano Badge",                lambda e: badge(e, 6)),
    ("30_earth_badge",   "Earth Badge",                  lambda e: badge(e, 7)),
    ("31_victory_road",  "Entered Victory Road",         lambda e: on_map(e, 0x6C)),
    ("32_indigo",        "Reached Indigo Plateau",       lambda e: on_map(e, 0x09)),
    ("33_elite_four",    "Facing the Elite Four",        lambda e: on_map(e, 0xF5)),
    ("34_champion",      "Reached the Champion",         lambda e: on_map(e, 0x78)),
    ("35_hall_of_fame",  "Hall of Fame!",                lambda e: on_map(e, 0x76)),
]
LABELS = {k: label for k, label, _ in MILESTONES}


def _atomic_write(path, data, mode="wb"):
    tmp = path.with_name(path.name + f".tmp{uuid.uuid4().hex[:6]}")
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


class Archive:
    """Folder of milestone saves. Best = fewest total steps."""

    def __init__(self, root, base_state_name):
        self.root = Path(root)
        self.base = base_state_name
        self.root.mkdir(parents=True, exist_ok=True)

    def saves(self, key):
        out = []
        for meta in (self.root / key).glob("*.json"):
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue  # half-written by another process; it'll be readable next time
            state = meta.with_suffix(".state")
            lineage = meta.with_suffix(".lineage.npz")
            if (m.get("base_state") == self.base and m.get("format") == SAVE_FORMAT
                    and state.exists() and lineage.exists()):
                m["state_path"] = str(state)
                m["lineage_path"] = str(lineage)
                out.append(m)
        return sorted(out, key=lambda m: m["total_steps"])

    def best(self, key, k=2):
        return self.saves(key)[:k]

    def reached(self):
        """Indices of milestones that have at least one save, ascending."""
        return [i for i, (key, _, _) in enumerate(MILESTONES) if self.best(key, 1)]

    def add(self, key, state_bytes, meta, lineage, keep=5):
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=True)
        existing = self.saves(key)
        if len(existing) >= keep and meta["total_steps"] >= existing[keep - 1]["total_steps"]:
            return False  # slower than everything we keep; not worth storing
        name = f"{meta['total_steps']:012d}_a{meta['agent']:02d}_{uuid.uuid4().hex[:6]}"
        meta = dict(meta, format=SAVE_FORMAT)
        buf = io.BytesIO()
        np.savez_compressed(buf, coords=lineage[0], maps=lineage[1])
        _atomic_write(folder / f"{name}.lineage.npz", buf.getvalue())
        _atomic_write(folder / f"{name}.state", state_bytes)
        _atomic_write(folder / f"{name}.json", json.dumps(meta, indent=1), mode="w")  # last: marks it complete
        for old in self.saves(key)[keep:]:  # prune the slowest
            for suffix in (".json", ".state", ".lineage.npz"):
                try:
                    Path(old["state_path"]).with_suffix(suffix).unlink()
                except OSError:
                    pass
        return True


class Ladder:
    """Per-agent start assignment, persisted in runs/archive/assignments/."""

    def __init__(self, archive, rank, n_envs, n_base=4, window=5):
        self.archive = archive
        self.rank = rank
        self.n_rotating = max(n_envs - n_base, 0)
        self.rotating = rank >= n_base and self.n_rotating > 0
        self.slot = rank - n_base
        self.window = window
        self.path = archive.root / "assignments" / f"agent_{rank:02d}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.state = json.loads(self.path.read_text())
        except Exception:
            self.state = {"runs": 0, "milestone": None, "alternate": 0}

    def _save(self):
        _atomic_write(self.path, json.dumps(self.state), mode="w")

    def next_start(self):
        """Called at every reset. Returns (save metadata or None for base, reason)."""
        self.state["runs"] += 1
        run = self.state["runs"]
        if not self.rotating:
            self._save()
            return None, "base"

        reached = self.archive.reached()
        newest = reached[-1] if reached else None

        # Your rotation: from run 2, slot (run-2) % n_rotating moves to the newest.
        if run >= 2 and newest is not None and (run - 2) % self.n_rotating == self.slot:
            self.state["milestone"] = newest

        idx = self.state["milestone"]
        if idx is not None and newest is not None:
            allowed = [i for i in reached if i > newest - self.window]
            if idx not in allowed:  # retired: move to the oldest milestone still in the window
                idx = self.state["milestone"] = allowed[0]

        meta = None
        if idx is not None:
            best = self.archive.best(MILESTONES[idx][0], k=2)
            if best:
                meta = best[self.state["alternate"] % len(best)]
                self.state["alternate"] += 1
        self._save()
        return meta, (MILESTONES[idx][0] if meta else "base")


class MilestoneTracker:
    """Lives inside one environment: detects milestones and makes saves."""

    CHECK_EVERY = 8          # steps between checks (cheap memory reads)
    QUALITY_PATIENCE = 400   # steps to wait for a "good moment" before giving up

    def __init__(self, env, archive=None, ladder=None, min_hp_fraction=0.5):
        self.env = env
        self.archive = archive
        self.ladder = ladder
        self.min_hp = min_hp_fraction
        self.offset = 0          # total steps of the save this run started from
        self.start_key = "base"
        self.done = set()
        self.pending = {}        # milestone index -> step first seen
        self.reached_this_run = set()  # for the one-time milestone reward

    @property
    def highest(self):
        """Index of the furthest milestone this agent's game has reached (-1 = none)."""
        reached = self.done | self.reached_this_run
        return max(reached) if reached else -1

    def choose_start(self):
        self.start_meta = None
        if self.ladder is None:
            self.offset, self.start_key = 0, "base"
            return None
        meta, key = self.ladder.next_start()
        self.offset = meta["total_steps"] if meta else 0
        self.start_key = key
        self.start_meta = meta
        return meta["state_path"] if meta else None

    def after_load(self):
        """Mark milestones already true in the loaded game so they aren't 're-reached'."""
        keys = [k for k, _, _ in MILESTONES]
        start_idx = keys.index(self.start_key) if self.start_key in keys else -1
        self.env._ms_seen_mt_moon_b2f = start_idx >= keys.index("07_out_of_mt_moon")
        # Everything up to the start milestone counts as done (going back to an
        # earlier place isn't "reaching" it), plus anything already true in the save.
        self.done = {i for i, (_, _, check) in enumerate(MILESTONES)
                     if i <= start_idx or self._safe(check)}
        self.pending = {}
        self.reached_this_run = set()
        # Preload where this save's line of agents has already been
        meta = getattr(self, "start_meta", None)
        if meta and self.start_key != "base":
            try:
                with np.load(meta["lineage_path"]) as z:
                    self.env.apply_lineage(z["coords"], z["maps"])
            except Exception as e:
                print(f"[milestone] agent {self.env.agent_id}: no lineage loaded ({e})")

    def _safe(self, check):
        try:
            return bool(check(self.env))
        except Exception:
            return False

    def good_moment(self):
        e = self.env
        try:
            return e.read_m(IS_IN_BATTLE) == 0 and e.read_hp_fraction() >= self.min_hp
        except Exception:
            return False

    def step(self):
        e = self.env
        if e.read_m(CUR_MAP) == 0x3D:
            e._ms_seen_mt_moon_b2f = True
        if e.step_count % self.CHECK_EVERY:
            return
        for i, (key, label, check) in enumerate(MILESTONES):
            if i in self.done:
                continue
            if i not in self.pending:
                if not self._safe(check):
                    continue
                self.pending[i] = e.step_count
                self.reached_this_run.add(i)
            # Reached: save at the first good moment (out of battle, healthy enough)
            if self.good_moment():
                self._save(i, key, label)
                self.done.add(i)
                self.pending.pop(i, None)
            elif e.step_count - self.pending[i] > self.QUALITY_PATIENCE:
                self.done.add(i)          # never got a good moment; skip this one
                self.pending.pop(i, None)

    def _save(self, i, key, label):
        e = self.env
        if self.archive is None:
            return
        buf = io.BytesIO()
        e.pyboy.save_state(buf)
        meta = {
            "milestone": key,
            "label": label,
            "total_steps": int(self.offset + e.step_count),
            "run_steps": int(e.step_count),
            "started_from": self.start_key,
            "agent": int(e.agent_id),
            "hp_fraction": round(float(e.read_hp_fraction()), 2),
            "levels": [int(e.read_m(a)) for a in (0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268)][: e.read_m(0xD163)],
            "badges": int(e.get_badges()),
            "base_state": self.archive.base,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        coords = np.array(list(e.seen_coords.keys()), dtype=np.int16).reshape(-1, 3)
        maps = np.array(sorted(e.maps_visited), dtype=np.int16)
        meta["lineage_tiles"] = int(len(coords))
        meta["lineage_maps"] = int(len(maps))
        if self.archive.add(key, buf.getvalue(), meta, (coords, maps)):
            print(f"\n[milestone] agent {e.agent_id}: {label} in {meta['total_steps']:,} total steps "
                  f"(from {self.start_key})")
