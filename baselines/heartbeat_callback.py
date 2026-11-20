"""Heartbeat for overnight runs.

Every `every_n_calls` training steps this appends one JSON line to
runs/heartbeat.jsonl with progress, game stats and memory use. Two things
read it:

* run_overnight.py (the watchdog) watches the file's modification time. If it
  stops changing, some emulator has stalled and the run gets restarted.
* morning_summary.py turns it into a short report of what happened overnight.

Gathering the stats asks every environment for its data, so if one emulator
is stuck the heartbeat stops too. That is deliberate: a missing heartbeat is
exactly the signal the watchdog needs.
"""
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import psutil
except ImportError:  # memory numbers are nice to have, not required
    psutil = None


class HeartbeatCallback(BaseCallback):
    def __init__(self, heartbeat_path, session_id, every_n_calls=256, stop_after_hours=0, verbose=0):
        super().__init__(verbose)
        self.stop_after_hours = stop_after_hours
        self.heartbeat_path = Path(heartbeat_path)
        self.session_id = session_id
        self.every_n_calls = every_n_calls
        self.t_start = None
        self.steps_at_start = 0

    def _on_training_start(self):
        self.t_start = time.time()
        self.steps_at_start = self.num_timesteps
        self._write(event="start")

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_calls == 0:
            self._write(event="beat")
            if self.stop_after_hours and time.time() - self.t_start > self.stop_after_hours * 3600:
                print(f"\nReached the {self.stop_after_hours} hour limit, stopping.")
                self._write(event="time_limit")
                return False
        return True

    def _on_training_end(self):
        self._write(event="end")

    def _memory(self):
        if psutil is None:
            return {}
        try:
            me = psutil.Process()
            kids = me.children(recursive=True)
            kid_rss = [k.memory_info().rss for k in kids]
            gb = 1024 ** 3
            return {
                "main_rss_gb": round(me.memory_info().rss / gb, 2),
                "workers_rss_gb": round(sum(kid_rss) / gb, 2),
                "max_worker_rss_gb": round(max(kid_rss) / gb, 2) if kid_rss else 0,
                "system_used_pct": psutil.virtual_memory().percent,
            }
        except Exception:
            return {}

    def _game_stats(self):
        try:
            stats = [s[0] for s in self.training_env.get_attr("agent_stats") if s]
        except Exception:
            return {}
        if not stats:
            return {}

        def col(key):
            return np.array([s.get(key, 0) for s in stats], dtype=float)

        badges = col("badge")
        progress = col("max_map_progress")
        coords = col("coord_count")
        levels = col("levels_sum")
        return {
            "badges_max": int(badges.max()),
            "badges_mean": round(float(badges.mean()), 2),
            "map_progress_max": int(progress.max()),
            "map_progress_mean": round(float(progress.mean()), 1),
            "coords_max": int(coords.max()),
            "levels_sum_max": int(levels.max()),
            "deaths_mean": round(float(col("deaths").mean()), 2),
        }

    def _write(self, event):
        elapsed = max(time.time() - (self.t_start or time.time()), 1e-6)
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix": round(time.time(), 1),
            "event": event,
            "session": self.session_id,
            "timesteps": int(self.num_timesteps),
            "steps_per_sec": round((self.num_timesteps - self.steps_at_start) / elapsed, 1),
        }
        if event != "start":
            record.update(self._game_stats())
        record.update(self._memory())
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.heartbeat_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
