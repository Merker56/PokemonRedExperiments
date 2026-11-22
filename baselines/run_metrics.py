"""Run metrics: milestone timing, learning health and the best checkpoint.

Three things, all written under runs/ (the folder holding heartbeat.jsonl):

* runs/milestone_times.jsonl - the first time (total steps) any agent reached each
  milestone and each badge count in this run. One line per first. This is the
  number to compare runs on: "Brock at 41.2M steps" instead of eyeballing reports.

* runs/learning.jsonl - how the learner itself is doing, every ~30 seconds:
  entropy ("curiosity": how much the policy still tries different buttons),
  explained variance (how well it predicts its own rewards), plus the average
  agent's milestone, new tiles and team size. Entropy usually falls well before
  behaviour collapses, so the check-in warns when it drops far below its peak.

* runs/best/best_model.zip - a copy of the policy whenever the average agent's
  milestone beats every earlier moment of this run. If a run peaks and then
  collapses (Nov 21: best around 10M steps, then four hours of circling Pallet),
  you can restart from the peak:
      python run_overnight.py ... --checkpoint runs/best/best_model.zip

"A run" starts with a fresh launch (no --resume/--checkpoint) and continues
through watchdog restarts. runs/current_run.json holds its id.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from milestones import MILESTONES

TRAIN_KEYS = {
    "train/entropy_loss": "entropy",          # SB3 logs -entropy; flipped below
    "train/explained_variance": "explained_variance",
    "train/approx_kl": "approx_kl",
    "train/value_loss": "value_loss",
    "train/clip_fraction": "clip_fraction",
}


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line cut off by a crash
    except FileNotFoundError:
        pass
    return rows


def current_run_id(runs_dir):
    try:
        return json.loads((Path(runs_dir) / "current_run.json").read_text())["run_id"]
    except Exception:
        return None


class RunMetricsCallback(BaseCallback):
    def __init__(self, runs_dir, fresh, session_id, settings=None, every_n_calls=2048,
                 min_save_gap_s=300, verbose=0):
        super().__init__(verbose)
        self.runs_dir = Path(runs_dir)
        self.fresh = fresh
        self.session_id = session_id
        self.settings = settings or {}
        self.every_n_calls = every_n_calls
        self.min_save_gap_s = min_save_gap_s
        self.train_stats = {}
        self.firsts = set()
        self.best_score = -1.0
        self.last_save = 0.0

    # ------------------------------------------------------------ setup
    def _on_training_start(self):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        best_dir = self.runs_dir / "best"
        run_id = None if self.fresh else current_run_id(self.runs_dir)
        if run_id is None:
            run_id = time.strftime("%Y-%m-%d_%H%M")
            (self.runs_dir / "current_run.json").write_text(json.dumps(
                {"run_id": run_id, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "settings": self.settings}, indent=1))
            if best_dir.exists():  # the previous run's best is kept, one level back
                prev = self.runs_dir / "best_previous"
                shutil.rmtree(prev, ignore_errors=True)
                best_dir.rename(prev)
        self.run_id = run_id
        try:
            self.start_level = json.loads((self.runs_dir / "current_run.json").read_text()).get("start_milestone")
        except Exception:
            self.start_level = None
        # Resuming: remember what this run already recorded.
        for r in read_jsonl(self.runs_dir / "milestone_times.jsonl"):
            if r.get("run_id") == run_id:
                self.firsts.add(r["key"])
        try:
            best = json.loads((best_dir / "best.json").read_text())
            if best.get("run_id") == run_id:
                self.best_score = best["score"]
        except Exception:
            pass

    def _save_run_info(self, **extra):
        path = self.runs_dir / "current_run.json"
        try:
            info = json.loads(path.read_text())
        except Exception:
            info = {"run_id": self.run_id}
        info.update(extra)
        path.write_text(json.dumps(info, indent=1))

    # ------------------------------------------------------------ hooks
    def _on_rollout_start(self):
        # PPO.train() has just logged its numbers and they haven't been dumped yet.
        try:
            vals = self.model.logger.name_to_value
            stats = {short: float(vals[k]) for k, short in TRAIN_KEYS.items() if k in vals}
            if "entropy" in stats:
                stats["entropy"] = -stats["entropy"]
            if stats:
                self.train_stats = stats
        except Exception:
            pass

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_calls == 0:
            try:
                self._tick()
            except Exception as e:  # metrics must never stop training
                print(f"\n[metrics] skipped: {e}")
        return True

    # ------------------------------------------------------------ work
    def _tick(self):
        stats = [s[0] for s in self.training_env.get_attr("agent_stats") if s]
        if not stats:
            return
        col = lambda k: np.array([s.get(k, 0) for s in stats], dtype=float)
        milestone, badges = col("milestone"), col("badge")
        tiles, party = col("coord_count"), col("pcount")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        steps = int(self.num_timesteps)

        # 1. Firsts. Milestones the start state already has (Bulbasaur.state is past
        # the Pokedex) are noted once as the run's starting point, not as firsts.
        if self.start_level is None:
            self.start_level = int(milestone.min())
            self._save_run_info(start_milestone=self.start_level)
        new = []
        for k in range(self.start_level + 1, int(milestone.max()) + 1):
            key = f"m{k:02d}"
            if key not in self.firsts:
                label = MILESTONES[k - 1][1] if k <= len(MILESTONES) else key
                new.append({"key": key, "label": label})
        for b in range(1, int(badges.max()) + 1):
            key = f"badge{b}"
            if key not in self.firsts:
                new.append({"key": key, "label": f"Badge #{b}"})
        if new:
            with open(self.runs_dir / "milestone_times.jsonl", "a", encoding="utf-8") as f:
                for n in new:
                    self.firsts.add(n["key"])
                    f.write(json.dumps({"run_id": self.run_id, "time": now, "timesteps": steps,
                                        "per_agent": steps // max(len(stats), 1), **n}) + "\n")

        # 2. Learning health
        record = {"run_id": self.run_id, "time": now, "timesteps": steps,
                  "milestone_mean": round(float(milestone.mean()), 2),
                  "milestone_max": int(milestone.max()),
                  "tiles_mean": round(float(tiles.mean()), 1),
                  "party_mean": round(float(party.mean()), 2),
                  **{k: round(v, 4) for k, v in self.train_stats.items()}}
        with open(self.runs_dir / "learning.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 3. Best checkpoint (average agent's milestone; badges count for more)
        score = round(float(milestone.mean()) + 10 * float(badges.mean()), 2)
        if score > self.best_score and time.time() - self.last_save >= self.min_save_gap_s:
            best_dir = self.runs_dir / "best"
            best_dir.mkdir(exist_ok=True)
            self.model.save(best_dir / "best_model_tmp")
            (best_dir / "best_model_tmp.zip").replace(best_dir / "best_model.zip")
            (best_dir / "best.json").write_text(json.dumps(
                {**record, "score": score, "badges_mean": round(float(badges.mean()), 2),
                 "session": self.session_id}, indent=1))
            self.best_score = score
            self.last_save = time.time()


# ---------------------------------------------------------------- for the check-in page
def health_summary(runs_dir):
    """Small dict the check-in report turns into a 'How training is going' section."""
    runs_dir = Path(runs_dir)
    run_id = current_run_id(runs_dir)
    if run_id is None:
        return None
    learn = [r for r in read_jsonl(runs_dir / "learning.jsonl") if r.get("run_id") == run_id]
    firsts = [r for r in read_jsonl(runs_dir / "milestone_times.jsonl") if r.get("run_id") == run_id]
    out = {"run_id": run_id, "firsts": firsts}
    ent = [r["entropy"] for r in learn if "entropy" in r]
    if ent:
        peak = max(ent)
        out["entropy_now"] = ent[-1]
        out["entropy_peak"] = peak
        out["entropy_ratio"] = ent[-1] / peak if peak > 0 else 1.0
    if learn:
        out["explained_variance"] = learn[-1].get("explained_variance")
    try:
        out["best"] = json.loads((runs_dir / "best" / "best.json").read_text())
    except Exception:
        out["best"] = None
    try:
        out["settings"] = json.loads((runs_dir / "current_run.json").read_text()).get("settings", {})
    except Exception:
        out["settings"] = {}
    return out
