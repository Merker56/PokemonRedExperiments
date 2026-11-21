"""Check-in reports for reviewing training with the kids.

Every `every_minutes` (default 60) this writes a small web page to
runs/checkins/<date_time>/report.html with:

  * what every agent is looking at right now (their Game Boy screens)
  * a leaderboard: badges, where they are, tiles explored, party levels
  * each agent's Pokemon team with levels and moves
  * the combined map of everywhere the agents have walked
  * how this compares with the previous check-in

runs/checkins/index.html lists every check-in, newest first, so you can open
one file and click through the night. Training never pauses for this: it reads
data the agents already keep, and any problem making a report is printed and
skipped rather than stopping training.
"""
import html
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

from map_lookup import MapIds
from milestones import MILESTONES, Archive


def start_label(key):
    for k, label, _ in MILESTONES:
        if k == key:
            return label
    return "the beginning"


def milestone_label(n):
    """n is 1-based (0 = none yet)."""
    return MILESTONES[n - 1][1] if 1 <= n <= len(MILESTONES) else "-"


def cut_cell(a):
    if a.get("trees_cut"):
        return f"✂️ {a['trees_cut']} tree(s)"
    if a.get("knows_cut"):
        return f"knows Cut, tried {a.get('cut_tries', 0)}x"
    return "-"


def map_name(map_n):
    try:
        return MapIds(int(map_n)).name.replace("_", " ").title()
    except Exception:
        return f"Map {map_n}"


class CheckinReportCallback(BaseCallback):
    def __init__(self, out_dir, session_id, every_minutes=60, archive_dir=None, base_state=None, verbose=0):
        super().__init__(verbose)
        self.archive = Archive(archive_dir, base_state) if archive_dir else None
        self.out_dir = Path(out_dir)
        self.session_id = session_id
        self.every_seconds = every_minutes * 60
        self.last_report = None

    def _on_training_start(self):
        # First report ~10 minutes in, so a fresh start shows something early.
        self.last_report = time.time() - self.every_seconds + 600

    def _on_step(self) -> bool:
        # Checking the clock is cheap; only do it every 1000 calls anyway.
        if self.n_calls % 1000 == 0 and time.time() - self.last_report >= self.every_seconds:
            self.last_report = time.time()
            try:
                self.write_report()
            except Exception as e:  # never let a report stop training
                print(f"\n[check-in] could not write report: {e}")
        return True

    # ---------------------------------------------------------------- data
    def collect(self):
        env = self.training_env
        stats = [s[0] if s else {} for s in env.get_attr("agent_stats")]
        parties = env.get_attr("party_info")
        screens = env.env_method("full_screen")
        explore = np.array(env.get_attr("explore_map"))
        try:
            starts = env.get_attr("start_key")
        except Exception:
            starts = ["base"] * len(stats)
        agents = []
        for i, (st, party, scr, start) in enumerate(zip(stats, parties, screens, starts)):
            agents.append({
                "id": i,
                "badges": int(st.get("badge", 0)),
                "map_progress": int(st.get("max_map_progress", 0)),
                "milestone": int(st.get("milestone", 0)),
                "location": map_name(st.get("map", -1)),
                # New ground this run (agents starting from a milestone inherit
                # their line's explored tiles; those don't count here)
                "tiles": int(st.get("new_tiles", st.get("coord_count", 0))),
                "levels_sum": int(st.get("levels_sum", 0)),
                "deaths": int(st.get("deaths", 0)),
                "caught": int(st.get("pokemon_caught", 0) or 0),
                "trees_cut": int(st.get("trees_cut", 0)),
                "cut_tries": int(st.get("cut_tries", 0)),
                "knows_cut": bool(st.get("knows_cut", False)),
                "party": [
                    {"name": p[0], "level": int(p[1]), "moves": [m for m in p[2:6] if m and str(m) != "None"],
                     "hp": p[6]}
                    for p in (party or {}).values() if p
                ],
                "screen": np.asarray(scr),
                "started_from": start,
            })
        return agents, explore

    # -------------------------------------------------------------- output
    def write_report(self):
        stamp = time.strftime("%Y-%m-%d_%H%M")
        folder = self.out_dir / stamp
        folder.mkdir(parents=True, exist_ok=True)
        agents, explore = self.collect()

        # Screens: grayscale 144x160 -> PNG, scaled up 2x so they're easy to see
        for a in agents:
            img = a.pop("screen")
            if img.ndim == 3:
                img = img[:, :, 0]
            Image.fromarray(img.astype(np.uint8)).resize((320, 288), Image.NEAREST).save(
                folder / f"agent_{a['id']}.png")

        # Combined explored map: every tile any agent has stepped on
        if explore.ndim == 3 and explore.size:
            m = explore.max(axis=0)
            ys, xs = np.nonzero(m)
            if len(ys):
                pad = 10
                m = m[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
            Image.fromarray(m.astype(np.uint8)).resize(
                (m.shape[1] * 3, m.shape[0] * 3), Image.NEAREST).save(folder / "explored_map.png")

        # Rank by furthest milestone (covers the whole game), then badges, then new ground
        ranked = sorted(agents, key=lambda a: (a["milestone"], a["badges"], a["tiles"]), reverse=True)
        summary = {
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "session": self.session_id,
            "timesteps": int(self.num_timesteps),
            "best_badges": max((a["badges"] for a in agents), default=0),
            "best_milestone": ranked[0]["milestone"] if ranked else 0,
            "best_location": ranked[0]["location"] if ranked else "",
            "best_tiles": max((a["tiles"] for a in agents), default=0),
            "best_levels": max((a["levels_sum"] for a in agents), default=0),
            "agents": agents,
            "ladder": self._ladder(agents),
        }
        (folder / "data.json").write_text(json.dumps(summary, indent=1))

        previous = self._previous_summary(stamp)
        (folder / "report.html").write_text(self._html(summary, ranked, previous), encoding="utf-8")
        self._write_index()
        print(f"\n[check-in] report written: {folder / 'report.html'}")

    def _ladder(self, agents):
        if self.archive is None:
            return []
        rows = []
        for key, label, _ in MILESTONES:
            saves = self.archive.saves(key)
            if not saves:
                continue
            best = saves[0]
            rows.append({
                "key": key, "label": label, "best_steps": best["total_steps"], "best_agent": best["agent"],
                "best_time": best["time"], "saves": len(saves),
                "starting_here": [a["id"] for a in agents if a.get("started_from") == key],
            })
        return rows

    def _previous_summary(self, current_stamp):
        folders = sorted(p for p in self.out_dir.iterdir() if p.is_dir() and p.name < current_stamp)
        for p in reversed(folders):
            try:
                return json.loads((p / "data.json").read_text())
            except Exception:
                continue
        return None

    def _html(self, s, ranked, prev):
        e = html.escape

        def change(key):
            if not prev:
                return ""
            d = s[key] - prev.get(key, 0)
            return f" <span class='up'>(+{d:,})</span>" if d > 0 else ""

        rows = []
        for place, a in enumerate(ranked, 1):
            team = ", ".join(f"{e(str(p['name']))} L{p['level']}" for p in a["party"]) or "-"
            rows.append(
                f"<tr><td>{place}</td><td>Agent {a['id']}</td><td>{e(milestone_label(a['milestone']))}</td><td>{'🏅' * a['badges'] or '-'}</td>"
                f"<td>{e(a['location'])}</td><td>{a['tiles']:,}</td><td>{a['levels_sum']}</td>"
                f"<td>{a['deaths']}</td><td>{cut_cell(a)}</td><td>{team}</td></tr>")

        cards = []
        for a in ranked:
            moves = "".join(
                f"<li><b>{e(str(p['name']))}</b> L{p['level']} ({e(str(p['hp']))} HP)"
                f"<br><small>{e(', '.join(str(m) for m in p['moves']))}</small></li>" for p in a["party"])
            cards.append(
                f"<div class='card'><img src='agent_{a['id']}.png'><h3>Agent {a['id']}</h3>"
                f"<p>{e(a['location'])} · {a['badges']} badge(s)<br><small>started this run from: "
                f"{e(start_label(a.get('started_from', 'base')))}</small></p><ul>{moves}</ul></div>")

        prev_best = {r["key"]: r["best_steps"] for r in (prev or {}).get("ladder", [])}
        ladder_rows = []
        for r in s.get("ladder", []):
            if r["key"] not in prev_best:
                note = " <span class='up'>NEW!</span>"
            elif r["best_steps"] < prev_best[r["key"]]:
                note = f" <span class='up'>faster by {prev_best[r['key']] - r['best_steps']:,}</span>"
            else:
                note = ""
            here = ", ".join(f"Agent {i}" for i in r["starting_here"]) or "-"
            ladder_rows.append(
                f"<tr><td>{e(r['label'])}{note}</td><td>{r['best_steps']:,}</td><td>Agent {r['best_agent']}</td>"
                f"<td>{e(r['best_time'])}</td><td>{here}</td></tr>")
        ladder_html = ""
        if ladder_rows:
            ladder_html = ("<h2>Milestone ladder</h2><p>Fastest route found so far to each milestone "
                           "(total steps from the start of the game), and which agents are practising from it.</p>"
                           "<table><tr><th>Milestone</th><th>Fastest (steps)</th><th>Found by</th><th>When</th>"
                           "<th>Starting here now</th></tr>" + "".join(ladder_rows) + "</table>")

        hours = ""
        if prev:
            hours = f"<p>Since the last check-in ({e(prev['time'])}): " \
                    f"{s['timesteps'] - prev['timesteps']:,} more steps of practice.</p>"

        return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Check-in {e(s['time'])}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f7fb;color:#222}}
h1{{margin-bottom:4px}} .up{{color:#1a7f37;font-weight:600}}
.big{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.stat{{background:#fff;border-radius:10px;padding:12px 18px;box-shadow:0 1px 3px #0002}}
.stat b{{font-size:1.6em;display:block}}
table{{border-collapse:collapse;background:#fff;box-shadow:0 1px 3px #0002}}
td,th{{padding:6px 10px;border-bottom:1px solid #eee;text-align:left;font-size:14px}}
.grid{{display:flex;flex-wrap:wrap;gap:14px}}
.card{{background:#fff;border-radius:10px;padding:10px;width:340px;box-shadow:0 1px 3px #0002}}
.card img{{width:320px;image-rendering:pixelated;border-radius:4px}}
.card ul{{padding-left:18px;font-size:13px}} .map{{image-rendering:pixelated;max-width:100%;background:#000}}
</style></head><body>
<h1>Pokémon Red training check-in</h1>
<p>{e(s['time'])} · {s['timesteps']:,} total steps{change('timesteps')}</p>
{hours}
<div class='big'>
 <div class='stat'><b>{e(milestone_label(s.get('best_milestone', 0)))}</b>furthest milestone{change('best_milestone')}</div>
 <div class='stat'><b>{s['best_badges']}</b>most badges{change('best_badges')}</div>
 <div class='stat'><b>{e(s['best_location'])}</b>leader is here</div>
 <div class='stat'><b>{s['best_tiles']:,}</b>most new tiles in a run{change('best_tiles')}</div>
 <div class='stat'><b>{s['best_levels']}</b>best team level total{change('best_levels')}</div>
</div>
<h2>Leaderboard</h2>
<table><tr><th>#</th><th>Agent</th><th>Furthest milestone</th><th>Badges</th><th>Where</th><th>New tiles</th><th>Levels</th><th>Faints</th><th>Cut</th><th>Team</th></tr>
{''.join(rows)}</table>
{ladder_html}
<h2>Everywhere they've walked</h2>
<img class='map' src='explored_map.png'>
<h2>What each agent sees right now</h2>
<div class='grid'>{''.join(cards)}</div>
</body></html>"""

    def _write_index(self):
        entries = []
        for p in sorted((p for p in self.out_dir.iterdir() if p.is_dir()), reverse=True):
            try:
                d = json.loads((p / "data.json").read_text())
            except Exception:
                continue
            entries.append(
                f"<tr><td><a href='{p.name}/report.html'>{html.escape(d['time'])}</a></td>"
                f"<td>{d['timesteps']:,}</td><td>{html.escape(milestone_label(d.get('best_milestone', 0)))}</td><td>{d['best_badges']}</td>"
                f"<td>{html.escape(d['best_location'])}</td><td>{d['best_tiles']:,}</td><td>{d['best_levels']}</td></tr>")
        (self.out_dir / "index.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Training check-ins</title>"
            "<style>body{font-family:system-ui;margin:24px}td,th{padding:6px 12px;text-align:left}</style>"
            "<h1>Training check-ins</h1><table><tr><th>When</th><th>Steps</th><th>Furthest milestone</th><th>Best badges</th>"
            "<th>Leader location</th><th>Most tiles</th><th>Best levels</th></tr>"
            + "".join(entries) + "</table>", encoding="utf-8")
