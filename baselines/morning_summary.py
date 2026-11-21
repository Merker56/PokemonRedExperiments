"""What happened overnight? Run from the baselines folder:

    python morning_summary.py            # last 12 hours
    python morning_summary.py --hours 24

Reads runs/heartbeat.jsonl and runs/watchdog.log and prints a short report:
how long it trained, how far the best agent got, how memory held up, and
whether the watchdog had to rescue anything.
"""
import argparse
import json
import time
from pathlib import Path

try:
    from milestones import MILESTONES
except Exception:
    MILESTONES = []


def milestone_label(n):
    return MILESTONES[n - 1][1] if 1 <= n <= len(MILESTONES) else "-"

RUNS = Path("runs")


def load_beats(since):
    path = RUNS / "heartbeat.jsonl"
    if not path.exists():
        return []
    beats = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            b = json.loads(line)
        except json.JSONDecodeError:
            continue
        if b.get("unix", 0) >= since:
            beats.append(b)
    return beats


def load_watchdog(since):
    path = RUNS / "watchdog.log"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            t = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if t >= since:
            out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    args = ap.parse_args()
    since = time.time() - args.hours * 3600

    beats = load_beats(since)
    watchdog = load_watchdog(since)
    print(f"\n=== Pokemon Red training: last {args.hours:g} hours ===\n")

    if not beats:
        print("No heartbeats in that window, so no training was recorded.")
        print("(Check that it was started with run_overnight.py from the baselines folder.)")
        return

    with_stats = [b for b in beats if "badges_max" in b]
    first, last = beats[0], beats[-1]
    hours = (last["unix"] - first["unix"]) / 3600
    gained = last["timesteps"] - first["timesteps"]
    print(f"Trained from {first['time']} to {last['time']} ({hours:.1f} h)")
    print(f"Steps: {first['timesteps']:,} -> {last['timesteps']:,} (+{gained:,})")

    speeds = [b["steps_per_sec"] for b in beats if b.get("event") == "beat" and b.get("steps_per_sec")]
    if speeds:
        print(f"Speed: about {sorted(speeds)[len(speeds) // 2]:,.0f} steps/sec")

    age_min = (time.time() - last["unix"]) / 60
    if age_min > 15:
        print(f"Last heartbeat was {age_min:.0f} min ago, so training is NOT running now.")
    else:
        print("Training is still running.")

    if with_stats:
        best = lambda k: max(b.get(k, 0) for b in with_stats)
        print("\nBest agent so far:")
        print(f"  badges:           {best('badges_max')}")
        print(f"  furthest milestone: {milestone_label(best('milestone_max'))}")
        print(f"  tiles explored:   {best('coords_max'):,}")
        print(f"  party level sum:  {best('levels_sum_max')}")

        # A few snapshots across the night, so you can see the trend
        print("\nThrough the night (best agent at each point):")
        print("  time                 steps         badges  milestone                    tiles   levels")
        n = len(with_stats)
        picks = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        for i in picks:
            b = with_stats[i]
            print(f"  {b['time']}  {b['timesteps']:>12,}  {b['badges_max']:>6}  {milestone_label(b.get('milestone_max', 0)):28.28}"
                  f"  {b['coords_max']:>6,}  {b['levels_sum_max']:>6}")

    mem = [b for b in beats if "system_used_pct" in b]
    if mem:
        peak = max(mem, key=lambda b: b["system_used_pct"])
        print(f"\nMemory: peak {peak['system_used_pct']:.0f}% of WSL RAM in use "
              f"(workers {peak.get('workers_rss_gb', '?')} GB, largest worker {peak.get('max_worker_rss_gb', '?')} GB)")
        if len(mem) > 10:
            early = sum(b.get("workers_rss_gb", 0) for b in mem[:5]) / 5
            late = sum(b.get("workers_rss_gb", 0) for b in mem[-5:]) / 5
            if late > early * 1.3:
                print(f"  Note: worker memory grew from {early:.1f} GB to {late:.1f} GB, which looks like a leak.")

    restarts = [l for l in watchdog if "RESTART" in l]
    print(f"\nWatchdog restarts: {len(restarts)}")
    for l in restarts:
        print("  " + l)
    print()


if __name__ == "__main__":
    main()
