"""Overnight runner with a watchdog.

Starts training, watches runs/heartbeat.jsonl, and if the heartbeat stops
(an emulator stalled, a worker died, anything else) it stops the whole run
and restarts it from the newest checkpoint. You wake up to a run that
recovered instead of one that froze at midnight.

Usage (from the baselines folder, with the venv active):
    python run_overnight.py                 # fresh start, run until Ctrl+C
    python run_overnight.py --hours 9       # stop by itself after ~9 hours
    python run_overnight.py --resume        # continue from the newest checkpoint
    python run_overnight.py --stream        # also show agents on the live map

Everything it does is logged to runs/watchdog.log. Ctrl+C stops it cleanly.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs")
HEARTBEAT = RUNS / "heartbeat.jsonl"
LOG = RUNS / "watchdog.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(f"[watchdog] {line}", flush=True)
    RUNS.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def heartbeat_age():
    try:
        return time.time() - HEARTBEAT.stat().st_mtime
    except FileNotFoundError:
        return None


def stop(proc):
    """Stop training and every emulator worker it started."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)  # polite first: lets PyTorch/PyBoy clean up
        proc.wait(timeout=60)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # a stalled emulator won't answer politely
    except ProcessLookupError:
        pass
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0, help="total hours to train (0 = until Ctrl+C)")
    ap.add_argument("--resume", action="store_true", help="first launch continues from the newest checkpoint")
    ap.add_argument("--num-envs", type=int, default=12)
    ap.add_argument("--stream", action="store_true", help="broadcast to the live map (slower)")
    ap.add_argument("--stall-minutes", type=float, default=10,
                    help="restart if no heartbeat for this long (heartbeat is ~every 30 s)")
    ap.add_argument("--startup-minutes", type=float, default=10,
                    help="grace period for a launch to write its first heartbeat")
    ap.add_argument("--max-restarts", type=int, default=1000,
                    help="effectively unlimited: keep training no matter how many times it needs rescuing")
    ap.add_argument("--checkin-minutes", type=float, default=60)
    ap.add_argument("--base-agents", type=int, default=None,
                    help="agents that always start from the base state (passed to training)")
    ap.add_argument("--ladder-window", type=int, default=None)
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--map", action="store_true", help="show agents on the shared live map")
    ap.add_argument("--map-name", default=None)
    ap.add_argument("--map-color", default=None)
    ap.add_argument("--map-sprite", type=int, default=None)
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600 if args.hours else None
    restarts = 0
    resume = args.resume
    log(f"=== overnight run starting (hours={args.hours or 'until stopped'}, envs={args.num_envs}, "
        f"stream={args.stream}, resume={resume}) ===")

    while True:
        cmd = [sys.executable, "baseline_fast_v2.py", "--num-envs", str(args.num_envs),
               "--checkin-minutes", str(args.checkin_minutes)]
        if args.base_agents is not None:
            cmd += ["--base-agents", str(args.base_agents)]
        if args.ladder_window is not None:
            cmd += ["--ladder-window", str(args.ladder_window)]
        if args.no_ladder:
            cmd.append("--no-ladder")
        if args.map:
            cmd.append("--map")
        for flag, val in (("--map-name", args.map_name), ("--map-color", args.map_color), ("--map-sprite", args.map_sprite)):
            if val is not None:
                cmd += [flag, str(val)]
        if resume:
            cmd.append("--resume")
        if args.stream:
            cmd.append("--stream")
        if deadline:
            remaining_h = (deadline - time.time()) / 3600
            if remaining_h <= 0.05:
                log("time budget used up, not relaunching")
                break
            cmd += ["--hours", f"{remaining_h:.2f}"]

        log("launching: " + " ".join(cmd[1:]))
        launched = time.time()
        # Own process group, so a restart takes the emulator workers down too.
        proc = subprocess.Popen(cmd, start_new_session=True)

        reason = None
        try:
            while proc.poll() is None:
                time.sleep(15)
                age = heartbeat_age()
                fresh = age is not None and (time.time() - age) >= launched
                if fresh and age > args.stall_minutes * 60:
                    reason = f"no heartbeat for {age / 60:.0f} min (stalled)"
                elif not fresh and time.time() - launched > args.startup_minutes * 60:
                    reason = f"no heartbeat within {args.startup_minutes:.0f} min of launch"
                if reason:
                    break
        except KeyboardInterrupt:
            log("Ctrl+C received, stopping training")
            stop(proc)
            log("=== stopped by user ===")
            return

        if reason:
            log(f"RESTART: {reason}")
            stop(proc)
        elif proc.returncode == 0:
            log("training finished normally")
            break
        else:
            log(f"RESTART: training exited with code {proc.returncode}")

        if restarts >= args.max_restarts:
            log(f"giving up after {restarts} restarts; check the log above")
            break
        restarts += 1
        resume = True  # every relaunch continues from the newest checkpoint
        time.sleep(20)

    log(f"=== overnight run ended after {restarts} restart(s) ===")


if __name__ == "__main__":
    main()
