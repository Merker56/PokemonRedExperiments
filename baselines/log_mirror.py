"""Copy the run's logs and check-in reports to a Windows folder.

Files inside WSL are awkward to reach from Windows, so the watchdog calls
mirror() every few minutes (run_overnight.py --mirror-dir). Only small files
are copied: logs, check-in reports with their screenshots, and the milestone
archive's .json summaries. Checkpoints, save states and lineage files stay in
WSL. A file is copied again only when its size or time changed.

Run once by hand:   python log_mirror.py /mnt/c/Users/<you>/Documents/pokered-logs
"""
import shutil
import sys
from pathlib import Path

LOG_FILES = ["heartbeat.jsonl", "learning.jsonl", "milestone_times.jsonl", "watchdog.log",
             "current_run.json", "best/best.json", "best_previous/best.json"]
TREES = {  # folder under runs/ -> file patterns to copy
    "checkins": ["*.html", "*.png", "*.jpg", "*.json"],
    "stalls": ["*.txt"],
    "archive": ["*.json"],
}


def _changed(src: Path, dst: Path) -> bool:
    try:
        d = dst.stat()
    except FileNotFoundError:
        return True
    s = src.stat()
    return s.st_size != d.st_size or s.st_mtime > d.st_mtime + 1


def _copy(src: Path, dst: Path) -> int:
    if not src.is_file() or not _changed(src, dst):
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def mirror(runs_dir, dest) -> int:
    """Copy what changed from runs_dir to dest. Returns the number of files copied."""
    runs_dir, dest = Path(runs_dir), Path(dest)
    copied = 0
    for name in LOG_FILES:
        copied += _copy(runs_dir / name, dest / name)
    for folder, patterns in TREES.items():
        root = runs_dir / folder
        if not root.is_dir():
            continue
        for pattern in patterns:
            for src in root.rglob(pattern):
                copied += _copy(src, dest / folder / src.relative_to(root))
    (dest / "README.txt").write_text(
        "Copied here from WSL by run_overnight.py every few minutes.\n"
        "Open checkins/index.html for the hourly reports.\n"
        "The run itself lives in ~/PokemonRedExperiments/baselines/runs inside Ubuntu.\n")
    return copied


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"copied {mirror('runs', sys.argv[1])} files")
