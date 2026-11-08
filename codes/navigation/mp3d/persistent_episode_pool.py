"""Cross-process episode claims for long-lived evaluation workers."""

from __future__ import annotations

import fcntl
import glob
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def valid_episode_summary(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and "steps_taken" in payload
        and "goal_reached" in payload
    )


def episode_index_from_summary(path: Path) -> int | None:
    try:
        return int(path.parent.name.split("_", 1)[0])
    except (TypeError, ValueError):
        return None


class PersistentEpisodePool:
    def __init__(self, pool_dir: str | Path, log_root: str | Path) -> None:
        self.pool_dir = Path(pool_dir)
        self.log_root = Path(log_root)
        self.done_dir = self.pool_dir / "done"
        self.claim_dir = self.pool_dir / "claims"
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.claim_dir.mkdir(parents=True, exist_ok=True)

    def done_path(self, episode_index: int) -> Path:
        return self.done_dir / f"{episode_index}.done"

    def is_done(self, episode_index: int) -> bool:
        return self.done_path(episode_index).is_file()

    def _find_valid_summary(self, episode_index: int) -> Path | None:
        pattern = str(
            self.log_root
            / "*_of_*"
            / f"{episode_index}_*"
            / "episode_summary.json"
        )
        for raw_path in glob.iglob(pattern):
            path = Path(raw_path)
            if valid_episode_summary(path):
                return path
        return None

    def mark_done(self, episode_index: int, summary_path: str | Path) -> bool:
        summary = Path(summary_path)
        if not valid_episode_summary(summary):
            return False
        done_path = self.done_path(episode_index)
        tmp = done_path.with_name(
            f".{done_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        tmp.write_text(str(summary) + "\n", encoding="utf-8")
        os.replace(tmp, done_path)
        return True

    def seed_from_summaries(self) -> int:
        seed_lock_path = self.pool_dir / "seed.lock"
        seeded = 0
        with seed_lock_path.open("a+", encoding="utf-8") as seed_lock:
            fcntl.flock(seed_lock.fileno(), fcntl.LOCK_EX)
            pattern = str(self.log_root / "*_of_*" / "*" / "episode_summary.json")
            for raw_path in glob.iglob(pattern):
                path = Path(raw_path)
                episode_index = episode_index_from_summary(path)
                if episode_index is None or self.is_done(episode_index):
                    continue
                if self.mark_done(episode_index, path):
                    seeded += 1
            fcntl.flock(seed_lock.fileno(), fcntl.LOCK_UN)
        return seeded

    @contextmanager
    def try_claim(self, episode_index: int) -> Iterator[bool]:
        claim_path = self.claim_dir / f"{episode_index}.lock"
        claim_file = claim_path.open("a+", encoding="utf-8")
        acquired = False
        try:
            try:
                fcntl.flock(
                    claim_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                acquired = True
            except BlockingIOError:
                yield False
                return

            if self.is_done(episode_index):
                yield False
                return
            summary = self._find_valid_summary(episode_index)
            if summary is not None:
                self.mark_done(episode_index, summary)
                yield False
                return
            yield True
        finally:
            if acquired:
                fcntl.flock(claim_file.fileno(), fcntl.LOCK_UN)
            claim_file.close()
