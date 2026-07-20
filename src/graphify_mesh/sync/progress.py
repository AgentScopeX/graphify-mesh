"""Minimal dependency-free terminal progress bar for the per-repo indexing
loop. No third-party progress-bar library — this is small enough not to
justify a new dependency, and pipeline.py's log.info calls already cover the
non-interactive (systemd/redirected-output) case.

Renders only when stderr is a real terminal; otherwise `tick()` is a no-op
so piping output to a file or a systemd journal never fills up with
carriage-return noise.
"""
from __future__ import annotations

import sys

BAR_WIDTH = 30


class ProgressBar:
    def __init__(self, total: int, label: str = "indexing", stream=None) -> None:
        self.total = total
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = self.stream.isatty()

    def tick(self, i: int, detail: str) -> None:
        if not self.enabled or self.total <= 0:
            return
        filled = int(BAR_WIDTH * i / self.total)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        line = f"\r{self.label} [{bar}] {i}/{self.total} {detail}"
        pad = max(0, self._last_len - len(line)) if hasattr(self, "_last_len") else 0
        self.stream.write(line + (" " * pad))
        self.stream.flush()
        self._last_len = len(line)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.stream.write("\n")
        self.stream.flush()
