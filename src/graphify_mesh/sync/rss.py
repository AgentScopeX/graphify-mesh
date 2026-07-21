"""Per-stage RSS instrumentation for the sync pipeline.

Reads VmRSS/VmHWM from /proc/self/status (Linux-only, like the sync service
itself). VmHWM is the process's high-water mark and never decreases, so the
tracker attributes peak growth to a stage by diffing HWM between consecutive
marks: if HWM rose while a stage ran, that stage owned a new process-wide
peak. This costs one small file read per stage — no sampling thread, no
third-party profiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROC_STATUS = Path("/proc/self/status")
_UNKNOWN_KB = 0


@dataclass
class RssSnapshot:
    vm_rss_kb: int
    vm_hwm_kb: int


def _parse_kb(line: str) -> int:
    parts = line.split()
    if len(parts) < 3:
        return _UNKNOWN_KB
    if not parts[1].isdigit():
        return _UNKNOWN_KB
    if parts[2] != "kB":
        return _UNKNOWN_KB
    return int(parts[1])


def snapshot() -> RssSnapshot:
    rss_kb = _UNKNOWN_KB
    hwm_kb = _UNKNOWN_KB
    try:
        content = _PROC_STATUS.read_text(encoding="ascii")
    except OSError:
        return RssSnapshot(vm_rss_kb=_UNKNOWN_KB, vm_hwm_kb=_UNKNOWN_KB)
    for line in content.splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = _parse_kb(line)
        if line.startswith("VmHWM:"):
            hwm_kb = _parse_kb(line)
    return RssSnapshot(vm_rss_kb=rss_kb, vm_hwm_kb=hwm_kb)


class StageRssTracker:
    def __init__(self) -> None:
        self._stages: dict[str, dict] = {}
        self._last_hwm_kb = snapshot().vm_hwm_kb

    def mark(self, stage_name: str) -> None:
        snap = snapshot()
        growth = max(snap.vm_hwm_kb - self._last_hwm_kb, 0)
        self._last_hwm_kb = snap.vm_hwm_kb
        self._stages[stage_name] = {
            "rss_kb_at_end": snap.vm_rss_kb,
            "hwm_kb_at_end": snap.vm_hwm_kb,
            "hwm_growth_kb": growth,
        }

    def to_dict(self) -> dict[str, dict]:
        return dict(self._stages)
