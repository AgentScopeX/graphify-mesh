"""Whole-transaction flock so concurrent graphify-mesh-sync runs cannot overlap."""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterator
from pathlib import Path


class LockHeldError(RuntimeError):
    """Raised when another run already holds the transaction lock."""


@contextlib.contextmanager
def transaction_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LockHeldError(f"another graphify-mesh-sync run holds {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()
