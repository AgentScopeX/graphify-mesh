"""Float32 numpy container for per-repo embedding vectors.

Replaces the pure-Python ``dict[str, list[float]]`` representation with a
sorted-key matrix so downstream code (sync shard build, overlay ANN
similarity, server query scoring) can operate on contiguous float32 rows
instead of Python-level float lists.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


def _is_empty_vector(vector: list[float] | np.ndarray) -> bool:
    """`bool(ndarray)` raises for arrays with more than one element, so a
    plain `not vector` truthiness check (fine for lists) cannot be reused
    here — check `.size` for ndarrays, length/truthiness for everything
    else."""
    if isinstance(vector, np.ndarray):
        return vector.size == 0
    return not vector


class _NormalizedRows:
    """Lazy L2-normalized view over a ``RepoVectors`` matrix.

    Replaces the eagerly-materialized normalized float32 copy ``normalized()``
    used to build and cache: that copy pulled every (possibly mmap'd) shard
    matrix fully into RAM for the process lifetime, defeating the mmap
    read path entirely (the server is one process per client session). This
    view keeps the backing matrix untouched — mmap'd or not — and holds only
    the per-row safe-norms vector (n float32), computing normalized values
    on demand for the two operations callers actually perform:

      * ``view @ query`` (server scoring, retrieval.py) —
        ``(matrix @ query) / norms``, one (n,) allocation;
      * ``view[row]`` / ``view[slice]`` (bucket gathering,
        embed_similarity.py) — divides just the requested rows.

    ``__array__`` exists purely as a compatibility net for any consumer that
    insists on a full ndarray — it materializes the complete normalized
    matrix, so nothing on the hot paths should hit it.
    """

    def __init__(self, matrix: np.ndarray, safe_norms: np.ndarray) -> None:
        self.matrix = matrix
        # shape (n,), float32; zero-norm rows carry 1.0 so all-zero vectors
        # stay all-zero instead of producing NaN/inf (same rule as before).
        self.safe_norms = safe_norms

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return (self.matrix @ other) / self.safe_norms

    def __getitem__(self, index):
        rows = self.matrix[index]
        if rows.ndim == 1:
            return rows / self.safe_norms[index]
        return rows / self.safe_norms[index][:, np.newaxis]

    def __len__(self) -> int:
        return self.matrix.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    def __array__(self, dtype=None) -> np.ndarray:
        full = self.matrix / self.safe_norms[:, np.newaxis]
        if dtype is None:
            return full.astype(np.float32, copy=False)
        return full.astype(dtype, copy=False)


@dataclass
class RepoVectors:
    keys: list[str]  # sorted; row i of matrix belongs to keys[i]
    matrix: np.ndarray  # float32, shape (len(keys), dim); may be a read-only mmap
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _normalized: _NormalizedRows | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._index:
            return
        self._index = {key: i for i, key in enumerate(self.keys)}

    @classmethod
    def from_mapping(cls, vectors: Mapping[str, list[float] | np.ndarray]) -> RepoVectors:
        """Build a RepoVectors from a ``{key: [float, ...]}`` mapping.

        Values may be plain float lists or 1-D numpy arrays (e.g. rows
        passed straight through from a previous ``RepoVectors.matrix`` on
        the reuse path) — both are accepted without an intermediate
        Python-list copy.

        Reference dimension is taken from the first non-empty vector in
        sorted-key order (deterministic); vectors of any other length are
        dropped with a warning — mixed dimensions only occur when a shard
        was embedded under two different models, and a dropped vector would
        score zero everywhere anyway (see cosine_similarity's mixed-dim
        rule).

        Defense-in-depth (this is the single choke point every caller goes
        through, not just the currently-known ones): a value that is not a
        non-string 1-D numeric sequence — a scalar, a string, a nested
        (2-D-producing) list, a dict, anything `np.asarray(..., dtype=
        float32)` cannot coerce — is dropped with the same mixed-dim-style
        warning rather than being allowed to reach `np.asarray(kept_rows,
        ...)` below, where it would either raise or silently produce a
        higher-rank matrix.
        """
        sorted_keys = sorted(vectors.keys())

        expected_dim: int | None = None
        kept_keys: list[str] = []
        kept_rows: list[object] = []

        for key in sorted_keys:
            vector = vectors[key]
            try:
                candidate = np.asarray(vector, dtype=np.float32)
            except (ValueError, TypeError, OverflowError):
                log.warning(
                    "RepoVectors.from_mapping: dropping key %r with non-numeric value (type %s)",
                    key,
                    type(vector).__name__,
                )
                continue
            if candidate.ndim != 1:
                log.warning(
                    "RepoVectors.from_mapping: dropping key %r with non-1D value "
                    "(ndim=%d, expected 1)",
                    key,
                    candidate.ndim,
                )
                continue
            if _is_empty_vector(candidate):
                continue
            if expected_dim is None:
                expected_dim = candidate.shape[0]
            if candidate.shape[0] != expected_dim:
                log.warning(
                    "RepoVectors.from_mapping: dropping key %r with mismatched "
                    "dimension %d (expected %d)",
                    key,
                    candidate.shape[0],
                    expected_dim,
                )
                continue
            kept_keys.append(key)
            kept_rows.append(vector)

        if not kept_keys:
            return cls.empty(dim=expected_dim or 0)

        matrix = np.asarray(kept_rows, dtype=np.float32)
        return cls(keys=kept_keys, matrix=matrix)

    @classmethod
    def from_rows(cls, keys_by_row: list[str], matrix: np.ndarray) -> RepoVectors:
        """Builds a RepoVectors from on-disk row order, canonicalizing to the
        sorted-key invariant (`keys` sorted; row i belongs to `keys[i]`).

        Shared by both v2 shard readers (sync's `_read_v2_shard` and the
        server's `_load_v2_shard_vectors`) so the canonicalization logic
        exists exactly once — callers run their own validation (shape,
        dtype, row-range, duplicate-row checks) BEFORE calling this.

        - Already-sorted `keys_by_row` (the canonical case: our own writer,
          `stage_embeddings`, always emits rows in sorted-key order): use
          `matrix` as-is — no copy — preserving a read-only mmap.
        - Out-of-order `keys_by_row` (a hand-edited or otherwise
          out-of-band-written shard): argsort-permute keys and matrix rows
          together. Fancy indexing materializes a full copy of the matrix
          here — the price of correctness for the rare non-canonical shard.
        """
        if keys_by_row == sorted(keys_by_row):
            return cls(keys=keys_by_row, matrix=matrix)

        order = np.argsort(np.array(keys_by_row))
        sorted_keys = [keys_by_row[i] for i in order]
        return cls(keys=sorted_keys, matrix=matrix[order])

    @classmethod
    def empty(cls, dim: int = 0) -> RepoVectors:
        return cls(keys=[], matrix=np.zeros((0, dim), dtype=np.float32))

    def get(self, key: str) -> np.ndarray | None:
        row_index = self._index.get(key)
        if row_index is None:
            return None
        return self.matrix[row_index]

    def __len__(self) -> int:
        return len(self.keys)

    def to_mapping(self) -> dict[str, list[float]]:
        return {key: [float(x) for x in self.matrix[i]] for i, key in enumerate(self.keys)}

    @property
    def dim(self) -> int:
        if self.matrix.ndim != 2:
            return 0
        return self.matrix.shape[1]

    def normalized(self) -> _NormalizedRows:
        """L2-normalized view of ``matrix``: row i behaves as ``matrix[i]``
        divided by its own L2 norm. Rows with zero norm (all-zero vectors —
        e.g. a trivial-skip placeholder) are never divided by zero; they stay
        all-zero instead of producing NaN/inf.

        Unlike the previous implementation this no longer materializes a
        full float32 copy of the matrix (which defeated the mmap'd-shard
        read path for the process lifetime): only the per-row norms vector
        (n float32) is computed, and the returned `_NormalizedRows` view
        divides on demand (`view @ query` for scoring, `view[row]` for
        bucket gathering). Built once and cached on this instance
        (`is`-stable across calls), so per-query/per-bucket work never
        recomputes norms."""
        if self._normalized is not None:
            return self._normalized
        norms = np.linalg.norm(self.matrix, axis=1)
        safe_norms = np.where(norms == 0.0, 1.0, norms).astype(np.float32)
        self._normalized = _NormalizedRows(self.matrix, safe_norms)
        return self._normalized
