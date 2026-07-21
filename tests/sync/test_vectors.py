import numpy as np
import pytest

from graphify_mesh.sync.vectors import RepoVectors


def test_from_mapping_sorted_rows_and_dtype():
    rv = RepoVectors.from_mapping({"b": [1.0, 0.0], "a": [0.0, 1.0]})
    assert rv.keys == ["a", "b"]
    assert rv.matrix.dtype == np.float32
    assert rv.matrix.shape == (2, 2)
    assert rv.get("a") is not None
    assert rv.get("a")[1] == pytest.approx(1.0)
    assert rv.get("missing") is None
    assert len(rv) == 2
    assert rv.dim == 2


def test_from_mapping_skips_empty_and_mixed_dim():
    rv = RepoVectors.from_mapping({"a": [1.0, 0.0], "b": [], "c": [1.0, 2.0, 3.0]})
    assert rv.keys == ["a"]
    assert rv.matrix.shape == (1, 2)


def test_from_mapping_first_sorted_key_sets_reference_dim():
    # "a" is first in sorted-key order and has dim 2; "b" and "c" both have
    # dim 3 (a numeric majority) but must still be dropped, because the
    # reference dimension is the first non-empty vector in sorted-key order,
    # not a majority vote.
    rv = RepoVectors.from_mapping(
        {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0], "c": [4.0, 5.0, 6.0]}
    )
    assert rv.keys == ["a"]
    assert rv.matrix.shape == (1, 2)


def test_empty_container():
    rv = RepoVectors.empty()
    assert len(rv) == 0
    assert rv.get("x") is None
    assert rv.to_mapping() == {}


def test_from_mapping_accepts_ndarray_values():
    # Reuse path passes previous rows through as ndarrays (no per-key list
    # copy) — from_mapping must accept a mix of ndarray and list values.
    rv = RepoVectors.from_mapping(
        {"a": np.array([1.0, 2.0], dtype=np.float32), "b": [3.0, 4.0]}
    )
    assert rv.keys == ["a", "b"]
    assert rv.matrix.dtype == np.float32
    assert rv.matrix.shape == (2, 2)
    assert rv.get("a")[0] == pytest.approx(1.0)


def test_from_mapping_drops_empty_ndarray():
    rv = RepoVectors.from_mapping(
        {"a": [1.0, 2.0], "b": np.array([], dtype=np.float32)}
    )
    assert rv.keys == ["a"]


def test_to_mapping_roundtrip():
    source = {"a": [0.5, 0.25], "b": [1.0, 2.0]}
    mapping = RepoVectors.from_mapping(source).to_mapping()
    assert set(mapping) == {"a", "b"}
    assert mapping["a"] == pytest.approx(source["a"], abs=1e-6)
    assert all(isinstance(v, float) for v in mapping["a"])  # JSON-able


def test_normalized_caches_same_object():
    rv = RepoVectors.from_mapping({"a": [3.0, 4.0], "b": [1.0, 0.0]})
    first = rv.normalized()
    second = rv.normalized()
    assert first is second


def test_normalized_zero_row_stays_zero():
    rv = RepoVectors.from_mapping({"a": [3.0, 4.0]})
    # Overwrite with a genuine all-zero row post-construction (from_mapping
    # itself drops empty vectors, so this is the only way to get one into
    # the matrix for this test) — a trivial-skip placeholder row would look
    # exactly like this.
    rv.matrix[0] = np.array([0.0, 0.0], dtype=np.float32)
    normalized = rv.normalized()
    assert np.all(normalized[0] == 0.0)


def test_normalized_unit_length_for_nonzero_rows():
    rv = RepoVectors.from_mapping({"a": [3.0, 4.0]})
    normalized = rv.normalized()
    assert float(np.linalg.norm(normalized[0])) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# from_rows: shared canonicalization used by both the sync and server v2
# shard readers — sorted-key invariant enforced in exactly one place.
# ---------------------------------------------------------------------------


def test_from_rows_already_sorted_keeps_matrix_as_is_no_copy():
    matrix = np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32)
    rv = RepoVectors.from_rows(["a", "z"], matrix)
    assert rv.keys == ["a", "z"]
    assert rv.matrix is matrix  # no copy — mmap zero-copy path preserved
    assert list(rv.get("a")) == pytest.approx([1.0, 2.0])
    assert list(rv.get("z")) == pytest.approx([9.0, 9.0])


def test_from_rows_permuted_reorders_keys_and_matrix_together():
    # row 0 -> "z", row 1 -> "a" — reverse of sorted order.
    matrix = np.array([[9.0, 9.0], [1.0, 2.0]], dtype=np.float32)
    rv = RepoVectors.from_rows(["z", "a"], matrix)
    assert rv.keys == ["a", "z"]  # canonicalized to sorted order
    assert list(rv.get("a")) == pytest.approx([1.0, 2.0])
    assert list(rv.get("z")) == pytest.approx([9.0, 9.0])


def test_from_rows_preserves_mmap_type_on_sorted_fast_path(tmp_path):
    matrix_path = tmp_path / "m.npy"
    np.save(matrix_path, np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32))
    mmap_matrix = np.load(matrix_path, mmap_mode="r")

    rv = RepoVectors.from_rows(["a", "z"], mmap_matrix)
    assert isinstance(rv.matrix, np.memmap)


# ---------------------------------------------------------------------------
# from_mapping: defense-in-depth against non-1D / non-numeric values —
# every caller (not just server/store.py's v1 loader) benefits.
# ---------------------------------------------------------------------------


def test_from_mapping_skips_scalar_string_nested_and_dict_values(caplog):
    with caplog.at_level("WARNING"):
        rv = RepoVectors.from_mapping(
            {
                "good_a": [1.0, 2.0],
                "scalar": 5.0,
                "stringy": "not-a-vector",
                "nested": [[1.0, 2.0], [3.0, 4.0]],
                "mapping": {"x": 1.0},
                "good_b": [3.0, 4.0],
            }
        )
    assert rv.keys == ["good_a", "good_b"]
    assert rv.matrix.shape == (2, 2)
    assert list(rv.get("good_a")) == pytest.approx([1.0, 2.0])
    assert list(rv.get("good_b")) == pytest.approx([3.0, 4.0])
    assert rv.get("scalar") is None
    assert rv.get("stringy") is None
    assert rv.get("nested") is None
    assert rv.get("mapping") is None
    assert len(caplog.records) >= 4


def test_from_mapping_skips_overflowing_huge_int(caplog):
    huge_int = 10**400
    with caplog.at_level("WARNING"):
        rv = RepoVectors.from_mapping(
            {
                "good_a": [1.0, 2.0],
                "huge": [huge_int],
                "good_b": [3.0, 4.0],
            }
        )
    assert rv.keys == ["good_a", "good_b"]
    assert rv.matrix.shape == (2, 2)
    assert list(rv.get("good_a")) == pytest.approx([1.0, 2.0])
    assert list(rv.get("good_b")) == pytest.approx([3.0, 4.0])
    assert rv.get("huge") is None
    assert len(caplog.records) >= 1
