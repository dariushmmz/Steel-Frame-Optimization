"""Layer 1 tests for canonical serialization and deterministic hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steel_frame_optimization.serialization import (
    canonical_json_dumps,
    sha256_config,
    sha256_file,
)

pytestmark = pytest.mark.unit


def test_canonical_json_is_order_independent_with_fixed_floats() -> None:
    """Equivalent mappings must yield identical bytes and documented float lexemes."""

    first = {"z": -0.0, "a": [1.0, 1.2e20, 0.000001]}
    second = {"a": [1.0, 1.2e20, 0.000001], "z": 0.0}
    expected = '{"a":[1.0,1.2e20,9.9999999999999995e-7],"z":0.0}\n'
    assert canonical_json_dumps(first) == expected
    assert canonical_json_dumps(second) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_floats(value: float) -> None:
    """Prevent non-standard float tokens from entering hashed data."""

    with pytest.raises(ValueError, match="rejects"):
        canonical_json_dumps({"value": value})


def test_canonical_json_rejects_non_string_mapping_keys() -> None:
    """Avoid language-dependent coercion of mapping keys."""

    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_dumps({1: "value"})


def test_canonical_output_is_valid_json() -> None:
    """Keep the custom deterministic representation standards-compliant."""

    rendered = canonical_json_dumps({"unicode": "فولاد", "number": 2.5})
    assert json.loads(rendered) == {"unicode": "فولاد", "number": 2.5}


def test_config_hash_is_deterministic() -> None:
    """Mapping insertion order must not change configuration identity."""

    assert sha256_config({"b": 2, "a": 1.0}) == sha256_config({"a": 1.0, "b": 2})


def test_file_hash_matches_known_sha256(tmp_path: Path) -> None:
    """Hash exact bytes without newline or encoding transformations."""

    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_lock_hash_matches_recorded_phase_two_dependency_value(project_root: Path) -> None:
    """Keep the lock identity stable after the reviewed Phase 2 xlrd additions."""

    assert sha256_file(project_root / "uv.lock") == (
        "5ee37da8381518fde8f0106f7470492328483ee2b6775d98b6ee680fc9c75671"
    )
