"""Define canonical JSON and SHA-256 identity rules for reproducible artifacts.

Python's normal JSON encoder is deterministic for key order when configured, but it does not expose
a public fixed float formatter. This module therefore encodes the small JSON data model directly and
uses exactly 17 significant decimal digits for finite IEEE-754 binary64 values.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

CANONICAL_FLOAT_SIGNIFICANT_DIGITS = 17
HASH_CHUNK_SIZE_BYTES = 1024 * 1024


def canonical_float(value: float) -> str:
    """Return one finite float using the project's fixed 17-digit JSON number rule."""

    if not math.isfinite(value):
        raise ValueError("canonical JSON rejects NaN and infinite floats")
    if value == 0.0:
        return "0.0"
    rendered = format(value, f".{CANONICAL_FLOAT_SIGNIFICANT_DIGITS}g").lower()
    if "e" in rendered:
        mantissa, exponent = rendered.split("e", maxsplit=1)
        return f"{mantissa}e{int(exponent)}"
    if "." not in rendered:
        return f"{rendered}.0"
    return rendered


def canonical_json_dumps(value: Any) -> str:
    """Encode supported JSON data with sorted keys, fixed floats, and one final newline."""

    def encode(item: Any) -> str:
        if isinstance(item, BaseModel):
            return encode(item.model_dump(mode="json"))
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, int):
            return str(item)
        if isinstance(item, float):
            return canonical_float(item)
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("canonical JSON mapping keys must be strings")
            fields = [f"{encode(key)}:{encode(item[key])}" for key in sorted(item)]
            return "{" + ",".join(fields) + "}"
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return "[" + ",".join(encode(element) for element in item) + "]"
        raise TypeError(f"unsupported canonical JSON type: {type(item).__name__}")

    return encode(value) + "\n"


def write_canonical_json(path: Path, value: Any) -> None:
    """Write canonical UTF-8 JSON to the requested path without implicit timestamps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_dumps(value), encoding="utf-8", newline="\n")


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks without newline or encoding transformations."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_config(value: Any) -> str:
    """Hash configuration data only after canonical JSON serialization."""

    return sha256_bytes(canonical_json_dumps(value).encode("utf-8"))
