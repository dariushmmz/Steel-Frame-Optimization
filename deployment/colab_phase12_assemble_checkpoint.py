"""Assemble and verify the selected Phase 12 Colab recovery checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

EXPECTED_SHA256 = "5ca1b959d1f2a724f58c39d3a59188565bc4cb5ef2ddc482570041cf20ef03c3"
EXPECTED_BYTES = 66_628_525


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Concatenate seven transferred parts and verify the canonical bytes."""

    parts = sorted(Path("/content").glob("seed10.part.*"))
    if len(parts) != 7:
        raise RuntimeError(f"expected 7 checkpoint parts, found {len(parts)}")
    destination = Path("/content/case1_filtered_seed_10_source.json")
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    actual_sha256 = sha256_file(destination)
    if destination.stat().st_size != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError("assembled recovery checkpoint differs from its source")
    print(
        json.dumps(
            {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": actual_sha256,
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
