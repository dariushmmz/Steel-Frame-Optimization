"""Verify durable Google Drive storage for the Phase 12 Colab backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

PROBE_PAYLOAD = b"steel-frame-phase12-colab-drive-durability-v1\n"


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of an in-memory payload."""

    return hashlib.sha256(payload).hexdigest()


def write_probe(path: Path) -> dict[str, object]:
    """Write and independently reread the deterministic Drive probe."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(PROBE_PAYLOAD)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    persisted = path.read_bytes()
    return {
        "operation": "write_and_verify",
        "path": str(path),
        "bytes": len(persisted),
        "sha256": sha256_bytes(persisted),
        "expected_sha256": sha256_bytes(PROBE_PAYLOAD),
        "verified": persisted == PROBE_PAYLOAD,
    }


def read_probe(path: Path) -> dict[str, object]:
    """Reread an existing Drive probe without modifying it."""

    persisted = path.read_bytes()
    return {
        "operation": "read_only_verify",
        "path": str(path),
        "bytes": len(persisted),
        "sha256": sha256_bytes(persisted),
        "expected_sha256": sha256_bytes(PROBE_PAYLOAD),
        "verified": persisted == PROBE_PAYLOAD,
    }


def main() -> None:
    """Run the requested write or read-only Drive durability operation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        nargs="?",
        choices=("write", "read"),
        default=os.environ.get("PHASE12_DRIVE_PROBE_OPERATION", "read"),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(
            os.environ.get(
                "PHASE12_DRIVE_PROBE_PATH",
                "/content/drive/MyDrive/steel-frame-phase12-v1/probes/drive_probe.bin",
            )
        ),
    )
    # ``colab exec`` evaluates the file inside an existing Jupyter kernel whose
    # process arguments belong to the kernel launcher, not to this probe.
    arguments = parser.parse_args([])
    result = (
        write_probe(arguments.path)
        if arguments.operation == "write"
        else read_probe(arguments.path)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
