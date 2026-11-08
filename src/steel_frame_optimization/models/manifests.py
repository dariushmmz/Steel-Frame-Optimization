"""Define immutable, strict manifests for artifacts, seeds, version control, and runtimes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictFrozenModel(BaseModel):
    """Reject unknown fields and mutation so persisted identity data cannot drift silently."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactFile(StrictFrozenModel):
    """Identify one immutable file by portable relative path, byte count, and SHA-256."""

    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactManifest(StrictFrozenModel):
    """Describe the immutable files and input identities belonging to one declared run."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[ArtifactFile, ...]


class SeedManifest(StrictFrozenModel):
    """Persist explicit random seeds and the delegated status of their policy."""

    schema_version: Literal["1.0"] = "1.0"
    generator: Literal["numpy.random.PCG64"]
    decision_status: Literal["agent_decided_pending_ratification_DEC-012"]
    purpose: str = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_unique_nonnegative_seeds(self) -> SeedManifest:
        """Reject ambiguous duplicates and negative seed values at the configuration boundary."""

        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        return self


class VcsProvenance(StrictFrozenModel):
    """Record the Git identity observed when provenance collection starts."""

    kind: Literal["git", "none"]
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    dirty: bool | None = None

    @model_validator(mode="after")
    def ensure_consistent_vcs_fields(self) -> VcsProvenance:
        """Require commit and dirtiness for Git and forbid them when no VCS exists."""

        if self.kind == "git" and (self.commit is None or self.dirty is None):
            raise ValueError("Git provenance requires commit and dirty fields")
        if self.kind == "none" and (self.commit is not None or self.dirty is not None):
            raise ValueError("No-VCS provenance cannot contain Git fields")
        return self


class RuntimeProvenance(StrictFrozenModel):
    """Capture reproducibility facts while isolating declared variable run metadata."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    uv_version: str
    package_versions: dict[str, str]
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vcs: VcsProvenance

    def comparable_payload(self) -> dict[str, Any]:
        """Return fields that must match across runs in an unchanged environment."""

        return self.model_dump(exclude={"run_id", "created_at_utc"}, mode="json")
