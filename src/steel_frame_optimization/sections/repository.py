"""Load and query immutable section datasets without implicit global state."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from steel_frame_optimization.sections.records import SectionDataset, SectionRecord


class SectionRepositoryError(ValueError):
    """Indicate an invalid lookup, duplicate input, or unsupported section family."""


class SectionRepository:
    """Provide explicit source, operator, name, and family views over one dataset."""

    def __init__(self, dataset: SectionDataset) -> None:
        """Index one validated dataset while retaining all ordering as instance state."""

        self._dataset = dataset
        self._by_name = {record.name: record for record in dataset.records}

    @classmethod
    def from_json(cls, path: Path) -> SectionRepository:
        """Load a canonical dataset file and revalidate every persisted field."""

        content = json.loads(path.read_text(encoding="utf-8"))
        return cls(SectionDataset.model_validate(content))

    @property
    def dataset(self) -> SectionDataset:
        """Return the immutable validated dataset backing this repository."""

        return self._dataset

    def get(self, name: str) -> SectionRecord:
        """Return one section by canonical case-insensitive name."""

        canonical_name = name.upper()
        try:
            return self._by_name[canonical_name]
        except KeyError as error:
            raise SectionRepositoryError(f"unknown section: {name}") from error

    def source_order(self) -> tuple[SectionRecord, ...]:
        """Return records in immutable workbook-row order for provenance and serialization."""

        return self._dataset.records

    def operator_order(
        self, records: Iterable[SectionRecord] | None = None
    ) -> tuple[SectionRecord, ...]:
        """Order mutation/crossover neighborhoods by weight, breaking ties by source row."""

        selected = self._dataset.records if records is None else tuple(records)
        return tuple(
            sorted(selected, key=lambda record: (record.weight_n_per_m, record.source_row_id))
        )

    def family(self, family: str) -> tuple[SectionRecord, ...]:
        """Return the all-W or W14 base family in operator order."""

        if family == "all_w":
            selected = self._dataset.records
        elif family == "w14":
            selected = tuple(
                record for record in self._dataset.records if record.name.startswith("W14X")
            )
        else:
            raise SectionRepositoryError(f"unsupported section family: {family}")
        return self.operator_order(selected)
