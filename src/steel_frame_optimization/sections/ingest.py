"""Convert the approved local AISC v14.1 workbook into deterministic SI section records.

The raw workbook remains ignored and local. This module emits only the project-required W-shape
fields plus provenance, using explicit command arguments rather than hidden paths or timestamps.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import xlrd

from steel_frame_optimization.sections.records import (
    SectionDataset,
    SectionRecord,
    SectionSource,
)
from steel_frame_optimization.serialization import sha256_file, write_canonical_json

INCH_TO_M: Final = 0.0254
FOOT_TO_M: Final = 0.3048
POUND_MASS_TO_KG: Final = 0.45359237
POUND_FORCE_TO_N: Final = 4.4482216152605


class SectionIngestionError(ValueError):
    """Indicate an invalid source workbook, missing field, or provenance mismatch."""


def _positive_number(sheet: Any, row_index: int, column_index: int, field: str) -> float:
    """Read one required finite positive numeric cell with a source-row error message."""

    value = sheet.cell_value(row_index, column_index)
    if not isinstance(value, int | float):
        raise SectionIngestionError(f"row {row_index + 1} field {field} is not numeric")
    result = float(value)
    if not result > 0.0:
        raise SectionIngestionError(f"row {row_index + 1} field {field} must be positive")
    return result


def _imperial_column_indices(headers: list[str]) -> dict[str, int]:
    """Resolve the first, U.S.-customary copy of every required workbook column."""

    required = (
        "Type",
        "EDI_Std_Nomenclature",
        "W",
        "A",
        "d",
        "bf",
        "tw",
        "tf",
        "bf/2tf",
        "h/tw",
        "Ix",
        "Zx",
        "Sx",
        "rx",
        "Iy",
        "Zy",
        "Sy",
        "ry",
        "J",
        "Cw",
        "rts",
        "ho",
    )
    indices: dict[str, int] = {}
    for field in required:
        try:
            indices[field] = headers.index(field)
        except ValueError as error:
            raise SectionIngestionError(f"missing required workbook column: {field}") from error
    return indices


def _record_from_row(sheet: Any, row_index: int, columns: dict[str, int]) -> SectionRecord:
    """Convert one source W row from inches/pounds to an immutable canonical-SI record."""

    def number(field: str) -> float:
        return _positive_number(sheet, row_index, columns[field], field)

    inch2_to_m2 = INCH_TO_M**2
    inch3_to_m3 = INCH_TO_M**3
    inch4_to_m4 = INCH_TO_M**4
    inch6_to_m6 = INCH_TO_M**6
    source_weight = number("W")
    name = str(sheet.cell_value(row_index, columns["EDI_Std_Nomenclature"])).strip().upper()
    return SectionRecord(
        source_row_id=row_index + 1,
        name=name,
        source_weight_lb_per_ft=source_weight,
        mass_kg_per_m=source_weight * POUND_MASS_TO_KG / FOOT_TO_M,
        weight_n_per_m=source_weight * POUND_FORCE_TO_N / FOOT_TO_M,
        area_m2=number("A") * inch2_to_m2,
        depth_m=number("d") * INCH_TO_M,
        flange_width_m=number("bf") * INCH_TO_M,
        web_thickness_m=number("tw") * INCH_TO_M,
        flange_thickness_m=number("tf") * INCH_TO_M,
        ix_m4=number("Ix") * inch4_to_m4,
        zx_m3=number("Zx") * inch3_to_m3,
        sx_m3=number("Sx") * inch3_to_m3,
        rx_m=number("rx") * INCH_TO_M,
        iy_m4=number("Iy") * inch4_to_m4,
        zy_m3=number("Zy") * inch3_to_m3,
        sy_m3=number("Sy") * inch3_to_m3,
        ry_m=number("ry") * INCH_TO_M,
        torsional_j_m4=number("J") * inch4_to_m4,
        warping_cw_m6=number("Cw") * inch6_to_m6,
        rts_m=number("rts") * INCH_TO_M,
        flange_centroid_distance_ho_m=number("ho") * INCH_TO_M,
        flange_slenderness=number("bf/2tf"),
        web_slenderness=number("h/tw"),
    )


def build_section_dataset(
    workbook_path: Path,
    *,
    expected_sha256: str,
    retrieval_url: str,
    retrieved_at_utc: str,
    authority_limitation: str,
) -> SectionDataset:
    """Build the exact DEV-001 W dataset after verifying the raw workbook identity."""

    actual_sha256 = sha256_file(workbook_path)
    if actual_sha256 != expected_sha256:
        raise SectionIngestionError(
            f"workbook SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    workbook = xlrd.open_workbook(str(workbook_path), on_demand=True)
    try:
        sheet = workbook.sheet_by_name("Database v14.1")
    except xlrd.biffh.XLRDError as error:
        raise SectionIngestionError("workbook is missing sheet 'Database v14.1'") from error
    headers = [str(sheet.cell_value(0, column)).strip() for column in range(sheet.ncols)]
    columns = _imperial_column_indices(headers)
    records = tuple(
        _record_from_row(sheet, row_index, columns)
        for row_index in range(1, sheet.nrows)
        if str(sheet.cell_value(row_index, columns["Type"])).strip().upper() == "W"
    )
    return SectionDataset(
        domain_label="273-section domain, pending client confirmation",
        source=SectionSource(
            filename=workbook_path.name,
            sha256=actual_sha256,
            retrieval_url=retrieval_url,
            retrieved_at_utc=retrieved_at_utc,
            authority_limitation=authority_limitation,
        ),
        records=records,
    )


def _parser() -> argparse.ArgumentParser:
    """Create the deterministic ingestion command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument("--retrieved-at-utc", required=True)
    parser.add_argument("--authority-limitation", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic workbook ingestion and write canonical JSON."""

    arguments = _parser().parse_args(argv)
    dataset = build_section_dataset(
        arguments.workbook,
        expected_sha256=arguments.expected_sha256,
        retrieval_url=arguments.retrieval_url,
        retrieved_at_utc=arguments.retrieved_at_utc,
        authority_limitation=arguments.authority_limitation,
    )
    write_canonical_json(arguments.output, dataset)
    print(
        f"wrote {len(dataset.records)} sections ({dataset.w14_section_count} W14) "
        f"to {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
