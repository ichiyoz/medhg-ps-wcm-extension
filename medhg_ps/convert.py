"""Regenerate the A1..A5 + bulk-features parquet files from the headerless
CSV batch.

Cogito exports each table WITHOUT a header. We read the CSV header-agnostic
(names come from the SQL-derived *_COLUMNS schema in config, applied
positionally; data._read_table also coerces ID columns to str), then write
parquet -- which bakes the column names and dtypes into the file schema.

CSV is lossless when headerless, so no row is dropped in the process; the
resulting parquet carries a proper header for fast, typed reloads.

    python -m medhg_ps.convert            # convert every table in DATA_DIR
"""
from __future__ import annotations

from pathlib import Path

from . import config as C
from . import data as D

# (target parquet path, schema). Source CSV is the same stem with .csv.
# ENC_FEATURES_CSV / LABELS_PARQUET point at the same bulk file, so it is
# listed once.
_TABLES = [
    (C.ENCOUNTERS_PARQUET, C.A1_ENCOUNTERS_COLUMNS),
    (C.PROV_EDGES_PARQUET, C.A2_PROV_EDGES_COLUMNS),
    (C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS),
    (C.PROV_ATTRS_PARQUET, C.A4_PROV_ATTRS_COLUMNS),
    (C.UNIT_ATTRS_PARQUET, C.A5_UNIT_ATTRS_COLUMNS),
    (C.ENC_FEATURES_CSV, C.BULK_FEATURES_COLUMNS),
]


def main() -> None:
    seen: set[Path] = set()
    for path, schema in _TABLES:
        parquet = Path(path).with_suffix(".parquet")
        if parquet in seen:
            continue
        seen.add(parquet)

        csv = parquet.with_suffix(".csv")
        if not csv.exists():
            print(f"[skip] {csv.name} not found")
            continue

        df = D._read_table(csv, schema=schema)   # header-agnostic + typed IDs
        df.to_parquet(parquet)                    # header + dtypes baked in
        print(f"[ok]   {csv.name:32s} -> {parquet.name:32s} {df.shape}")


if __name__ == "__main__":
    main()
