
"""Generic data validation for Pandas tables."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_loading import load_tables


def read_yaml(path):
    if not path:
        return {}

    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def missing_mask(series):
    """Find NULL values and blank strings."""
    mask = series.isna()

    if pd.api.types.is_object_dtype(series) or (
        pd.api.types.is_string_dtype(series)
    ):
        mask = mask | series.astype("string").str.strip().eq(
            ""
        ).fillna(False)

    return mask


def invalid_type_mask(series, role):
    """Check values against the expected semantic datatype."""
    role = str(role).lower()
    missing = missing_mask(series)
    values = series[~missing]

    if values.empty:
        return pd.Series(False, index=series.index)

    if role in ("numerical", "numeric", "integer", "float"):
        valid = pd.to_numeric(
            values, errors="coerce"
        ).notna()

    elif role in ("datetime", "date", "timestamp"):
        try:
            parsed = pd.to_datetime(
                values.astype("string"),
                errors="coerce",
                format="mixed",
            )
        except (TypeError, ValueError):
            parsed = pd.to_datetime(
                values.astype("string"),
                errors="coerce",
            )
        valid = parsed.notna()

    elif role in ("boolean", "bool"):
        allowed = {
            "true", "false", "yes", "no",
            "y", "n", "1", "0", "t", "f"
        }
        valid = values.astype("string").str.lower().str.strip().isin(
            allowed
        )

    else:
        # Other roles need no generic value conversion.
        return pd.Series(False, index=series.index)

    invalid = pd.Series(False, index=series.index)
    invalid.loc[values.index] = ~valid.fillna(False)
    return invalid


def validate_tables(tables, metadata=None, config=None):
    """Return issues without modifying any DataFrame."""
    metadata = metadata or {}
    config = config or {}
    issues = []

    def report(table, column, check, count, message):
        issues.append({
            "table": table,
            "column": column,
            "check": check,
            "count": int(count),
            "message": message,
        })

    # 1. General table and column checks
    for table_name, df in tables.items():
        if df.empty:
            report(
                table_name, "", "empty_table", 1,
                "Table has no rows"
            )
            continue

        duplicates = df.columns[df.columns.duplicated()].tolist()

        for column in set(duplicates):
            report(
                table_name, column, "duplicate_column", 1,
                "Column name appears more than once"
            )

        table_metadata = metadata.get(
            "tables", {}
        ).get(table_name, {})

        for column in df.columns.unique():
            if column not in df.columns:
                continue

            # Avoid ambiguous duplicate-name columns
            if list(df.columns).count(column) > 1:
                continue

            series = df[column]
            missing = missing_mask(series)
            null_count = int(missing.sum())

            if null_count:
                report(
                    table_name, column, "missing_values",
                    null_count, "NULL or blank values found"
                )

            info = table_metadata.get(column, {})
            role = (
                info.get("manual_override")
                or info.get("final_type")
                or info.get("detected_type")
            ) if isinstance(info, dict) else info

            if not role:
                continue

            invalid = invalid_type_mask(series, role)

            if invalid.any():
                examples = (
                    series[invalid]
                    .astype(str)
                    .drop_duplicates()
                    .head(3)
                    .tolist()
                )

                report(
                    table_name, column, "datatype",
                    invalid.sum(),
                    f"Expected {role}; invalid examples: {examples}"
                )

    # 2. Explicit primary key checks
    for table_name, key_columns in config.get(
        "primary_keys", {}
    ).items():
        if table_name not in tables:
            report(
                table_name, "", "missing_table", 1,
                "Declared PK table not loaded"
            )
            continue

        df = tables[table_name]
        if isinstance(key_columns, str):
            key_columns = [key_columns]

        if not all(c in df.columns for c in key_columns):
            report(
                table_name, "", "missing_pk_column", 1,
                f"Missing PK columns: "
                f"{[c for c in key_columns if c not in df.columns]}"
            )
            continue

        if df.columns.duplicated().any():
            report(
                table_name, "", "pk_not_checked", 1,
                "Resolve duplicate column names first"
            )
            continue

        missing = pd.concat(
            [missing_mask(df[c]) for c in key_columns],
            axis=1
        ).any(axis=1)

        if missing.any():
            report(
                table_name, ",".join(key_columns), "pk_null",
                missing.sum(), "Primary key contains NULLs"
            )

        duplicated = df.loc[
            ~missing, key_columns
        ].duplicated(keep=False)

        if duplicated.any():
            report(
                table_name, ",".join(key_columns),
                "pk_duplicate", duplicated.sum(),
                "Duplicate primary key values"
            )

    # 3. Explicit foreign key checks
    for relation in config.get("relationships", []):
        parent = relation.get("parent_table_name", relation.get("parent_table"))
        parent_col = relation.get("parent_primary_key", relation.get("parent_column"))
        child = relation.get("child_table_name", relation.get("child_table"))
        child_col = relation.get("child_foreign_key", relation.get("child_column"))
        if not all(isinstance(value, str) and value.strip() for value in (parent, parent_col, child, child_col)):
            raise ValueError("Each relationship must specify parent/child table names and key columns")

        if (
            parent not in tables
            or child not in tables
            or parent_col not in tables[parent].columns
            or child_col not in tables[child].columns
        ):
            report(
                child, child_col, "fk_configuration", 1,
                "Required table or FK/PK column is missing"
            )
            continue

        parent_df = tables[parent]
        child_df = tables[child]

        if (
            parent_df.columns.duplicated().any()
            or child_df.columns.duplicated().any()
        ):
            report(
                child, child_col, "fk_not_checked", 1,
                "Resolve duplicate column names first"
            )
            continue

        parent_values = parent_df[parent_col]
        child_values = child_df[child_col]

        parent_keys = set(
            parent_values[~missing_mask(parent_values)]
            .astype("string").str.strip()
        )

        child_non_null = child_values[
            ~missing_mask(child_values)
        ].astype("string").str.strip()

        orphans = ~child_non_null.isin(parent_keys)

        if orphans.any():
            report(
                child, child_col, "orphan_fk",
                orphans.sum(),
                f"Foreign key values missing from "
                f"{parent}.{parent_col}"
            )

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Data folder or YAML")
    parser.add_argument(
        "--metadata", help="Detected schema YAML", default=None
    )
    parser.add_argument(
        "--rules", help="PK/FK validation YAML", default=None
    )
    args = parser.parse_args()

    source = Path(args.source)
    data_dir = args.source
    source_config = {}

    if source.is_file() and source.suffix.lower() in (".yaml", ".yml"):
        source_config = read_yaml(source)
        if not isinstance(source_config, dict) or not isinstance(source_config.get("data_dir"), str) or not source_config["data_dir"].strip():
            parser.error("YAML must contain a non-empty data_dir string")
        data_dir = source_config["data_dir"]

    tables = load_tables(data_dir)

    metadata_path = args.metadata or (source_config.get("dtype_detection") or {}).get("output_path")
    metadata = read_yaml(metadata_path)
    rules = read_yaml(args.rules) if args.rules else source_config

    issues = validate_tables(tables, metadata, rules)

    print("\n=== DATASET SUMMARY ===")
    for name, df in tables.items():
        print(
            f"{name}: {len(df)} rows, {len(df.columns)} columns"
        )

    print("\n=== VALIDATION RESULTS ===")
    if not issues:
        print("No issues found by enabled checks.")
    else:
        print(pd.DataFrame(issues).to_string(index=False))
        print(f"\nTotal reported issues: {len(issues)}")


if __name__ == "__main__":
    main()
