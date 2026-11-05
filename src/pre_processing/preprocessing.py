"""Shared YAML/schema selection and output handling for preprocessing stages."""
import argparse
from pathlib import Path

import yaml

from src.data_handling.data_loading import load_tables


def preprocess_tables(tables, schema, config, stage, transform):
    settings = config.get("preprocessing") or {}
    options = settings.get(stage) or {}
    excluded = {(table, column) for table, columns in (config.get("pii_columns") or {}).items()
                for column in columns}
    for table, columns in (config.get("primary_keys") or {}).items():
        excluded.update((table, column) for column in ([columns] if isinstance(columns, str) else columns))
    for table, spec in (config.get("tables") or {}).items():
        excluded.update((table, column) for column in (spec.get("id_cols") or []))
        excluded.update((table, column) for column in (spec.get("exclude_columns") or []))
    for table, columns in (config.get("exclude_from_training_columns") or {}).items():
        excluded.update((table, column) for column in columns)
    for relation in config.get("relationships") or []:
        excluded.add((relation.get("parent_table_name", relation.get("parent_table")),
                      relation.get("parent_primary_key", relation.get("parent_column"))))
        excluded.add((relation.get("child_table_name", relation.get("child_table")),
                      relation.get("child_foreign_key", relation.get("child_column"))))
    result = {name: df.copy() for name, df in tables.items()}
    schema_tables = schema.get("tables")
    if not isinstance(schema_tables, dict):
        raise ValueError("Detected schema must contain a tables mapping")
    for table, df in result.items():
        if df.columns.duplicated().any():
            raise ValueError(f"Duplicate columns in {table}")
        if table not in schema_tables:
            raise ValueError(f"No detected schema for table {table}; run dtype_detection first")
        for column in df.columns:
            if column not in schema_tables[table]:
                raise ValueError(f"No detected schema for {table}.{column}")
            info = schema_tables[table][column]
            role = (info.get("manual_override") or info.get("final_type")) if isinstance(info, dict) else info
            if role != stage or (table, column) in excluded:
                continue
            try:
                df[column] = transform(df[column], options)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{table}.{column}: {exc}") from exc
    return result


def run_stage(stage, transform):
    parser = argparse.ArgumentParser(description=f"Run {stage} preprocessing using YAML.")
    parser.add_argument("config", help="project YAML; relative paths use the current directory")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error("Project YAML must contain a mapping")
    settings = config.get("preprocessing") or {}
    source = settings.get("input_dir", config.get("data_dir"))
    output = settings.get("output_dir")
    schema_path = (config.get("dtype_detection") or {}).get("output_path")
    if not all(isinstance(value, str) and value.strip() for value in (source, output, schema_path)):
        parser.error("YAML requires data_dir (or preprocessing.input_dir), preprocessing.output_dir and dtype_detection.output_path")
    destination = Path(output) / stage
    source_path = Path(source).resolve()
    if destination.resolve() == source_path or source_path in destination.resolve().parents:
        parser.error("Preprocessing output must be outside its input folder")
    with open(schema_path, encoding="utf-8-sig") as stream:
        schema = yaml.safe_load(stream)
    if not isinstance(schema, dict):
        parser.error("Detected schema must contain a mapping")
    result = preprocess_tables(load_tables(source), schema, config, stage, transform)
    destination.mkdir(parents=True, exist_ok=True)
    for table, df in result.items():
        df.to_parquet(destination / f"{table}.parquet", index=False)
    print(f"Saved {len(result)} tables after {stage} preprocessing: {destination}")
