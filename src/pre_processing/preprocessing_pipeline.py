"""Split related tables, fit preprocessing on training rows, and save both partitions."""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.pre_processing import boolean, categorical, date_handling as datehandling, numericalpreprocessing as numerical
from src.data_handling.data_loading import load_tables
from src.data_handling.data_splitting import save_splits, split_tables


def _date_numbers(series, options):
    dates = datehandling.preprocess_column(series, options)
    return dates.map(lambda value: float("nan") if pd.isna(value) else value.value / 1e9).astype("float64")


def fit_preprocessing(train, schema, config):
    """Return portable YAML state, including feature lists and training-only statistics."""
    settings = config.get("preprocessing") or {}
    state = {"tables": {}}
    keys = {(table, column) for table, columns in (config.get("primary_keys") or {}).items()
            for column in ([columns] if isinstance(columns, str) else columns)}
    for relation in config.get("relationships") or []:
        keys.add((relation.get("parent_table_name", relation.get("parent_table")),
                  relation.get("parent_primary_key", relation.get("parent_column"))))
        keys.add((relation.get("child_table_name", relation.get("child_table")),
                  relation.get("child_foreign_key", relation.get("child_column"))))
    for table, df in train.items():
        spec = (config.get("tables") or {}).get(table) or {}
        pii = set((config.get("pii_columns") or {}).get(table) or {})
        excluded = set(spec.get("exclude_columns") or [])
        excluded.update((config.get("exclude_from_training_columns") or {}).get(table) or [])
        declared_ids = set(spec.get("id_cols") or [])
        table_state = {"features": [], "keys": [], "drop_columns": [], "columns": {}}
        for column in df.columns:
            if table not in schema.get("tables", {}) or column not in schema["tables"][table]:
                raise ValueError(f"Missing detected schema: {table}.{column}")
            info = schema["tables"][table][column]
            role = (info.get("manual_override") or info.get("final_type")) if isinstance(info, dict) else info
            is_key = (table, column) in keys or column in declared_ids or role == "id"
            if is_key:
                if column in pii or column in excluded:
                    raise ValueError(f"Key is also excluded or declared PII: {table}.{column}")
                table_state["keys"].append(column)
                continue
            if column in pii or column in excluded:
                table_state["drop_columns"].append(column)
                continue
            if role not in ("numerical", "categorical", "boolean", "datetime"):
                raise ValueError(f"Unsupported feature type {role}: {table}.{column}; override or exclude it")
            options = dict(settings.get(role) or {})
            try:
                if role == "categorical":
                    learned = categorical.fit_column(df[column], options)
                elif role == "boolean":
                    values = boolean.preprocess_column(df[column], {**options, "fill_missing": None})
                    fill = options.get("fill_missing")
                    if fill is None:
                        if values.dropna().empty:
                            raise ValueError("All training booleans are missing; configure fill_missing")
                        fill = int(values.mode().iloc[0])
                    if fill not in (0, 1):
                        raise ValueError("Boolean fill_missing must be 0 or 1")
                    learned = {"fill_value": int(fill)}
                else:
                    values = _date_numbers(df[column], options) if role == "datetime" else df[column]
                    numeric_options = {**options, "errors": "raise"} if role == "datetime" else options
                    learned = numerical.fit_column(values, numeric_options)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{table}.{column}: {exc}") from exc
            table_state["features"].append(column)
            table_state["columns"][column] = {"role": role, "options": options, "learned": learned}
        state["tables"][table] = table_state
    return state


def transform_tables(tables, state):
    """Apply saved statistics; never learn anything from these rows."""
    result = {}
    for table, spec in state["tables"].items():
        df = tables[table]
        output = df[spec["keys"] + spec["features"]].copy()
        for column, column_state in spec["columns"].items():
            role, options, learned = (column_state[key] for key in ("role", "options", "learned"))
            try:
                if role == "categorical":
                    values = categorical.transform_column(df[column], options, learned)
                elif role == "boolean":
                    values = boolean.preprocess_column(df[column], {**options, "fill_missing": learned["fill_value"]})
                else:
                    raw = _date_numbers(df[column], options) if role == "datetime" else df[column]
                    numeric_options = {**options, "errors": "raise"} if role == "datetime" else options
                    values = numerical.transform_column(raw, numeric_options, learned)
                if values.isna().any() or values.isin([float("inf"), -float("inf")]).any():
                    raise ValueError("Feature contains missing or infinite values after preprocessing")
                output[column] = values
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{table}.{column}: {exc}") from exc
        result[table] = output
    return result


def main():
    parser = argparse.ArgumentParser(description="Run training/validation preprocessing using YAML.")
    parser.add_argument("config", help="project YAML; relative paths use the current directory")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error("Project YAML must contain a mapping")
    settings = config.get("preprocessing") or {}
    source = settings.get("input_dir", config.get("data_dir"))
    output = settings.get("pipeline_output_dir")
    schema_path = (config.get("dtype_detection") or {}).get("output_path")
    if not all(isinstance(value, str) and value.strip() for value in (source, output, schema_path)):
        parser.error("YAML requires data_dir, preprocessing.pipeline_output_dir and dtype_detection.output_path")
    with open(schema_path, encoding="utf-8-sig") as stream:
        schema = yaml.safe_load(stream)
    if not isinstance(schema, dict):
        parser.error("Detected schema must contain a mapping")
    train, validation = split_tables(load_tables(source), config)
    state = fit_preprocessing(train, schema, config)
    processed_train = transform_tables(train, state)
    processed_validation = transform_tables(validation, state)
    save_splits(processed_train, processed_validation, output, source)
    with (Path(output) / "preprocessing_state.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(state, stream, sort_keys=False)
    for table in train:
        print(f"{table}: train={len(train[table])}, validation={len(validation[table])}, features={len(state['tables'][table]['features'])}")
    print(f"Saved combined tables and training-fitted settings: {output}")


if __name__ == "__main__":
    main()
