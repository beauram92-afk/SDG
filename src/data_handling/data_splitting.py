"""Split parent entities and carry all their descendants into the same partition."""
import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data_handling.data_loading import load_tables


def split_tables(tables, config):
    options = config.get("data_splitting") or {}
    fraction = options.get("validation_fraction", 0.2)
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    primary = config.get("primary_keys") or {}
    for table, df in tables.items():
        if df.columns.duplicated().any():
            raise ValueError(f"Duplicate column names: {table}")
        keys = primary.get(table)
        keys = [keys] if isinstance(keys, str) else keys
        if not keys or not set(keys) <= set(df.columns):
            raise ValueError(f"Declare valid primary_keys for {table} in YAML")
        values = df[keys]
        blank = values.astype("string").apply(lambda s: s.str.strip().eq("")).any(axis=1)
        if values.isna().any(axis=None) or blank.any() or values.duplicated().any():
            raise ValueError(f"Primary key is null, blank or duplicated: {table}")
    parents = {}
    for relation in config.get("relationships") or []:
        parent = relation.get("parent_table_name", relation.get("parent_table"))
        child = relation.get("child_table_name", relation.get("child_table"))
        pk = relation.get("parent_primary_key", relation.get("parent_column"))
        fk = relation.get("child_foreign_key", relation.get("child_column"))
        if parent not in tables or child not in tables or pk not in tables[parent] or fk not in tables[child]:
            raise ValueError("Relationship references a missing table or column")
        if child in parents:
            raise ValueError(f"Multiple parents for {child}; this splitter requires a forest")
        if not tables[parent][pk].is_unique:
            raise ValueError(f"Relationship parent key is not unique: {parent}.{pk}")
        if tables[child][fk].isna().any() or not tables[child][fk].isin(tables[parent][pk]).all():
            raise ValueError(f"Missing or orphan foreign keys: {child}.{fk}")
        parents[child] = (parent, pk, fk)
    order = []
    pending = set(tables)
    while pending:
        ready = sorted(t for t in pending if t not in parents or parents[t][0] in order)
        if not ready:
            raise ValueError("Cyclic relationships are not supported")
        order.extend(ready)
        pending.difference_update(ready)
    rng = np.random.default_rng(options.get("seed", config.get("seed", 42)))
    train, validation = {}, {}
    for table in order:
        df = tables[table]
        if table not in parents:
            if len(df) < 2:
                raise ValueError(f"Root table {table} needs at least two rows")
            count = min(len(df) - 1, max(1, int(np.ceil(len(df) * fraction))))
            positions = rng.permutation(len(df))[:count]
            mask = np.zeros(len(df), dtype=bool)
            mask[positions] = True
        else:
            parent, pk, fk = parents[table]
            mask = df[fk].isin(validation[parent][pk]).to_numpy()
        train[table] = df.iloc[np.flatnonzero(~mask)].copy()
        validation[table] = df.iloc[np.flatnonzero(mask)].copy()
        if train[table].empty:
            raise ValueError(f"No training rows left in {table}; change the split seed or fraction")
    return train, validation


def save_splits(train, validation, output, source):
    destination = Path(output).resolve()
    source = Path(source).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Output must be separate from the input directory")
    # Never mix a new run with stale files or partially overwrite an existing run.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output is not empty; choose a new output_dir: {destination}")
    for name, tables in (("train", train), ("validation", validation)):
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        for table, df in tables.items():
            df.to_parquet(folder / f"{table}.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split related tables using YAML.")
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    source = (config.get("preprocessing") or {}).get("input_dir", config["data_dir"])
    train, validation = split_tables(load_tables(source), config)
    save_splits(train, validation, config["data_splitting"]["output_dir"], source)
    for table in train:
        print(f"{table}: train={len(train[table])}, validation={len(validation[table])}")
