"""Build the same graph representation for GNN-VAE and GNN-DDPM."""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data_loading import load_tables


def build_graph(tables, state, config):
    if set(tables) != set(state["tables"]):
        raise ValueError("Prepared tables do not match preprocessing state")
    features = {}
    for table, df in tables.items():
        columns = state["tables"][table]["features"]
        if df.columns.duplicated().any() or not set(columns) <= set(df.columns):
            raise ValueError(f"Missing or duplicate feature columns: {table}")
        values = torch.tensor(df[columns].to_numpy(dtype="float32"), dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError(f"Non-finite graph features: {table}")
        features[table] = values
    edges = {}
    for relation in config.get("relationships") or []:
        parent = relation.get("parent_table_name", relation.get("parent_table"))
        child = relation.get("child_table_name", relation.get("child_table"))
        pk = relation.get("parent_primary_key", relation.get("parent_column"))
        fk = relation.get("child_foreign_key", relation.get("child_column"))
        if parent not in tables or child not in tables or pk not in tables[parent] or fk not in tables[child]:
            raise ValueError("Relationship references a missing table or column")
        parent_values = tables[parent][pk]
        if parent_values.isna().any() or not parent_values.is_unique:
            raise ValueError(f"Invalid parent key: {parent}.{pk}")
        if tables[child][fk].isna().any():
            raise ValueError(f"Null foreign key: {child}.{fk}")
        positions = pd.Index(parent_values).get_indexer(tables[child][fk])
        if (positions < 0).any():
            raise ValueError(f"Orphan foreign keys: {child}.{fk}")
        child_rows = torch.arange(len(tables[child]), dtype=torch.long)
        parent_rows = torch.tensor(positions, dtype=torch.long)
        # Include column names so multiple relationships between two tables never collide.
        forward = (parent, f"{pk}->{fk}", child)
        reverse = (child, f"{fk}<-{pk}", parent)
        if forward in edges or reverse in edges or forward == reverse:
            raise ValueError("Duplicate relationship types")
        edges[forward] = torch.stack([parent_rows, child_rows])
        edges[reverse] = torch.stack([child_rows, parent_rows])
    return {"node_features": features, "edge_indices": edges,
            "feature_columns": {t: list(s["features"]) for t, s in state["tables"].items()}}


def main():
    parser = argparse.ArgumentParser(description="Build shared GNN graphs from prepared tables.")
    parser.add_argument("config", help="project YAML; relative paths use the current directory")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    prepared = Path(config["preprocessing"]["pipeline_output_dir"])
    output = Path(config["graph"]["output_dir"])
    source = Path(config["data_dir"]).resolve()
    if output.resolve() == source or source in output.resolve().parents or output.resolve() in source.parents:
        parser.error("Graph output must be separate from source data")
    if output.resolve() == prepared.resolve() or prepared.resolve() in output.resolve().parents or output.resolve() in prepared.resolve().parents:
        parser.error("Graph output must be separate from prepared data")
    if output.exists() and any(output.iterdir()):
        parser.error("Graph output is not empty; choose a new graph.output_dir")
    with (prepared / "preprocessing_state.yaml").open(encoding="utf-8-sig") as stream:
        state = yaml.safe_load(stream)
    graphs = {split: build_graph(load_tables(str(prepared / split)), state, config)
              for split in ("train", "validation")}
    if list(graphs["train"]["edge_indices"]) != list(graphs["validation"]["edge_indices"]):
        raise ValueError("Training and validation relation types differ")
    output.mkdir(parents=True, exist_ok=True)
    for split, graph in graphs.items():
        torch.save(graph, output / f"{split}.pt")
        print(f"{split}: {sum(len(x) for x in graph['node_features'].values())} nodes, "
              f"{sum(e.shape[1] for e in graph['edge_indices'].values())} directed edges")
    # Preserve vocabulary and feature order alongside graph tensors.
    with (output / "preprocessing_state.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(state, stream, sort_keys=False)


if __name__ == "__main__":
    main()
