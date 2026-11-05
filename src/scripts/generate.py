"""Generate synthetic relational tables from a trained VAE or DDPM checkpoint."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.gnn.graph_builder import build_graph
from src.scripts.evaluation import compare_features
from src.post_processing.inverse_preprocessing import inverse_tables
from src.pre_processing.pii import replace_pii
from src.scripts.train import build_model


def plan_tables(state, config, training_graph, options, rng):
    """Sample a forest topology from empirical children-per-parent counts."""
    parents = {}
    for relation in config.get("relationships") or []:
        parent = relation.get("parent_table_name", relation.get("parent_table"))
        child = relation.get("child_table_name", relation.get("child_table"))
        pk = relation.get("parent_primary_key", relation.get("parent_column"))
        fk = relation.get("child_foreign_key", relation.get("child_column"))
        if child in parents:
            raise ValueError("Generation currently supports one parent relationship per child table")
        if parent not in state["tables"] or child not in state["tables"]:
            raise ValueError("Unknown relationship table")
        parents[child] = (parent, pk, fk)
    root_rows = options.get("root_rows", 10)
    if not isinstance(root_rows, int) or isinstance(root_rows, bool) or root_rows < 1:
        raise ValueError("generation.root_rows must be a positive integer")
    max_rows = options.get("max_rows_per_table", 100000)
    if not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows_per_table must be positive")
    order, tables = [], {}
    run_token = uuid.UUID(int=int.from_bytes(rng.bytes(16), "big")).hex
    remaining = set(state["tables"])
    while remaining:
        ready = sorted(t for t in remaining if t not in parents or parents[t][0] in tables)
        if not ready:
            raise ValueError("Cyclic relationships are not supported")
        for table in ready:
            spec = state["tables"][table]
            foreign_key = None
            if table in parents:
                parent, pk, foreign_key = parents[table]
                edge_type = (parent, f"{pk}->{foreign_key}", table)
                if edge_type not in training_graph["edge_indices"]:
                    raise ValueError(f"Missing training relationship: {edge_type}")
                parent_count = len(training_graph["node_features"][parent])
                if not parent_count:
                    raise ValueError(f"No training parents for {table}")
                targets = training_graph["edge_indices"][edge_type][0]
                empirical = torch.bincount(targets, minlength=parent_count).cpu().numpy()
                counts = rng.choice(empirical, size=len(tables[parent]), replace=True)
                rows = int(counts.sum())
                if rows > max_rows:
                    raise ValueError(f"Planned row count exceeds max_rows_per_table: {table}")
                pointers = np.repeat(np.arange(len(tables[parent])), counts)
            else:
                rows = root_rows
            if rows > max_rows:
                raise ValueError(f"Planned row count exceeds max_rows_per_table: {table}")
            df = pd.DataFrame(index=range(rows))
            for column in spec["keys"]:
                if column == foreign_key:
                    if pk not in tables[parent]:
                        raise ValueError(f"Parent key absent from saved state: {parent}.{pk}")
                    df[column] = tables[parent][pk].iloc[pointers].to_numpy()
                else:
                    # IDs are fresh strings, independent of observed source identifiers.
                    df[column] = [f"SYN_{run_token}_{table}_{column}_{i}" for i in range(rows)]
            for column in spec["features"]:
                df[column] = 0.0
            tables[table] = df
            order.append(table)
            remaining.remove(table)
    return tables, order


def sample_features(model, graph, order, options, training_graph):
    """Sample only from model parameters and synthetic graph context."""
    temperature = options.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("generation.temperature must be positive")
    missing_allowed = {table: {index: bool((training_graph["node_features"][table][:, index] == -1).any())
                              for index, role, _, _ in model.layout.entries[table] if role == "categorical"}
                       for table in model.layout.tables}
    with torch.no_grad():
        if model.__class__.__name__ == "GNNVAE":
            for i, table in enumerate(model.layout.tables):
                rows = len(graph["node_features"][table])
                width = model.layout.widths[table]
                if not width or not rows:
                    continue
                latent = torch.randn(rows, model.latent_dim, device=next(model.parameters()).device)
                packed = model.decoders[i](latent)
                for column, role, start, size in model.layout.entries[table]:
                    if role == "categorical":
                        logits = packed[:, start:start + size] / temperature
                        logits[:, 0] = -torch.inf  # unknown is not an output category
                        if not missing_allowed[table][column]:
                            logits[:, 1] = -torch.inf
                        sampled = torch.multinomial(logits.softmax(dim=1), 1).flatten()
                        packed[:, start:start + size] = -1
                        packed[torch.arange(rows, device=packed.device), start + sampled] = 1
                    elif role == "boolean":
                        packed[:, start] = torch.bernoulli((packed[:, start] / temperature).sigmoid()) * 2 - 1
                graph["node_features"][table] = model.layout.unpack(table, packed, logits=True)
        else:
            sweeps = options.get("ddpm_sweeps", 2)
            if not isinstance(sweeps, int) or isinstance(sweeps, bool) or sweeps < 1:
                raise ValueError("ddpm_sweeps must be a positive integer")
            # Initialize synthetic context; no real feature rows are donors.
            for table, values in graph["node_features"].items():
                for index, role, _, size in model.layout.entries[table]:
                    if role == "categorical":
                        values[:, index] = torch.randint(max(size - 2, 1), (len(values),), device=values.device) if size > 2 else -1
                    elif role == "boolean":
                        values[:, index] = torch.randint(2, (len(values),), device=values.device)
                    else:
                        values[:, index] = torch.randn(len(values), device=values.device)
            for _ in range(sweeps):
                for table in order:
                    values = model.sample_table(graph, table, categorical_missing=missing_allowed[table])
                    for index, role, _, _ in model.layout.entries[table]:
                        if role == "categorical":
                            values[:, index].clamp_(min=-1)
                    graph["node_features"][table] = values
    return graph


def generate(config):
    options = config.get("generation") or {}
    checkpoint = options.get("checkpoint", "latest")
    if checkpoint == "latest":
        candidates = sorted(Path(config["training"]["output_dir"]).glob("*/best.pt"))
        if not candidates:
            raise ValueError("No trained checkpoints found")
        checkpoint = candidates[-1]
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("model_type") != config.get("model_type"):
        raise ValueError("Checkpoint model_type does not match YAML")
    trained = payload["config"]
    for key in ("primary_keys", "relationships", "pii_columns"):
        if config.get(key) != trained.get(key):
            raise ValueError(f"YAML {key} differs from checkpoint; use the trained schema")
    seed = options.get("seed", config.get("seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("training", {}).get("num_threads", 2))
    requested = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; set device: auto or cpu")
    state = payload["preprocessing_state"]
    # Older checkpoints need their saved training graph for count and range statistics.
    training_graph = torch.load(Path(trained["graph"]["output_dir"]) / "train.pt", map_location="cpu", weights_only=True)
    expected = {t: s["features"] for t, s in state["tables"].items()}
    if training_graph["feature_columns"] != expected or set(training_graph["edge_indices"]) != {tuple(e) for e in payload["edge_types"]}:
        raise ValueError("Training graph no longer matches checkpoint")
    tables, order = plan_tables(state, trained, training_graph, options, rng)
    graph = build_graph(tables, state, trained)
    graph["node_features"] = {t: x.to(device) for t, x in graph["node_features"].items()}
    graph["edge_indices"] = {e: x.to(device) for e, x in graph["edge_indices"].items()}
    model = build_model(state, payload["edge_types"], trained).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    print(f"Generating {payload['model_type']} from {checkpoint}", flush=True)
    graph = sample_features(model, graph, order, options, training_graph)
    clipped = {}
    for table, spec in state["tables"].items():
        values = graph["node_features"][table].cpu()
        if not torch.isfinite(values).all():
            raise ValueError(f"Model generated non-finite features: {table}")
        for index, column in enumerate(spec["features"]):
            role = spec["columns"][column]["role"]
            if options.get("clip_to_training_range", True) and role in ("numerical", "datetime"):
                observed = training_graph["node_features"][table][:, index]
                if observed.numel():
                    before = values[:, index].clone()
                    values[:, index].clamp_(float(observed.min()), float(observed.max()))
                    clipped[f"{table}.{column}"] = int((before != values[:, index]).sum())
            tables[table][column] = values[:, index].numpy()
    schema = None
    schema_path = (trained.get("dtype_detection") or {}).get("output_path")
    if schema_path and Path(schema_path).is_file():
        with open(schema_path, encoding="utf-8-sig") as stream:
            schema = yaml.safe_load(stream)
    tables = inverse_tables(tables, state, schema)
    for table, spec in state["tables"].items():
        for column in spec["drop_columns"]:
            tables[table][column] = pd.NA
    if options.get("restore_fake_pii", True):
        report = {"tables": {}}
        for table, columns in (trained.get("pii_columns") or {}).items():
            if table not in tables:
                raise ValueError(f"Unknown PII table: {table}")
            report["tables"][table] = {}
            for column, kind in columns.items():
                if column not in state["tables"][table]["drop_columns"]:
                    raise ValueError(f"PII column was not excluded at training: {table}.{column}")
                tables[table][column] = [f"synthetic-{table}-{column}-{i}" for i in range(len(tables[table]))]
                report["tables"][table][column] = {"pii_type": kind}
        tables = replace_pii(tables, report, {**trained, "pii_seed": seed})
    from src.data_handling.data_validation import validate_tables
    issues = validate_tables(tables, config=trained)
    fatal = [issue for issue in issues if issue["check"] in
             ("pk_null", "pk_duplicate", "orphan_fk", "fk_configuration", "missing_table", "missing_pk_column")]
    if fatal:
        raise ValueError(f"Generated key validation failed: {fatal}")
    reference = {table: pd.DataFrame(training_graph["node_features"][table].numpy(), columns=spec["features"])
                 for table, spec in state["tables"].items()}
    reference = inverse_tables(reference, state, schema)
    quality = compare_features(reference, tables, state)
    output_root = Path(options["output_dir"]).resolve()
    for source in (Path(trained["data_dir"]).resolve(), Path(trained["graph"]["output_dir"]).resolve(), checkpoint.parent.resolve()):
        if output_root == source or source in output_root.parents or output_root in source.parents:
            raise ValueError("Generation output must be separate from data, graphs and checkpoints")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = output_root / run_id
    data_output = output / "tables"
    data_output.mkdir(parents=True, exist_ok=False)
    for table, df in tables.items():
        df.to_csv(data_output / f"{table}.csv", index=False)
    report = {"checkpoint": str(checkpoint), "seed": seed, "model_type": payload["model_type"],
              "row_counts": {t: len(df) for t, df in tables.items()}, "clipped_values": clipped,
              "key_validation": "passed", "validation_issues": issues,
              "limitations": ["Synthetic identifiers are strings.",
                              "Child counts are sampled empirically, not learned by a neural count model.",
                              "Imputed source missingness cannot be reconstructed.",
                              "VAE prior rows are independent; DDPM uses approximate conditional sweeps.",
                              "No statistical quality or privacy guarantee; inspect clipped_values and evaluate outputs."]}
    with (output / "generation_report.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    with (output / "quality_report.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(quality, stream, sort_keys=False)
    with (output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Saved synthetic tables and generation report: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic tables from a YAML-selected checkpoint.")
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error("Project YAML must contain a mapping")
    generate(config)
