"""Train either model with the shared GNN, saved graphs and YAML settings."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import yaml

from src.models.gnnddpm import GNNDDPM
from src.models.gnnvae import GNNVAE


def build_model(state, edge_types, config):
    family = config.get("model_type")
    if family not in ("gnnvae", "gnnddpm"):
        raise ValueError("model_type must be gnnvae or gnnddpm")
    return (GNNVAE if family == "gnnvae" else GNNDDPM)(state, edge_types, config)


def _move_graph(graph, state, device):
    expected = {table: spec["features"] for table, spec in state["tables"].items()}
    if graph.get("feature_columns") != expected:
        raise ValueError("Graph feature order does not match its preprocessing state; rebuild graphs")
    if set(graph["node_features"]) != set(expected):
        raise ValueError("Graph node types do not match preprocessing state")
    return {"node_features": {table: values.to(device=device, dtype=torch.float32)
                              for table, values in graph["node_features"].items()},
            "edge_indices": {tuple(edge): values.to(device=device)
                             for edge, values in graph["edge_indices"].items()},
            "feature_columns": graph["feature_columns"]}


def _save_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config):
    options = config.get("training") or {}
    epochs = options.get("epochs", 50)
    patience = options.get("patience", 10)
    threads = options.get("num_threads", 2)
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (epochs, patience, threads)):
        raise ValueError("training epochs, patience and num_threads must be positive integers")
    learning_rate = options.get("learning_rate", 0.001)
    weight_decay = options.get("weight_decay", 0.00001)
    clip = options.get("gradient_clip", 1.0)
    if not learning_rate > 0 or weight_decay < 0 or not clip > 0:
        raise ValueError("Invalid learning rate, weight decay or gradient clipping")
    if not isinstance(options.get("output_dir"), str) or not options["output_dir"].strip():
        raise ValueError("YAML must contain training.output_dir")
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    requested = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; set device: cpu or device: auto in YAML")
    graph_dir = Path(config["graph"]["output_dir"])
    with (graph_dir / "preprocessing_state.yaml").open(encoding="utf-8-sig") as stream:
        state = yaml.safe_load(stream)
    training_graph = _move_graph(torch.load(graph_dir / "train.pt", map_location="cpu", weights_only=True), state, device)
    validation_graph = _move_graph(torch.load(graph_dir / "validation.pt", map_location="cpu", weights_only=True), state, device)
    edge_types = list(training_graph["edge_indices"])
    if set(validation_graph["edge_indices"]) != set(edge_types):
        raise ValueError("Training and validation relation types differ")
    model = build_model(state, edge_types, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_root = Path(options["output_dir"]).resolve()
    for source in (Path(config["data_dir"]).resolve(), graph_dir.resolve()):
        if output_root == source or source in output_root.parents or output_root in source.parents:
            raise ValueError("training.output_dir must be separate from input data and graphs")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    best, stale, best_epoch = float("inf"), 0, 0
    print(f"Training {config['model_type']} on {device}; run: {output}")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            training = model.loss(training_graph)
            if not all(torch.isfinite(value) for value in training.values()):
                raise ValueError("Non-finite training loss")
            training["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            # Reuse the same validation randomness each epoch without disturbing training RNG.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(seed + 1)
                validation = model.loss(validation_graph)
            if not all(torch.isfinite(value) for value in validation.values()):
                raise ValueError("Non-finite validation loss")
            metrics = {"epoch": epoch, **{f"train_{k}": float(v.detach()) for k, v in training.items()},
                       **{f"validation_{k}": float(v.detach()) for k, v in validation.items()}}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(metrics))
                writer.writeheader()
            writer.writerow(metrics)
            stream.flush()
            score = metrics["validation_loss"]
            improved = score < best
            if improved:
                best, stale, best_epoch = score, 0, epoch
            else:
                stale += 1
            payload = {"format_version": 1, "model_type": config["model_type"], "epoch": epoch,
                       "best_epoch": best_epoch, "best_validation_loss": best,
                       "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "preprocessing_state": state, "edge_types": edge_types, "config": config}
            _save_checkpoint(payload, output / "last.pt")
            if improved:
                _save_checkpoint(payload, output / "best.pt")
            print(f"epoch={epoch} train_loss={metrics['train_loss']:.6f} validation_loss={score:.6f}")
            if stale >= patience:
                print(f"Early stopping; best epoch={best_epoch}")
                break
    print(f"Saved best.pt, last.pt and metrics.csv: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN-VAE or GNN-DDPM using YAML.")
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error("Project YAML must contain a mapping")
    train(config)
