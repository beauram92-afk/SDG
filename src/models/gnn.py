"""Shared heterogeneous, relation-aware GNN backbone for VAE and DDPM heads.

Edges always use row 0 = source and row 1 = destination.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SharedGNN(nn.Module):
    """Per-table feature projections followed by relation-wise mean aggregation.

    Returns {table: row_embeddings}. VAE and diffusion heads consume these
    embeddings independently; this module contains no model-specific loss.
    """

    def __init__(self, table_specs, edge_types, hidden_dim=64, out_dim=64,
                 num_layers=2, dropout=0.1, categorical_embedding_dim=8):
        super().__init__()
        if not table_specs:
            raise ValueError("At least one table is required")
        for value in (hidden_dim, out_dim, num_layers, categorical_embedding_dim):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("GNN dimensions and num_layers must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        self.table_names = list(table_specs)
        self.feature_columns = {t: list(s["features"]) for t, s in table_specs.items()}
        self.edge_types = [tuple(edge) for edge in edge_types]
        if len(set(self.edge_types)) != len(self.edge_types):
            raise ValueError("Duplicate edge types")
        for edge in self.edge_types:
            if len(edge) != 3 or edge[0] not in table_specs or edge[2] not in table_specs:
                raise ValueError(f"Invalid edge type: {edge}")
        self.out_dim = out_dim
        self.dropout = dropout
        self.cat_positions = {}
        self.embeddings = nn.ModuleList()
        self.input_projections = nn.ModuleList()
        for table in self.table_names:
            positions = []
            embeddings = nn.ModuleList()
            for position, column in enumerate(self.feature_columns[table]):
                info = table_specs[table]["columns"][column]
                if info["role"] == "categorical":
                    positions.append(position)
                    # -2 unknown and -1 missing become embedding indices 0 and 1.
                    embeddings.append(nn.Embedding(len(info["learned"]["categories"]) + 2,
                                                   categorical_embedding_dim))
            self.cat_positions[table] = positions
            self.embeddings.append(embeddings)
            width = len(self.feature_columns[table]) + len(positions) * (categorical_embedding_dim - 1)
            self.input_projections.append(nn.Linear(max(width, 1), hidden_dim))
        self.self_layers = nn.ModuleList()
        self.relation_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for layer in range(num_layers):
            width = out_dim if layer == num_layers - 1 else hidden_dim
            self.self_layers.append(nn.Linear(hidden_dim, width))
            self.relation_layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, width, bias=False) for _ in self.edge_types
            ]))
            self.norms.append(nn.LayerNorm(width))
            self.residuals.append(nn.Identity() if hidden_dim == width else nn.Linear(hidden_dim, width, bias=False))

    def forward(self, node_features, edge_indices):
        if set(node_features) != set(self.table_names) or set(edge_indices) != set(self.edge_types):
            raise ValueError("Graph table and relation types must match the encoder")
        parameter = next(self.parameters())
        x = {}
        for index, table in enumerate(self.table_names):
            values = node_features[table]
            if values.ndim != 2 or values.shape[1] != len(self.feature_columns[table]):
                raise ValueError(f"Wrong feature shape for {table}")
            if values.device != parameter.device or values.dtype != parameter.dtype:
                raise ValueError("Move graph tensors and encoder to the same device and floating dtype")
            if not torch.isfinite(values).all():
                raise ValueError(f"Non-finite features in {table}")
            categorical = self.cat_positions[table]
            numeric = [j for j in range(values.shape[1]) if j not in categorical]
            parts = [values[:, numeric]]
            for position, embedding in zip(categorical, self.embeddings[index]):
                codes = values[:, position]
                if (codes != codes.round()).any() or (codes < -2).any() or (codes >= embedding.num_embeddings - 2).any():
                    raise ValueError(f"Invalid categorical codes in {table}")
                parts.append(embedding(codes.long() + 2))
            features = torch.cat(parts, dim=1)
            if features.shape[1] == 0:
                features = values.new_ones((len(values), 1))
            x[table] = F.gelu(self.input_projections[index](features))
        for edge_type, edges in edge_indices.items():
            source, _, target = edge_type
            if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
                raise ValueError("Edges must be torch.long tensors of shape (2, E)")
            if edges.device != parameter.device:
                raise ValueError("Edges and encoder must use the same device")
            if edges.numel() and ((edges < 0).any() or (edges[0] >= len(x[source])).any() or (edges[1] >= len(x[target])).any()):
                raise ValueError(f"Edge index outside node bounds: {edge_type}")
        for self_layer, relation_layers, norm, residual in zip(
                self.self_layers, self.relation_layers, self.norms, self.residuals):
            neighbors = {table: x[table].new_zeros((len(x[table]), self_layer.out_features))
                         for table in self.table_names}
            active = {table: x[table].new_zeros((len(x[table]), 1)) for table in self.table_names}
            for edge_type, relation_layer in zip(self.edge_types, relation_layers):
                source, _, target = edge_type
                src, dst = edge_indices[edge_type]
                if src.numel() == 0:
                    continue
                messages = relation_layer(x[source].index_select(0, src))
                aggregate = messages.new_zeros(neighbors[target].shape)
                aggregate.index_add_(0, dst, messages)
                degree = messages.new_zeros((len(x[target]), 1))
                degree.index_add_(0, dst, messages.new_ones((len(dst), 1)))
                neighbors[target] = neighbors[target] + aggregate / degree.clamp_min(1)
                active[target] = active[target] + (degree > 0).to(messages.dtype)
            x = {table: F.dropout(F.gelu(norm(self_layer(values)
                     + neighbors[table] / active[table].clamp_min(1))) + residual(values),
                     p=self.dropout, training=self.training) for table, values in x.items()}
        return x


def build_shared_gnn(state, edge_types, config):
    """Both model families use this factory and the same YAML gnn settings."""
    options = dict(config.get("gnn") or {})
    return SharedGNN(state["tables"], edge_types, **options)
