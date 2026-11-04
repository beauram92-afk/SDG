"""Conditional Gaussian diffusion over numerical and one-hot categorical features."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.features import FeatureLayout
from src.models.gnn import build_shared_gnn


class GNNDDPM(nn.Module):
    def __init__(self, state, edge_types, config):
        super().__init__()
        self.gnn = build_shared_gnn(state, edge_types, config)
        self.layout = FeatureLayout(state)
        options = config.get("ddpm") or {}
        self.num_steps = options.get("num_steps", 1000)
        hidden = options.get("hidden_dim", 64)
        self.time_dim = options.get("time_dim", 32)
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (self.num_steps, hidden, self.time_dim)):
            raise ValueError("DDPM dimensions and num_steps must be positive integers")
        if self.time_dim < 2 or self.num_steps < 2:
            raise ValueError("DDPM time_dim and num_steps must be at least 2")
        start, end = options.get("beta_start", 0.0001), options.get("beta_end", 0.02)
        if not 0 < start <= end < 1:
            raise ValueError("DDPM betas must satisfy 0 < beta_start <= beta_end < 1")
        betas = torch.linspace(start, end, self.num_steps)
        alphas = 1 - betas
        cumulative = alphas.cumprod(0)
        previous = torch.cat([torch.ones(1), cumulative[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", cumulative)
        self.register_buffer("posterior_variance", betas * (1 - previous) / (1 - cumulative))
        self.denoisers = nn.ModuleList([
            nn.Sequential(nn.Linear(width + self.gnn.out_dim + self.time_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, width))
            if width else nn.Identity() for width in self.layout.widths.values()
        ])

    def time_embedding(self, steps):
        half = self.time_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=steps.device) / max(half - 1, 1))
        phase = steps[:, None] * frequencies[None, :]
        return F.pad(torch.cat([phase.sin(), phase.cos()], dim=1), (0, self.time_dim - 2 * half))

    def context(self, graph, target_table):
        # Hide the whole target table, including its outgoing messages, to prevent clean-target leakage.
        features = dict(graph["node_features"])
        masked = torch.zeros_like(features[target_table])
        for index, role, _, _ in self.layout.entries[target_table]:
            if role == "categorical":
                masked[:, index] = -1
        features[target_table] = masked
        return self.gnn(features, graph["edge_indices"])[target_table]

    def loss(self, graph):
        losses = []
        for i, table in enumerate(self.layout.tables):
            values = graph["node_features"][table]
            if not len(values) or not self.layout.widths[table]:
                continue
            clean = self.layout.pack(table, values)
            steps = torch.randint(self.num_steps, (len(clean),), device=clean.device)
            noise = torch.randn_like(clean)
            cumulative = self.alpha_bar[steps, None]
            noisy = cumulative.sqrt() * clean + (1 - cumulative).sqrt() * noise
            condition = self.context(graph, table)
            prediction = self.denoisers[i](torch.cat([noisy, condition, self.time_embedding(steps)], dim=1))
            losses.append(F.mse_loss(prediction, noise))
        if not losses:
            raise ValueError("No non-empty feature tables for DDPM loss")
        loss = torch.stack(losses).mean()
        return {"loss": loss, "noise_mse": loss}

    @torch.no_grad()
    def sample_table(self, graph, table, categorical_missing=None):
        """Sample a table conditioned on the other tables and supplied topology."""
        if self.training:
            raise ValueError("Call model.eval() before sampling")
        index = self.layout.tables.index(table)
        rows = len(graph["node_features"][table])
        values = self.betas.new_empty((rows, self.layout.widths[table])).normal_()
        if not rows or not self.layout.widths[table]:
            return self.layout.unpack(table, values)
        condition = self.context(graph, table)
        for step in reversed(range(self.num_steps)):
            steps = torch.full((rows,), step, device=values.device, dtype=torch.long)
            noise = self.denoisers[index](torch.cat([values, condition, self.time_embedding(steps)], dim=1))
            values = (values - self.betas[step] / (1 - self.alpha_bar[step]).sqrt() * noise) / self.alphas[step].sqrt()
            if step:
                values = values + self.posterior_variance[step].sqrt() * torch.randn_like(values)
        if categorical_missing is not None:
            for column, role, start, width in self.layout.entries[table]:
                if role == "categorical":
                    values[:, start] = -torch.inf  # unknown is never a generated label
                    if not categorical_missing[column]:
                        values[:, start + 1] = -torch.inf
        return self.layout.unpack(table, values)
