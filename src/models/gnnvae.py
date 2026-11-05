"""Row-level VAE with the shared relational GNN encoder."""
import torch
from torch import nn

from src.models.gnn.features import FeatureLayout
from src.models.gnn.gnn import build_shared_gnn


class GNNVAE(nn.Module):
    def __init__(self, state, edge_types, config):
        super().__init__()
        self.gnn = build_shared_gnn(state, edge_types, config)
        self.layout = FeatureLayout(state)
        options = config.get("vae") or {}
        latent = options.get("latent_dim", 16)
        hidden = options.get("hidden_dim", 64)
        self.kl_weight = options.get("kl_weight", 0.01)
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (latent, hidden)):
            raise ValueError("VAE dimensions must be positive integers")
        if not isinstance(self.kl_weight, (int, float)) or self.kl_weight < 0:
            raise ValueError("vae.kl_weight must be non-negative")
        self.latent_dim = latent
        self.means = nn.ModuleList([nn.Linear(self.gnn.out_dim, latent) for _ in self.layout.tables])
        self.logvars = nn.ModuleList([nn.Linear(self.gnn.out_dim, latent) for _ in self.layout.tables])
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, width))
            if width else nn.Identity() for width in self.layout.widths.values()
        ])

    def forward(self, graph, sample=None):
        embeddings = self.gnn(graph["node_features"], graph["edge_indices"])
        sample = self.training if sample is None else sample
        predictions, means, logvars = {}, {}, {}
        for i, table in enumerate(self.layout.tables):
            mean = self.means[i](embeddings[table])
            logvar = self.logvars[i](embeddings[table]).clamp(-10, 10)
            latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
            predictions[table] = self.decoders[i](latent) if self.layout.widths[table] else latent[:, :0]
            means[table], logvars[table] = mean, logvar
        return predictions, means, logvars

    def loss(self, graph):
        predictions, means, logvars = self(graph)
        reconstruction, kl = [], []
        for table in self.layout.tables:
            values = graph["node_features"][table]
            if len(values) == 0 or not self.layout.widths[table]:
                continue
            # Validate targets before CE/BCE; category indices include reserved codes.
            self.layout.pack(table, values)
            reconstruction.append(self.layout.reconstruction_loss(table, predictions[table], values))
            kl.append(-0.5 * (1 + logvars[table] - means[table].square() - logvars[table].exp()).mean())
        if not reconstruction:
            raise ValueError("No non-empty feature tables for VAE loss")
        rec, divergence = torch.stack(reconstruction).mean(), torch.stack(kl).mean()
        return {"loss": rec + self.kl_weight * divergence, "reconstruction": rec, "kl": divergence}

    @torch.no_grad()
    def reconstruct(self, graph):
        predictions, _, _ = self(graph, sample=False)
        return {table: self.layout.unpack(table, pred, logits=True) for table, pred in predictions.items()}
