"""Shared mixed-type feature layout for reconstruction and diffusion heads."""
import torch
from torch.nn import functional as F


class FeatureLayout:
    def __init__(self, state):
        self.tables = list(state["tables"])
        self.entries, self.widths = {}, {}
        for table, spec in state["tables"].items():
            entries, offset = [], 0
            for index, column in enumerate(spec["features"]):
                info = spec["columns"][column]
                role = info["role"]
                if role not in ("numerical", "datetime", "boolean", "categorical"):
                    raise ValueError(f"Unsupported feature role: {role}")
                width = len(info["learned"]["categories"]) + 2 if role == "categorical" else 1
                entries.append((index, role, offset, width))
                offset += width
            self.entries[table], self.widths[table] = entries, offset
        if not any(self.widths.values()):
            raise ValueError("No model features in preprocessing state")

    def pack(self, table, values):
        parts = []
        for index, role, _, width in self.entries[table]:
            column = values[:, index]
            if role == "categorical":
                codes = column.long() + 2
                if (column != column.round()).any() or (codes < 0).any() or (codes >= width).any():
                    raise ValueError(f"Invalid categorical targets: {table}")
                parts.append(F.one_hot(codes, width).to(values.dtype))
            else:
                if role == "boolean" and ((column != 0) & (column != 1)).any():
                    raise ValueError(f"Invalid boolean targets: {table}")
                parts.append(column[:, None])
        return torch.cat(parts, dim=1) if parts else values[:, :0]

    def reconstruction_loss(self, table, predictions, values):
        losses = []
        for index, role, offset, width in self.entries[table]:
            target = values[:, index]
            if role == "categorical":
                losses.append(F.cross_entropy(predictions[:, offset:offset + width], target.long() + 2))
            elif role == "boolean":
                losses.append(F.binary_cross_entropy_with_logits(predictions[:, offset], target))
            else:
                losses.append(F.mse_loss(predictions[:, offset], target))
        return torch.stack(losses).mean()

    def unpack(self, table, packed, logits=False):
        columns = []
        for _, role, offset, width in self.entries[table]:
            if role == "categorical":
                value = (packed[:, offset:offset + width].argmax(dim=1) - 2).to(packed.dtype)
            elif role == "boolean":
                value = (packed[:, offset] >= (0.0 if logits else 0.5)).to(packed.dtype)
            else:
                value = packed[:, offset]
            columns.append(value)
        return torch.stack(columns, dim=1) if columns else packed[:, :0]
