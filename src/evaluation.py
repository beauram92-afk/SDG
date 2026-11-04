"""Basic marginal comparisons; these are not a privacy or downstream utility audit."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.datehandling import preprocess_column as parse_dates


def compare_features(reference, synthetic, state):
    """Compare generated marginals with training features, without reporting row values."""
    report = {"reference": "training partition", "tables": {},
              "not_checked": ["downstream usefulness", "privacy", "cross-table feature dependencies", "temporal ordering"]}
    for table, spec in state["tables"].items():
        columns = {}
        for column, info in spec["columns"].items():
            real, fake = reference[table][column], synthetic[table][column]
            role = info["role"]
            metrics = {"role": role, "reference_rows": len(real), "synthetic_rows": len(fake),
                       "reference_missing": int(real.isna().sum()), "synthetic_missing": int(fake.isna().sum())}
            if role in ("categorical", "boolean"):
                # Missing is a separate bin and cannot collide with a category label.
                left = real.astype("string").dropna().value_counts() / max(len(real), 1)
                right = fake.astype("string").dropna().value_counts() / max(len(fake), 1)
                labels = left.index.union(right.index)
                distance = (left.reindex(labels, fill_value=0) - right.reindex(labels, fill_value=0)).abs().sum()
                distance += abs(real.isna().sum() / max(len(real), 1) - fake.isna().sum() / max(len(fake), 1))
                metrics["total_variation_distance"] = float(distance / 2) if len(real) and len(fake) else None
            else:
                if role == "datetime":
                    def seconds(values):
                        parsed = parse_dates(values, {**info["options"], "errors": "raise"})
                        return parsed.dropna().map(lambda v: v.value / 1e9).astype(float)
                    real, fake = seconds(real), seconds(fake)
                else:
                    real, fake = pd.to_numeric(real).dropna(), pd.to_numeric(fake).dropna()
                if len(real) and len(fake):
                    metrics.update({"ks_statistic": float(ks_2samp(real, fake).statistic),
                                    "reference_mean": float(real.mean()), "synthetic_mean": float(fake.mean()),
                                    "reference_std": float(np.std(real)), "synthetic_std": float(np.std(fake))})
                else:
                    metrics["ks_statistic"] = None
            columns[column] = metrics
        report["tables"][table] = columns
    return report
