"""Restore model feature values using saved training preprocessing statistics."""
import numpy as np
import pandas as pd


def inverse_tables(tables, state, schema=None):
    """Restore labels, units and dtypes; imputed source missingness is not recoverable."""
    result = {}
    for table, spec in state["tables"].items():
        output = tables[table].copy()
        metadata = (schema or {}).get("tables", {}).get(table, {})
        for column, info in spec["columns"].items():
            role, learned = info["role"], info["learned"]
            values = pd.to_numeric(output[column], errors="raise")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite model output: {table}.{column}")
            if role == "categorical":
                categories = learned["categories"]
                if ((values != values.round()) | (values < -2) | (values >= len(categories))).any():
                    raise ValueError(f"Invalid categorical codes: {table}.{column}")
                output[column] = values.map(dict(enumerate(categories))).astype("string")
            elif role == "boolean":
                if not values.isin([0, 1]).all():
                    raise ValueError(f"Invalid boolean output: {table}.{column}")
                output[column] = values.astype("boolean")
            else:
                restored = values * learned["scale"] + learned["center"]
                if role == "datetime":
                    dates = pd.to_datetime(restored, unit="s", utc=True, errors="coerce")
                    if dates.isna().any():
                        raise ValueError(f"Generated dates exceed supported bounds: {table}.{column}")
                    date_format = info.get("options", {}).get("format", "mixed")
                    output[column] = dates.dt.strftime(date_format) if date_format != "mixed" else dates
                else:
                    original = info.get("original_dtype") or (metadata.get(column) or {}).get("physical_type", "")
                    if str(original).lower().startswith(("int", "uint")):
                        rounded = restored.round()
                        if ((rounded < np.iinfo(np.int64).min) | (rounded >= 2**63)).any():
                            raise ValueError(f"Generated integer exceeds int64 bounds: {table}.{column}")
                        output[column] = rounded.astype("Int64")
                    else:
                        output[column] = restored
        result[table] = output
    return result
