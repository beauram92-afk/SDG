"""Numerical conversion and optional missing-value filling."""
import pandas as pd


def preprocess_column(series, options):
    result = pd.to_numeric(series.replace(r"^\s*$", pd.NA, regex=True), errors=options.get("errors", "raise"))
    method = options.get("fill_missing", "median")
    if method not in ("median", "mean", "constant", None):
        raise ValueError("numerical.fill_missing must be median, mean, constant or null")
    if method == "constant":
        value = options["fill_value"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("numerical.fill_value must be a number")
        result = result.fillna(value)
    elif method and result.notna().any():
        value = result.median() if method == "median" else result.mean()
        result = result.astype("float64").fillna(value)
    return result


def fit_column(series, options):
    """Learn fill/scaling statistics from training rows only."""
    raw = preprocess_column(series, {**options, "fill_missing": None}).astype("float64")
    if raw.isin([float("inf"), -float("inf")]).any():
        raise ValueError("Infinite numerical values are not supported")
    method = options.get("fill_missing", "median")
    if method not in ("median", "mean", "constant", None):
        raise ValueError("Invalid numerical fill_missing setting")
    if method == "constant":
        fill = float(options["fill_value"])
    elif method:
        if not raw.notna().any():
            raise ValueError("Training column is all missing; configure constant filling or exclude it")
        fill = float(raw.median() if method == "median" else raw.mean())
    else:
        fill = None
    filled = raw.fillna(fill) if fill is not None else raw
    scaling = options.get("scaling", "standard")
    if scaling not in ("standard", "none"):
        raise ValueError("scaling must be standard or none")
    center = float(filled.mean()) if scaling == "standard" else 0.0
    scale = float(filled.std(ddof=0)) if scaling == "standard" else 1.0
    if pd.isna(center):
        raise ValueError("Training column has no usable values")
    return {"fill_value": fill, "center": center, "scale": scale if scale > 0 else 1.0}


def transform_column(series, options, state):
    raw = preprocess_column(series, {**options, "fill_missing": None}).astype("float64")
    if raw.isin([float("inf"), -float("inf")]).any():
        raise ValueError("Infinite numerical values are not supported")
    if state["fill_value"] is not None:
        raw = raw.fillna(state["fill_value"])
    return (raw - state["center"]) / state["scale"]


if __name__ == "__main__":
    from src.preprocessing import run_stage
    run_stage("numerical", preprocess_column)
