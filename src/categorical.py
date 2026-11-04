"""Clean categorical text while preserving category labels."""
import pandas as pd


def preprocess_column(series, options):
    result = series.astype("string").str.strip().replace("", pd.NA)
    if options.get("lowercase", False):
        result = result.str.lower()
    fill = options.get("fill_missing")
    if fill is not None:
        if not isinstance(fill, str):
            raise ValueError("categorical.fill_missing must be a string or null")
        result = result.fillna(fill)
    return result


def fit_column(series, options):
    """Ordinal vocabulary learned only from training rows."""
    return {"categories": sorted(preprocess_column(series, options).dropna().unique().tolist()),
            "missing_code": -1, "unknown_code": -2}


def transform_column(series, options, state):
    values = preprocess_column(series, options)
    mapping = {value: code for code, value in enumerate(state["categories"])}
    encoded = values.map(mapping).fillna(state["unknown_code"])
    return encoded.where(values.notna(), state["missing_code"]).astype("int64")


if __name__ == "__main__":
    from src.preprocessing import run_stage
    run_stage("categorical", preprocess_column)
