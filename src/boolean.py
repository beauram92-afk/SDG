"""Convert boolean tokens to nullable 1/0 values."""
import pandas as pd


def preprocess_column(series, options):
    errors = options.get("errors", "raise")
    if errors not in ("raise", "coerce"):
        raise ValueError("boolean.errors must be raise or coerce")
    text = series.astype("string").str.strip().str.lower().replace("", pd.NA)
    tokens = {"true": 1, "yes": 1, "y": 1, "t": 1, "1": 1, "1.0": 1,
              "false": 0, "no": 0, "n": 0, "f": 0, "0": 0, "0.0": 0}
    result = text.map(tokens).astype("Int64")
    if errors == "raise" and (text.notna() & result.isna()).any():
        raise ValueError("Unrecognized boolean values; correct the input or set errors: coerce")
    fill = options.get("fill_missing")
    if fill is not None:
        if fill not in (0, 1):
            raise ValueError("boolean.fill_missing must be 0, 1 or null")
        result = result.fillna(fill)
    return result


if __name__ == "__main__":
    from src.preprocessing import run_stage
    run_stage("boolean", preprocess_column)
