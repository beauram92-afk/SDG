"""Parse dates using the format declared in YAML."""
import pandas as pd


def preprocess_column(series, options):
    errors = options.get("errors", "raise")
    if errors not in ("raise", "coerce"):
        raise ValueError("datetime.errors must be raise or coerce")
    text = series.astype("string").str.strip().replace("", pd.NA)
    return pd.to_datetime(text, format=options.get("format", "mixed"),
                          errors=errors, dayfirst=options.get("dayfirst", False),
                          utc=options.get("utc", True))


if __name__ == "__main__":
    from src.pre_processing.preprocessing import run_stage
    run_stage("datetime", preprocess_column)
