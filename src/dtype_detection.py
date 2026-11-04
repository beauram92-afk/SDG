
import pandas as pd
import yaml
from pathlib import Path


class SchemaDetector:
    def detect(self, series):
        physical = str(series.dtype)
        values = series.dropna()

        if values.empty:
            return physical, "unknown"

        if pd.api.types.is_datetime64_any_dtype(series):
            return physical, "datetime"

        if pd.api.types.is_bool_dtype(series):
            return physical, "boolean"

        if pd.api.types.is_numeric_dtype(series):
            return physical, "numerical"

        text = values.astype(str)

        if text.str.fullmatch(
            r"\d{4}-\d{2}-\d{2}"
        ).all():
            dates = pd.to_datetime(text, errors="coerce")
            if dates.notna().all():
                return physical, "datetime"

        if text.str.lower().isin(["true", "false", "yes", "no"]).all():
            return physical, "boolean"

        return physical, "categorical"


class ColumnClassifier:
    def classify(self, name, series, semantic_type):
        name = name.lower()

        if name == "id" or name.endswith("_id"):
            return "id"

        return semantic_type


class ConfidenceScorer:
    def score(self, column_name, detected_type):
        if detected_type == "id":
            return 0.70, "Column name suggests an ID"

        if detected_type == "datetime":
            return 0.95, "Values match a date format or dtype"

        if detected_type == "unknown":
            return 0.0, "No non-null values"

        return 0.80, "Values match the detected datatype"


class SchemaMetadataManager:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, metadata):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file, sort_keys=False)

    def override(self, metadata, table, column, new_type):
        metadata["tables"][table][column]["manual_override"] = new_type
        metadata["tables"][table][column]["final_type"] = new_type
        return metadata


def detect_table(table_name, df):
    detector = SchemaDetector()
    classifier = ColumnClassifier()
    scorer = ConfidenceScorer()

    metadata = {"tables": {table_name: {}}}

    for column in df.columns:
        physical, semantic = detector.detect(df[column])
        final_type = classifier.classify(column, df[column], semantic)
        confidence, reason = scorer.score(column, final_type)

        metadata["tables"][table_name][column] = {
            "physical_type": physical,
            "semantic_type": semantic,
            "final_type": final_type,
            "confidence": confidence,
            "reason": reason
        }

    return metadata


if __name__ == "__main__":
    import argparse
    from src.data_loading import load_tables

    parser = argparse.ArgumentParser(description="Detect column types using a YAML config.")
    parser.add_argument("config", help="project YAML; relative paths use the current directory")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("data_dir"), str) or not config["data_dir"].strip():
        parser.error("YAML must contain a non-empty data_dir string")
    output_path = (config.get("dtype_detection") or {}).get("output_path")
    if not isinstance(output_path, str) or not output_path.strip():
        parser.error("YAML must contain dtype_detection.output_path")
    tables = load_tables(config["data_dir"])

    manager = SchemaMetadataManager(
        output_path
    )

    metadata = {"tables": {}}

    for name, df in tables.items():
        result = detect_table(name, df)
        metadata["tables"].update(result["tables"])

    manager.save(metadata)

    print(yaml.safe_dump(metadata, sort_keys=False))
