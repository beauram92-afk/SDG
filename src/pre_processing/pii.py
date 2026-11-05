"""Detect PII candidates and export copies with fake replacements.

Run: python -m src.pre_processing.pii configs/gnnvae.yaml
"""
import argparse
from pathlib import Path

import yaml
from faker import Faker

from src.data_handling.data_loading import load_tables


def detect_pii(tables, config):
    """YAML declarations take priority over column-name hints, which need review."""
    declared = config.get("pii_columns") or {}
    if not isinstance(declared, dict):
        raise ValueError("pii_columns must map table names to column/type mappings")
    for table, columns in declared.items():
        if table not in tables:
            raise ValueError(f"PII table not loaded: {table}")
        if not isinstance(columns, dict):
            raise ValueError(f"pii_columns.{table} must map columns to PII types")
        for column, kind in columns.items():
            if column not in tables[table].columns:
                raise ValueError(f"PII column not found: {table}.{column}")
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError(f"PII type must be a non-empty string: {table}.{column}")

    hints = {
        "name": "name", "full_name": "name", "first_name": "name",
        "last_name": "name", "customer_name": "name",
        "email": "email", "email_address": "email",
        "phone": "phone", "phone_number": "phone", "contact_no": "phone",
        "mobile": "phone", "mobile_number": "phone",
        "address": "address", "street_address": "address",
        "pincode": "postal_code", "postal_code": "postal_code", "zip_code": "postal_code",
        "city": "city", "state": "state",
        "account_number": "bank_account_number", "bank_account_number": "bank_account_number",
        "ifsc_code": "ifsc_code", "payment_reference": "payment_reference",
        "date_of_birth": "date_of_birth", "dob": "date_of_birth",
        "ssn": "government_id", "aadhaar": "government_id",
        "aadhaar_number": "government_id", "pan_number": "government_id",
        "passport_number": "government_id", "ip_address": "ip_address",
    }
    report = {"tables": {}}
    for table, df in tables.items():
        columns = {}
        for column in df.columns:
            explicit = (declared.get(table) or {}).get(column)
            normalized = str(column).strip().lower().replace(" ", "_").replace("-", "_")
            kind = explicit or hints.get(normalized)
            if kind:
                columns[str(column)] = {
                    "pii_type": kind,
                    "source": "yaml" if explicit else "column_name_hint",
                    "needs_review": not bool(explicit),
                }
        report["tables"][table] = columns
    return report


def main():
    parser = argparse.ArgumentParser(description="Report PII columns using a project YAML.")
    parser.add_argument("config", help="project YAML; relative paths use the current directory")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("data_dir"), str) or not config["data_dir"].strip():
        parser.error("YAML must contain a non-empty data_dir string")
    settings = config.get("pii") or {}
    if not isinstance(settings, dict):
        parser.error("pii must contain an output_path setting")
    output = settings.get("output_path")
    if not isinstance(output, str) or not output.strip():
        parser.error("YAML must contain pii.output_path")
    path = Path(output)
    data_dir = Path(config["data_dir"]).resolve()
    if path.resolve() == data_dir or data_dir in path.resolve().parents:
        parser.error("pii.output_path must be outside data_dir")
    output_dir = settings.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        parser.error("YAML must contain pii.output_dir")
    folder = Path(output_dir)
    if folder.resolve() == data_dir or data_dir in folder.resolve().parents:
        parser.error("pii.output_dir must be outside data_dir")
    tables = load_tables(config["data_dir"])
    report = detect_pii(tables, config)
    replaced = replace_pii(tables, report, config)
    if path.resolve() in {(folder / f"{table}.csv").resolve() for table in tables}:
        parser.error("PII report path conflicts with an output table")
    folder.mkdir(parents=True, exist_ok=True)
    for table, df in replaced.items():
        df.to_csv(folder / f"{table}.csv", index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    count = sum(len(columns) for columns in report["tables"].values())
    print(f"Reported {count} configured or suspected PII columns. Saved: {path}")
    print(f"Saved tables with fake replacements: {folder}")
    print("Column-name hints require review; unlisted columns may also contain PII.")


def replace_pii(tables, report, config):
    """Replace candidates on copies; reuse replacements of the same type/value."""
    fake = Faker(config.get("pii_locale", "en_IN"))
    fake.seed_instance(config.get("pii_seed", config.get("seed", 42)))
    providers = {
        "name": fake.name, "email": fake.email, "phone": fake.phone_number,
        "address": fake.address, "postal_code": fake.postcode,
        "city": fake.city, "state": fake.state,
        "bank_account_number": lambda: fake.numerify("################"),
        "ifsc_code": lambda: fake.bothify("????0######").upper(),
        "payment_reference": lambda: fake.bothify("FAKE-????????########"),
        "date_of_birth": lambda: fake.date_of_birth().isoformat(),
        "government_id": lambda: fake.bothify("FAKE-????????########"),
        "ip_address": fake.ipv4,
    }
    keys = set()
    for table, columns in (config.get("primary_keys") or {}).items():
        keys.update((table, column) for column in ([columns] if isinstance(columns, str) else columns))
    for table, spec in (config.get("tables") or {}).items():
        keys.update((table, column) for column in (spec.get("id_cols") or []))
    for relation in config.get("relationships") or []:
        keys.add((relation.get("parent_table_name", relation.get("parent_table")),
                  relation.get("parent_primary_key", relation.get("parent_column"))))
        keys.add((relation.get("child_table_name", relation.get("child_table")),
                  relation.get("child_foreign_key", relation.get("child_column"))))
    for table, columns in report["tables"].items():
        for column, info in columns.items():
            if (table, column) in keys:
                raise ValueError(f"PII replacement would change a declared key: {table}.{column}")
            if info["pii_type"] not in providers:
                raise ValueError(f"Unsupported PII type for {table}.{column}: {info['pii_type']}")
    result = {table: df.copy() for table, df in tables.items()}
    replacements = {}
    for table, columns in report["tables"].items():
        for column, info in columns.items():
            series = tables[table][column]
            kind = info["pii_type"]
            values = series.astype("string")
            present = series.notna() & values.str.strip().ne("").fillna(False)
            mapping = {}
            for value in values[present].unique():
                key = (kind, value)
                if key not in replacements:
                    replacement = providers[kind]()
                    if replacement == value:
                        replacement = "FAKE-" + replacement
                    replacements[key] = replacement
                mapping[value] = replacements[key]
            replaced = series.astype(object).copy()
            replaced.loc[present] = values[present].map(mapping)
            result[table][column] = replaced
    return result


if __name__ == "__main__":
    main()
