# src/data_loading.py
"""Step 1: read the table files from a folder. Nothing else.

Run from the project root: python -m src.data_loading <data_folder>
Or use a YAML's data_dir: python -m src.data_loading configs/gnnvae.yaml
"""
import csv

from src.imports.lib import Dict, List, Optional, Path, argparse, logging, pd

logger = logging.getLogger("sdg")

TEXT_FILES = (".csv", ".tsv", ".txt", ".csv.gz")      # delimited text; the separator is detected
SUPPORTED = TEXT_FILES + (".parquet", ".xlsx")


def _suffix(path: Path) -> str:
    name = path.name.lower()
    return ".csv.gz" if name.endswith(".csv.gz") else path.suffix.lower()


def _table_name(path: Path) -> str:
    return path.name[: -len(_suffix(path))]


def _read_text_file(path: Path) -> pd.DataFrame:
    """CSV / TSV / TXT: detects the separator (, ; tab |) and falls back to latin-1 if the file is not UTF-8."""
    with open(path, "rb") as f:
        head = f.read(65536)
    sample = head.decode("utf-8-sig", errors="replace") if not path.name.lower().endswith(".gz") else ""
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter if sample else ","
    except csv.Error:
        sep = ","
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig")     # utf-8-sig removes the hidden marker Excel adds
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin-1")


def load_tables(data_dir: str, tables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Returns {table_name: DataFrame}. The file name without its extension is the table name."""
    folder = Path(data_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Data folder not found: {folder.resolve()}")

    files: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith(("~$", ".")) or _suffix(p) not in SUPPORTED:
            continue                                                # skips folders, Excel lock files, hidden files
        name = _table_name(p)
        if name in files:
            raise ValueError(f"Two files for table '{name}': {files[name].name} and {p.name}")
        files[name] = p
    if tables is not None:
        missing = [t for t in tables if t not in files]
        if missing:
            raise FileNotFoundError(f"Tables not found in {folder}: {missing}. Files present: {sorted(files)}")
        files = {t: files[t] for t in tables}
    if not files:
        raise ValueError(f"No {SUPPORTED} files found in {folder.resolve()}")

    result: Dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        suffix = _suffix(path)
        try:
            if suffix in TEXT_FILES:
                df = _read_text_file(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_excel(path)                            # first sheet only
        except Exception as e:
            raise ValueError(f"Could not read {path.name}: {type(e).__name__}: {e}") from e
        df.columns = [str(c).strip() for c in df.columns]
        result[name] = df
        logger.info("loaded %-25s rows=%-9d cols=%d", name, len(df), df.shape[1])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load tables from a folder or a YAML config's data_dir.")
    parser.add_argument("source", help="data folder or YAML config file; relative data_dir paths use the current directory")
    args = parser.parse_args()
    source = Path(args.source)
    data_dir = args.source
    if source.is_file() and source.suffix.lower() in (".yaml", ".yml"):
        import yaml

        with source.open(encoding="utf-8-sig") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict) or not isinstance(config.get("data_dir"), str) or not config["data_dir"].strip():
            parser.error("YAML must contain a non-empty 'data_dir' string")
        data_dir = config["data_dir"]
    tables = load_tables(data_dir)

    for table_name, df in tables.items():
        print(f"\nTable: {table_name}")
        print(df)