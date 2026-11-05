# SDG

YAML-driven synthetic relational data generation with GNN-VAE and GNN-DDPM.
Both models use the shared encoder in `src/models/gnn/gnn.py`.

## Setup

Create a Python environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activate that environment or use its Python executable for the commands below.
Run commands from the project root. Datasets and generated artifacts are not
included in this repository. Set `data_dir` in the YAML to your dataset folder,
and review primary keys, relationships, PII mappings and detected column types.

## Workflow

Example for GNN-VAE; use `configs/gnnddpm.yaml` for GNN-DDPM:

```powershell
python -m src.data_handling.data_loading configs/gnnvae.yaml
python -m src.data_handling.dtype_detection configs/gnnvae.yaml
python -m src.data_handling.data_validation configs/gnnvae.yaml
python -m src.pre_processing.preprocessing_pipeline configs/gnnvae.yaml
python -m src.models.gnn.graph_builder configs/gnnvae.yaml
python -m src.scripts.train configs/gnnvae.yaml
python -m src.scripts.generate configs/gnnvae.yaml
```

Review the detected schema before preprocessing. Run `python -m src.pre_processing.pii
configs/gnnvae.yaml` separately to export source-table copies with fake PII.
The generation command restores fake PII directly on synthetic tables.

Split, prepared-data and graph output folders must be empty for a new build.
Training and generation create a separate run folder each time. Generation uses
the latest training run's best checkpoint by default; configure an explicit
checkpoint path to select another run.

## Current limits

- Processing and training use full in-memory tables and graphs.
- Splitting and generation support forests with one parent per child table.
- VAE prior generation samples rows independently. DDPM uses approximate
  conditional sweeps over a synthetic graph.
- Child counts are sampled from training-graph statistics. Generation currently
  requires the selected checkpoint's saved training graph.
- Fresh identifiers are strings. Imputed source missingness is not recovered.
- Review generation and quality reports, particularly range-clipping counts.
  Valid keys do not establish statistical quality, privacy or downstream utility.
