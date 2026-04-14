# GEMINI.md / CLAUDE.md guidelines

This file provides guidance to AI assistants when working with code in this repository.

## Project Overview

Privacy evaluation framework for synthetic data publishing, based on the paper "Synthetic Data - Anonymisation Groundhog Day" (Stadler, Oprisanu, Troncoso, 2020). Evaluates the privacy-utility tradeoff of different synthetic data generation and sanitisation techniques.

## Environment Setup

The project uses **uv** for dependency management (Python 3.9+). A `.venv` already exists in the repo root. Dependencies are declared in `pyproject.toml`. Note that this project uses a forked version of CTGAN (`git+https://github.com/SDC4Society/CTGAN.git`), which is included in `pyproject.toml`.

## Running Evaluations

There is a newly architected unified evaluation runner (`all_cli.py`) spanning the three components. Alternatively, `utility_cli.py`, `linkage_cli.py`, and `inference_cli.py` can be used individually.
Each entry point follows this pattern: `-D` for local data path, `-RC` for run config JSON, `-O` for output directory, `-W` for worker count, `--device` for computing unit designation.

```bash
# ALL-IN-ONE (Recommended)
uv run python all_cli.py -D data/texas -O outputs/all --device cpu

# Linkage privacy (MIA-based)
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage --device cpu

# Inference privacy (attribute inference)
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cuda:0

# Utility evaluation
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

## Running Tests

```bash
uv run python -m unittest discover tests/
uv run python -m unittest tests/test_gms.py
uv run python -m unittest tests/test_gms.TestGenerativeModel.test_bayesian_net
```
Test data lives in `tests/` (germancredit_test.csv). Main evaluation data is in `data/` (texas dataset). Some datasets are fetched from S3 on first use via `--s3name`.

## Architecture & Optimizations

### 1. Data Pipeline & Parallelization (`joblib`)
- **Shared IPC**: Evaluations dispatch `multiprocessing` processes using `joblib.Parallel` via the `loky` backend. This ensures incredibly fast memory-mapped views instead of duplicating pandas DataFrames for every task.
- **Resource Constraints**: CPU-based models automatically detect system boundaries, whereas GPU PyTorch/TensorFlow models (`PrivMRF`, `CTGAN`, `GEM`) force sequential queues (`max_workers=1`) whenever the flag `--device cuda:X` forces GPU usage, averting CUDA Context fragmentation.

### 2. Preprocessing & Sklearn Pipeline
- **Unified Pipeline Module**: Instead of messy custom for-loop encoding routines scattered around attack packages, datasets are ingested through `preprocess_common/pipeline.py` leveraging `sklearn.compose.ColumnTransformer`. 

### 3. Module Hierarchy

- **Generative Models** (`generative_models/`): Inherit from `GenerativeModel`. E.g., `CTGAN`, `PATEGAN`, `PrivSyn`, `TabDDPM`.
- **Attack Models** (`attack_models/`): Inherit from `PrivacyAttack`. Includes MIA classifiers (`mia_classifier.py`) and attribute reconstructors (`reconstruction.py`).
- **Feature Sets** (`feature_sets/`): MIA extraction layers (`NaiveFeatureSet`, `HistogramFeatureSet`).
- **Predictive Models** (`predictive_models/`): Classifiers or regressors measuring practical utility bounds against fake datasets.

## Development Guidelines (Instructions for AI)
1. **Dependency management**: Always suggest `uv run` or `uv add`. Never use standard `pip` directly.
2. **Device Awareness**: Always incorporate `--device` functionality or consider `SYNTHETIC_DATA_DEVICE` in environment variables when initializing PyTorch and ML instances.
3. **Extend with Standards**: Inject new classifiers and sanitizers seamlessly by passing properties down to standard `sklearn` factories unless intrinsically unsupported.