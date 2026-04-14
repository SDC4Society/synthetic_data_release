# GEMINI.md

This file provides guidance to Gemini Code Assist when working with code in this repository.

## Project Overview

Privacy evaluation framework for synthetic data publishing, based on the paper "Synthetic Data - Anonymisation Groundhog Day" (Stadler, Oprisanu, Troncoso, 2020). Evaluates the privacy-utility tradeoff of different synthetic data generation and sanitisation techniques.

## Environment Setup

The project uses **uv** for dependency management (Python 3.9+). A `.venv` already exists in the repo root. Dependencies are declared in `pyproject.toml`. Note that this project uses a forked version of CTGAN (`git+https://github.com/SDC4Society/CTGAN.git`), which is included in `pyproject.toml`.

## Running Evaluations

Three CLI entry points, all following the same pattern: `-D` for local data path, `-RC` for run config JSON, `-O` for output directory. A `--device` option can be used to specify `cpu` or `cuda:N`. If omitted, it defaults to `cuda:0` if a GPU is available, otherwise `cpu`.

```bash
# Linkage privacy (MIA-based) on CPU
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage --device cpu

# Inference privacy (attribute inference) on GPU
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cuda:0

# Utility evaluation (auto-detect device)
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

## Running Tests

```bash
# All tests
uv run python -m unittest discover tests/

# Single test file
uv run python -m unittest tests/test_gms.py

# Single test method
uv run python -m unittest tests/test_gms.TestGenerativeModel.test_bayesian_net
```

Test data lives in `tests/` (germancredit_test.csv + .json). Main evaluation data is in `data/` (texas dataset). Some datasets are fetched from S3 on first use via `--s3name`.

## Architecture

### Data Format
Datasets consist of a `.csv` file paired with a `.json` metadata file describing column types (`Categorical`, `Ordinal`, `Integer`, `Float`) and value mappings. The `load_local_data_as_df()` function in `utils/datagen.py` loads both and returns a DataFrame + metadata dict.

### Module Hierarchy

**Generative Models** (`generative_models/`): All inherit from `GenerativeModel` base class with `fit(data)` and `generate_samples(nsamples)` interface. Implementations: `IndependentHistogram`, `BayesianNet`, `PrivBayes` (in `data_synthesiser.py`), `CTGAN` (`ctgan.py`), `PATEGAN` (`pate_gan.py`).

**Attack Models** (`attack_models/`): All inherit from `PrivacyAttack` base class with `train()` and `attack()` interface. MIA classifier (`mia_classifier.py`) for linkage attacks, attribute reconstruction attacks (`reconstruction.py`) for inference attacks.

**Feature Sets** (`feature_sets/`): Feature extraction layers for MIA attacks. `NaiveFeatureSet`, `HistogramFeatureSet`, `CorrelationsFeatureSet`, `EnsembleFeatureSet` — all inherit from `FeatureSet` with an `extract(data)` method.

**Predictive Models** (`predictive_models/`): Utility task classifiers/regressors (RandomForest, LogReg, LinReg) used to measure data utility.

**Sanitisation Techniques** (`sanitisation_techniques/`): Traditional anonymisation methods (k-anonymity style `SanitiserNHS`).

### Evaluation Flow
Each CLI follows the same pattern:
1. Load data + metadata, parse run config JSON
2. Instantiate generative models and/or sanitisers from config params
3. Run a privacy/utility "game" over `nIter` iterations: sample raw data, train models, generate synthetic data, run attacks/utility tasks
4. Write results as JSON to the output directory

### Run Config
JSON files in `tests/{linkage,inference,utility}/runconfig.json` control experiment parameters: number of iterations, dataset sizes, target selection, which generative models/sanitisers to use and their hyperparameters (passed as positional args).

### Results Analysis
Results JSON files can be parsed with functions in `utils/analyse_results.py`. Jupyter notebooks in `notebooks/` provide visualization.

## Development Guidelines (Instructions for Gemini)

### Python Standards
- Enforce strict type hinting and Google-style docstrings.
- Use `uv` for all dependency management tasks.

### Extension Points
- **New Attacks:** Must inherit from `PrivacyAttack` in `attack_models/`.
- **New Features:** Must inherit from `FeatureSet` in `feature_sets/`.

### Testing Requirements
- Every new feature must include a corresponding test case in `tests/`.
- Use `germancredit_test.csv` for small-scale CI tests.

### Forbidden Practices
- Do not use standard `pip` commands; suggest `uv run` or `uv add`.
- Do not hardcode device IDs; always use the `--device` logic defined in CLI files.