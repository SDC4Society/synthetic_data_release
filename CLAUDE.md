# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Privacy evaluation framework for synthetic data publishing, based on the paper "Synthetic Data - Anonymisation Groundhog Day" (Stadler, Oprisanu, Troncoso, 2020). Evaluates the privacy-utility tradeoff of different synthetic data generation and sanitisation techniques.

## Environment Setup

The project uses **uv** for dependency management (Python 3.9+). A `.venv` already exists in the repo root. Dependencies are declared in `pyproject.toml` (core) and `requirements.txt` (full, includes TensorFlow/PyTorch).

The CTGAN model requires a forked version at `CTGAN/` (subdir). Install it with `cd CTGAN && make install` and ensure `CTGAN/` is on `PYTHONPATH`.

## Running Evaluations

Three CLI entry points, all following the same pattern: `-D` for local data path, `-RC` for run config JSON, `-O` for output directory.

```bash
# Linkage privacy (MIA-based)
python3 linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage

# Inference privacy (attribute inference)
python3 inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference

# Utility evaluation
python3 utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

## Running Tests

```bash
# All tests
python3 -m unittest discover tests/

# Single test file
python3 -m unittest tests/test_gms.py

# Single test method
python3 -m unittest tests.test_gms.TestGenerativeModel.test_bayesian_net
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
