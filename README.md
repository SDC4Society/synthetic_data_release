# Privacy evaluation framework for synthetic data publishing
A practical framework to evaluate the privacy-utility tradeoff of synthetic data publishing 

Based on "Stadler, T., Oprisanu, B., and Troncoso, C. (2022). In 31st USENIX Security Symposium (USENIX Security22), pages 1451–1468, Boston, MA. USENIX Association.", [official paper](https://www.usenix.org/conference/usenixsecurity22/presentation/stadler), [arXiv](https://arxiv.org/abs/2011.07018), [github](https://github.com/spring-epfl/synthetic_data_release)

# Attack models
The module `attack_models` so far includes

A privacy adversary to test for privacy gain with respect to linkage attacks modelled as a membership inference attack `MIAAttackClassifier`.

A simple attribute inference attack `AttributeInferenceAttack` that aims to infer a target's sensitive value given partial knowledge about the target record

# Generative models
The module `generative_models` so far includes:   
- `IndependentHistogram`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNet`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `PrivBayes`: A differentially private version of the BayesianNet model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PATE-GAN`: A differentially private generative adversarial network adapted from its original implementation by the [MLforHealth Lab](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)
- Additional models including `AIM`, `GEM`, `PrivMRF`, `PrivSyn`, `DP_MERF`, `TabDDPM`, `PrivateGSD`, and `RAPpp`.

# Setup

## Direct Installation

### Requirements
The framework and its building blocks have been developed and tested under Python 3.9+.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. All dependencies (including the CTGAN fork) are declared in `pyproject.toml`.

**For standard (CPU-only) environments:**
```bash
uv sync
```

**For GPU-accelerated environments (CUDA 12):**
If you have a compatible NVIDIA GPU, you can drastically speed up operations by installing the optional GPU dependencies (which pull `cuml-cu12`, `cudf-cu12`, and `cupy-cuda12x` directly from the NVIDIA package registry):
```bash
uv sync --extra gpu
```

To test your installation try to run:
```bash
uv run python -c "import ctgan"
uv run python -c "import cuml; print('GPU support ready!')"  # Only if you used --extra gpu
```

## Docker Distribution (not recommended)

For your convenience, Synthetic Data is also distributed as a Docker image containing Python 3.9 and CUDA 11.4.2.

**Note:** This distribution includes CUDA binaries, before downloading the image, ensure to read [its EULA](https://docs.nvidia.com/cuda/eula/index.html) and to agree to its terms.

```
docker pull springepfl/synthetic-data:latest
docker run -it --rm -v "$(pwd)/output:/output" -p 8888:8888 springepfl/synthetic-data
```

# Example runs

This repository utilizes a consolidated execution engine (`all_cli.py`) to easily string together Utility, Linkage (MIA), and Inference privacy evaluations without needing to repeatedly load common datasets into memory. Alternatively, you can run the individual tests separately.

### Consolidated Run (Recommended)
You can run all three evaluation games seamlessly. The CLI automatically schedules appropriate workers and dynamically manages PyTorch/CUDA resources.

```bash
uv run python all_cli.py -D data/texas -O outputs/texas -W 4 --device cpu
```
This single command triggers `-RCU` (Utility), `-RCL` (Linkage), and `-RCI` (Inference) configurations (which default to the `tests/*/runconfig.json` templates).

### Single Evaluation Runs
```bash
# Linkage privacy evaluation (MIA-based)
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage --device cpu

# Inference privacy evaluation (attribute inference)
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cuda:0

# Utility evaluation
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

### Note on Device Optimization & Hardware (GPU/CPU)
Almost all models (`AIM`, `GEM`, `TabDDPM`, `DP_MERF`, `CTGAN`, `PATEGAN`, `PrivMRF`, etc.) have been verified and modified for CPU/GPU parallelization.

1. **`--device` Flag**: Specifies device. Supports `cpu`, `cuda:0`, `cuda:1`, etc. If no device is specified, it will dynamically select CUDA if available.
2. **Environment Variable**: You can globally set the computation context via `export SYNTHETIC_DATA_DEVICE="cuda:0"`.
3. **Smart GPU Allocation**: For heavy PyTorch/JAX models, if the `SYNTHETIC_DATA_DEVICE` states "gpu" or "cuda" and the `GPU_MODELS` list covers the current algorithm, the framework will automatically restrict `multiprocessing` to `max_workers=1` serialization to strictly avoid CUDA Context conflicts and VRAM Out-Of-Memory thrashing.
4. **Shared IPC**: This framework uses `joblib.Parallel (loky)` to transmit massive input datasets to multiprocessing workers via OS Shared Memory, rather than generating gigabytes of duplicate pickle allocations.
5. **Sklearn Pipelines**: Internal data representation operates natively over `sklearn.compose.ColumnTransformer`, eliminating Python loop overhead for attribute One-Hot operations and StandardScaler application.

## Analyzing Results
The JSON files produced can be parsed within notebooks using the helper functions `load_results_linkage`, `load_results_inference`, and `load_results_utility` provided inside `utils/analyse_results.py` to plot ROC curves, utilities, and advantage differences.
