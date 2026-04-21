"""Generate synthetic low-rank ordinal datasets for the synthetic_data_release experiments.

Reproduces the data-generation procedure used in the
`Mondorian_vs_effective_rank` repository (`generate_discrete_lowrank`) and
writes the results in the CSV + JSON metadata format consumed by
`utils/datagen.load_local_data_as_df`.

Two datasets are produced, both with `p=10` columns and `K=5` ordinal bins:

* ``eff_rank_best``  -- rank ``r=3``  (low effective rank, Mondrian-favourable).
* ``eff_rank_worst`` -- rank ``r=10`` (full rank, Mondrian-unfavourable).

Running this script with the default arguments always produces byte-identical
output because the random seed is fixed per dataset. To regenerate, simply run

    python generate_synthetic_lowrank.py

from any working directory. The files are written next to this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core generator (copied from Mondorian_vs_effective_rank/comparison_experiment.py
# so the script is self-contained and does not depend on that repository).
# ---------------------------------------------------------------------------
def generate_discrete_lowrank(
    n: int,
    p: int,
    r: int,
    K: int = 5,
    snr: float = 2.0,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Generate an n x p integer matrix with a rank-r latent signal.

    Mirrors the function of the same name in
    ``Mondorian_vs_effective_rank/comparison_experiment.py``.

    Args:
        n: Number of rows (samples).
        p: Number of columns (attributes).
        r: Target rank of the latent signal.
        K: Number of discrete ordinal bins per attribute.
        snr: Signal-to-noise ratio (variance ratio); 0 means noise equals signal std.
        seed: Random seed for reproducibility.

    Returns:
        X: Integer array of shape (n, p) with values in {0, ..., K-1}.
        stable_rank: Stable rank of the noiseless signal (sum(sv^2) / sv_max^2).
    """
    if seed is not None:
        np.random.seed(seed)

    U = np.random.randn(n, r) / np.sqrt(r)
    V = np.random.randn(p, r) / np.sqrt(r)
    Eta_signal = U @ V.T  # n x p matrix with rank exactly r

    signal_std = np.std(Eta_signal)
    noise_std = signal_std / np.sqrt(snr) if snr > 0 else signal_std
    Eta = Eta_signal + np.random.randn(n, p) * noise_std

    X = np.zeros((n, p), dtype=int)
    for j in range(p):
        thresholds = np.percentile(Eta[:, j], np.linspace(0, 100, K + 1)[1:-1])
        X[:, j] = np.digitize(Eta[:, j], thresholds)

    sv = np.linalg.svd(Eta_signal, compute_uv=False)
    stable_rank = float((sv ** 2).sum() / sv[0] ** 2)
    return X, stable_rank


# ---------------------------------------------------------------------------
# Dataset specifications.
#
# Each entry describes one dataset; seeds are fixed so repeated runs of this
# script produce byte-identical CSV/JSON output.
# ---------------------------------------------------------------------------
DEFAULT_N = 20000
DEFAULT_P = 10
DEFAULT_K = 5
DEFAULT_SNR = 2.0

DATASETS = {
    "eff_rank_best": {
        "r": 3,
        "seed": 42,
        "description": "Low effective rank (r=3) -- Mondrian-favourable case.",
    },
    "eff_rank_worst": {
        "r": 10,
        "seed": 43,
        "description": "Full rank (r=10) -- Mondrian-unfavourable case.",
    },
}


def _column_names(p: int) -> list[str]:
    return [f"attr_{j}" for j in range(p)]


def _build_metadata(K: int, col_names: list[str]) -> dict:
    """Schema understood by ``utils/datagen.load_local_data_as_df``."""
    return {
        "columns": [
            {
                "name": name,
                "type": "Ordinal",
                "size": K,
                "i2s": [str(k) for k in range(K)],
            }
            for name in col_names
        ]
    }


def _write_dataset(
    out_dir: Path,
    name: str,
    X: np.ndarray,
    col_names: list[str],
    K: int,
    stable_rank: float,
) -> None:
    """Persist a single dataset as ``<name>.csv`` + ``<name>.json``."""
    df = pd.DataFrame(X, columns=col_names)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    df.to_csv(csv_path, index=False)

    metadata = _build_metadata(K, col_names)
    with json_path.open("w") as fh:
        json.dump(metadata, fh, indent=4)

    print(
        f"[{name}] rows={len(df):>6d} cols={len(col_names):>2d} "
        f"K={K} stable_rank={stable_rank:.3f} -> {csv_path.name}, {json_path.name}"
    )


def generate_all(
    out_dir: Path | None = None,
    n: int = DEFAULT_N,
    p: int = DEFAULT_P,
    K: int = DEFAULT_K,
    snr: float = DEFAULT_SNR,
) -> None:
    """Generate all datasets defined in ``DATASETS`` into ``out_dir``."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    col_names = _column_names(p)

    for name, spec in DATASETS.items():
        X, stable_rank = generate_discrete_lowrank(
            n=n, p=p, r=spec["r"], K=K, snr=snr, seed=spec["seed"],
        )
        _write_dataset(out_dir, name, X, col_names, K, stable_rank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: this script's directory).",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Rows per dataset.")
    parser.add_argument("--p", type=int, default=DEFAULT_P, help="Columns per dataset.")
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="Ordinal bins per column.")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR, help="Signal-to-noise ratio.")
    args = parser.parse_args()

    generate_all(out_dir=args.out_dir, n=args.n, p=args.p, K=args.K, snr=args.snr)


if __name__ == "__main__":
    main()
