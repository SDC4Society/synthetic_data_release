# MIA Target Selection Notebook for MovieLens

**Date:** 2026-04-16
**Status:** Approved

## Goal

Create a notebook `notebooks/select_mia_target_movielens.ipynb` that selects outlier records (MIA attack targets) from MovieLens `ncols_10/` datasets using frequency-based rarity scoring.

## Context

- The existing `notebooks/select_mia_target_adult.ipynb` performs outlier target selection for the Adult dataset using a 5-step pipeline (numerical p95 scoring, categorical rarity scoring, normalization, column grouping, diversity selection).
- MovieLens data differs fundamentally: all columns are Ordinal (0-10 integers, 0 = no rating), no categorical/numerical distinction, no label column.
- The notebook is **not** shared as a common module — it is a self-contained, independently readable notebook for research reproducibility.

## Design Decisions

| Decision | Rationale |
|---|---|
| **Frequency-based rarity only** (no p95) | 0-10 discrete values have too few distinct values for p95 to provide meaningful granularity |
| **No column grouping** | Genre columns vary across datasets, making stable groups impractical; datasets have only 10 columns each |
| **Single notebook, parameterized** | The 3 `ncols_10/` datasets share the same 200,948 rows (users) — only column selection differs |
| **No common module extraction** | Only 2-4 datasets total; target selection is a key research design decision best kept visible in-notebook |

## Scope

### In scope
- `ncols_10/eff_rank_best`, `ncols_10/eff_rank_middle`, `ncols_10/eff_rank_worst`

### Out of scope
- `ml32m_full` (20 columns)
- Changes to the existing Adult notebook
- Common Python module extraction

## Notebook Structure

### Cell 1 — Parameters

```python
DATA_DIR = "data/movielens/ncols_10/eff_rank_best"  # change to switch dataset
RARE_THRESHOLD = 0.01   # frequency threshold for rare values
N_TARGETS = 5           # number of outlier records to select
```

No `EXCLUDE_COLS` needed (MovieLens has no label/fnlwgt equivalent).

### Cell 2 — Data loading

- Read `{DATA_DIR}.csv` and `{DATA_DIR}.json`
- Assign `ID{i}` index (consistent with SDR's `load_local_data_as_df`)
- Classify all columns as feature columns

### Cell 3 — Distribution visualization

- Bar chart per column showing value distribution (0-10 discrete)
- Grid layout matching Adult notebook style

### Cell 4 (Markdown) — Method explanation

Explain the frequency-based rarity criterion: values with frequency < `RARE_THRESHOLD` are considered rare; score = `1 / frequency`.

### Cell 5 — Rarity scoring

For each column:
1. Compute value frequencies (`value_counts(normalize=True)`)
2. Identify rare values (frequency < `RARE_THRESHOLD`)
3. Assign score `1 / frequency` to records with rare values, 0 otherwise

Output: summary table of rare values per column (column, value, frequency%, record count, score).

### Cell 6 (Markdown) — Score normalization explanation

### Cell 7 — Score normalization and combined matrix

- Normalize each column's scores to [0, 1] by dividing by column max
- Add tiebreaker from raw scores (same approach as Adult notebook)
- Display outlier candidate count and top 15 candidates with heatmap

### Cell 8 (Markdown) — Selection explanation

Explain: no column grouping; select top N by total normalized score, with ties broken by number of outlier columns.

### Cell 9 — Top N selection

- Rank candidates by total normalized score
- Tiebreak: number of columns with score > 0 (more outlier columns = higher priority)
- Select top `N_TARGETS` records
- Display: selected ID, primary column (highest score), value, normalized score, list of outlier columns

### Cell 10 — Result inspection

- Display raw data of selected records
- Print `selected_ids` list for use in downstream evaluation configs
