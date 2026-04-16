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

## Consistency with Adult Notebook

Both notebooks must follow the **same cell structure, heading style, and comment conventions** so that readers can see at a glance that the same selection methodology is applied across datasets. Differences should only arise where the data characteristics require it.

### Shared conventions

- **Markdown cells** use `## Step N — Title` headings (matching Adult's `### Step N` pattern)
- **Code cells** begin with `# ── Section title ──` banner comments for major blocks
- **Parameter names** use the same identifiers where applicable (`RARE_THRESHOLD`, `N_TARGETS`, `EXCLUDE_COLS`, `QUANTILE`)
- **ID assignment** uses the same `ID{i}` convention
- **Display style** uses the same `display()` + `.style.background_gradient()` for score matrices

### Structural mapping (Adult → MovieLens)

| Adult notebook | MovieLens notebook | Difference |
|---|---|---|
| Cell: Data loading + parameters | Cell: Data loading + parameters | `DATA_DIR` parameterized; no `EXCLUDE_COLS`; `QUANTILE` unused |
| Cell: Distribution visualization | Cell: Distribution visualization | Bar chart for discrete 0-10 instead of histogram+bar mix |
| Markdown: Outlier Target Selection intro | Markdown: Outlier Target Selection intro | Same text structure, adapted for frequency-only approach |
| Step 1 — Numerical p95 scoring | *(omitted)* | MovieLens has no continuous numerical columns |
| Step 2 — Categorical rarity scoring | Step 1 — Rarity scoring | Same logic, same variable names (`cat_scores`, `cat_rare_vals`, `rare_df`) |
| Step 3 — Combined score matrix | Step 2 — Score normalization | Same normalization + tiebreaker logic; input is rarity scores only |
| Step 4 — Column groups preview | *(omitted)* | No column grouping for MovieLens |
| Step 5 — Final selection (per-group) | Step 3 — Final selection (top-N) | Simplified: rank by total score, tiebreak by outlier column count |
| Cell: Result inspection | Cell: Result inspection | Same `display(df.loc[selected_ids])` pattern |

## Notebook Structure

### Cell 1 (Code) — Data loading + parameters

```python
import pandas as pd
import numpy as np

# ── Parameters ─────────────────────────────────
DATA_DIR        = "data/movielens/ncols_10/eff_rank_best"  # change to switch dataset
RARE_THRESHOLD  = 0.01   # frequency threshold for rare values
N_TARGETS       = 5      # number of outlier records to select

# ── Data loading ───────────────────────────────
df_raw = pd.read_csv(f"{DATA_DIR}.csv")
df_raw["ID"] = [f"ID{i}" for i in range(len(df_raw))]
df = df_raw.set_index("ID")

feat_cols = df.columns.tolist()
```

Same banner comment style (`# ── ... ──`) and ID assignment as Adult.

### Cell 2 (Code) — Distribution visualization

- Bar chart per column showing value distribution (0-10 discrete)
- Grid layout (`nrows/ncols=3`, `figsize`) matching Adult notebook style
- Same `plt.tight_layout()` / `plt.show()` pattern

### Cell 3 (Markdown) — Outlier Target Selection intro

Same structure as Adult's intro markdown cell:

```markdown
## Outlier Target Selection (MIA Paper §4.3)

Selection criterion from the paper:

> *"records that either have rare categorical attribute values or numerical values outside the attribute's 95% quantile"*

**Adaptation for MovieLens:**
All columns are Ordinal (0-10 integers). Since the value space is discrete and small,
we use **frequency-based rarity scoring only** (no p95 threshold).

**Selection pipeline:**

1. **Rarity scoring** — Flag records with rare values (frequency < 1%); score by inverse frequency
2. **Score matrix** — Normalize scores to [0, 1] and combine
3. **Top-N selection** — Pick the top-scoring records by total normalized score
```

### Cell 4 (Markdown) — Step 1 heading

```markdown
### Step 1 — Rarity scoring

- **Criterion**: value frequency < `RARE_THRESHOLD` (default 1%)
- **Score**: `1 / frequency`  (rarer values score higher)
```

Same formatting as Adult's Step 2 markdown.

### Cell 5 (Code) — Step 1: Rarity scoring

- Same variable names as Adult: `cat_scores`, `cat_rare_vals`, `freq`, `rare_df`
- Same summary table structure (column, rare value, frequency%, # records, score)
- Same `display(rare_df)` output

### Cell 6 (Markdown) — Step 2 heading

```markdown
### Step 2 — Score normalization

Normalize each column's scores to [0, 1] and combine into a single matrix.
```

### Cell 7 (Code) — Step 2: Score normalization and combined matrix

- Same logic as Adult's Step 3: `col_max`, `scores_norm`, tiebreaker via raw scores
- Same candidate count print and top-15 display with `.style.background_gradient(cmap="YlOrRd")`

### Cell 8 (Markdown) — Step 3 heading

```markdown
### Step 3 — Final selection: top N records

Select the top `N_TARGETS` records by total normalized score.
Ties are broken by number of outlier columns (more = higher priority).
```

### Cell 9 (Code) — Step 3: Top N selection

- Rank by total normalized score
- Tiebreak: number of columns with score > 0
- Same result table columns as Adult: selected ID, primary column, value, norm. score, outlier columns
- Same `display(result_df)` + `print("\nSelected IDs:", selected_ids)` pattern

### Cell 10 (Code) — Result inspection

```python
# Inspect the raw data of selected records
display(df.loc[selected_ids])
```

Same as Adult's final cell.
