# MovieLens MIA Target Selection Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/select_mia_target_movielens.ipynb` — a parameterized notebook that selects outlier MIA targets from MovieLens `ncols_10/` datasets using frequency-based rarity scoring, following the same cell structure and conventions as the Adult notebook.

**Architecture:** Single Jupyter notebook with 10 cells (5 code + 5 markdown), mirroring the Adult notebook's structure. Steps 1/4 from Adult (p95 numerical scoring, column grouping) are omitted since MovieLens is all-Ordinal with no meaningful groups. The notebook reads a CSV + JSON pair and produces a list of `ID{i}` target identifiers.

**Tech Stack:** Python 3.9+, pandas, numpy, matplotlib

**Spec:** `docs/superpowers/specs/2026-04-16-select-mia-target-movielens-design.md`

**Reference:** `notebooks/select_mia_target_adult.ipynb` (the source notebook to mirror)

---

## File Map

- **Create:** `notebooks/select_mia_target_movielens.ipynb`

No other files are created or modified.

---

### Task 1: Create the notebook with all cells

Since this is a single Jupyter notebook with no module dependencies, we create it in one task. The notebook has 10 cells that mirror the Adult notebook's structure.

**Files:**
- Create: `notebooks/select_mia_target_movielens.ipynb`

- [ ] **Step 1: Create the notebook file**

Create `notebooks/select_mia_target_movielens.ipynb` with the following cells. Use the `NotebookEdit` tool to build cells one by one, or write the full `.ipynb` JSON.

**Cell 1 (code) — Data loading + parameters:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ─────────────────────────────────
DATA_DIR        = "data/movielens/ncols_10/eff_rank_best"  # change to switch dataset
RARE_THRESHOLD  = 0.01   # frequency threshold for rare values
N_TARGETS       = 5      # number of outlier records to select

# ── Data loading ───────────────────────────────
# Assign sequential IDs compatible with SDR's load_local_data_as_df
df_raw = pd.read_csv(f"{DATA_DIR}.csv")
df_raw["ID"] = [f"ID{i}" for i in range(len(df_raw))]
df = df_raw.set_index("ID")

feat_cols = df.columns.tolist()

print(f"Dataset: {DATA_DIR}")
print(f"Shape: {df.shape}")
print(f"Feature columns ({len(feat_cols)}): {feat_cols}")
```

**Cell 2 (code) — Distribution visualization:**

```python
# ── Distribution of each column ────────────────
n = len(feat_cols)

fig, axes = plt.subplots(
    nrows=(n + 2) // 3,
    ncols=3,
    figsize=(15, 4 * ((n + 2) // 3))
)
axes = axes.flatten()

# All columns are Ordinal (0-10) → bar chart
for ax, col in zip(axes, feat_cols):
    df[col].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title(col)
    ax.tick_params(axis='x', rotation=0)

# Hide unused subplots
for ax in axes[n:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
```

**Cell 3 (markdown) — Outlier Target Selection intro:**

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

**Cell 4 (markdown) — Step 1 heading:**

```markdown
### Step 1 — Rarity scoring

- **Criterion**: value frequency < `RARE_THRESHOLD` (default 1%)
- **Score**: `1 / frequency`  (rarer values score higher)
```

**Cell 5 (code) — Step 1: Rarity scoring:**

```python
cat_scores = pd.DataFrame(0.0, index=df.index, columns=feat_cols)
cat_rare_vals = {}

for col in feat_cols:
    freq = df[col].value_counts(normalize=True)
    rare_vals = freq[freq < RARE_THRESHOLD]
    cat_rare_vals[col] = rare_vals
    freq_series = df[col].map(freq.to_dict()).astype(float)
    cat_scores[col] = np.where(
        freq_series < RARE_THRESHOLD,
        1.0 / freq_series.clip(lower=1e-10),
        0.0,
    )

# Summary table of rare values
rare_summary_rows = []
for col, rare_vals in cat_rare_vals.items():
    for val, freq in rare_vals.items():
        rare_summary_rows.append({
            "column": col, "rare value": val,
            "frequency (%)": round(freq * 100, 3),
            "# records": round(freq * len(df)),
            "score": round(1.0 / freq, 1),
        })

rare_df = pd.DataFrame(rare_summary_rows).sort_values("frequency (%)").reset_index(drop=True)
display(rare_df)
```

**Cell 6 (markdown) — Step 2 heading:**

```markdown
### Step 2 — Score normalization

Normalize each column's scores to [0, 1] and combine into a single matrix.
Records with at least one non-zero score are **outlier candidates**.
```

**Cell 7 (code) — Step 2: Score normalization and combined matrix:**

```python
scores_raw = cat_scores[feat_cols]

# Normalize each column to [0, 1] by dividing by its maximum score
col_max = scores_raw.max()
scores_norm = scores_raw.div(col_max.where(col_max > 0, 1.0))

# Tiebreaker: add a tiny fraction of the raw score so that when two records
# share the same normalized score, the one with higher raw rarity wins
global_max = scores_raw.max().max()
scores = scores_norm + scores_raw * (1e-6 / global_max) if global_max > 0 else scores_norm

is_candidate = scores.gt(0).any(axis=1)
candidates = scores[is_candidate].copy()
candidates["total score"] = candidates[feat_cols].sum(axis=1)

print(f"Outlier candidates: {is_candidate.sum()} / {len(df)} records")

# Top 15 candidates (display with [0,1] normalized scores)
display(
    scores_norm[is_candidate]
    .assign(**{"total score": candidates["total score"]})
    .sort_values("total score", ascending=False)
    .head(15)
    .round(3)
    .style.background_gradient(cmap="YlOrRd", axis=0, subset=feat_cols)
)
```

**Cell 8 (markdown) — Step 3 heading:**

```markdown
### Step 3 — Final selection: top N records

Select the top `N_TARGETS` records by total normalized score.
Ties are broken by number of outlier columns (more = higher priority).
```

**Cell 9 (code) — Step 3: Top N selection:**

```python
candidates_sorted = candidates.copy()
candidates_sorted["n_outlier_cols"] = candidates_sorted[feat_cols].gt(0).sum(axis=1)
candidates_sorted = candidates_sorted.sort_values(
    ["total score", "n_outlier_cols"], ascending=[False, False]
)

selected_ids = candidates_sorted.index[:N_TARGETS].tolist()
result_rows = []

for sid in selected_ids:
    rec = scores.loc[sid]
    outlier_cols = rec[rec > 0].index.tolist()
    primary_col = rec.idxmax()
    result_rows.append({
        "selected ID":     sid,
        "primary column":  primary_col,
        "value":           df.loc[sid, primary_col],
        "norm. score":     round(rec[primary_col], 3),
        "outlier columns": ", ".join(outlier_cols),
    })

result_df = pd.DataFrame(result_rows).set_index("selected ID")
display(result_df)

print("\nSelected IDs:", selected_ids)
```

**Cell 10 (code) — Result inspection:**

```python
# Inspect the raw data of selected records
display(df.loc[selected_ids])
```

- [ ] **Step 2: Run the notebook to verify it executes without errors**

Run:
```bash
cd /Users/ktanaka/Developer/synthetic_data_release
uv run jupyter nbconvert --to notebook --execute notebooks/select_mia_target_movielens.ipynb --output select_mia_target_movielens_executed.ipynb
```

Expected: exits 0, produces output notebook with all cells executed.

- [ ] **Step 3: Verify output sanity**

Open the executed notebook and check:
- Cell 1 prints dataset shape `(200948, 10)` and 10 column names
- Cell 2 renders 10 bar charts
- Cell 5 produces a `rare_df` table with rare values
- Cell 7 shows outlier candidate count and heatmap
- Cell 9 shows 5 selected IDs with outlier columns
- Cell 10 shows raw data for 5 records

- [ ] **Step 4: Clean up executed notebook and commit**

Remove the executed copy (it was only for verification), then commit:

```bash
rm notebooks/select_mia_target_movielens_executed.ipynb
git add notebooks/select_mia_target_movielens.ipynb
git commit -m "feat: add MovieLens MIA target selection notebook

Frequency-based rarity scoring for ncols_10/ datasets,
mirroring the Adult notebook's structure and conventions."
```
