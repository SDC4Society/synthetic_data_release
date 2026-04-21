# MovieLens dataset

A user-by-genre aggregated table generated from MovieLens ml-32m. One row = one user, one column = one genre, and each value is the user's most frequent (doubled) rating for that genre — an integer `Ordinal` in the range 1–10, obtained as the mode of `round(rating × 2)` over the user's ratings within that genre. `0` is reserved for unrated.

---

## Directory layout

```
data/movielens/
├── README.md
├── README_JP.md
├── ml32m_full.csv                       # rows=200,948, cols=20 (all genres)
├── ml32m_full.json                      # metadata
└── ncols_10/
    ├── eff_rank_best.csv                # rows=200,948, cols=10 (highest effective rank)
    ├── eff_rank_best.json
    ├── eff_rank_middle.csv              # rows=200,948, cols=10 (middle effective rank)
    ├── eff_rank_middle.json
    ├── eff_rank_worst.csv               # rows=200,948, cols=10 (lowest effective rank)
    └── eff_rank_worst.json
```

Four datasets are provided: `ml32m_full`, `ncols_10/eff_rank_best`, `ncols_10/eff_rank_middle`, and `ncols_10/eff_rank_worst`.

With `ncols_20` (= all genres) the best / middle / worst selections all collapse onto the same 20-column set, so they are consolidated into the single `ml32m_full` dataset.

---

## File meanings

### `*.csv`

- Rows = users (obtained by taking `userId` from `ratings.csv`, deduplicating, and sorting in ascending order; row `i` ⇔ `uids[i]`)
- Columns = genres
  - `ml32m_full`: all 20 genre columns of MovieLens ml-32m (19 named genres plus a `(no genres listed)` column)
  - `ncols_10/eff_rank_best`: the 10-genre subset that **maximizes** the effective rank of the user-by-genre rating matrix, found by searching over subsets
  - `ncols_10/eff_rank_worst`: the 10-genre subset that **minimizes** the effective rank
  - `ncols_10/eff_rank_middle`: the 10-genre subset whose effective rank is closest to the mean of the maximum and minimum effective ranks
- Values = the user's most frequent (doubled) rating for that genre (integer in 1–10, computed as the mode of `round(rating × 2)` over the user's ratings in that genre; `0` if the user has no rating in that genre)

### `*.json`

Metadata for the data loader. All columns are `Ordinal` with `size=11` and `i2s=["0",...,"10"]`. 

---

## MIA target selection

MIA target records are selected with the notebook [notebooks/select_mia_target_movielens.ipynb](../../notebooks/select_mia_target_movielens.ipynb). 

---

## Statistics summary

| Dataset | Rows | Cols |
|---|---|---|
| `ml32m_full`              | 200,948 | 20 |
| `ncols_10/eff_rank_best`  | 200,948 | 10 |
| `ncols_10/eff_rank_middle`| 200,948 | 10 |
| `ncols_10/eff_rank_worst` | 200,948 | 10 |
