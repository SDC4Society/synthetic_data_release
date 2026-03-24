# Dataset Format and Custom Dataset Guide

This document describes the dataset format used in the evaluation framework and how to prepare your own custom datasets.

## Overview

Every dataset in this framework consists of exactly two files:

1. **A CSV file** (`.csv`) — contains the actual data as a flat table
2. **A JSON metadata file** (`.json`) — describes each column's type and value domain

Both files must share the same base name and reside in the same directory (e.g., `data/texas.csv` and `data/texas.json`).

## CSV File Format

A standard CSV with a header row. The data must be a single flat table — no nested structures or multi-table relationships.

```csv
DISCHARGE,TYPE_OF_ADMISSION,SEX_CODE,LENGTH_OF_STAY,TOTAL_CHARGES
2013Q4,3,F,5,12345.67
2013Q1,1,M,3,8901.23
```

Key points:
- One header row, followed by data rows
- No index column required (an `ID` column is added automatically at load time)
- Missing values for categorical/ordinal columns are handled as `FILLNA_VALUE_CAT` internally

## JSON Metadata Format

The JSON file contains a single `columns` array. Each entry defines one column with a `name` and `type`, plus type-specific fields.

### Column Types

There are four supported column types:

#### Categorical

Unordered discrete values. Requires `i2s` (index-to-string mapping) listing all possible values.

```json
{
  "name": "SEX_CODE",
  "type": "Categorical",
  "size": 3,
  "i2s": ["F", "M", "U"]
}
```

#### Ordinal

Ordered discrete values. Same structure as Categorical, but the order of `i2s` is meaningful (first = lowest, last = highest).

```json
{
  "name": "RISK_MORTALITY",
  "type": "Ordinal",
  "size": 5,
  "i2s": ["0", "1", "2", "3", "4"]
}
```

#### Integer

Integer-valued continuous column. Requires `min` and `max`.

```json
{
  "name": "LENGTH_OF_STAY",
  "type": "Integer",
  "min": 1,
  "max": 986
}
```

#### Float

Floating-point continuous column. Requires `min` and `max`.

```json
{
  "name": "TOTAL_CHARGES",
  "type": "Float",
  "min": 0.0,
  "max": 3293072.0
}
```

### Full JSON Example

```json
{
  "columns": [
    {
      "name": "Sex",
      "type": "Categorical",
      "size": 2,
      "i2s": ["male", "female"]
    },
    {
      "name": "Job",
      "type": "Ordinal",
      "size": 4,
      "i2s": ["unemployed", "unskilled", "skilled", "management"]
    },
    {
      "name": "Age",
      "type": "Float",
      "min": 19.0,
      "max": 75.0
    },
    {
      "name": "Credit amount",
      "type": "Float",
      "min": 250.0,
      "max": 18424.0
    }
  ]
}
```

## Using a Custom Dataset

### Step 1: Prepare the Data

If your data comes from multiple tables (e.g., a relational database), flatten it into a single table first by joining the tables. The result should be one row per record with all attributes as columns.

### Step 2: Create the CSV

Export the flat table as a CSV file with a header row. Place it in the `data/` directory (or any directory of your choice).

```
data/mydataset.csv
```

### Step 3: Create the JSON Metadata

Create a JSON file with the same base name, describing every column in the CSV:

```
data/mydataset.json
```

For each column, determine:

| Question | Type to use | Required fields |
|---|---|---|
| Discrete values, no inherent order? | `Categorical` | `name`, `type`, `size`, `i2s` |
| Discrete values, with a meaningful order? | `Ordinal` | `name`, `type`, `size`, `i2s` |
| Whole numbers? | `Integer` | `name`, `type`, `min`, `max` |
| Decimal numbers? | `Float` | `name`, `type`, `min`, `max` |

- `size` = number of distinct values
- `i2s` = list of all possible values (as strings)
- `min` / `max` = value range in the data

### Step 4: Run Evaluations

Use the `-D` flag to point to your dataset (without file extension):

```bash
# Linkage evaluation
python3 linkage_cli.py -D data/mydataset -RC tests/linkage/runconfig.json -O output/linkage

# Inference evaluation
python3 inference_cli.py -D data/mydataset -RC tests/inference/runconfig.json -O output/inference

# Utility evaluation
python3 utility_cli.py -D data/mydataset -RC tests/utility/runconfig.json -O output/utility
```

You may also need to adjust the run config JSON files (e.g., target columns, sensitive attributes) to match your dataset's columns.

### Step 5: Load Programmatically (Optional)

You can also load your dataset directly in Python:

```python
from utils.datagen import load_local_data_as_df, load_local_data_as_array

# As a pandas DataFrame
df, metadata = load_local_data_as_df('data/mydataset')

# As a NumPy array
data, metadata = load_local_data_as_array('data/mydataset')
```

The returned `metadata` dict is augmented with `categorical_columns`, `ordinal_columns`, and `continuous_columns` — lists of column indices grouped by type.

## Existing Datasets

| Dataset | Location | Description |
|---|---|---|
| Texas hospital discharge | `data/texas` | Main evaluation dataset (18 columns: demographics, medical, charges) |
| German Credit | `data/germancredit` | Credit risk dataset (10 columns: personal, financial, risk) |
| German Credit (test subset) | `tests/germancredit_test` | Small subset used for unit tests |

Additional datasets can be fetched from S3 using the `--s3name` CLI option.
