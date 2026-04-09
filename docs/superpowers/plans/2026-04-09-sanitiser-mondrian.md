# SanitiserMondrian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mondrian k-anonymization as an evaluation target alongside SanitiserNHS.

**Architecture:** New `SanitiserMondrian` class in `sanitisation_techniques/sanitiser_mondrian.py` wraps `k_anonymization` library's `ClassicMondrian` with `MEAN_MODE` strategy. A duck-typed `_InMemoryDataset` adapts our DataFrame+metadata to the library's `Dataset` interface. A `_FlatHierarchy` stub satisfies `get_max_ranges` for categorical QIDs. The base `Sanitiser` class gains a `get_output_metadata()` hook for future extensibility. CLI workers are wired to call it.

**Tech Stack:** Python 3.12, `k-anonymization` library (`ClassicMondrian`, `GroupAnonymizationBuiltIn`), pandas, scikit-learn (`SimpleImputer`), unittest.

**Spec:** `docs/superpowers/specs/2026-04-09-sanitiser-mondrian-design.md`

---

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Modify | `sanitisation_techniques/sanitiser.py` | Add `get_output_metadata()` to base class |
| Create | `sanitisation_techniques/sanitiser_mondrian.py` | `_FlatHierarchy`, `_InMemoryDataset`, `SanitiserMondrian` |
| Modify | `utils/parallel.py:11-22` | Import + registry entry |
| Modify | `linkage_cli.py:56-91` | Wire `get_output_metadata` in 2 workers |
| Modify | `inference_cli.py:103-155` | Wire `get_output_metadata` in 1 worker |
| Modify | `utility_cli.py:86-134` | Wire `get_output_metadata` in 1 worker |
| Modify | `tests/linkage/runconfig.default.json:15-22` | Add Mondrian entries |
| Modify | `tests/inference/runconfig.default.json:14-20` | Add Mondrian entries |
| Modify | `tests/utility/runconfig.default.json:13-19` | Add Mondrian entries |
| Modify | `tests/test_sanitisation.py` | Add Mondrian unit tests |
| Modify | `tests/test_parallel.py` | Add registry test |

---

### Task 1: Add `get_output_metadata` to base Sanitiser

**Files:**
- Modify: `sanitisation_techniques/sanitiser.py:4-8`

- [ ] **Step 1: Add the method to `Sanitiser` base class**

In `sanitisation_techniques/sanitiser.py`, add the `get_output_metadata` method after the `sanitise` method:

```python
""" Parent class for sanitisers """


class Sanitiser(object):

    def sanitise(self, data):
        """ Apply a privacy policy to the data. """
        return NotImplementedError('Method needs to be overwritten by a subclass')

    def get_output_metadata(self, input_metadata):
        """Return metadata describing the sanitiser's output schema.

        Default: identity (output schema matches input).
        Override in subclasses that transform the schema.
        """
        return input_metadata
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `uv run python3 -m unittest tests.test_sanitisation -v 2>&1 | tail -5`

Expected: Same result as before (tests may error due to LFS issue, but no new failures).

- [ ] **Step 3: Commit**

```bash
git add sanitisation_techniques/sanitiser.py
git commit -m "feat: add get_output_metadata hook to Sanitiser base class"
```

---

### Task 2: Create `SanitiserMondrian` with `_FlatHierarchy` and `_InMemoryDataset`

**Files:**
- Create: `sanitisation_techniques/sanitiser_mondrian.py`

- [ ] **Step 1: Create the file with all three classes**

Create `sanitisation_techniques/sanitiser_mondrian.py`:

```python
"""Mondrian k-anonymization sanitiser."""
from pandas import DataFrame
from sklearn.impute import SimpleImputer

from k_anonymization.algorithms.local_recoding.mondrian import ClassicMondrian
from k_anonymization.algorithms.local_recoding.local_recoding_algorithm import (
    GroupAnonymizationBuiltIn,
)
from k_anonymization.core.frame import ITableDF

from sanitisation_techniques.sanitiser import Sanitiser
from utils.constants import CATEGORICAL, ORDINAL, NUMERICAL


class _FlatHierarchy:
    """Minimal hierarchy stub for get_max_ranges().

    ClassicMondrian + MEAN_MODE only accesses hierarchies[idx].height
    via get_max_ranges. A flat hierarchy with height=1 is sufficient.
    """

    height = 1


class _InMemoryDataset:
    """Duck-typed substitute for k_anonymization.core.Dataset.

    Provides only the attributes accessed by ClassicMondrian and
    LocalRecodingAlgorithm.__init__, without disk I/O.
    """

    def __init__(self, df, qids, qids_idx, is_categorical, hierarchies):
        self.df = ITableDF(df)
        self.qids = qids
        self.qids_idx = qids_idx
        self.is_categorical = is_categorical
        self.hierarchies = hierarchies
        self.target = None


class SanitiserMondrian(Sanitiser):
    """Mondrian k-anonymization sanitiser (MEAN_MODE strategy).

    Parameters
    ----------
    metadata : dict
        Dataset metadata (texas.json format).
    k : int
        Privacy parameter k for k-anonymity.
    quids : list[str]
        Column names of quasi-identifiers (required, non-empty).
    drop_cols : list[str] or None
        Columns to exclude before anonymization.
    """

    def __init__(self, metadata, k=5, quids=None, drop_cols=None):
        if not quids:
            raise ValueError("SanitiserMondrian requires at least one QID")
        self.metadata = metadata
        self.k = int(k)
        self.quids = list(quids)
        self.drop_cols = list(drop_cols) if drop_cols else []
        self.datatype = DataFrame
        self.histogram_size = 10  # default for HistogramFeatureSet compatibility

        self.ImputerCat = SimpleImputer(strategy="most_frequent")
        self.ImputerNum = SimpleImputer(strategy="median")

        self.trained = False
        self.__name__ = f"SanitiserMondrianK{self.k}"

    def sanitise(self, data):
        """Sanitise a dataset using Mondrian k-anonymization.

        :param data: DataFrame: Raw dataset
        :return: DataFrame: Anonymized dataset
        """
        work = data.drop(columns=self.drop_cols, errors="ignore").copy()
        original_index = work.index
        original_columns = list(work.columns)

        missing = [q for q in self.quids if q not in work.columns]
        if missing:
            raise ValueError(f"QIDs not found in data: {missing}")

        # Impute missing values
        work = self._impute(work)

        # Build in-memory dataset for ClassicMondrian
        qids_idx = [original_columns.index(q) for q in self.quids]
        is_categorical = self._resolve_is_categorical()
        hierarchies = {idx: _FlatHierarchy() for idx in qids_idx}

        dataset = _InMemoryDataset(
            df=work.reset_index(drop=True),
            qids=self.quids,
            qids_idx=qids_idx,
            is_categorical=is_categorical,
            hierarchies=hierarchies,
        )

        # Run Mondrian
        algo = ClassicMondrian(
            dataset=dataset,
            k=self.k,
            group_anonymization=GroupAnonymizationBuiltIn.MEAN_MODE,
        )
        algo.anonymize()

        # Reconstruct DataFrame with original index
        anon = DataFrame(algo.anon_data.values, columns=original_columns)
        anon.index = original_index[: len(anon)]
        return anon

    def _resolve_is_categorical(self):
        """Determine is_categorical flags for each QID from metadata."""
        type_lookup = {}
        for cdict in self.metadata["columns"]:
            type_lookup[cdict["name"]] = cdict["type"]

        result = []
        for q in self.quids:
            qtype = type_lookup.get(q)
            if qtype in (CATEGORICAL, ORDINAL):
                result.append(True)
            elif qtype in NUMERICAL:
                result.append(False)
            else:
                # Default to categorical for safety
                result.append(True)
        return result

    def _impute(self, df):
        """Impute missing values: most_frequent for categorical, median for numerical."""
        cat_cols, num_cols = [], []
        for cdict in self.metadata["columns"]:
            col = cdict["name"]
            if col not in df.columns:
                continue
            if cdict["type"] in (CATEGORICAL, ORDINAL):
                cat_cols.append(col)
            elif cdict["type"] in NUMERICAL:
                num_cols.append(col)
        if cat_cols:
            df[cat_cols] = self.ImputerCat.fit_transform(df[cat_cols])
        if num_cols:
            df[num_cols] = self.ImputerNum.fit_transform(df[num_cols])
        return df
```

- [ ] **Step 2: Verify the module imports without error**

Run: `uv run python3 -c "from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add sanitisation_techniques/sanitiser_mondrian.py
git commit -m "feat: add SanitiserMondrian with ClassicMondrian + MEAN_MODE"
```

---

### Task 3: Verify `_FlatHierarchy` and `_InMemoryDataset` work with ClassicMondrian

This task validates that the duck-typed stubs actually work with the library before writing formal tests.

**Files:**
- None (verification only)

- [ ] **Step 1: Run a minimal Mondrian anonymization in REPL**

```bash
uv run python3 -c "
import pandas as pd
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian

# Inline test data: 8 rows, 3 categorical columns
metadata = {
    'columns': [
        {'name': 'Sex', 'type': 'Categorical', 'i2s': ['male', 'female'], 'size': 2},
        {'name': 'Job', 'type': 'Ordinal', 'i2s': ['unskilled', 'skilled', 'highly_skilled'], 'size': 3},
        {'name': 'Risk', 'type': 'Categorical', 'i2s': ['good', 'bad'], 'size': 2},
    ]
}
df = pd.DataFrame({
    'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'Job': ['unskilled', 'skilled', 'skilled', 'unskilled', 'skilled', 'highly_skilled', 'unskilled', 'skilled'],
    'Risk': ['good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'bad'],
})

san = SanitiserMondrian(metadata, k=2, quids=['Sex', 'Job'])
result = san.sanitise(df)
print(f'Input rows: {len(df)}, Output rows: {len(result)}')
print(f'Columns: {list(result.columns)}')
print(result)

# Verify all output values are from original categories
for col in ['Sex', 'Job', 'Risk']:
    i2s = [c for c in metadata['columns'] if c['name'] == col][0]['i2s']
    unexpected = set(result[col].unique()) - set(i2s)
    assert not unexpected, f'{col}: unexpected values {unexpected}'
print('All values within original categories: OK')
"
```

Expected: Prints input/output row counts, DataFrame contents, and "All values within original categories: OK".

- [ ] **Step 2: If `_FlatHierarchy` fails, switch to dummy with `get_lowest_common_ancestor`**

If step 1 errors on `hierarchies[idx]` accessing attributes beyond `.height`, replace `_FlatHierarchy` with:

```python
class _FlatHierarchy:
    height = 1

    def get_lowest_common_ancestor(self, values, get_type=None):
        if get_type == "height":
            return 1
        return "*"
```

Re-run step 1 to verify.

- [ ] **Step 3: If `ITableDF` wrapping fails, fall back to plain DataFrame**

If step 1 errors because `ITableDF` causes issues (e.g., `ipywidgets` import failure in some environments), change `_InMemoryDataset.__init__` to use `self.df = DataFrame(df)` instead of `self.df = ITableDF(df)`.

Re-run step 1 to verify.

---

### Task 4: Write unit tests for `SanitiserMondrian`

**Files:**
- Modify: `tests/test_sanitisation.py`

- [ ] **Step 1: Add Mondrian test class with inline test data**

Note: The existing `germancredit_test.csv` is behind git-lfs and may not be available. The Mondrian tests use inline DataFrame construction to avoid this dependency.

Add the following to `tests/test_sanitisation.py` after the existing `TestSanitisation` class (before the `write_to_dict` function at line 64):

```python
class TestSanitiserMondrian(TestCase):
    """Tests for SanitiserMondrian using inline test data."""

    @classmethod
    def setUpClass(cls):
        cls.metadata = {
            'columns': [
                {'name': 'Sex', 'type': 'Categorical', 'i2s': ['male', 'female'], 'size': 2},
                {'name': 'Job', 'type': 'Ordinal',
                 'i2s': ['unemployed', 'unskilled', 'skilled', 'highly_skilled'], 'size': 4},
                {'name': 'Housing', 'type': 'Categorical', 'i2s': ['own', 'free', 'rent'], 'size': 3},
                {'name': 'Age', 'type': 'Integer', 'min': 19, 'max': 75},
                {'name': 'Risk', 'type': 'Categorical', 'i2s': ['good', 'bad'], 'size': 2},
            ],
            'categorical_columns': [0, 1, 2, 4],
            'ordinal_columns': [1],
            'continuous_columns': [3],
        }
        cls.quids = ['Sex', 'Job', 'Housing']
        cls.raw = pd.DataFrame({
            'Sex': ['male', 'female', 'male', 'female', 'male', 'female',
                    'male', 'female', 'male', 'female', 'male', 'female'],
            'Job': ['unskilled', 'skilled', 'skilled', 'unskilled', 'skilled',
                    'highly_skilled', 'unskilled', 'skilled', 'skilled', 'unskilled',
                    'highly_skilled', 'skilled'],
            'Housing': ['own', 'rent', 'free', 'own', 'rent', 'free',
                        'own', 'rent', 'free', 'own', 'rent', 'free'],
            'Age': [25, 30, 45, 22, 50, 35, 28, 40, 55, 33, 60, 27],
            'Risk': ['good', 'bad', 'good', 'bad', 'good', 'bad',
                     'good', 'bad', 'good', 'bad', 'good', 'bad'],
        })
        cls.raw.index = [f'ID{i}' for i in range(len(cls.raw))]

    def test_sanitise_basic(self):
        """Anonymized output preserves columns and has <= input rows."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids)
        result = san.sanitise(self.raw)
        self.assertEqual(list(result.columns), list(self.raw.columns))
        self.assertLessEqual(len(result), len(self.raw))
        self.assertGreater(len(result), 0)

    def test_output_values_within_categories(self):
        """MEAN_MODE output for categorical QIDs stays within original i2s."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids)
        result = san.sanitise(self.raw)
        for cdict in self.metadata['columns']:
            if cdict['type'] in (CATEGORICAL, ORDINAL) and 'i2s' in cdict:
                col = cdict['name']
                valid = set(cdict['i2s'])
                actual = set(result[col].unique())
                self.assertTrue(actual.issubset(valid),
                    f'{col}: unexpected values {actual - valid}')

    def test_k_anonymity_holds(self):
        """All equivalence classes have at least k records."""
        k = 3
        san = SanitiserMondrian(self.metadata, k=k, quids=self.quids)
        result = san.sanitise(self.raw)
        if len(result) > 0:
            counts = result.groupby(self.quids).size()
            self.assertTrue((counts >= k).all(),
                f'Groups smaller than k={k}: {counts[counts < k].to_dict()}')

    def test_drop_cols(self):
        """Dropped columns are absent from output."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids, drop_cols=['Risk'])
        result = san.sanitise(self.raw)
        self.assertNotIn('Risk', result.columns)

    def test_quids_empty_raises(self):
        """Empty or None quids raises ValueError."""
        with self.assertRaises(ValueError):
            SanitiserMondrian(self.metadata, k=2, quids=None)
        with self.assertRaises(ValueError):
            SanitiserMondrian(self.metadata, k=2, quids=[])

    def test_quids_missing_col_raises(self):
        """QID referencing nonexistent column raises ValueError."""
        san = SanitiserMondrian(self.metadata, k=2, quids=['NONEXISTENT'])
        with self.assertRaises(ValueError):
            san.sanitise(self.raw)

    def test_get_output_metadata_identity(self):
        """get_output_metadata returns input metadata unchanged."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids)
        result = san.get_output_metadata(self.metadata)
        self.assertIs(result, self.metadata)

    def test_name_includes_k(self):
        """Model name includes k value for result identification."""
        san = SanitiserMondrian(self.metadata, k=7, quids=self.quids)
        self.assertEqual(san.__name__, 'SanitiserMondrianK7')

    def test_histogram_size_attribute(self):
        """histogram_size attribute exists for HistogramFeatureSet compatibility."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids)
        self.assertIsInstance(san.histogram_size, int)
```

Also add the necessary imports at the top of the file. After the existing import of `SanitiserNHS`, add:

```python
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian
import pandas as pd
```

- [ ] **Step 2: Run the Mondrian tests**

Run: `uv run python3 -m unittest tests.test_sanitisation.TestSanitiserMondrian -v 2>&1`

Expected: All tests PASS. If any fail, fix the implementation in `sanitiser_mondrian.py` per the error.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sanitisation.py
git commit -m "test: add SanitiserMondrian unit tests"
```

---

### Task 5: Register `SanitiserMondrian` in `MODEL_REGISTRY`

**Files:**
- Modify: `utils/parallel.py:11-22`

- [ ] **Step 1: Add import and registry entry**

In `utils/parallel.py`, add the import after line 11 (`from sanitisation_techniques.sanitiser_nhs import SanitiserNHS`):

```python
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian
```

Add the registry entry to `MODEL_REGISTRY` after line 21 (`"SanitiserNHS": SanitiserNHS,`):

```python
    "SanitiserMondrian": SanitiserMondrian,
```

The result should look like:

```python
from sanitisation_techniques.sanitiser_nhs import SanitiserNHS
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian

MODEL_REGISTRY = {
    "IndependentHistogram": IndependentHistogram,
    "BayesianNet": BayesianNet,
    "PrivBayes": PrivBayes,
    "CTGAN": CTGAN,
    "PATEGAN": PATEGAN,
    "SanitiserNHS": SanitiserNHS,
    "SanitiserMondrian": SanitiserMondrian,
}
```

- [ ] **Step 2: Add registry test to `tests/test_parallel.py`**

Add the following test method to the `TestCreateModel` class in `tests/test_parallel.py` (after the `test_create_sanitiser` method):

```python
    def test_create_sanitiser_mondrian(self):
        from utils.parallel import create_model, is_generative_model
        model = create_model(
            ("SanitiserMondrian", 5, ["Sex", "Job"], []),
            self.metadata
        )
        self.assertEqual(model.__name__, 'SanitiserMondrianK5')
        self.assertFalse(is_generative_model(model))
```

- [ ] **Step 3: Run the registry tests**

Run: `uv run python3 -m unittest tests.test_parallel.TestCreateModel -v 2>&1`

Expected: `test_create_sanitiser_mondrian` PASSES. Other tests may error due to LFS, but the new test should pass.

- [ ] **Step 4: Commit**

```bash
git add utils/parallel.py tests/test_parallel.py
git commit -m "feat: register SanitiserMondrian in MODEL_REGISTRY"
```

---

### Task 6: Wire `get_output_metadata` in CLI workers

**Files:**
- Modify: `linkage_cli.py:56-91` (attack worker, sanitiser branch)
- Modify: `linkage_cli.py:94-143` (eval worker, sanitiser branch)
- Modify: `inference_cli.py:103-155` (sanitiser eval worker)
- Modify: `utility_cli.py:86-134` (sanitiser eval worker)

The goal: in each worker's sanitiser branch, replace bare `metadata` with `attack_metadata` obtained from `model.get_output_metadata(metadata)`. Under the current identity implementation this is a no-op, but it wires up the hook for future use and ensures correctness.

- [ ] **Step 1: Modify `linkage_attack_worker` in `linkage_cli.py`**

In `linkage_cli.py`, the function `linkage_attack_worker` (starting line 56). After line 59 (`model.multiprocess = False`), add:

```python
    attack_metadata = metadata if is_generative_model(model) else model.get_output_metadata(metadata)
```

Then in the `else` branch (line 74-89), replace all occurrences of `metadata` with `attack_metadata`. The `else` branch should become:

```python
    else:
        sanA, labelsA = generate_mia_anon_data(
            model, target, rawA,
            runconfig['sizeRawT'],
            runconfig['nShadows'] * runconfig['nSynA'])

        for Feature in [NaiveFeatureSet(DataFrame),
                        HistogramFeatureSet(DataFrame, attack_metadata,
                                           nbins=model.histogram_size, quids=model.quids),
                        CorrelationsFeatureSet(DataFrame, attack_metadata, quids=model.quids),
                        EnsembleFeatureSet(DataFrame, attack_metadata,
                                          nbins=model.histogram_size,
                                          quasi_id_cols=model.quids)]:
            Attack = MIAttackClassifierRandomForest(metadata=attack_metadata, FeatureSet=Feature, quids=model.quids)
            Attack.train(sanA, labelsA)
            trained_attacks[Feature.__name__] = Attack
```

Note: The generative model branch (lines 62-73) keeps using `metadata` directly — no change needed there.

- [ ] **Step 2: Modify `linkage_eval_worker` in `linkage_cli.py`**

In function `linkage_eval_worker` (starting line 94). After line 98 (`model.multiprocess = False`), add:

```python
    attack_metadata = metadata if is_generative_model(model) else model.get_output_metadata(metadata)
```

The eval worker does not directly instantiate feature extractors or attacks (it uses pre-trained `attacks_for_model`), so no further changes are needed in this function. The `attack_metadata` variable is wired for future use.

- [ ] **Step 3: Modify `inference_eval_san_worker` in `inference_cli.py`**

In `inference_cli.py`, function `inference_eval_san_worker` (starting line 103). After line 108 (`model = create_model(model_config, metadata)`), add:

```python
    attack_metadata = model.get_output_metadata(metadata)
```

Then replace `metadata` with `attack_metadata` in the attack instantiation block (lines 111-115):

```python
    attacks = {}
    for sa, atype in sensitive_attrs.items():
        if atype == 'LinReg':
            attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=attack_metadata, quids=model.quids)
        elif atype == 'Classification':
            attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=attack_metadata, quids=model.quids)
```

- [ ] **Step 4: Modify `utility_eval_san_worker` in `utility_cli.py`**

In `utility_cli.py`, function `utility_eval_san_worker` (starting line 86). After line 92 (`model = create_model(model_config, metadata)`), add:

```python
    attack_metadata = model.get_output_metadata(metadata)
```

Then replace `metadata` in the utility task creation (line 93):

```python
    utility_tasks = [create_utility_task(cfg, attack_metadata) for cfg in utility_task_configs]
```

- [ ] **Step 5: Verify import is available**

In `linkage_cli.py`, `is_generative_model` is already imported at line 28. In `inference_cli.py` and `utility_cli.py`, verify `is_generative_model` is imported (it is, at their respective import lines from `utils.parallel`).

- [ ] **Step 6: Commit**

```bash
git add linkage_cli.py inference_cli.py utility_cli.py
git commit -m "feat: wire get_output_metadata in CLI workers for sanitiser metadata hook"
```

---

### Task 7: Update runconfig files

**Files:**
- Modify: `tests/linkage/runconfig.default.json`
- Modify: `tests/inference/runconfig.default.json`
- Modify: `tests/utility/runconfig.default.json`

- [ ] **Step 1: Add `SanitiserMondrian` to linkage runconfig**

In `tests/linkage/runconfig.default.json`, add `SanitiserMondrian` entries inside the `sanitisationTechniques` object, after the `SanitiserNHS` array:

```json
  "sanitisationTechniques": {
    "SanitiserNHS": [
      [10, 1, 0.99, 2, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 5, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 10, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 25, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]]
    ],
    "SanitiserMondrian": [
      [2, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [5, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [10, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [25, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []]
    ]
  }
```

- [ ] **Step 2: Add `SanitiserMondrian` to inference runconfig**

Same pattern in `tests/inference/runconfig.default.json`:

```json
  "sanitisationTechniques": {
    "SanitiserNHS": [
      [10, 1, 0.99, 2, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 5, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 10, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 25, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]]
    ],
    "SanitiserMondrian": [
      [2, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [5, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [10, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [25, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []]
    ]
  }
```

- [ ] **Step 3: Add `SanitiserMondrian` to utility runconfig**

Same pattern in `tests/utility/runconfig.default.json`:

```json
  "sanitisationTechniques": {
    "SanitiserNHS": [
      [10, 1, 0.99, 2, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 5, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 10, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]],
      [10, 1, 0.99, 25, [], ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"]]
    ],
    "SanitiserMondrian": [
      [2, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [5, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [10, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []],
      [25, ["PAT_STATE", "SEX_CODE", "RACE", "ETHNICITY", "PAT_AGE"], []]
    ]
  }
```

- [ ] **Step 4: Verify JSON is valid**

Run:
```bash
uv run python3 -c "
import json
for f in ['tests/linkage/runconfig.default.json', 'tests/inference/runconfig.default.json', 'tests/utility/runconfig.default.json']:
    json.load(open(f))
    print(f'{f}: valid')
"
```

Expected: All three print "valid".

- [ ] **Step 5: Commit**

```bash
git add tests/linkage/runconfig.default.json tests/inference/runconfig.default.json tests/utility/runconfig.default.json
git commit -m "config: add SanitiserMondrian to default runconfigs"
```

---

### Task 8: Run all Mondrian-related tests

**Files:** None (verification only)

- [ ] **Step 1: Run Mondrian unit tests**

Run: `uv run python3 -m unittest tests.test_sanitisation.TestSanitiserMondrian -v 2>&1`

Expected: All 9 tests PASS.

- [ ] **Step 2: Run registry tests**

Run: `uv run python3 -m unittest tests.test_parallel.TestCreateModel.test_create_sanitiser_mondrian -v 2>&1`

Expected: PASS.

- [ ] **Step 3: Run full test suite for regressions**

Run: `uv run python3 -m unittest discover tests/ -v 2>&1`

Expected: No new failures. Pre-existing failures (LFS-dependent tests) are acceptable.
