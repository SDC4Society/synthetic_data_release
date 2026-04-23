"""Mondrian k-anonymization sanitiser."""
import copy

from pandas import DataFrame
from sklearn.impute import SimpleImputer

from k_anonymization.algorithms.local_recoding.mondrian import ClassicMondrian
from k_anonymization.algorithms.local_recoding.local_recoding_algorithm import (
    GroupAnonymizationBuiltIn,
)
from k_anonymization.core.frame import ITableDF

from sanitisation_techniques.sanitiser import Sanitiser
from utils.constants import CATEGORICAL, FLOAT, INTEGER, ORDINAL, NUMERICAL


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
        self.k = int(k)
        self.quids = list(quids)
        self.drop_cols = list(drop_cols) if drop_cols else []
        self.metadata = self._read_meta(metadata)
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

    def _read_meta(self, input_metadata):
        """Integer QIDs become Float after MEAN_MODE generalization."""
        output = {"columns": [col for col in input_metadata["columns"] if col["name"] not in self.drop_cols]}
        for col in output["columns"]:
            if col["name"] in self.quids and col["type"] == INTEGER:
                col["type"] = FLOAT
        return output

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
        """Impute missing values: most_frequent for categorical, median for numerical.

        Returns a new DataFrame; does not mutate the input.
        """
        df = df.copy()
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

class SanitiserNHSMondrian(SanitiserMondrian):

    def __init__(self, metadata, k=5, quids=None, drop_cols=None, thresh_rare=0, max_quantile = 1):
        super().__init__(metadata, k, quids, drop_cols)
        self.unique_threshold = thresh_rare
        self.max_quantile = max_quantile
        
        self.__name__ = f"SanitiserNHSMondrianK{self.k}"

    def sanitise(self, data):
        
        work = data.drop(columns=self.drop_cols, errors="ignore").copy()

        missing = [q for q in self.quids if q not in work.columns]
        if missing:
            raise ValueError(f"QIDs not found in data: {missing}")

        # Impute missing values
        work = self._impute(work)

        drop_records = []

        for cdict in self.metadata["columns"]:
            col = cdict["name"]
            coltype = cdict['type']
            col_data = work[col].copy()

            if coltype == FLOAT or coltype == INTEGER:
                col_data = col_data.astype(int)

                # Cap numerical attributes
                cap = col_data.quantile(self.max_quantile)
                idx = col_data[col_data > cap].index
                col_data.loc[idx] = int(cap)

            elif coltype == CATEGORICAL or coltype == ORDINAL:
                # Remove any records with rare categories
                frequencies = col_data.value_counts()
                drop_cats = frequencies[frequencies <= self.unique_threshold].index
                
                if not drop_cats.empty:
                    ridx = col_data[col_data.isin(drop_cats)].index
                    drop_records.extend(ridx.tolist())

            work[col] = col_data.values

        drop_records = list(set(drop_records))
        work = work.drop(drop_records)

        return super().sanitise(work)
