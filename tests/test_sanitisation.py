"""A template file for writing a simple test for a sanitisation technique"""
from unittest import TestCase

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from sanitisation_techniques.sanitiser_nhs import SanitiserNHS
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian
import pandas as pd

from utils.datagen import load_local_data_as_df
from utils.constants import *


class TestSanitisation(TestCase):

    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.sizeS = len(self.raw)

    def test_sanitise_nhs(self):
        print('\nTest SanitiserNHS')

        ## Test default params
        sanitiser = SanitiserNHS(self.metadata)
        san = sanitiser.sanitise(self.raw)

        # Expect no columns to be dropped or rows removed
        self.assertTupleEqual(san.shape, self.raw.shape)

        ## Test dropping columns
        sanitiser = SanitiserNHS(self.metadata, drop_cols=['Purpose'])
        san = sanitiser.sanitise(self.raw)

        # Purpose should be dropped
        self.assertTrue('Purpose' not in list(san))

        ## Test rare value threshold
        sanitiser = SanitiserNHS(self.metadata, thresh_rare=2)
        san = sanitiser.sanitise(self.raw)

        for cdict in self.metadata['columns']:
            if cdict['type'] == CATEGORICAL or cdict['type'] == ORDINAL:
                counts = san[cdict['name']].value_counts()
                self.assertTrue(len(counts[counts > 2]) == len(counts))

        ## Test converting numerical into categorical attributes
        demographics = ['Age', 'Sex', 'Job', 'Housing']
        sanitiser = SanitiserNHS(self.metadata, quids=demographics)
        san = sanitiser.sanitise(self.raw)

        self.assertListEqual([type(str) for _ in demographics], list(san[demographics].dtypes))

        ## Test k-anonymity constraint
        sanitiser = SanitiserNHS(self.metadata, quids=demographics, anonymity_set_size=7)
        san = sanitiser.sanitise(self.raw)

        counts = san.groupby(demographics).size()
        self.assertTrue(len(counts[counts >= 7]) == len(counts))


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

    def test_get_output_metadata_no_integer_qid(self):
        """get_output_metadata returns equal metadata when no Integer QIDs."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids)
        result = san.get_output_metadata(self.metadata)
        self.assertEqual(result, self.metadata)
        self.assertIsNot(result, self.metadata)  # deep copy

    def test_get_output_metadata_integer_qid_becomes_float(self):
        """get_output_metadata converts Integer QIDs to Float."""
        san = SanitiserMondrian(self.metadata, k=2, quids=['Sex', 'Age'])
        result = san.get_output_metadata(self.metadata)
        age_col = next(c for c in result['columns'] if c['name'] == 'Age')
        self.assertEqual(age_col['type'], 'Float')
        # non-QID Integer columns should remain unchanged
        sex_col = next(c for c in result['columns'] if c['name'] == 'Sex')
        self.assertEqual(sex_col['type'], 'Categorical')

    def test_name_includes_k(self):
        """Model name includes k value for result identification."""
        san = SanitiserMondrian(self.metadata, k=7, quids=self.quids)
        self.assertEqual(san.__name__, 'SanitiserMondrianK7')

    def test_histogram_size_attribute(self):
        """histogram_size attribute exists for HistogramFeatureSet compatibility."""
        san = SanitiserMondrian(self.metadata, k=2, quids=self.quids)
        self.assertIsInstance(san.histogram_size, int)


def write_to_dict(nr, results):
    results[nr] = 'a'

