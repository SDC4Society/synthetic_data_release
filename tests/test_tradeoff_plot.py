"""Tests for utils.tradeoff_plot aggregation filters."""
from unittest import TestCase

import pandas as pd

from utils.tradeoff_plot import aggregate_tradeoff, add_method_param


def _mk_linkage():
    """Two FeatureSets with different PG so the filter is observable."""
    rows = []
    for fs, pg in [('Correlations', 0.9), ('Naive', 0.1)]:
        for param in [2, 5]:
            rows.append({
                'TargetModel': f'BayesianNetBins{param}',
                'FeatureSet': fs,
                'PrivacyGain': pg,
            })
    return pd.DataFrame(rows)


def _mk_inference():
    """Two SensitiveAttributes with different PG so the filter is observable."""
    rows = []
    for sa, pg in [('LengthOfStay', 0.8), ('RACE', -0.2)]:
        for param in [2, 5]:
            rows.append({
                'TargetModel': f'BayesianNetBins{param}',
                'SensitiveAttribute': sa,
                'PrivacyGain': pg,
            })
    return pd.DataFrame(rows)


def _mk_utility():
    rows = []
    for param in [2, 5]:
        rows.append({
            'TargetModel': f'BayesianNetBins{param}',
            'PredictionModel': 'LogisticRegression',
            'LabelVar': 'Income',
            'Accuracy': 0.7 + 0.01 * param,
        })
    return pd.DataFrame(rows)


class TestAggregateTradeoff(TestCase):

    def setUp(self):
        self.rl = _mk_linkage()
        self.ri = _mk_inference()
        self.ru = _mk_utility()

    def test_parse_target_model_columns(self):
        parsed = add_method_param(self.rl)
        self.assertTrue((parsed['Method'] == 'BayesianNet').all())
        self.assertSetEqual(set(parsed['Param']), {2.0, 5.0})

    def test_featureset_filter_selects_correlations(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter='Correlations',
            sensitive_attr_filter=None,
        )
        # With only 'Correlations' rows surviving, PG_link should equal 0.9.
        self.assertTrue((merged['PG_link'] == 0.9).all())

    def test_featureset_filter_selects_naive(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter='Naive',
            sensitive_attr_filter=None,
        )
        self.assertTrue((merged['PG_link'] == 0.1).all())

    def test_featureset_filter_none_averages_all(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter=None,
            sensitive_attr_filter=None,
        )
        # Average of 0.9 and 0.1 -> 0.5
        self.assertAlmostEqual(merged['PG_link'].iloc[0], 0.5)

    def test_sensitive_attr_filter_selects_lengthofstay(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter='Correlations',
            sensitive_attr_filter='LengthOfStay',
        )
        self.assertFalse(merged.empty)
        self.assertTrue((merged['PG_inf'] == 0.8).all())

    def test_sensitive_attr_filter_selects_race(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter='Correlations',
            sensitive_attr_filter='RACE',
        )
        self.assertTrue((merged['PG_inf'] == -0.2).all())

    def test_sensitive_attr_filter_none_averages_all(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter='Correlations',
            sensitive_attr_filter=None,
        )
        # Mean of 0.8 and -0.2 -> 0.3
        self.assertAlmostEqual(merged['PG_inf'].iloc[0], 0.3)

    def test_sensitive_attr_filter_unknown_produces_nan(self):
        merged = aggregate_tradeoff(
            self.rl, self.ri, self.ru,
            featureset_filter='Correlations',
            sensitive_attr_filter='DoesNotExist',
        )
        # Outer-joined with empty pg_inf -> PG_inf column is all NaN.
        self.assertTrue(merged['PG_inf'].isna().all())
        # PG_link side should still be populated.
        self.assertTrue((merged['PG_link'] == 0.9).all())
