import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.datagen import load_local_data_as_df


class _FakeTorchTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self


class _FakeGEMMechanism:
    def __init__(self):
        self.mean = 'mean'
        self.std = 'std'
        self.calls = 0
        self.transformer = types.SimpleNamespace(
            inverse_transform=lambda tensor, _: DataFrame({'workclass': tensor.values[:, 0]})
        )

    def generate_fake_data(self, mean, std, resample=False):
        self.calls += 1
        return np.array([[self.calls], [self.calls]], dtype=float)

    def get_onehot(self, fake_data):
        return _FakeTorchTensor(fake_data)


class _FakeSynMechanism:
    def __init__(self, frame):
        self.frame = frame

    def syn(self, nsamples, preprocesser=None, path=None, **kwargs):
        preprocesser.reverse_data(self.frame.head(nsamples).copy(), path)
        return None


class TestWrapperRegressions(unittest.TestCase):
    def test_aim_decode_tolerates_float_codes(self):
        from generative_models.aim import AIM

        _, metadata = load_local_data_as_df('data/adult')
        model = AIM(metadata)
        model._reverse_maps = {'workclass': {0: 'Private', 1: 'Self-emp-not-inc'}}

        decoded = model._decode_data(DataFrame({'workclass': [1.0, 0.0]}))
        self.assertEqual(decoded['workclass'].tolist(), ['Self-emp-not-inc', 'Private'])

    def test_dp_merf_preserves_numeric_dtype_when_cat_present(self):
        from generative_models.dp_merf import DP_MERF

        _, metadata = load_local_data_as_df('data/adult')
        model = DP_MERF(metadata)
        model._numeric_columns = ['age']
        model._categorical_columns = ['workclass']
        model._original_columns = ['age', 'workclass']
        model._original_label_column = None

        synthetic = model._arrays_to_dataframe(
            x_num=np.array([[30.0], [40.0]]),
            x_cat=np.array([['Private'], ['State-gov']], dtype=object),
            y=np.array([0, 0]),
        )
        self.assertTrue(np.issubdtype(synthetic['age'].dtype, np.number))
        self.assertEqual(synthetic['workclass'].tolist(), ['Private', 'State-gov'])

    def test_gem_generate_samples_resamples_each_batch(self):
        from generative_models.gem import GEM

        _, metadata = load_local_data_as_df('data/adult')
        model = GEM(metadata, batch_size=2, device='cpu')
        model.trained = True
        model.mechanism = _FakeGEMMechanism()
        model._reverse_maps = {'workclass': {1: 'Self-emp-not-inc', 2: 'Local-gov'}}

        fake_torch = types.SimpleNamespace(
            cat=lambda tensors, dim=0: _FakeTorchTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))
        )

        with patch.dict(sys.modules, {'torch': fake_torch}):
            synthetic = model.generate_samples(3)

        self.assertEqual(model.mechanism.calls, 2)
        self.assertEqual(synthetic['workclass'].tolist(), ['Self-emp-not-inc', 'Self-emp-not-inc', 'Local-gov'])

    def test_private_gsd_uses_preprocesser_output_from_syn(self):
        from generative_models.private_gsd import PrivateGSD

        _, metadata = load_local_data_as_df('data/adult')
        model = PrivateGSD(metadata)
        model.trained = True
        model._reverse_maps = {
            'age': {0: 25},
            'workclass': {1: 'Self-emp-not-inc'},
        }
        model.mechanism = _FakeSynMechanism(DataFrame({'age': [0], 'workclass': [1]}))

        synthetic = model.generate_samples(1)
        self.assertEqual(synthetic.iloc[0].to_dict(), {'age': 25, 'workclass': 'Self-emp-not-inc'})

    def test_rappp_uses_preprocesser_output_from_syn(self):
        from generative_models.rappp import RAPpp

        _, metadata = load_local_data_as_df('data/adult')
        model = RAPpp(metadata)
        model.trained = True
        model._reverse_maps = {
            'age': {0: 25},
            'workclass': {1: 'Self-emp-not-inc'},
        }
        model.generator = _FakeSynMechanism(DataFrame({'age': [0], 'workclass': [1]}))

        synthetic = model.generate_samples(1)
        self.assertEqual(synthetic.iloc[0].to_dict(), {'age': 25, 'workclass': 'Self-emp-not-inc'})
