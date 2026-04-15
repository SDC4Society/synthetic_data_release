import unittest
from unittest.mock import patch

from pandas import DataFrame

from utils.datagen import load_local_data_as_df


class _FakePrivSynGenerator:
    def syn(self, nsamples, preprocesser, parent_dir=None, **kwargs):
        synthetic = DataFrame(
            {
                'age': [0] * nsamples,
                'workclass': [1] * nsamples,
                'education': [2] * nsamples,
            }
        )
        preprocesser.reverse_data(synthetic, parent_dir)


class TestPrivSyn(unittest.TestCase):
    @patch('generative_models.privsyn.privsyn_main')
    def test_fit_generate(self, mock_privsyn_main):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df[['age', 'workclass', 'education']].sample(20, random_state=0)

        mock_privsyn_main.return_value = {'privsyn_generator': _FakePrivSynGenerator()}

        from generative_models.privsyn import PrivSyn

        model = PrivSyn(metadata, epsilon=1.0)
        model.fit(sample)

        self.assertTrue(mock_privsyn_main.called)
        fit_args = mock_privsyn_main.call_args.args
        self.assertEqual(fit_args[1].shape, sample.shape)
        self.assertEqual(set(fit_args[2].keys()), set(sample.columns))

        syn = model.generate_samples(10)
        self.assertEqual(len(syn), 10)
        self.assertEqual(list(syn.columns), list(sample.columns))

        workclass_meta = next(column for column in metadata['columns'] if column['name'] == 'workclass')
        education_meta = next(column for column in metadata['columns'] if column['name'] == 'education')
        self.assertTrue((syn['workclass'] == workclass_meta['i2s'][1]).all())
        self.assertTrue((syn['education'] == education_meta['i2s'][2]).all())
