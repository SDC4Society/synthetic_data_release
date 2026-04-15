import unittest
from unittest.mock import patch

from pandas import DataFrame

from utils.datagen import load_local_data_as_df


class _FakeMechanism:
    def syn(self, nsamples, preprocesser, path=None):
        synthetic = DataFrame(
            {
                0: [0] * nsamples,
                1: [1] * nsamples,
                2: [0] * nsamples,
            }
        )
        preprocesser.reverse_data(synthetic, path)


class TestPrivMRF(unittest.TestCase):
    @patch('generative_models.privmrf.run_privmrf')
    def test_fit_generate(self, mock_run_privmrf):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df[['age', 'workclass', 'education']].sample(20, random_state=0)

        mock_run_privmrf.return_value = _FakeMechanism()

        from generative_models.privmrf import PrivMRF

        model = PrivMRF(
            metadata,
            epsilon=1.0,
            theta=2,
            max_measure_attr_num=2,
            estimation_iter_num=5,
        )
        model.fit(sample)

        self.assertTrue(mock_run_privmrf.called)
        fit_args = mock_run_privmrf.call_args.args
        self.assertEqual(fit_args[0].shape, sample.shape)
        self.assertEqual(list(fit_args[1].attr_list), [0, 1, 2])

        syn = model.generate_samples(10)
        self.assertEqual(len(syn), 10)
        self.assertEqual(list(syn.columns), list(sample.columns))
        workclass_meta = next(column for column in metadata['columns'] if column['name'] == 'workclass')
        self.assertTrue((syn['workclass'] == workclass_meta['i2s'][1]).all())
