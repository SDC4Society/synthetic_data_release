import unittest
from unittest.mock import patch

import numpy as np
from pandas import DataFrame

from utils.datagen import load_local_data_as_df


class _FakeSampler:
    def sample(self, num_sample=0, preprocesser=None, device='cpu', parent_dir=None, batch_size=2000, disbalance=None, seed=0):
        synthetic = np.array(
            [[37.2, 1, 2]] * num_sample,
            dtype=object,
        )
        preprocesser.reverse_data(synthetic, parent_dir)


class TestTabDDPM(unittest.TestCase):
    @patch('generative_models.tabddpm.ddpm_sampler')
    @patch('generative_models.tabddpm.finetune')
    @patch('generative_models.tabddpm.make_dataset_from_df')
    def test_fit_generate(self, mock_make_dataset, mock_finetune, mock_ddpm_sampler):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df[['age', 'workclass', 'education']].sample(20, random_state=0)

        mock_make_dataset.return_value = object()
        mock_finetune.return_value = object()
        mock_ddpm_sampler.return_value = _FakeSampler()

        from generative_models.tabddpm import TabDDPM

        model = TabDDPM(metadata, steps=5, batch_size=16, num_timesteps=10, device='cpu')
        model.fit(sample)

        self.assertTrue(mock_make_dataset.called)
        dataset_args = mock_make_dataset.call_args.args[0]
        self.assertIn('X_num', dataset_args)
        self.assertIn('X_cat', dataset_args)
        self.assertIn('y', dataset_args)

        syn = model.generate_samples(10)
        self.assertEqual(len(syn), 10)
        self.assertEqual(list(syn.columns), list(sample.columns))
        self.assertTrue(np.issubdtype(syn['age'].dtype, np.integer))

        workclass_meta = next(column for column in metadata['columns'] if column['name'] == 'workclass')
        education_meta = next(column for column in metadata['columns'] if column['name'] == 'education')
        self.assertTrue((syn['workclass'] == workclass_meta['i2s'][1]).all())
        self.assertTrue((syn['education'] == education_meta['i2s'][2]).all())
