# tests/test_gem.py
import unittest
from utils.datagen import load_local_data_as_df

class TestGEM(unittest.TestCase):
    def test_fit_generate(self):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df.sample(500)

        from generative_models.gem import GEM
        # Using fewer iterations for test speed
        model = GEM(metadata, batch_size=100, max_iters=10, T=5, device='cpu')
        model.fit(sample)

        syn = model.generate_samples(100)
        self.assertEqual(len(syn), 100)
        self.assertEqual(list(syn.columns), list(sample.columns))