# tests/test_private_gsd.py
import unittest
from utils.datagen import load_local_data_as_df

class TestPrivateGSD(unittest.TestCase):
    def test_fit_generate(self):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df.sample(500)

        from generative_models.private_gsd import PrivateGSD
        # Using fewer epochs for test speed
        model = PrivateGSD(metadata, epsilon=1.0, epochs=2)
        model.fit(sample)

        syn = model.generate_samples(100)
        self.assertEqual(len(syn), 100)
        self.assertEqual(list(syn.columns), list(sample.columns))