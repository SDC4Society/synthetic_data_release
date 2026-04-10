# tests/test_rappp.py
import unittest
from utils.datagen import load_local_data_as_df

class TestRAPpp(unittest.TestCase):
    def test_fit_generate(self):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df.sample(500)

        from generative_models.rappp import RAPpp
        model = RAPpp(metadata, epsilon=1.0)
        model.fit(sample)

        syn = model.generate_samples(100)
        self.assertEqual(len(syn), 100)
        self.assertEqual(list(syn.columns), list(sample.columns))
