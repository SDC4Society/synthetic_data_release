# tests/test_my_model.py
import unittest
from utils.datagen import load_local_data_as_df

class TestDP_MERF(unittest.TestCase):
    def test_fit_generate(self):
        df, metadata = load_local_data_as_df('data/adult')
        sample = df.sample(500)

        from generative_models.dp_merf import DP_MERF
        model = DP_MERF(metadata)
        model.fit(sample)

        syn = model.generate_samples(100)
        self.assertEqual(len(syn), 100)
        self.assertEqual(list(syn.columns), list(sample.columns))