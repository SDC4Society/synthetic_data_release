"""Tests for the parallel execution utilities"""
from unittest import TestCase
from os import path

from utils.datagen import load_local_data_as_df

cwd = path.dirname(__file__)


def _double(x):
    """Top-level function so it can be pickled for multiprocessing."""
    return x * 2


class TestCreateModel(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))

    def test_create_generative_model(self):
        from utils.parallel import create_model, is_generative_model
        model = create_model(("BayesianNet", 2, 1), self.metadata)
        self.assertEqual(model.__name__, 'BayesianNetBins2')
        self.assertTrue(is_generative_model(model))

    def test_create_sanitiser(self):
        from utils.parallel import create_model, is_generative_model
        model = create_model(
            ("SanitiserNHS", 10, 1, 0.99, 2, [], ["col1"]),
            self.metadata
        )
        self.assertTrue(model.__name__.startswith('SanitiserNHS'))
        self.assertFalse(is_generative_model(model))

    def test_create_model_unknown_raises(self):
        from utils.parallel import create_model
        with self.assertRaises(ValueError):
            create_model(("UnknownModel",), self.metadata)


class TestCreateUtilityTask(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))

    def test_create_utility_task(self):
        from utils.parallel import create_utility_task
        task = create_utility_task(("RandForestClass", "Sex"), self.metadata)
        self.assertIn('RandomForestClassifier', task.__name__)

    def test_create_utility_task_unknown_raises(self):
        from utils.parallel import create_utility_task
        with self.assertRaises(ValueError):
            create_utility_task(("UnknownTask",), self.metadata)


class TestRunParallelModels(TestCase):

    def test_serial_execution(self):
        from utils.parallel import run_parallel_models
        tasks = [(i,) for i in range(5)]
        results = run_parallel_models(_double, tasks, max_workers=1)
        self.assertEqual(results, [0, 2, 4, 6, 8])

    def test_parallel_execution(self):
        from utils.parallel import run_parallel_models
        tasks = [(i,) for i in range(5)]
        results = run_parallel_models(_double, tasks, max_workers=2)
        self.assertEqual(results, [0, 2, 4, 6, 8])

    def test_empty_tasks(self):
        from utils.parallel import run_parallel_models
        results = run_parallel_models(_double, [], max_workers=2)
        self.assertEqual(results, [])
