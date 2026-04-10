"""Parallel execution utilities for model evaluation"""
import os
from multiprocessing import Pool, cpu_count
from numpy.random import seed
from tqdm import tqdm

from generative_models.generative_model import GenerativeModel
from generative_models.data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from generative_models.aim import AIM
from generative_models.dp_merf import DP_MERF
from generative_models.gem import GEM
from generative_models.private_gsd import PrivateGSD
from generative_models.rappp import RAPpp

from sanitisation_techniques.sanitiser_nhs import SanitiserNHS
from sanitisation_techniques.sanitiser_mondrian import SanitiserMondrian
from predictive_models.predictive_model import RandForestClassTask, LogRegClassTask, LinRegTask


MODEL_REGISTRY = {
    "IndependentHistogram": IndependentHistogram,
    "BayesianNet": BayesianNet,
    "PrivBayes": PrivBayes,
    "CTGAN": CTGAN,
    "PATEGAN": PATEGAN,
    "SanitiserNHS": SanitiserNHS,
    "SanitiserMondrian": SanitiserMondrian,
    "AIM": AIM,
    "DP_MERF": DP_MERF,
    "GEM": GEM,
    "PrivateGSD": PrivateGSD,
    "RAPpp": RAPpp,
}

UTILITY_TASK_REGISTRY = {
    "RandForestClass": RandForestClassTask,
    "LogRegClass": LogRegClassTask,
    "LinReg": LinRegTask,
}


def create_model(config, metadata):
    """Create a model instance from a (class_name, *params) config tuple."""
    name, *params = config
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    return MODEL_REGISTRY[name](metadata, *params)


def create_utility_task(config, metadata):
    """Create a utility task instance from a (task_name, *params) config tuple."""
    name, *params = config
    if name not in UTILITY_TASK_REGISTRY:
        raise ValueError(f'Unknown utility task: {name}')
    return UTILITY_TASK_REGISTRY[name](metadata, *params)


def is_generative_model(model):
    """Check if a model is a GenerativeModel (vs a Sanitiser)."""
    return isinstance(model, GenerativeModel)


def _worker_init():
    """Initialize worker process with a unique random seed."""
    seed(os.getpid())


class _StarmapHelper:
    """Picklable wrapper that unpacks a tuple arg for imap_unordered."""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return self.fn(*args)


def run_parallel_models(worker_fn, tasks, max_workers=None, desc="Models"):
    """Run tasks in parallel using multiprocessing.Pool with tqdm progress bar.

    :param worker_fn: callable: Worker function to execute
    :param tasks: list[tuple]: List of argument tuples for worker_fn
    :param max_workers: int or None: Number of worker processes (default: min(cpu_count, len(tasks)))
    :param desc: str: Description for the progress bar
    :return: list: Results from each worker
    """
    if not tasks:
        return []
    if max_workers == 1:
        return [worker_fn(*task) for task in tqdm(tasks, desc=desc)]
    if max_workers is None:
        max_workers = min(cpu_count(), len(tasks))
    with Pool(max_workers, initializer=_worker_init) as pool:
        results = list(tqdm(
            pool.imap_unordered(_StarmapHelper(worker_fn), tasks),
            total=len(tasks),
            desc=desc
        ))
    return results
