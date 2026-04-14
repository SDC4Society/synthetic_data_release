"""Parallel execution utilities for model evaluation"""
import importlib
import os
import warnings
from multiprocessing import cpu_count, get_context

from numpy.random import seed
from tqdm import tqdm

from generative_models.generative_model import GenerativeModel


MODEL_REGISTRY = {
    "IndependentHistogram": "generative_models.data_synthesiser.IndependentHistogram",
    "BayesianNet": "generative_models.data_synthesiser.BayesianNet",
    "PrivBayes": "generative_models.data_synthesiser.PrivBayes",
    "CTGAN": "generative_models.ctgan.CTGAN",
    "PATEGAN": "generative_models.pate_gan.PATEGAN",
    "SanitiserNHS": "sanitisation_techniques.sanitiser_nhs.SanitiserNHS",
    "SanitiserMondrian": "sanitisation_techniques.sanitiser_mondrian.SanitiserMondrian",
    "AIM": "generative_models.aim.AIM",
    "DP_MERF": "generative_models.dp_merf.DP_MERF",
    "GEM": "generative_models.gem.GEM",
    "PrivateGSD": "generative_models.private_gsd.PrivateGSD",
    "RAPpp": "generative_models.rappp.RAPpp",
    "PrivMRF": "generative_models.privmrf.PrivMRF",
    "TabDDPM": "generative_models.tabddpm.TabDDPM",
    "PrivSyn": "generative_models.privsyn.PrivSyn",
}

UTILITY_TASK_REGISTRY = {
    "RandForestClass": "predictive_models.predictive_model.RandForestClassTask",
    "LogRegClass": "predictive_models.predictive_model.LogRegClassTask",
    "LinReg": "predictive_models.predictive_model.LinRegTask",
}


SANITISER_MODELS = {'SanitiserNHS', 'SanitiserMondrian'}


def _resolve_class(import_path):
    module_name, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_model(config, metadata):
    """Create a model instance from a (class_name, *params) config tuple."""
    name, *params = config
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    return _resolve_class(MODEL_REGISTRY[name])(metadata, *params)


def model_name_from_config(config, metadata=None):
    """Return a model name for a config, falling back when the model cannot be imported."""
    original_device = os.environ.get('SYNTHETIC_DATA_DEVICE')
    try:
        os.environ['SYNTHETIC_DATA_DEVICE'] = 'cpu'
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        model = create_model(config, metadata)
        return model.__name__
    except (ModuleNotFoundError, ImportError, RuntimeError, ValueError):
        name, *params = config
        if params:
            param_str = ','.join(str(p) for p in params)
            return f'{name}({param_str})'
        return name
    finally:
        if original_device is None:
            os.environ.pop('SYNTHETIC_DATA_DEVICE', None)
        else:
            os.environ['SYNTHETIC_DATA_DEVICE'] = original_device


def is_generative_model_config(config):
    """Determine model/sanitiser type from config without full instantiation."""
    name, *_ = config
    return name not in SANITISER_MODELS


def create_utility_task(config, metadata):
    """Create a utility task instance from a (task_name, *params) config tuple."""
    name, *params = config
    if name not in UTILITY_TASK_REGISTRY:
        raise ValueError(f'Unknown utility task: {name}')
    return _resolve_class(UTILITY_TASK_REGISTRY[name])(metadata, *params)


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


def _gpu_device_requested():
    device = os.environ.get('SYNTHETIC_DATA_DEVICE', '')
    return bool(device) and ('cuda' in device.lower() or device.lower().startswith('gpu'))


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
        if _gpu_device_requested():
            max_workers = min(1, len(tasks))
        else:
            max_workers = min(cpu_count(), len(tasks))
    elif _gpu_device_requested() and max_workers > 1:
        warnings.warn(
            'CUDA device requested via SYNTHETIC_DATA_DEVICE; using more than one worker may oversubscribe the GPU.',
            UserWarning
        )

    ctx = get_context('spawn') if _gpu_device_requested() else get_context()
    with ctx.Pool(max_workers, initializer=_worker_init) as pool:
        results = list(tqdm(
            pool.imap_unordered(_StarmapHelper(worker_fn), tasks),
            total=len(tasks),
            desc=desc
        ))
    return results
