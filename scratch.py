import joblib
from utils.parallel import create_model
import pandas as pd
from utils.logging import LOGGER

# Imports to mimic what utility_cli has
from os import mkdir, path
from numpy import mean
from numpy.random import choice, seed
from utils.evaluation_framework import EvaluationEngine

def run_ctgan():
    import os
    print("In worker: Creating CTGAN")
    # Instead of creating fake data, let's just do an empty sleep or similar
    # or just import torch here to mimic
    import torch
    import time
    time.sleep(1)
    return True

if __name__ == '__main__':
    print("Before joblib")
    # Try running 1 task in loky with 16 workers
    results = joblib.Parallel(n_jobs=16, backend='loky')(
        joblib.delayed(run_ctgan)() for _ in range(1)
    )
    print("Results:", results)
