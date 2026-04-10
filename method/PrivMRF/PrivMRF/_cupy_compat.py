import numpy as np

try:
    import cupy as cp
except ImportError:
    class _CupyCompat:
        ndarray = np.ndarray

        @staticmethod
        def asnumpy(values):
            return np.asarray(values)

        @staticmethod
        def get_array_module(values):
            return np

        def __getattr__(self, name):
            return getattr(np, name)

    cp = _CupyCompat()
