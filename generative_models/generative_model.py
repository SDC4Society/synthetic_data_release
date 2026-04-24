"""Parent class for all generative models"""

class GenerativeModel(object):

    seed = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'fit' in cls.__dict__:
            _orig = cls.__dict__['fit']
            def _checked_fit(self, *args, _orig=_orig, **kwargs):
                if getattr(self, '__name__', None) is None:
                    raise RuntimeError(
                        f'{type(self).__name__} did not set self.__name__ in __init__. '
                        f'Set it to include key hyperparameters, '
                        f'e.g. self.__name__ = f"{type(self).__name__}Eps{{self.epsilon}}".'
                    )
                return _orig(self, *args, **kwargs)
            cls.fit = _checked_fit

    def set_seed(self, seed: int | None):
        """Set a seed for reproducibility"""
        self.seed = seed

    def fit(self, data):
        """Fit a generative model to the input dataset"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def generate_samples(self, nsamples):
        """Generate a synthetic dataset of size nsamples"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')