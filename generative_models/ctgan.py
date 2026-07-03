from pandas import DataFrame
import torch

from utils.logging import LOGGER
from utils.device_utils import validate_and_get_device

from generative_models.generative_model import GenerativeModel
from ctgan import CTGANSynthesizer


class CTGAN(GenerativeModel):
    """A conditional generative adversarial network for tabular data"""
    def __init__(self, metadata,
                 embedding_dim=128, gen_dim=(256, 256),
                 dis_dim=(256, 256), l2scale=1e-6,
                 batch_size=500, epochs=300,
                 device=None,
                 multiprocess=False):
        # Set device with device_utils
        self.device, self.is_gpu = validate_and_get_device(device)
        
        self.synthesiser = CTGANSynthesizer(embedding_dim, gen_dim, dis_dim,
                                            l2scale, batch_size, epochs)
        # Try to set device on synthesiser if it supports it
        try:
            if hasattr(self.synthesiser, 'set_device'):
                self.synthesiser.set_device(self.device)
            elif hasattr(self.synthesiser, 'device'):
                self.synthesiser.device = torch.device(self.device)
        except Exception as e:
            LOGGER.warning(f"Could not set device on CTGANSynthesizer: {e}")

        self.metadata = metadata
        self.datatype = DataFrame

        self.multiprocess = bool(multiprocess)

        self.infer_ranges = True
        self.trained = False

        self.__name__ = 'CTGAN'
        
        LOGGER.debug(f"CTGAN initialized with device: {self.device} (GPU: {self.is_gpu})")

    def fit(self, data):
        """Train a generative adversarial network on tabular data.
        Input data is assumed to be of shape (n_samples, n_features)
        See https://github.com/DAI-Lab/SDGym for details"""
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        self.synthesiser.fit(data, self.metadata)

        LOGGER.debug(f'Finished fitting')
        self.trained = True

    def generate_samples(self, nsamples):
        """Generate random samples from the fitted Gaussian distribution"""
        assert self.trained, "Model must first be fitted to some data."

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_data = self.synthesiser.sample(nsamples)

        return synthetic_data

    def __del__(self):
        """Cleanup resources when the model object is destroyed."""
        if hasattr(self, 'synthesiser'):
            del self.synthesiser
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
