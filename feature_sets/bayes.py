"""A set of features that a Bayesian Net model is expected to extract from the raw data"""
from pandas import DataFrame, get_dummies, cut
from pandas.api.types import CategoricalDtype
from numpy import ndarray, all, concatenate, inf, zeros, zeros_like, triu_indices_from, linspace
from itertools import combinations

from utils.constants import *
from utils.logging import LOGGER
from feature_sets.feature_set import FeatureSet
from feature_sets.independent_histograms import HistogramFeatureSet


class CorrelationsFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype
        self.nfeatures = 0

        self.cat_attributes = []
        self.num_attributes = []

        self.category_codes = {}

        for cdict in metadata['columns']:
            attr_name = cdict['name']
            dtype = cdict['type']

            if dtype == FLOAT or dtype == INTEGER:
                self.num_attributes.append(attr_name)

            elif dtype == CATEGORICAL or dtype == ORDINAL:
                self.cat_attributes.append(attr_name)
                self.category_codes[attr_name] = cdict['i2s']
                self.nfeatures += len(cdict['i2s'])

        LOGGER.debug(f'Feature set will have length {self.nfeatures}')

        self.__name__ = 'Correlations'

    def extract(self, data, flatten=True):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'

        if len(data) == 0:
            # drop_first=True produces len(cats)-1 columns per attribute
            n_onehot = self.nfeatures - len(self.cat_attributes)
            n_encoded = len(self.num_attributes) + n_onehot
            n_corr = n_encoded * (n_encoded - 1) // 2
            if flatten:
                return zeros(n_corr)
            else:
                return zeros((n_encoded, n_encoded))

        assert all([c in list(data) for c in self.cat_attributes]), 'Missing some categorical attributes in input data'
        assert all([c in list(data) for c in self.num_attributes]), 'Missing some numerical attributes in input data'

        encoded = data[self.num_attributes].copy()
        for c in self.cat_attributes:
            col = data[c]
            col = col.astype(CategoricalDtype(categories=self.category_codes[c], ordered=True))
            encoded = encoded.merge(get_dummies(col, drop_first=True, prefix=c), left_index=True, right_index=True)

        col_names = list(encoded)
        self.feature_names = list(combinations(col_names, r=2))

        corr = encoded.corr().fillna(0).values

        mask = zeros_like(corr).astype(bool)
        mask[triu_indices_from(mask, k=1)] = True

        if flatten:
            features = corr[mask].flatten()
        else:
            features = corr

        return features

class BinnedCorrelationsFeatureSet(FeatureSet):
    # Same as CorrelationsFeatureSet but apply binning to numerical attributes
    # as stated in Groundhog Day paper.
    def __init__(self, datatype, metadata, nbins=10, quids=None):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype
        self.nfeatures = 0

        self.cat_attributes = []
        self.num_attributes = []

        self.histogram_bins = {}
        self.category_codes = {}

        if quids is None:
            quids = []

        for cdict in metadata['columns']:
            attr_name = cdict['name']
            dtype = cdict['type']

            if dtype == FLOAT or dtype == INTEGER:
                self.num_attributes.append(attr_name)
                # if nbins > attr range, keep the attr as-is
                if nbins > cdict["max"]-cdict["min"] + 1:
                    self.histogram_bins[attr_name] = []
                else:
                    self.histogram_bins[attr_name] = linspace(cdict["min"], cdict["max"], nbins).tolist()

            elif dtype == CATEGORICAL or dtype == ORDINAL:
                self.cat_attributes.append(attr_name)
                self.category_codes[attr_name] = cdict['i2s']
                self.nfeatures += len(cdict['i2s'])

        LOGGER.debug(f'Feature set will have length {self.nfeatures}')

        self.__name__ = 'Correlations'

    def extract(self, data, flatten=True):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'

        if len(data) == 0:
            # drop_first=True produces len(cats)-1 columns per attribute
            n_onehot = self.nfeatures - len(self.cat_attributes)
            n_encoded = len(self.num_attributes) + n_onehot
            n_corr = n_encoded * (n_encoded - 1) // 2
            if flatten:
                return zeros(n_corr)
            else:
                return zeros((n_encoded, n_encoded))

        assert all([c in list(data) for c in self.cat_attributes]), 'Missing some categorical attributes in input data'
        assert all([c in list(data) for c in self.num_attributes]), 'Missing some numerical attributes in input data'

        encoded = None
        for c in self.num_attributes:
            col = data[c]
            # if no histogram_bins provided (nbins > attr range), keep the attr as-is
            if self.histogram_bins[c] == []:
                F = col
            else:
                F = cut(col, bins=[-inf] + self.histogram_bins[c], labels=self.histogram_bins[c])
                F.astype(float)
            if encoded is None:
                encoded = DataFrame(F)
            else:
                encoded = encoded.merge(F, left_index=True, right_index=True)
        for c in self.cat_attributes:
            col = data[c]
            col = col.astype(CategoricalDtype(categories=self.category_codes[c], ordered=True))
            encoded = encoded.merge(get_dummies(col, drop_first=True, prefix=c), left_index=True, right_index=True)

        col_names = list(encoded)
        self.feature_names = list(combinations(col_names, r=2))

        corr = encoded.corr().fillna(0).values

        mask = zeros_like(corr).astype(bool)
        mask[triu_indices_from(mask, k=1)] = True

        if flatten:
            features = corr[mask].flatten()
        else:
            features = corr

        return features

class BayesFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, nbins=10):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype

        self.histograms  = HistogramFeatureSet(datatype, metadata, nbins)
        self.correlations = CorrelationsFeatureSet(datatype, metadata)

    def extract(self, data):
        Hist = self.histograms.extract(data)
        Corr = self.correlations.extract(data)

        return concatenate([Hist, Corr])
