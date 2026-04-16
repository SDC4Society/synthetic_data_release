""" Some predictive models to represent a simple analysis task. """
from sklearn.impute import SimpleImputer
from utils.classifier_fallback import get_linear_regression, get_logistic_regression, get_random_forest_classifier
from pandas import DataFrame
from numpy import empty, true_divide, zeros, arange

from sklearn.pipeline import Pipeline
from preprocess_common.pipeline import create_preprocessing_pipeline
from utils.logging import LOGGER
from utils.constants import *

class PredictiveModel(object):
    """ A predictive model. """
    def __init__(self, metadata, labelCol):
        """
        :param metadata: dict: Metadata description
        :param labelCol: str: Name of the target variable
        """
        self.metadata = metadata
        self.labelCol = labelCol
        self.nfeatures = self._get_num_features()

        self.pipeline = create_preprocessing_pipeline(metadata, labelCol)

        self.datatype = DataFrame
        self.trained = False

    def train(self, data):
        return NotImplementedError("Method needs to be overwritten by a subclass")

    def predict(self, features):
        return NotImplementedError("Method needs to be overwritten by a subclass")

    def evalute(self, data):
        return NotImplementedError("Method needs to be overwritten by a subclass")

        features_encoded = self.pipeline.fit_transform(data) 
        if not self.trained:
            # We only want to keep the fitted pipeline if it's the first time processing
            pass
        return features_encoded.astype('float32')

    def _get_num_features(self):
        nfeatures = 0

        for cdict in self.metadata['columns']:
            data_type = cdict['type']
            attr_name = cdict['name']

            if attr_name != self.labelCol:
                if data_type == FLOAT or data_type == INTEGER:
                    nfeatures += 1

                elif data_type == CATEGORICAL or data_type == ORDINAL:
                    nfeatures += len(cdict['i2s'])

                else:
                    raise ValueError(f'Unkown data type {data_type} for attribute {attr_name}')

        return nfeatures

    def _get_feature_names(self):
        # We can fetch directly from standard sklearn get_feature_names_out (or build backwards compatible output)
        return []


class ClassificationTask(PredictiveModel):
    """ A binary or multiclass classification model. """

    def __init__(self, Distinguisher, metadata, labelCol):
        """
        :param Distinguisher: sklearn.Classifier: A classification model
        :param metadata: dict: Metadata description
        :param labelCol: str: Name of the target variable
        """
        super().__init__(metadata, labelCol)
        self.Distinguisher = Distinguisher

        labels = self._get_labels()
        self.labels = {l:i for i, l in enumerate(labels)}
        self.labelsInv = {i:l for l, i in self.labels.items()}

        self.__name__ = f'{self.Distinguisher.__class__.__name__}{self.labelCol}'

    def set_seed(self, seed: int | None):
        """Set a seed for reproducibility"""
        self.seed = seed
        try:
            self.Distinguisher.random_state = seed
        except AttributeError:
            LOGGER.debug(f'{self.Distinguisher.__class__.__name__} does not support/need random_state, or it uses a different name.')

    def train(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        if data.empty:
            LOGGER.warning(f"Training data is empty for {self.__name__}. Skipping training.")
            self.trained = False
            return

        features = self.pipeline.fit_transform(data).astype('float32')
        labels = data[self.labelCol].apply(lambda x: self.labels[x]).values

        self.Distinguisher.fit(features, labels)

        LOGGER.debug('Finished training MIA distinguisher')
        self.trained = True

    def predict(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        features = self.pipeline.transform(data).astype('float32')
        labels = self.Distinguisher.predict(features)

        return [self.labelsInv[i] for i in labels]

    def evaluate(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        if not self.trained:
            # Return list of zeros (worst accuracy)
            return [0] * len(data)

        features = self.pipeline.transform(data).astype('float32')
        labelsTrue = data[self.labelCol].apply(lambda x: self.labels[x]).values
        labelsPred = self.Distinguisher.predict(features)

        return [int(l == p) for l, p in zip(labelsTrue, labelsPred)]

    def _get_accuracy(self, trueLabels, predLabels):
        return sum([g == l for g, l in zip(trueLabels, predLabels)])/len(trueLabels)

    def _get_labels(self):
        for cdict in self.metadata['columns']:
            if cdict['name'] == self.labelCol:
                if not cdict['type'] in [CATEGORICAL, ORDINAL]:
                    raise ValueError('Label column must be discrete data type.')

                return cdict['i2s']


class RandForestClassTask(ClassificationTask):
    def __init__(self, metadata, labelCol):
        super().__init__(get_random_forest_classifier(), metadata, labelCol)


class LogRegClassTask(ClassificationTask):
    def __init__(self, metadata, labelCol):
        super().__init__(get_logistic_regression(), metadata, labelCol)


class RegressionTask(PredictiveModel):
    """ A binary or multiclass classification model. """

    def __init__(self, Regressor, metadata, labelCol):
        """

        :param Regressor: sklearn.Regressor: A regression model
        :param metadata: dict: Metadata description
        :param labels: list: Label names
        :param FeatureSet: object: Feature extraction object
        """
        super().__init__(metadata, labelCol)
        self.Regressor = Regressor

        self.__name__ = f'{self.Regressor.__class__.__name__}{self.labelCol}'

    def set_seed(self, seed: int | None):
        """Set a seed for reproducibility"""
        self.seed = seed
        try:
            self.Regressor.random_state = seed
        except AttributeError:
            LOGGER.debug(f'{self.Regressor.__class__.__name__} does not support random_state, or it uses a different name.')

    def train(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        if data.empty:
            LOGGER.warning(f"Training data is empty for {self.__name__}. Skipping training.")
            self.trained = False
            return

        features = self.pipeline.fit_transform(data).astype('float32')
        labels = data[self.labelCol].values

        self.Regressor.fit(features, labels)

        LOGGER.debug('Finished training regression model')
        self.trained = True

    def predict(self, features):
        if not isinstance(features, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(features)}")

        features_arr = self.pipeline.transform(features).astype('float32')
        labels = self.Regressor.predict(features_arr)

        return list(labels)

    def evaluate(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        if not self.trained:
            # Return high error values
            return [1e6] * len(data)

        features = self.pipeline.transform(data).astype('float32')
        labelsTrue = data[self.labelCol].values
        labelsPred = self.Regressor.predict(features)

        return [true - pred for true, pred in zip(labelsTrue, labelsPred)]


class LinRegTask(RegressionTask):
    def __init__(self, metadata, labelCol):
        super().__init__(get_linear_regression(), metadata, labelCol)
