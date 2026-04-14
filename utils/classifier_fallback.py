import os
from utils.logging import LOGGER

def _use_gpu():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices == '':
        return False
        
    device_opt = os.environ.get('SYNTHETIC_DATA_DEVICE', 'cpu')
    if 'cuda' in device_opt.lower() or device_opt.lower() == 'gpu':
        try:
            import cuml
            return True
        except ImportError:
            LOGGER.warning("cuml is not installed but GPU was requested. Falling back to sklearn.")
            return False
    return False

def get_random_forest_classifier(**kwargs):
    if _use_gpu():
        try:
            from cuml.ensemble import RandomForestClassifier
            return RandomForestClassifier(**kwargs)
        except Exception as e:
            LOGGER.warning(f"cuml RandomForestClassifier fallback failed: {e}")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**kwargs)

def get_logistic_regression(**kwargs):
    if _use_gpu():
        try:
            from cuml.linear_model import LogisticRegression
            return LogisticRegression(**kwargs)
        except Exception as e:
            LOGGER.warning(f"cuml LogisticRegression fallback failed: {e}")
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**kwargs)

def get_linear_regression(**kwargs):
    if _use_gpu():
        try:
            from cuml.linear_model import LinearRegression
            return LinearRegression(**kwargs)
        except Exception as e:
            LOGGER.warning(f"cuml LinearRegression fallback failed: {e}")
    from sklearn.linear_model import LinearRegression
    return LinearRegression(**kwargs)

def get_svc(**kwargs):
    if _use_gpu():
        try:
            from cuml.svm import SVC
            return SVC(**kwargs)
        except Exception as e:
            LOGGER.warning(f"cuml SVC fallback failed: {e}")
    from sklearn.svm import SVC
    return SVC(**kwargs)

def get_knn_classifier(**kwargs):
    if _use_gpu():
        try:
            from cuml.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**kwargs)
        except Exception as e:
            LOGGER.warning(f"cuml KNeighborsClassifier fallback failed: {e}")
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(**kwargs)
