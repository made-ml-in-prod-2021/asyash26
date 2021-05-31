import sys
from pathlib import Path

import pytest

sys.path.append("..")

from src.data.data_preprocess import read_data
from src.data.data_preprocess import split_data
from src.entities.features_params import FeaturesParams
from src.entities.preprocess_params import PreprocessParams
from src.entities.split_params import SplitParams
from src.entities.train_params import LRTrainParams, RFTrainParams
from src.models.model_eval import check_metrics
from src.pipelines.build_pipelines import build_full_pipeline, serialize_pipeline


@pytest.fixture(scope='session')
def generated_data_path():
    return Path("../data/generation/gen_data.csv")


@pytest.fixture(scope='function')
def generated_data(generated_data_path):
    return read_data(generated_data_path)


@pytest.fixture(scope='session')
def drop_duplicates_preprocess_params():
    return PreprocessParams(False, True)


@pytest.fixture(scope='session')
def drop_nan_preprocess_params():
    return PreprocessParams(True, False)


@pytest.fixture(scope='session')
def target_col():
    return "target"


@pytest.fixture(scope='session')
def gen_data_size():
    return 100


@pytest.fixture(scope='session')
def split_params():
    return SplitParams(0.1, 42)


@pytest.fixture(scope='session')
def features_params(categorical_features, numerical_features, target_col):
    return FeaturesParams(categorical_features, numerical_features, target_col)


@pytest.fixture(scope='session')
def lr_train_params():
    return LRTrainParams("LogisticRegression", "saga", "l1", 10000, 0.0001)


@pytest.fixture(scope='session')
def rf_train_params():
    return RFTrainParams("RandomForestClassifier", 500, 3, 2)


@pytest.fixture(scope='session')
def full_pipeline(lr_train_params, features_params):
    return build_full_pipeline(lr_train_params, features_params)


@pytest.fixture(scope='session')
def serialized_pipeline_path(full_pipeline, out_path):
    serialize_pipeline(full_pipeline, out_path)
    return out_path


@pytest.fixture(scope='session')
def out_path():
    return "results/model.pkl"


@pytest.fixture(scope='session')
def preds_out_path():
    return "results/preds.csv"


@pytest.fixture(scope='session')
def metrics_out_path():
    return "results/metrics.json"


@pytest.fixture(scope='session')
def categorical_features():
    return [
        "sex",
        "cp",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "fbs",
    ]


@pytest.fixture(scope='session')
def numerical_features():
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope='function')
def preds_and_labels(full_pipeline, generated_data, features_params, split_params):
    X_train, X_test, y_train, y_test = split_data(
        generated_data, features_params.target_col, split_params)
    full_pipeline.fit(X_train, y_train)
    return full_pipeline.predict(X_test), y_test


@pytest.fixture(scope='function')
def metrics(preds_and_labels):
    return check_metrics(preds_and_labels[0], preds_and_labels[1])


@pytest.fixture(scope='session')
def train_config_path():
    return "../configs/test/train_config.yml"


@pytest.fixture(scope='session')
def model_config_path():
    return "../configs/test/model_config.yml"


@pytest.fixture(scope='session')
def predict_config_path():
    return "../configs/test/predict_config.yml"
