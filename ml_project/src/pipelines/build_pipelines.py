import logging
import pickle
from typing import NoReturn, Union

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.entities.features_params import FeaturesParams
from src.entities.train_params import RFTrainParams, LRTrainParams

logger = logging.getLogger("ml_in_prod")

TrainingParams = Union[RFTrainParams, LRTrainParams]
Model = Union[RandomForestClassifier, LogisticRegression]


def build_numerical_pipeline() -> Pipeline:
    logger.info("__Building numerical pipeline")
    numeric_pipeline = Pipeline(
        [
            (
                'imputer',
                SimpleImputer(missing_values=np.nan, strategy='median')
            ),
            (
                'scaler',
                StandardScaler()
            )
        ]
    )
    return numeric_pipeline


def build_categorical_pipeline() -> Pipeline:
    logger.info("__Building categorical pipeline")
    categorical_pipeline = Pipeline(
        [
            (
                'imputer',
                SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            ),
            (
                'ohe-hot',
                OneHotEncoder()
            ),
        ]
    )
    return categorical_pipeline


def build_transformer(params: FeaturesParams) -> ColumnTransformer:
    logger.info("__Building the transformer")
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                'numerical_pipeline',
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def create_model(train_params: TrainingParams) -> Model:
    logger.info("__Creating a model")
    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression(
            solver=train_params.solver,
            penalty=train_params.penalty,
            max_iter=train_params.max_iter,
            tol=train_params.tol,
        )

    elif train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            min_samples_split=train_params.min_samples_split,
            min_samples_leaf=train_params.min_samples_leaf,
        )
    else:
        raise NotImplementedError
    return model


def build_full_pipeline(train_params: TrainingParams, feature_params: FeaturesParams) -> Pipeline:
    model = create_model(train_params)
    transformer = build_transformer(feature_params)
    full_pipeline = Pipeline(
        (
            [
                (
                    'data_preprocess',
                    transformer,
                ),
                (
                    "model",
                    model
                ),
            ]
        )
    )
    return full_pipeline


def serialize_pipeline(pipeline: Pipeline, output_path: str) -> NoReturn:
    logger.info("__Serializing full pipeline")
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)


def deserialize_pipeline(input_path: str) -> Pipeline:
    logger.info("__Deserializing pipeline")
    with open(input_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
