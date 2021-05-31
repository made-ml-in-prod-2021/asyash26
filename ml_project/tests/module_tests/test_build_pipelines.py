import os
import pickle
from typing import Union

import numpy as np
import scipy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.pipelines.build_pipelines import (build_numerical_pipeline,
                                           build_categorical_pipeline,
                                           build_transformer,
                                           create_model,
                                           build_full_pipeline,
                                           serialize_pipeline,
                                           deserialize_pipeline)


def test_build_numerical_pipeline(generated_data, numerical_features):
    numerical_pipeline = build_numerical_pipeline()
    generated_data.iloc[0] = np.nan
    transformed_data = numerical_pipeline.fit_transform(generated_data[numerical_features])
    assert not np.isnan(transformed_data).any(), "Unexpected NAN values"
    assert np.allclose(transformed_data.mean(0), 0, rtol=1e-2), "Unexpected mean value"
    assert np.allclose(transformed_data.std(0), 1, rtol=1e-2), "Unexpected std value"


def test_build_categorical_pipeline(generated_data, categorical_features):
    categorical_pipeline = build_categorical_pipeline()
    transformed_data = categorical_pipeline.fit_transform(generated_data[categorical_features])
    assert type(transformed_data) == scipy.sparse.csr.csr_matrix, "Unexpected data type"
    assert transformed_data.shape[0] == generated_data.shape[0], "Unexpected transformed data shape"


def test_build_transformer(features_params):
    transformer = build_transformer(features_params)
    assert type(transformer) == ColumnTransformer, "Unexpected type"


def test_create_lr_model(lr_train_params):
    model = create_model(lr_train_params)
    assert type(model) == LogisticRegression, "Unexpected type"
    assert model.solver == lr_train_params.solver, "Unexpected parameter - solver"
    assert model.penalty == lr_train_params.penalty, "Unexpected parameter - penalty"
    assert model.max_iter == lr_train_params.max_iter, "Unexpected parameter - max_iter"
    assert model.tol == lr_train_params.tol, "Unexpected parameter - tol"


def test_create_rf_model(rf_train_params):
    model = create_model(rf_train_params)
    assert type(model) == RandomForestClassifier, "Unexpected type"
    assert model.n_estimators == rf_train_params.n_estimators, "Unexpected parameter - n_estimators"
    assert model.min_samples_split == rf_train_params.min_samples_split, "Unexpected parameter - min_samples_split"
    assert model.min_samples_leaf == rf_train_params.min_samples_leaf, "Unexpected parameter - min_samples_leaf"


def test_build_full_pipeline(rf_train_params, features_params):
    full_piplene = build_full_pipeline(rf_train_params, features_params)
    assert type(full_piplene) == Pipeline, "Unexpected type"


def test_serialize_pipeline(full_pipeline, out_path):
    serialize_pipeline(full_pipeline, out_path)
    assert os.path.isfile(out_path), "File was not created"


def test_deserialize_pipeline(serialized_pipeline_path):
    pipeline = deserialize_pipeline(serialized_pipeline_path)
    assert type(pipeline) == Pipeline, "Unexpected type"
