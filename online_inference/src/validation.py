import pandas as pd

COLUMNS_LIST = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal',
]
CATEGORICAL_FEATURES = {
    'sex': [1, 0],
    'cp': [3, 2, 1, 0],
    'restecg': [0, 1, 2],
    'exang': [0, 1],
    'slope': [0, 2, 1],
    'ca': [0, 2, 1, 3, 4],
    'thal': [1, 2, 3, 0],
    'fbs': [1, 0],
}
NUMERICAL_FEATURES_BOUNDARIES = {
    'age': [29, 77],
    'trestbps': [94, 200],
    'chol': [126, 564],
    'thalach': [71, 202],
    'oldpeak': [0.0, 6.2],
}


def validate_input_data(data: pd.DataFrame) -> bool:
    if not data.columns.tolist() == COLUMNS_LIST:
        return False
    for feature in CATEGORICAL_FEATURES:
        if not data[feature].unique().all() in CATEGORICAL_FEATURES[feature]:
            return False
    for feature in NUMERICAL_FEATURES_BOUNDARIES:
        [min_value, max_value] = NUMERICAL_FEATURES_BOUNDARIES[feature]
        if not all([min_value <= x <= max_value for x in data[feature]]):
            return False
    return True
