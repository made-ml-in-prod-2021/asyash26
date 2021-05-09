from dataclasses import dataclass, field


@dataclass()
class LRTrainParams:
    model_type: str = field(default='LogisticRegression')
    solver: str = field(default='lbfgs')
    penalty: str = field(default='l2')
    max_iter: int = field(default=100)
    tol: float = field(default=0.0001)


@dataclass()
class RFTrainParams:
    model_type: str = field(default='RandomForestClassifier')
    n_estimators: int = field(default=100)
    min_samples_split: int = field(default=2)
    min_samples_leaf: int = field(default=1)
