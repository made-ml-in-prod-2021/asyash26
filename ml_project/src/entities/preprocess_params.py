from dataclasses import dataclass, field


@dataclass()
class PreprocessParams:
    drop_nan: bool = field(default=False)
    drop_duplicates: bool = field(default=False)