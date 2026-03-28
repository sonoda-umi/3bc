from typing import Any, Dict, NamedTuple


class ExperimentSettings(NamedTuple):
    experiment_name: str
    tree_file: str
    dimension: int
    algorithm: str
    algorithm_parameters: Dict[str, Any]
    termination_criterion: Dict[str, Any]
    n_objectives: int = 0
    rotate_t: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()
