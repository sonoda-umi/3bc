import os

from jmetal.core.quality_indicator import GenerationalDistance, HyperVolume, InvertedGenerationalDistance

from utils.file_utils import load_n_evaluation_log, parse_exp_log_dir
from utils.reference_fronts import ReferenceFronts


class PerformanceEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def gd(reference, actual):
        return GenerationalDistance(reference).compute(actual)

    @staticmethod
    def igd(reference, actual):
        return InvertedGenerationalDistance(reference).compute(actual)

    @staticmethod
    def hv(reference, actual):
        return HyperVolume(reference).compute(actual)

    @staticmethod
    def match_experiment_file(
        data_base_path: str,
        solver: str,
        tree: str,
        dimension: int,
        termination: str,
    ):
        file_name_pattern = f"{solver}_{tree}_{dimension}_{termination}"
        files = [f for f in os.listdir(data_base_path) if os.path.isfile(os.path.join(data_base_path, f))]
        for file in files:
            if file.startswith(file_name_pattern):
                return os.path.join(data_base_path, file)

        # ファイルが見つからなかった場合のエラーメッセージ
        raise FileNotFoundError(f"No file found for pattern: {file_name_pattern} in {data_base_path}")

    def compute_indicator(
        self,
        reference_set,
        reference_front,
        actual_set,
        actual_front,
        indicator_type: str,
    ):
        if indicator_type == "GD":
            indicator = self.gd
            indicator_str = "GD"
        elif indicator_type == "IGD":
            indicator = self.igd
            indicator_str = "IGD"
        elif indicator_type == "HV":
            indicator = self.hv
            indicator_str = "HyperVolume"
            raise NotImplementedError("HyperVolume requires a reference point and therefore is not implemented yet @240918")
        else:
            raise NotImplementedError("Unrecognized Quality Indicator")

        return {
            "set_indicator": indicator(reference_set, actual_set),
            "front_indicator": indicator(reference_front, actual_front),
        }

    def get_evaluation_info(self, exp_dir):
        exp_info = parse_exp_log_dir(exp_dir=exp_dir)
        meta = exp_info["meta"]
        dimension = meta["dimension"]
        n_objectives = meta["n_objectives"]
        log_df = load_n_evaluation_log(file_path=exp_info["result"], return_df=True)
        n_ts = n_objectives - 1
        n_xs = dimension
        design_variable_header = [f"t{t + 1}" for t in range(n_ts)] + [f"x{x + 1}" for x in range(n_xs)]
        objective_value_header = [f"y{y + 1}" for y in range(n_objectives)]
        rf = ReferenceFronts()
        references = rf.get_n_obj_local_pareto_set(
            dimension=dimension,
            tree_file_path=exp_info["tree"],
            resolution=10,
            n_objectives=n_objectives,
        )
        try:
            population_size = meta["algorithm_parameters"]["population_size"]
        except KeyError:
            population_size = meta["algorithm_parameters"]["swarm_size"]
        design_variables = log_df[design_variable_header]
        objective_values = log_df[objective_value_header]
        return {
            "references": references,
            "design_variables": design_variables,
            "objective_values": objective_values,
            "population_size": population_size,
            "dimension": dimension,
            "n_objectives": n_objectives,
            "tree": meta["tree_file"].split("/")[-1],
            "solver": meta["algorithm"],
        }


def main():
    pe = PerformanceEvaluator()
    exp_dir = "../data/fully_managed_experiments_2025-08-05"

    start_step = 0
    end_step = start_step


if __name__ == "__main__":
    main()
