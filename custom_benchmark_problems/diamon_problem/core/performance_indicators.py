import math
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from config import base_path
from custom_benchmark_problems.diamon_problem.core import algs, evaluation
from custom_benchmark_problems.diamon_problem.data_structures.tree import Tree
from utils import file_utils


def get_local_pareto_set(dimension: int, tree_name: str):
    tree = Tree(dim_space=dimension)
    tree.from_json(f"experiment_trees/{tree_name}.json")
    sequence_info = tree.to_sequence()
    bmp = evaluation.BMP(sequence_info=sequence_info, dim_space=dimension)
    pareto_dict = {}

    def apply_computing(data: np.array):
        row_data = bmp.evaluate(data)
        return np.array(
            [
                row_data.t,
                row_data.y,
                # row_data.unrotated_value[0],
                # row_data.unrotated_value[1],
            ]
        )

    all_sets = []
    all_fronts = []
    for node in sequence_info:
        node_id = node["name"]
        symbols = node["attrs"]["symbol"]
        minimum = node["minima"]
        central_coordinates = bmp.compute_coordinates(symbol_sequence=symbols)
        minimal_time = len(symbols) + 1

        # Compute unrotated value
        step_back = bmp.evaluate(np.insert(central_coordinates, 0, minimal_time - 1))
        unrotated_y = step_back.unrotated_value[1]
        if minimum - unrotated_y <= -1:
            appearing_time = minimal_time
        else:
            appearing_time = minimal_time - 1

        t = np.linspace(appearing_time, bmp.t_upper_bound(), 100)
        central_coordinates = np.broadcast_to(central_coordinates, (100, len(central_coordinates)))
        pareto_set = np.insert(central_coordinates, 0, t, axis=1)
        pareto_front = np.apply_along_axis(apply_computing, axis=1, arr=pareto_set)
        pareto_dict[node_id] = {"pareto_set": pareto_set, "pareto_front": pareto_front}

        all_sets.append(pareto_set)
        all_fronts.append(pareto_front)
    # pareto_set_all = [pareto_dict[i]["pareto_set"] for i in range(len(pareto_dict))]
    # pareto_front_all = [pareto_dict[i]["pareto_front"] for i in range(len(pareto_dict))]
    all_sets = np.concatenate(all_sets, axis=0)
    all_fronts = np.concatenate(all_fronts, axis=0)
    return pareto_dict, all_sets, all_fronts
    # for node_id in pareto_dict.keys():
    #     pareto_set = pareto_dict[node_id]["pareto_set"]
    #     pareto_front = pareto_dict[node_id]["pareto_front"]


class PerformanceIndicators:
    def __init__(self):
        pass

    def compute_perpendicular_coordinates(self, sequence_info, solver_log: pd.DataFrame, dimension: int):
        x_s = ["t", "x1", "x2", "eval_node_id"]
        y_s = ["y1", "y2", "eval_node_id"]
        y_orgs = ["t_org", "y_org", "eval_node_id"]

        x_s = solver_log[x_s].values
        y_s = solver_log[y_s].values
        y_orgs = solver_log[y_orgs].values

        bmp = evaluation.BMP(sequence_info=sequence_info, dim_space=dimension)
        node_info = {}
        node_coef = {}
        max_t = 0
        maximal = -1
        minimal = -1
        node_count = 0
        for node in sequence_info:
            node_id = node["name"]
            symbols = node["attrs"]["symbol"]
            node_minimal = node["minima"]
            minimal_time = len(symbols) + 1
            if minimal_time > max_t:
                max_t = minimal_time
            if maximal < node_minimal:
                maximal = node_minimal
            if minimal > node_minimal:
                minimal = node_minimal
            central_coordinates = bmp.compute_coordinates(symbol_sequence=symbols)
            step_back = bmp.evaluate(np.insert(central_coordinates, 0, minimal_time - 1))

            # a = y2 - y1
            # b = x1 - x2
            # c = (x2 * y1) - (y2 * x1)

            x_1 = step_back.unrotated_value[0]
            y_1 = step_back.unrotated_value[1]
            x_2 = minimal_time
            y_2 = -1 if symbols == [] else node_minimal

            node_info[node_id] = {
                "node_id": node_id,
                "symbols": symbols,
                "minimal": -1 if symbols == [] else node_minimal,
                "minimal_time": minimal_time,
                "central_coordinates": central_coordinates.tolist(),
                "step_back": {
                    "t": step_back.t,
                    "y": step_back.y,
                    "unrotated_t": step_back.unrotated_value[0],
                    "unrotated_y": step_back.unrotated_value[1],
                },
            }
            a = y_2 - y_1
            b = x_1 - x_2
            c = (x_2 * y_1) - (y_2 * x_1)
            div = math.sqrt(a**2 + b**2)
            node_coef[node_id] = {
                "node_minimal": y_2,
                "a": a,
                "b": b,
                "c": c,
                "div": div,
            }
            node_count += 1

        solver_log[["node_minimal", "a", "b", "c", "div"]] = solver_log["eval_node_id"].map(node_coef).apply(pd.Series)

        print("solver log", solver_log)

        solver_log["d_1"] = (
            abs(solver_log["a"] * solver_log["t_org"] + solver_log["b"] * solver_log["y_org"] + solver_log["c"])
            / solver_log["div"]
        )

        solver_log["d_2"] = abs(solver_log["y_org"] - solver_log["node_minimal"])

        print("node_inf", node_info)
        print("solver log", solver_log)
        print(solver_log.columns)

    def IGD(self):
        pass

    def IGDx(self):
        pass

    def GD(self):
        pass

    def GDx(self):
        pass


def main():
    pi = PerformanceIndicators()
    test_data_path = Path(
        "data/pop100_50000iter/exp_csvs/GDE3_breadth_base_1_2_StoppingByEvaluations_2023-03-22T10-51-26.821709.csv"
    )
    demo_log = file_utils.load_evaluation_log(base_path / test_data_path, include_variables=True)
    demo_log = pd.DataFrame(demo_log)

    tree_name = "breadth_base_1"
    dimension = 2

    tree = Tree(dim_space=dimension)
    tree.from_json(base_path / f"experiment_trees/{tree_name}.json")
    pi.compute_perpendicular_coordinates(sequence_info=tree.to_sequence(), solver_log=demo_log, dimension=dimension)


if __name__ == "__main__":
    main()
