from pathlib import Path

import numpy as np

from custom_benchmark_problems.diamon_problem.core import evaluation, n_objectives_problem
from custom_benchmark_problems.diamon_problem.data_structures.tree import Tree


class ReferenceFronts:
    """This class is for compute the actual Pareto front (in theory)"""

    def __init__(self):
        pass

    def get_n_obj_local_pareto_set(
        self,
        dimension: int,
        n_objectives: int,
        tree_file_path: str,
        t_rotate: bool = False,
        decision_space_rotation: bool = False,
        resolution: int = 100,
    ):
        if not Path(tree_file_path).exists():
            raise ValueError(f"The tree file {tree_file_path} does not exist!")

        tree = Tree(dim_space=dimension)
        tree.from_json(tree_file_path)
        sequence_info = tree.to_sequence()
        n_bmp = n_objectives_problem.NBMP(
            sequence_info=sequence_info,
            dim_space=dimension,
            n_objectives=n_objectives,
            t_rotate=t_rotate,
        )
        # rot_matrix = n_bmp.t_rotation_matrix
        pareto_dict = {}

        def apply_computing(data: np.ndarray):
            row_data = n_bmp.n_evaluate(data)
            return row_data.objective_values

        all_sets = []
        all_fronts = []
        for node in sequence_info:
            node_id = node["name"]
            symbols = node["attrs"]["symbol"]
            minimum = node["minima"]
            central_coordinates = n_bmp.compute_coordinates(symbol_sequence=symbols)
            minimal_time = len(symbols) + 1

            # Compute unrotated value
            step_back = n_bmp.evaluate(np.insert(central_coordinates, 0, minimal_time - 1))
            unrotated_y = step_back.unrotated_value[1]

            # determine if 45 degrees rotation made tilt line Pareto front
            if minimum - unrotated_y <= -1:
                appearing_time = minimal_time
            else:
                appearing_time = minimal_time - 1

            # Create sampling points for t_1
            t = np.linspace(appearing_time, n_bmp.t_upper_bound(), resolution)
            # Create sampling points for the rest of the ts
            t_i_base = np.linspace(start=n_bmp.t_i_lower_bound, stop=0, num=resolution)

            mesh_source = [t] + [t_i_base for _ in range(n_objectives - 2)]

            # Create meshgrid for all arrays and then reshape to get all combinations
            mesh = np.meshgrid(*mesh_source)
            combinations = np.vstack([x.ravel() for x in mesh]).T
            # combinations = combinations @ rot_matrix

            central_coordinates = np.broadcast_to(central_coordinates, (len(combinations), len(central_coordinates)))

            pareto_set = np.concatenate((combinations, central_coordinates), axis=1)

            pareto_front = np.apply_along_axis(apply_computing, axis=1, arr=pareto_set)
            pareto_dict[node_id] = {
                "pareto_set": pareto_set,
                "pareto_front": pareto_front,
            }

            all_sets.append(pareto_set)
            all_fronts.append(pareto_front)

        all_sets = np.concatenate(all_sets, axis=0)
        all_fronts = np.concatenate(all_fronts, axis=0)
        return {
            "pareto_dict": pareto_dict,
            "all_sets": all_sets,
            "all_fronts": all_fronts,
        }

    def get_local_pareto_set(self, dimension: int, tree_name: str, resolution: int = 100):
        tree = Tree(dim_space=dimension)
        tree.from_json(f"../experiment_trees/{tree_name}.json")
        sequence_info = tree.to_sequence()
        bmp = evaluation.BMP(sequence_info=sequence_info, dim_space=dimension)
        pareto_dict = {}

        def apply_computing(data: np.ndarray):
            row_data = bmp.evaluate(data)
            return np.array(
                [
                    row_data.t,
                    row_data.y,
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

            t = np.linspace(appearing_time, bmp.t_upper_bound(), resolution)

            central_coordinates = np.broadcast_to(central_coordinates, (resolution, len(central_coordinates)))
            pareto_set = np.insert(central_coordinates, 0, t, axis=1)
            pareto_front = np.apply_along_axis(apply_computing, axis=1, arr=pareto_set)
            pareto_dict[node_id] = {
                "pareto_set": pareto_set,
                "pareto_front": pareto_front,
            }

            all_sets.append(pareto_set)
            all_fronts.append(pareto_front)

        all_sets = np.concatenate(all_sets, axis=0)
        all_fronts = np.concatenate(all_fronts, axis=0)
        return {
            "pareto_dict": pareto_dict,
            "all_sets": all_sets,
            "all_fronts": all_fronts,
        }


if __name__ == "__main__":
    rf = ReferenceFronts()
    res = rf.get_local_pareto_set(dimension=2, tree_name="breadth", resolution=10)
    # print(res[0])
    n_res = rf.get_n_obj_local_pareto_set(
        dimension=2,
        tree_file_path="../n_obj_experiment_trees/breadth.json",
        resolution=10,
        n_objectives=4,
    )
    print(n_res)
