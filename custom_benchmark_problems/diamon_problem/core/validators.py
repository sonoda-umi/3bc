import numpy as np

from custom_benchmark_problems.diamon_problem.core.evaluation import BMP


def validate_tree_minima(sequence_data: list, dim_space: int):
    """Validate the correctness of the tree, raise value error if the input do not meet the requirements.

    Parameters
    ----------
    sequence_data : list
        A list contains tree information.
    dim_space : int
        Dimension of the solution space

    Returns
    -------
    None
    """
    sorted_sequence = sorted(sequence_data, key=lambda node: len(node["attrs"]["symbol"]))
    t = len(sorted_sequence[-1]["attrs"]["symbol"]) + 5
    minimal_errs = []
    while sorted_sequence:
        current_node = sorted_sequence.pop()
        bmp = BMP(sequence_info=sorted_sequence, dim_space=dim_space, rotate=False)
        minimal_coordinates = bmp.compute_coordinates(current_node["attrs"]["symbol"])
        solution_variables = np.insert(minimal_coordinates, 0, t, axis=0)
        computed_minimal = bmp.evaluate(solution_variables=solution_variables)
        if computed_minimal[1] < current_node["minima"]:
            minimal_errs.append(f"Node: {current_node['name']}'s minima value should be less than {computed_minimal[1]}")
    if minimal_errs:
        raise ValueError(minimal_errs)


if __name__ == "__main__":
    a = np.array([1, 2, 3])
    print(a)
    print(np.insert(a, 0, 999, axis=0))
