from typing import List

import numpy as np

from config import project_base
from custom_benchmark_problems.diamon_problem.core import evaluation
from custom_benchmark_problems.diamon_problem.core.evaluation import (
    ParetoInfo,
)
from custom_benchmark_problems.diamon_problem.data_structures.tree import Tree


def compute_links(tree_data: dict) -> list:
    """Compute the link between nodes with nodes' symbol information

    Parameters
    ----------
    tree_data : dict
        Node dictionary, contains at least "symbol" and "id" information

    Returns
    -------
    list
        List of links in dictionary with keys: source, target

    """
    links_info = []
    sorted_tree = sorted(tree_data["nodes"], key=lambda x: x["symbol"])
    sorted_tree.reverse()
    for index, node in enumerate(sorted_tree):
        if node["id"] == 0:
            continue
        previous_node = find_previous_node(node, index, sorted_tree)
        if previous_node:
            links_info.append({"source": previous_node["id"], "target": node["id"]})
        else:
            links_info.append({"source": 0, "target": node["id"]})
    return links_info


def find_previous_node(current_node: dict, index: int, sorted_tree: list) -> dict:
    """Find the previous node in tree

    Parameters
    ----------
    current_node : dict
        Current node information
    index : int
        Current search index in tree
    sorted_tree : list
        Tree information, sorted by the length of "symbol" in reverse order (Longer one goes first)

    Returns
    -------
    dict
        Previous node of the current node

    """
    for i in range(index, len(sorted_tree) - 1):
        if check_sublist(sorted_tree[i + 1]["symbol"], current_node["symbol"]):
            return sorted_tree[i + 1]


def check_sublist(new_list: list, org_list: list) -> bool:
    # TODO: Naive implementation, may need to change
    iter_length = len(new_list) if len(new_list) < len(org_list) else len(org_list)
    for i in range(iter_length):
        if new_list[i] != org_list[i]:
            return False
    return True


def compute_intercept():
    # TODO: Confirm if this can be computed with per-dimension.
    # The diamond should be in the domain, what happen to the diamond at higher dim?
    pass


def compute_global_pareto_front(sequence_info: list, dimension: int):
    """Compute the global Pareto front for a given tree

    Returns
    -------

    """
    sorted_tree = sort_tree(sequence_info, dimension)
    s_lengths = list(reversed(sorted(sorted_tree.keys())))
    # Corner cases
    final_node = sorted_tree[s_lengths[0]][0]
    intersections = [[len(final_node.symbol) + 2, final_node.minima]]

    for index, s_length in enumerate(s_lengths):
        if index + 1 == len(s_lengths):
            break
        current_nodes = sorted_tree[s_length]
        pre_nodes = sorted_tree[s_lengths[index + 1]]
        # Add corner points of previous node
        intersections.append(pre_nodes[0].minima_coordinates)
        for node in current_nodes:
            intersections.append(node.minima_coordinates)
            if node.step_back_coordinates[1] < pre_nodes[0].minima_coordinates[1]:
                intersections.append(node.step_back_coordinates)
            else:
                slope = slope_(node.minima_coordinates, node.step_back_coordinates)
                intercept_ = intercept(slope, node.minima_coordinates)
                x = (pre_nodes[0].minima_coordinates[1] - intercept_) / slope
                intersections.append([x, pre_nodes[0].minima_coordinates[1]])
            if len(current_nodes) > 1:
                sub_sections = []
                for sub_node in current_nodes[1:]:
                    intersection = compute_intersection(node, sub_node)
                    sub_sections.append(intersection)
                sub_sections = get_non_dominated_points(sub_sections)
                intersections.extend(sub_sections)
            current_nodes.pop(0)
    intersections.append([0, 0])
    print("intersections", intersections)
    print(sorted(intersections, key=lambda x: x[0]))
    print("fronts", extract_fronts(intersections))
    return intersections


def extract_fronts(intersections: list[list[float]]):
    intersections = sorted(intersections, key=lambda x: x[0])
    fronts = []
    for index, intersection in enumerate(intersections):
        if index == len(intersections) - 1:
            break
        p1 = intersection
        p2 = intersections[index + 1]
        if p1[0] != p2[0]:
            slope = slope_(p1, p2)
            if slope > -1:
                fronts.append([p1, p2])
    return fronts


def compute_intersection(node_1: ParetoInfo, node_2: ParetoInfo) -> list[float]:
    slope_1 = slope_(node_1.minima_coordinates, node_1.step_back_coordinates)
    slope_2 = slope_(node_2.minima_coordinates, node_2.step_back_coordinates)
    intercept_1 = intercept(slope_1, node_1.minima_coordinates)
    intercept_2 = intercept(slope_2, node_2.minima_coordinates)
    x = (intercept_2 - intercept_1) / (slope_1 - slope_2)
    y = slope_1 * x + intercept_1
    return [x, y]


def slope_(p1: list[float], p2: list[float]) -> float:
    assert len(p1) == 2 and len(p2) == 2, "Inconsistent coordinates length (should be 2)"
    assert p1[0] != p2[0], "Infinite slope" + str(p1) + str(p2)
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def intercept(slope: float, p: list[float]) -> float:
    assert len(p) == 2, "Invalid point length"
    return p[1] - slope * p[0]


def sort_tree(sequence_info: list, dimension: int) -> dict:
    """Sort tree based on [node_minimum, symbol_length]

    Returns
    -------

    """
    s_dict = {}
    bmp = evaluation.BMP(sequence_info=sequence_info, dim_space=dimension)
    for node in sequence_info:
        node_id = node["name"]
        symbols = node["attrs"]["symbol"]
        node_minimal = node["minima"]
        minimal_time = len(symbols) + 1
        central_coordinates = bmp.compute_coordinates(symbol_sequence=symbols)
        len_s = len(symbols)
        step_back = bmp.evaluate(np.insert(central_coordinates, 0, minimal_time - 1))

        if len_s in s_dict:
            s_dict[len_s].append(
                ParetoInfo(
                    symbols,
                    node_minimal,
                    node_id,
                    [minimal_time, node_minimal],
                    [step_back.unrotated_value[0], step_back.unrotated_value[1]],
                )
            )
        else:
            s_dict[len_s] = [
                ParetoInfo(
                    symbols,
                    node_minimal,
                    node_id,
                    [minimal_time, node_minimal],
                    [step_back.unrotated_value[0], step_back.unrotated_value[1]],
                )
            ]
    for s_length in s_dict.keys():
        s_dict[s_length] = sorted(s_dict[s_length], key=lambda x: x.minima)
    return s_dict


def dominance_test(vector1: [float], vector2: [float]) -> int:
    """Implementation of dominance test.
    Original code: https://github.com/jMetal/jMetalPy/blob/c6007cad8aa0e12ddd6f8d2f749e3d2cfb6b1367/jmetal/util/comparator.py#L154

    Parameters
    ----------
    vector1
    vector2

    Returns
    -------

    """
    result = 0
    for i in range(len(vector1)):
        if vector1[i] > vector2[i]:
            if result == -1:
                return 0
            result = 1
        elif vector2[i] > vector1[i]:
            if result == 1:
                return 0
            result = -1

    return result


def get_non_dominated_points(points: List[list]):
    archive = NonDominatedPointsArchive()
    for point in points:
        archive.add(point)

    return archive.non_dominated_points


class NonDominatedPointsArchive:
    def __init__(self):
        self.comparator = dominance_test
        self.non_dominated_points = []

    def add(self, point: list) -> bool:
        is_dominated = False
        is_contained = False

        if len(self.non_dominated_points) == 0:
            self.non_dominated_points.append(point)
            return True
        else:
            number_of_deleted_solutions = 0

            # New copy of list and enumerate
            for index, current_point in enumerate(list(self.non_dominated_points)):
                is_dominated_flag = self.comparator(point, current_point)
                if is_dominated_flag == -1:
                    del self.non_dominated_points[index - number_of_deleted_solutions]
                    number_of_deleted_solutions += 1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break
                elif is_dominated_flag == 0:
                    if current_point == point:
                        is_contained = True
                        break

        if not is_dominated and not is_contained:
            self.non_dominated_points.append(point)
            return True

        return False


if __name__ == "__main__":
    # test_sequence = [1, 2, 1]
    # test_solution_coordinates = np.array([1.5, 1.5])
    # compute_distance(
    #     symbol_sequence=test_sequence,
    #     solution_coordinates=test_solution_coordinates,
    #     dim_space=2,
    #     diagonal_length=5,
    # )
    tree_path = project_base / "experiment_trees" / "sample.json"
    test_tree = Tree(dim_space=3)
    test_tree.from_json(tree_path)
    test_sequence_info = test_tree.to_sequence()
    compute_global_pareto_front(test_sequence_info, 3)
    points_list = [
        [1, -1],
        [1.5, -0.5],
        [2.5, -0.75],
        [2, -1.5],
        [3, -2],
        [4, -4],
        [2.5, -3],
    ]
