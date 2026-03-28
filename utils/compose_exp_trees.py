import json

breadth_node_minima = [-2, -2.1, -2.2, -2.3]
depth_node_minima = [-2, -3, -4, -5]

incremental_bases = list(range(1, 7))

for incremental_base in incremental_bases:
    breadth = {
        "nodes": [
            {"id": 1, "minima": -2, "symbol": [1] * incremental_base},
            {"id": 2, "minima": -2.1, "symbol": [-1] * incremental_base},
            {"id": 3, "minima": -2.2, "symbol": [2] * incremental_base},
            {"id": 4, "minima": -2.3, "symbol": [-2] * incremental_base},
        ]
    }
    depth = {
        "nodes": [
            {"id": 1, "minima": -2, "symbol": [1] * incremental_base},
            {"id": 2, "minima": -3, "symbol": [1] * (incremental_base + 2)},
            {"id": 3, "minima": -4, "symbol": [1] * (incremental_base + 4)},
            {"id": 4, "minima": -5, "symbol": [1] * (incremental_base + 6)},
        ]
    }
    with open(f"../experiment_trees/breadth_base{incremental_base}.json", "w") as breadth_file:
        json.dump(breadth, breadth_file)

    with open(f"../experiment_trees/depth_base{incremental_base}.json", "w") as depth_file:
        json.dump(depth, depth_file)
