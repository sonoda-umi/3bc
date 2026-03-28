import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import sample_file_path, solver_info
from custom_benchmark_problems.diamon_problem.core import algs, evaluation
from custom_benchmark_problems.diamon_problem.data_structures.tree import Tree
from utils import file_utils

app = FastAPI()

origins = ["http://localhost", "http://localhost:8080", "http://192.168.16.169:8080"]
data_base_path = os.getenv("EXP_DATA", "./data/sample_experiments")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/get_demo_problem")
def get_demo_problem():
    json_tree = file_utils.read_json_tree(str(sample_file_path))
    return json_tree


@app.get("/api/get_all_solvers")
def get_all_solvers():
    return solver_info


@app.post("/api/construct_problem")
def construct_problem(graph: dict):
    return graph


@app.get("/api/reeb_space")
def reeb_space_info(dimension: int, tree_name: str):
    tree = Tree(dim_space=dimension)
    tree.from_json(f"experiment_trees/{tree_name}.json")
    sequence_info = tree.to_sequence()
    bmp = evaluation.BMP(sequence_info=sequence_info, dim_space=dimension)
    node_info = []
    max_t = 0
    maximal = -1
    minimal = -1
    node_count = 0
    reeb_ids = []
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
        reeb_ids.append(node_id)
        node_info.append(
            {
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
        )
        node_count += 1
    node_info = sorted(node_info, key=lambda k: (k["minimal_time"], k["minimal"]))
    return JSONResponse(
        {
            "reeb_ids": reeb_ids,
            "nodeInfo": node_info,
            "treeInfo": {
                "nodeCount": node_count,
                "maxTime": max_t + 1,
                "minimal": minimal,
                "maximal": maximal,
                "minTime": 0,
            },
        }
    )


@app.get("/api/experiment_settings")
def get_experiment_parameters() -> Dict[str, Any]:
    """This function get all the available settings in the given data directory"""
    solvers: Set[str] = set()
    trees: Set[str] = set()
    dimensions: Set[int] = set()
    terminations: Set[str] = set()

    for entry in os.scandir(data_base_path):
        if not entry.is_dir():
            continue
        name = entry.name
        parts = name.split("_")
        # termination = last token
        termination = parts[-2]

        # dimension = last numeric token before termination
        dim_idx = None
        for i in range(len(parts) - 2, 0, -1):
            if parts[i].isdigit():
                dim_idx = i
                break
        if dim_idx is None:
            continue  # skip if no numeric dimension found

        solver = parts[0]
        tree = "_".join(parts[1:dim_idx])
        dimension = int(parts[dim_idx])

        solvers.add(solver)
        trees.add(tree)
        dimensions.add(dimension)
        terminations.add(termination)

    return {
        "solvers": sorted(solvers),
        "trees": sorted(trees),
        "dimensions": sorted(dimensions),
        "termination": sorted(terminations),
    }


def match_experiment_file(solver: str, tree: str, dimension: int, termination: str):
    file_name_pattern = f"{solver}_{tree}_{dimension}_{termination}"
    files = [f for f in os.listdir(data_base_path) if os.path.isfile(data_base_path + f) and f.endswith(".csv")]
    for file in files:
        if file.startswith(file_name_pattern):
            print("Data path: ", data_base_path + file)
            return data_base_path + file
        if file.split("_")[1].startswith(file_name_pattern):
            print("Data path: ", data_base_path + file)
            return data_base_path + file
    exp_info = file_utils.parse_exp_dir_with_meta(data_base_path, file_name_pattern)
    if exp_info:
        file, exp_tree, meta_data = exp_info
        return Path(data_base_path) / file, Path(data_base_path) / exp_tree, Path(data_base_path) / meta_data
    else:
        # Logger().debug.error(
        #     f"Experiment file for {solver}, {tree}, {dimension}, {termination} not found."
        # )
        return None, None, None


@app.get("/api/demo_data")
def demo_data(solver: str, tree_name: str, dimension: int, termination: str):
    log_path, exp_tree_path, meta_data_path = match_experiment_file(solver, tree_name, dimension, termination)
    if log_path is None:
        return JSONResponse(status_code=404, content={"message": "Experiment file not found."})
    demo_log = file_utils.load_evaluation_log(str(log_path))
    demo_tree = file_utils.read_json_tree(str(exp_tree_path))
    sequence_dict = {}
    for node in demo_tree["nodes"]:
        sequence_dict[node["id"]] = node
        sequence_dict[node["id"]]["label"] = f"ID: {node['id']},  Symbol: {node['symbol']},Best possible: {node['minima']}"
    link_map = {}
    for link in algs.compute_links(demo_tree):
        source_id = link["source"]
        if source_id in link_map.keys():
            link_map[source_id].append(link["target"])
        else:
            link_map[source_id] = [link["target"]]
    all_ids = list(sequence_dict.keys())
    all_ids.append(0)
    response = {
        "all_ids": all_ids,
        "tree": [
            {
                "id": 0,
                "label": "Root,  ID: 0, Best possible: -1.0",
                "children": construct_tree_structure(0, link_map, sequence_dict=sequence_dict),
            }
        ],
        "solver_log": demo_log,
    }
    return JSONResponse(content=jsonable_encoder(response))


def construct_tree_structure(current_key, links_map: dict, sequence_dict: dict):
    result = []
    for sub_key in links_map[current_key]:
        if sub_key in links_map.keys():
            sequence_dict[sub_key]["children"] = construct_tree_structure(sub_key, links_map, sequence_dict)
            result.append(sequence_dict[sub_key])
        else:
            result.append(sequence_dict[sub_key])
    return result


# @app.get("/api/available_exp_indexes")
# def get_exp_indexes():
#     data_base_path = (
#         # "/Volumes/l-liu/benchmark-visualizer-exp-data/pop100_50000iter/exp_csvs/"
#         # "data/pop100_50000iter/pop100_50000iter/"
#         "data/exp_20240202/"
#     )
#     files = [
#         f for f in os.listdir(data_base_path) if os.path.isfile(data_base_path + f)
#     ]
#     exp_indexes = []
#     for file in files:
#         exp_indexes.append(file.split("__")[0])
#     exp_indexes = list(set(exp_indexes))
#     return JSONResponse({"experiment_indexes": sorted(exp_indexes)})
