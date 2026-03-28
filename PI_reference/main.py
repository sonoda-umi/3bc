import os
from typing import Union

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


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/api/get_demo_problem")
def get_demo_problem():
    json_tree = file_utils.read_json_tree(sample_file_path)
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


def match_experiment_file(solver: str, tree: str, dimension: int, termination: str):
    file_name_pattern = f"{solver}_{tree}_{dimension}_{termination}"
    print(file_name_pattern)
    data_base_path = "/Volumes/l-liu/benchmark-visualizer-exp-data/pop100_50000iter/exp_csvs/"
    files = [f for f in os.listdir(data_base_path) if os.path.isfile(data_base_path + f)]
    for file in files:
        if file.startswith(file_name_pattern):
            return data_base_path + file


@app.get("/api/demo_data")
def demo_data(solver: str, tree_name: str, dimension: int, termination: str):
    log_path = match_experiment_file(solver, tree_name, dimension, termination)
    demo_log = file_utils.load_evaluation_log(log_path)
    demo_tree = file_utils.read_json_tree(f"experiment_trees/{tree_name}.json")
    sequence_dict = {}
    for node in demo_tree["nodes"]:
        sequence_dict[node["id"]] = node
        sequence_dict[node["id"]]["label"] = f"Node ID: {node['id']},  Symbol: {node['symbol']},Minimum: {node['minima']}"
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
                f"label": f"Root,  ID: 0, Minimum: -1.0",
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
