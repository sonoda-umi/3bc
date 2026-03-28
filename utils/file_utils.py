import json
import math
import os
from pathlib import Path

import pandas as pd

from utils.log import Logger


def read_json_tree(file_path: str) -> dict:
    with open(file_path, "r") as json_file:
        tree_info = json.load(json_file)
    return tree_info


def tree_to_json(file_path: str, tree_info: dict) -> None:
    with open(file_path, "w") as json_file:
        json.dump(tree_info, json_file)


def load_evaluation_log(file_path: str) -> list:
    Logger().debug.info(f"Loading data from disk, file size {get_file_size(file_path)}")
    eval_log = pd.read_csv(file_path, index_col=0)
    eval_log[["eval_node_id", "step"]] = eval_log[["eval_node_id", "step"]].astype(int)
    Logger().debug.info("Load complete, processing ...... ")
    return eval_log[["t", "y1", "y2", "eval_node_id", "diagonal_length", "step", "t_org", "y_org"]].to_dict(orient="records")


def load_n_evaluation_log(file_path: str, return_df: bool = False) -> list | pd.DataFrame:
    Logger().debug.info(f"Loading data from disk, file size {get_file_size(file_path)}")
    if not Path(file_path).exists():
        raise ValueError("Experiment log file not found")
    eval_log = pd.read_csv(file_path, index_col=0)
    Logger().debug.info("Load complete, processing ...... ")
    if return_df:
        return eval_log
    return eval_log.to_dict(orient="records")


def get_file_size(file_path):
    try:
        file_size = os.path.getsize(file_path)
        return file_size
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"An error occurred: {e}"


def convert_size(size_bytes):
    if size_bytes == "File not found" or isinstance(size_bytes, str):
        return size_bytes
    elif size_bytes == 0:
        return "0B"

    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def parse_exp_dir_with_meta(data_path, file_name_pattern: str) -> tuple[Path, Path, Path] | None:
    entries = os.listdir(data_path)
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(data_path, entry))]
    for sub_dir in subdirectories:
        if sub_dir.startswith(file_name_pattern):
            exp_file_path = Path(sub_dir) / (sub_dir + ".csv")
            exp_tree = Path(sub_dir) / "meta" / "experiment_tree.json"
            meta_data = Path(sub_dir) / "meta" / "meta.json"
            return exp_file_path, exp_tree, meta_data

    Logger().debug.error(f"File or directory {data_path} for pattern {file_name_pattern} not found.")
    return None


def parse_meta(exp_dir: str) -> list[dict]:
    exps_info = parse_exp_log_dir(exp_dir=exp_dir)
    exps_meta_info = []
    for exp_info in exps_info.values():
        try:
            meta = exp_info["meta"]
            dimension = meta["dimension"]
            n_objectives = meta["n_objectives"]
            try:
                population_size = meta["algorithm_parameters"]["population_size"]
            except KeyError:
                population_size = meta["algorithm_parameters"]["swarm_size"]
            exps_meta_info.append(
                {
                    "population_size": population_size,
                    "dimension": dimension,
                    "n_objectives": n_objectives,
                    "tree": meta["tree_file"].split("/")[-1],
                    "solver": meta["algorithm"],
                    "exp_result_file": exp_info["result"],
                }
            )
        except Exception as e:
            print(f"error dir: {exp_info}")
    return exps_meta_info


def parse_exp_log_dir(exp_dir: str) -> dict:
    exp_path = Path(exp_dir)
    if not (exp_path.exists() and exp_path.is_dir()):
        raise ValueError(f"Invalid experiment experiment path. \n Path: {exp_path}")
    subdirectories = [d for d in exp_path.iterdir() if d.is_dir()]
    if subdirectories:
        exp_parse_data = {}
        for exp_data_dir in subdirectories:
            meta_dir = exp_data_dir / "meta"
            metadata = meta_dir / "meta.json"
            experiment_tree_file = meta_dir / "experiment_tree.json"
            exp_result_file = exp_data_dir / (exp_data_dir.name + ".csv")
            with open(metadata, "r") as meta_file:
                meta_data = json.load(meta_file)
            exp_parse_data[str(exp_data_dir.name)] = {
                "meta": meta_data,
                "tree": experiment_tree_file,
                "result": str(exp_result_file),
            }
        return exp_parse_data
    else:
        raise ValueError(f"No experiment data found @ {exp_path}")


if __name__ == "__main__":
    # print(parse_exp_dir_with_meta("../data/test_exp_v8", "NSGAII"))
    # print(load_evaluation_log("../test_runx_2023-01-16T10-30-27.298413.csv"))
    res = parse_exp_log_dir("../data/fully_managed_experiments_2025-08-05")
    print(res)
