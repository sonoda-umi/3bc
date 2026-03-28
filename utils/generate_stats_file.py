"""
Utility to generate statistics files from experiment results.

Results are in CSV format and contain counts of evaluations at different tree nodes
"""

import multiprocessing
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.file_utils import parse_meta


def get_exps_meta(search_dir, exp_dir_pattern):
    # subdirs = [
    #     os.path.join(search_dir, d)
    #     for d in os.listdir(search_dir)
    #     if os.path.isdir(os.path.join(search_dir, d)) and d.startswith(exp_dir_pattern)
    # ]

    subdirs = []
    for root, sub_dirs, files in os.walk(search_dir):
        for sub_dir in sub_dirs:
            cwd = os.path.join(root, sub_dir)
            if sub_dir.startswith(exp_dir_pattern):
                subdirs.append(cwd)
            else:
                for sub_root, sub_sub_dirs, sub_files in os.walk(cwd):
                    for sub_sub_dir in sub_sub_dirs:
                        if sub_sub_dir.startswith(exp_dir_pattern):
                            subdirs.append(os.path.join(cwd, sub_sub_dir))

    print("Walking subdirectories done, found ", len(subdirs), " experiment directories. \n Parsing metadata...")

    meta_list = []
    for subdir in subdirs:
        meta_list += parse_meta(exp_dir=subdir)

    exp_df = pd.DataFrame(meta_list)
    exp_df.to_csv("experiment_metadata.csv", index=False)
    return exp_df


def parse_result_file(exp_file_path: str):
    result_df = pd.read_csv(exp_file_path)
    return result_df


def run_data(dimension, n_objectives, tree, generation: int, solvers, exp_df):
    print(f"Processing dimension {dimension}, n_objectives {n_objectives}, tree {tree}, generation {generation}")
    naming_prefix = f"dim{dimension}_objs{n_objectives}_tree_{tree.split('.')[0]}"
    stat_res = []
    for solver in solvers:
        filtered_df = exp_df[
            (exp_df["dimension"] == dimension)
            & (exp_df["n_objectives"] == n_objectives)
            & (exp_df["solver"] == solver)
            & (exp_df["tree"] == tree)
        ]
        for i, row in filtered_df.iterrows():
            eval_info = parse_result_file(row["exp_result_file"])
            try:
                vc = eval_info["eval_node_id"][generation * 100 : (generation + 1) * 100].value_counts()
                stat_res.append(
                    {
                        "solver": solver,
                        "exp_index": i,
                        "root": vc.get(0, 0),
                        "node_1": vc.get(1, 0),
                        "node_2": vc.get(2, 0),
                        "node_3": vc.get(3, 0),
                        "node_4": vc.get(4, 0),
                    }
                )
            except Exception:
                continue
    stat_res = pd.DataFrame(stat_res)
    stat_res.to_csv(f"stats_output/gen_{generation}/" + naming_prefix + ".csv")


def main():
    parser = ArgumentParser()
    parser.add_argument("--search_dir", type=str, default="data")
    parser.add_argument("--exp_dir_pattern", type=str, default="N-obj")
    parser.add_argument("--output_dir", type=str, default="stats_output")
    parser.add_argument("--max_gen", type=int, default=50)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    search_dir = args.search_dir
    exp_dir_pattern = args.exp_dir_pattern
    output_dir = args.output_dir
    max_gen = args.max_gen
    step = args.step

    dimensions = [2, 3, 4, 5]
    n_objectives_list = [2, 3, 4, 5]
    trees = ["breadth.json", "depth.json"]
    solvers = ["MOEAD", "NSGAII", "GDE3", "OMOPSO", "IBEA"]
    gens = range(2, max_gen + 1, step)
    total_tasks = len(dimensions) * len(n_objectives_list) * len(trees) * len(gens)

    exp_df = get_exps_meta(search_dir, exp_dir_pattern)

    cpus = multiprocessing.cpu_count()
    pool = Pool(processes=cpus)
    pbar = tqdm(total=total_tasks)
    pbar.set_description("Parsing Progress")

    def pbar_update(_):
        pbar.update()

    def print_err(value):
        print(f"ERR! {value}")
        pbar.update()

    os.makedirs(f"{output_dir}", exist_ok=True)
    for gen in gens:
        os.makedirs(f"{output_dir}/gen_{gen}", exist_ok=True)
        for dimension in dimensions:
            for n_objectives in n_objectives_list:
                for tree in trees:
                    pool.apply_async(
                        run_data,
                        args=(
                            dimension,
                            n_objectives,
                            tree,
                            gen,
                            solvers,
                            exp_df,
                        ),
                        error_callback=print_err,
                        callback=pbar_update,
                    )
    pool.close()
    pool.join()
    pbar.close()


if __name__ == "__main__":
    main()
