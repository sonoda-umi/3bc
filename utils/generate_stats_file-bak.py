"""
Utility to generate statistics files from experiment results.

Results are in CSV format and contain counts of evaluations at different tree nodes
"""

import multiprocessing
import os
import sys
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.arg_utils import parse_gen_range
from utils.file_utils import parse_meta


def get_exps_meta(search_dir, exp_dir_pattern):
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


def run_data(dimension, n_objectives, tree, generation: int, solvers: list, exp_meta_list: list, gen_output_dir: str):
    naming_prefix = f"dim{dimension}_objs{n_objectives}_tree_{tree.split('.')[0]}"
    stat_res = []
    for solver in solvers:
        # Filter metadata list instead of DataFrame for better multiprocessing compatibility
        try:
            filtered_meta = [
                item for item in exp_meta_list
                if item["dimension"] == dimension
                and item["n_objectives"] == n_objectives
                and item["solver"] == solver
                and item["tree"] == tree
            ]
        except (KeyError, TypeError) as e:
            print(f"Error filtering metadata: {e}")
            continue
            
        for idx, item in enumerate(filtered_meta):
            try:
                eval_info = parse_result_file(item["exp_result_file"])
                if len(eval_info) == 0:
                    continue
                vc = eval_info["eval_node_id"][generation * 100 : (generation + 1) * 100].value_counts()
                stat_res.append(
                    {
                        "solver": solver,
                        "exp_index": idx,
                        "root": vc.get(0, 0),
                        "node_1": vc.get(1, 0),
                        "node_2": vc.get(2, 0),
                        "node_3": vc.get(3, 0),
                        "node_4": vc.get(4, 0),
                    }
                )
            except Exception as e:
                # Silently skip experiments with issues
                pass
                
    if stat_res:
        stat_res = pd.DataFrame(stat_res)
        stat_res.to_csv(f"{gen_output_dir}/{naming_prefix}.csv")


def main():
    parser = ArgumentParser()
    parser.add_argument("--search_dir", type=str, default="data")
    parser.add_argument("--exp_dir_pattern", type=str, default="N-obj")
    parser.add_argument("--output_dir", type=str, default="stats_output")
    parser.add_argument("--gens", type=str)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    search_dir = args.search_dir
    exp_dir_pattern = args.exp_dir_pattern
    output_dir = args.output_dir
    step = args.step

    dimensions = [2, 3, 4, 5, 6, 7, 8, 9]
    n_objectives_list = [2, 3, 4, 5, 6, 7, 8, 9]
    trees = ["breadth.json", "depth.json"]
    solvers = ["MOEAD", "NSGAII", "GDE3", "OMOPSO", "IBEA", "NSGAIII"]
    start, end = parse_gen_range(args.gens)
    gens = range(start, end, step)
    total_tasks = len(dimensions) * len(n_objectives_list) * len(trees) * len(gens)

    print(f"Loading metadata from {search_dir}...")
    exp_df = get_exps_meta(search_dir, exp_dir_pattern)
    
    # Convert DataFrame to list of dicts for better multiprocessing compatibility
    exp_meta_list = exp_df.to_dict('records')
    
    print(f"Loaded metadata for {len(exp_meta_list)} experiments")
    if exp_meta_list:
        print(f"Metadata columns: {list(exp_meta_list[0].keys())}")
    print(f"Total tasks to process: {total_tasks}")

    cpus = multiprocessing.cpu_count()
    pool = Pool(processes=cpus)
    pbar = tqdm(total=total_tasks)
    pbar.set_description("Parsing Progress")

    def pbar_update(_):
        pbar.update()

    def print_err(value):
        print(f"ERR! {type(value).__name__}: {value}")
        if hasattr(value, '__traceback__'):
            traceback.print_exception(type(value), value, value.__traceback__)
        pbar.update()

    os.makedirs(f"{output_dir}", exist_ok=True)
    for gen in gens:
        gen_output_dir = f"{output_dir}/gen_{gen}"
        os.makedirs(gen_output_dir, exist_ok=True)
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
                            exp_meta_list,
                            gen_output_dir,
                        ),
                        error_callback=print_err,
                        callback=pbar_update,
                    )
    pool.close()
    pool.join()
    pbar.close()
    print(f"Saved stats to {output_dir}")


if __name__ == "__main__":
    main()
