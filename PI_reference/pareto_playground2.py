import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jmetal.core.quality_indicator import GenerationalDistance, InvertedGenerationalDistance
from utils.local_pareto import get_local_pareto_set

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })


# 仮定: pareto_dictには複数のParetoフロントが含まれている
# pareto_fronts = [pareto_dict[key]["pareto_front"] for key in pareto_dict.keys()]


def gd(reference, actual):
    return GenerationalDistance(reference).compute(actual)


def igd(reference, actual):
    return InvertedGenerationalDistance(reference).compute(actual)


# GD
def compute_indicators(
    reference_set,
    reference_front,
    generation_set,
    generation_front,
    pareto_dict: dict,
    generation_points_node_x,
    generation_points_node_y,
    indicator_type,
):
    if indicator_type == "GD":
        indicator = gd
        # indicator_str = "Generational Distance"
        indicator_str = "GD"
    elif indicator_type == "IGD":
        indicator = igd
        # indicator_str = "Inverted Generational Distance"
        indicator_str = "IGD"
    else:
        raise NotImplementedError("Unrecognized Quality Indicator")

    # Generational Distanceの計算
    gd_value = indicator(reference=reference_front, actual=generation_front)
    gdx_value = indicator(reference=reference_set, actual=generation_set)

    # print(f"{indicator_str}: ", gd_value)
    # print(f"{indicator_str} X: ", gdx_value)
    global_indicators = {"gd": gd_value, "gdx": gdx_value}

    node_indicators = {}
    for node_id in pareto_dict.keys():
        # Generational Distanceの計算
        set = generation_points_node_x[generation_points_node_x[:, 3] == node_id, :][:, 0:3]
        front = generation_points_node_y[generation_points_node_y[:, 2] == node_id, :][:, 0:2]

        if front.size > 0:
            gd_value = indicator(reference=pareto_dict[node_id]["pareto_front"], actual=front)
            gdx_value = indicator(reference=pareto_dict[node_id]["pareto_set"], actual=set)
            # print(f"{indicator_str} for node[{node_id}]: ", gd_value)
            # print(f"{indicator_str} X for node[{node_id}]: ", gdx_value)
            node_indicators[node_id] = {"gd": gd_value, "gdx": gdx_value}
        else:
            # print(f"{indicator_str} for node[{node_id}]: NaN")
            node_indicators[node_id] = {"gd": None, "gdx": None}
    return {"global": global_indicators, "node": node_indicators}


# def match_experiment_file(data_base_path:str, solver: str, tree: str, dimension: int, termination: str):
#     file_name_pattern = f"{solver}_{tree}_{dimension}_{termination}"
#     files = [
#         f for f in os.listdir(data_base_path) if os.path.isfile(data_base_path + f)
#     ]
#     for file in files:
#         if file.startswith(file_name_pattern):
#             return data_base_path + file


def match_experiment_file(data_base_path: str, solver: str, tree: str, dimension: int, termination: str):
    file_name_pattern = f"{solver}_{tree}_{dimension}_{termination}"
    files = [f for f in os.listdir(data_base_path) if os.path.isfile(os.path.join(data_base_path, f))]
    for file in files:
        if file.startswith(file_name_pattern):
            return os.path.join(data_base_path, file)

    # ファイルが見つからなかった場合のエラーメッセージ
    raise FileNotFoundError(f"No file found for pattern: {file_name_pattern} in {data_base_path}")


def main():
    # trees = ["breadth_base_1", "depth_base_1"]
    # trees = ["breadth_base_conn_1", "depth_base_conn_1"]
    # trees = ["depth_base_2"]
    trees = ["breadth_base_1"]
    # generations = list(range(200))
    generations = list(range(50, 200, 5))
    dim_spaces = [2]
    solvers = ["NSGAII", "MOEAD", "IBEA", "GDE3", "OMOPSO"]
    termination_criterias = ["StoppingByEvaluations"]
    indicators = ["GD", "IGD"]
    population_size = 100
    # data_base_path="data/pop100_50000iter/exp_csvs/"
    # data_base_path="data/exp_230727/"
    data_base_path = "data/benchmark-visualizer-exp-data/pop100_50000iter/exp_csvs"

    # Create a DataFrame to store the results
    results_global_df = pd.DataFrame(index=["GD", "IGD", "GDX", "IGDX"], columns=solvers)

    # Create a DataFrame to store the GD values for each solver and indicator
    gd_values_df = pd.DataFrame()
    gdx_values_df = pd.DataFrame()

    # Big loop to compute all possiblities
    for tree_name in trees:
        for dim_space in dim_spaces:
            for solver_name in solvers:
                for termination_criteria in termination_criterias:
                    for indicator_type in indicators:
                        data_file = match_experiment_file(
                            data_base_path=data_base_path,
                            solver=solver_name,
                            tree=tree_name,
                            dimension=dim_space,
                            termination=termination_criteria,
                        )

                        variables_header_x = [f"x{x + 1}" for x in range(dim_space)]
                        variables_header_x.insert(0, "t")
                        variables_header_node_x = variables_header_x + ["eval_node_id"]
                        variables_header_y = ["y1", "y2"]
                        variables_header_node_y = variables_header_y + ["eval_node_id"]

                        pareto_dict, all_sets, all_fronts = get_local_pareto_set(dimension=dim_space, tree_name=tree_name)
                        experiment_record = pd.read_csv(data_file, index_col=0)

                        for generation in generations:
                            # print()
                            # print(f"*** Computing indicators for tree: {tree_name}, dimension: {dim_space}, solver: {solver_name}, generation: {generation}, termination: {termination_criteria} ***")
                            # print()
                            starting_point = generation * population_size
                            generation_df = experiment_record.iloc[starting_point : starting_point + population_size]
                            generation_points_x = generation_df[variables_header_x].values
                            generation_points_node_x = generation_df[variables_header_node_x].values
                            generation_points_y = generation_df[variables_header_y].values
                            generation_points_node_y = generation_df[variables_header_node_y].values

                            result = compute_indicators(
                                reference_set=all_sets,
                                reference_front=all_fronts,
                                generation_set=generation_points_x,
                                generation_front=generation_points_y,
                                pareto_dict=pareto_dict,
                                generation_points_node_x=generation_points_node_x,
                                generation_points_node_y=generation_points_node_y,
                                indicator_type=indicator_type,
                            )

                            # Store the GD value in the gd_values_df DataFrame
                            gd_values_df = gd_values_df.append(
                                {
                                    "tree_name": tree_name,
                                    "solver": solver_name,
                                    "indicator": indicator_type,
                                    "generation": generation,
                                    "gd_value": result["global"]["gd"],
                                    "gd_value_node": result["node"],
                                },
                                ignore_index=True,
                            )

                            gdx_values_df = gdx_values_df.append(
                                {
                                    "tree_name": tree_name,
                                    "solver": solver_name,
                                    "indicator": indicator_type,
                                    "generation": generation,
                                    "gdx_value": result["global"]["gdx"],
                                    "gdx_value_node": result["node"],
                                },
                                ignore_index=True,
                            )

                        generation = 199
                        starting_point = generation * population_size
                        generation_df = experiment_record.iloc[starting_point : starting_point + population_size]
                        generation_points_x = generation_df[variables_header_x].values
                        generation_points_node_x = generation_df[variables_header_node_x].values
                        generation_points_y = generation_df[variables_header_y].values
                        generation_points_node_y = generation_df[variables_header_node_y].values

                        result = compute_indicators(
                            reference_set=all_sets,
                            reference_front=all_fronts,
                            generation_set=generation_points_x,
                            generation_front=generation_points_y,
                            pareto_dict=pareto_dict,
                            generation_points_node_x=generation_points_node_x,
                            generation_points_node_y=generation_points_node_y,
                            indicator_type=indicator_type,
                        )

                        # Store the results in the DataFrame
                        results_global_df.loc[indicator_type, solver_name] = result["global"]["gd"]
                        results_global_df.loc[indicator_type + "X", solver_name] = result["global"]["gdx"]

        # Display the table
        plt.figure(figsize=(10, 6))  # Adjust the figure size
        plt.table(
            cellText=results_global_df.values,
            rowLabels=results_global_df.index,
            colLabels=results_global_df.columns,
            loc="center",
        )
        plt.axis("off")  # Hide the axis
        plt.title(f"Quality Indicators for Different Solvers for {tree_name}")
        plt.show()

        # Print the DataFrame in LaTeX format
        print(results_global_df.to_latex())

        # Print the DataFrame in LaTeX format and highlight the minimum in each column
        # print(results_df.style.highlight_min(axis=0).to_latex())

    for tree_name in trees:
        for solver_name in solvers:
            for indicator_type in indicators:
                # プロットの準備
                plt.figure()
                plt.ylim(-0.05, 2.5)
                plt.xlim(50, 200)

                # 各ノードに対するGD, IGD値の折れ線グラフを描く
                for node_id in pareto_dict.keys():
                    # 各世代に対する特定のノードのGD, IGD値を取得
                    gd_values_df_gd = gd_values_df[
                        (gd_values_df["tree_name"] == tree_name)
                        & (gd_values_df["solver"] == solver_name)
                        & (gd_values_df["indicator"] == indicator_type)
                    ]
                    gd_values_node = gd_values_df_gd["gd_value_node"].apply(
                        lambda x: x[node_id]["gd"] if x[node_id] is not None else np.nan
                    )

                    # 折れ線グラフを描く
                    plt.plot(generations, gd_values_node, marker=".", label=f"Node {node_id}")

                # グラフのタイトルと軸ラベルを設定
                # plt.title(f'{indicator_type} values for each node over generations for {tree_name} by {solver_name}')
                plt.xlabel("Generation")
                plt.ylabel(f"{indicator_type} value")

                # 凡例を表示
                plt.legend()

                # グラフを表示
                plt.show()

                plt.savefig(f"{solver_name}_{tree_name}_{indicator_type}_for_each_node.png")

                # プロットの準備
                plt.figure()
                plt.ylim(-0.05, 2.5)
                plt.xlim(50, 200)

                # 各ノードに対するGDX, IGDX値の折れ線グラフを描く
                for node_id in pareto_dict.keys():
                    # 各世代に対する特定のノードのGDX, IGDX値を取得
                    gdx_values_df_gd = gdx_values_df[
                        (gdx_values_df["tree_name"] == tree_name)
                        & (gdx_values_df["solver"] == solver_name)
                        & (gdx_values_df["indicator"] == indicator_type)
                    ]
                    gdx_values_node = gdx_values_df_gd["gdx_value_node"].apply(
                        lambda x: x[node_id]["gdx"] if x[node_id] is not None else np.nan
                    )

                    # 折れ線グラフを描く
                    plt.plot(
                        generations,
                        gdx_values_node,
                        marker=".",
                        label=f"Node {node_id}",
                    )

                # グラフのタイトルと軸ラベルを設定
                # plt.title(f'{indicator_type}X values for each node over generations for {tree_name} by {solver_name}')
                plt.xlabel("Generation")
                plt.ylabel(f"{indicator_type}X value")

                # 凡例を表示
                plt.legend()

                # グラフを表示
                plt.show()

                plt.savefig(f"{solver_name}_{tree_name}_{indicator_type}X_for_each_node.png")


if __name__ == "__main__":
    main()
