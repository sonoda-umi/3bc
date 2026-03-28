from argparse import ArgumentParser

import yaml


def compose_solver_settings(solver_name) -> dict:
    if solver_name == "GDE3":
        settings = {"population_size": 100, "cr": 0.5, "f": 0.5}
    elif solver_name == "NSGAII":
        settings = {
            "population_size": 100,
            "offspring_population_size": 100,
            "mutation": {
                "mutation": "PolynomialMutation",
                "probability": "n_variables",
                "distribution_index": 20,
            },
            "crossover": {
                "crossover": "SBXCrossover",
                "probability": 1.0,
                "distribution_index": 20,
            },
        }
    elif solver_name == "IBEA":
        settings = {
            "kappa": 1.0,
            "population_size": 100,
            "offspring_population_size": 100,
            "mutation": {
                "mutation": "PolynomialMutation",
                "probability": "n_variables",
                "distribution_index": 20,
            },
            "crossover": {
                "crossover": "SBXCrossover",
                "probability": 1.0,
                "distribution_index": 20,
            },
        }
    elif solver_name == "MOEAD":
        settings = {
            "population_size": 100,
            "mutation": {
                "mutation": "PolynomialMutation",
                "probability": "n_variables",
                "distribution_index": 20,
            },
            "crossover": {
                "crossover": "DifferentialEvolutionCrossover",
                "CR": 1.0,
                "F": 0.5,
                "K": 0.5,
            },
            "aggregative_function": {
                "aggregative_function": "Tschebycheff",
                "dimension": "n_variables",
            },
            "neighbor_size": 20,
            "neighbourhood_selection_probability": 0.9,
            "max_number_of_replaced_solutions": 2,
            "weight_files_path": "resources/MOEAD_weights",
        }
    elif solver_name == "OMOPSO":
        settings = {
            "swarm_size": 100,
            "epsilon": 0.0075,
            "uniform_mutation": {
                "uniform_mutation": "UniformMutation",
                "probability": "n_variables",
                "perturbation": 0.5,
            },
            "non_uniform_mutation": {
                "non_uniform_mutation": "NonUniformMutation",
                "probability": "n_variables",
                "perturbation": 0.5,
                "max_iterations": "max_evaluations/swarm_size",
            },
            "leaders": {"leaders": "CrowdingDistanceArchive", "maximum_size": 100},
        }
    else:
        raise NotImplementedError
    return settings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_base_name", type=str, default="N-obj-test")
    parser.add_argument("--output_path", type=str, default="exp_config.yaml")
    args = parser.parse_args()
    exp_base_name = args.exp_base_name
    output_path = args.output_path
    # trees = ["experiment_trees/diverse_tree.json"]
    # trees = []
    # depth_trees = [f"experiment_trees/depth_base_{i}.json" for i in range(1, 7)]
    # breadth_trees = [f"experiment_trees/breadth_base_{i}.json" for i in range(1, 7)]
    # trees.extend(breadth_trees)
    # trees.extend(depth_trees)
    trees = [
        "n_obj_experiment_trees/breadth.json",
        "n_obj_experiment_trees/depth.json",
    ]
    solvers = ["MOEAD", "GDE3", "NSGAII", "IBEA", "OMOPSO"]
    dimensions = [2, 3, 4, 5]
    dimensions.reverse()
    n_objectives = [2, 3, 4, 5]
    n_objectives.reverse()
    termination_criterions = [
        # {"criterion_name": "StoppingByTime", "termination_parameter": 200},
        {"criterion_name": "StoppingByEvaluations", "termination_parameter": 100000},
    ]
    counter = 0
    exp_settings = []
    rotate_t = [True]
    for tree in trees:
        for solver in solvers:
            if solver == "MOEAD":
                pass
            else:
                n_objectives = [2]
            for dimension in dimensions:
                for n_objective in n_objectives:
                    for termination_criterion in termination_criterions:
                        for if_rotate_t in rotate_t:
                            exp_settings.append(
                                {
                                    "experiment_name": exp_base_name + f"_v{counter}",
                                    "tree_file": tree,
                                    "dimension": dimension,
                                    "algorithm": solver,
                                    "n_objectives": n_objective,
                                    "algorithm_parameters": compose_solver_settings(solver),
                                    "termination_criterion": dict(termination_criterion),
                                    "rotate_t": if_rotate_t,
                                }
                            )
                            counter += 1
    with open(f"{output_path}", "w") as file:
        documents = yaml.safe_dump(exp_settings, file)
