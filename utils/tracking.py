import argparse
import datetime
import json
import os
import shutil
import sys
import traceback
import typing

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config import data_dir, mlflow_tracking_uri
from utils.data_structures import ExperimentSettings
from utils.log import Logger

ArgParserFunc = typing.Callable[[typing.Optional[argparse.ArgumentParser]], argparse.Namespace]


def batch_create_experiments(exp_names: set) -> dict:
    exps = {}
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    for exp_name in exp_names:
        try:
            exp_id = mlflow.create_experiment(exp_name)
            Logger().debug.info(f"Created experiment {exp_name}")

        except MlflowException as e:
            exp_id = mlflow.get_experiment_by_name("exp_name")
            Logger().debug.info(f"Experiment {exp_name} exists, skipping ...")
        exps[exp_name] = exp_id
    return exps


class MlflowTracker:
    def __init__(self, run_name: str, experiment_config: ExperimentSettings):
        self.run_name = run_name
        self.experiment_config = experiment_config
        self.step = 0
        self.headers = None
        self.step_metrics = []
        self.exp_name = None

    def __enter__(self):
        try:
            client = MlflowClient()
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            if self.experiment_config.experiment_name:
                experiment_name = self.experiment_config.experiment_name
            else:
                Logger().debug.info("No experiment name set, attempting acquiring from ENV")
                experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

            if experiment_name:
                mlflow.set_experiment(experiment_name)
            else:
                Logger().debug.info(
                    "Experiment not found, using default ... "
                    "(Set MLFLOW_EXPERIMENT_NAME in environment variable to change this behavior)"
                )
            mlflow.start_run(run_name=self.run_name)
            artifact_dir = data_dir / "mlflow_artifacts" / "MLproject"
            os.makedirs(artifact_dir, exist_ok=True)
            mlflow.log_artifact(str(artifact_dir))
            mlflow.log_params(self.experiment_config._asdict())

            self.exp_name = experiment_name

            return self
        except Exception as e:
            self.__exit__(*sys.exc_info())
            raise

    def log_step(
        self,
        variables: list,
        objectives: list,
        eval_node_id: int,
        diagonal_length: float,
        org_objectives: list,
        constrains: list = [],
    ):
        if constrains:
            raise NotImplementedError("Constrains logging is not available yet")
        if not self.headers:
            self.create_headers(variables=variables, objectives=objectives)
        if type(diagonal_length) is not int:
            diagonal_length = diagonal_length.tolist()
        self.step_metrics.append(variables + objectives + [eval_node_id, diagonal_length, self.step] + org_objectives)
        self.step += 1

    def create_headers(self, variables: list, objectives: list, constrains: list = []) -> None:
        if len(objectives) == 2:
            variable_header = [f"x{x + 1}" for x in range(len(variables) - 1)]
            variable_header.insert(0, "t")
        else:
            # Case for N objectives
            # Type for variables: [t_1, t_2, ... t_(n-1), x_1, x_2, ..., x_n]
            # Number of t: n_objectives - 1
            variable_header = [f"t{x + 1}" for x in range(len(objectives) - 1)]
            variable_header += [f"x{x + 1}" for x in range(len(variables) - len(objectives) + 1)]
        objective_header = [f"y{x + 1}" for x in range(len(objectives))]
        self.headers = variable_header + objective_header + ["eval_node_id", "diagonal_length", "step", "t_org", "y_org"]

    def send_data(self):
        step_metrics_df = pd.DataFrame(self.step_metrics, columns=self.headers)
        algorithm = self.experiment_config.algorithm
        tree_file = self.experiment_config.tree_file.split(".")[0]
        dimension = self.experiment_config.dimension
        termination_criterion = self.experiment_config.termination_criterion["criterion_name"]

        exp_name = f"{self.exp_name}" if self.exp_name else "default"
        exp_base_path = data_dir / exp_name

        dir_name = (
            f"{algorithm}_"
            f"{tree_file.split('/')[1]}_"
            f"{dimension}_"
            f"{termination_criterion}_"
            f"{datetime.datetime.now().isoformat().replace(':', '-')}"
        )
        dir_path = exp_base_path / dir_name
        meta_dir = dir_path / "meta"
        meta_dir.mkdir(exist_ok=True, parents=True)

        tree_src = self.experiment_config.tree_file
        shutil.copy(tree_src, meta_dir / "experiment_tree.json")

        exp_settings = self.experiment_config.to_dict()
        with open(meta_dir / "meta.json", "w", encoding="utf-8") as json_file:
            json.dump(exp_settings, json_file, indent=4)

        file_name = dir_name + ".csv"
        step_metrics_df.to_csv(dir_path / file_name)
        mlflow.log_artifact(local_path=str(dir_path / file_name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # executed when an error occurred
        if exc_type is not None:
            # Send error logs to Mlflow server
            traceback.print_exc(file=open("error.log", "w"))
            mlflow.log_artifact("error.log")
        self.send_data()
        sys.stdout.flush()
        sys.stderr.flush()
        mlflow.end_run()
