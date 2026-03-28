from typing import Optional

import mlflow
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from custom_benchmark_problems.diamon_problem.core import evaluation, n_objectives_problem
from utils.tracking import MlflowTracker


class Diamond(FloatProblem):
    def __init__(
        self,
        dim_space: int,
        sequence_info: list[dict],
        enable_tracking: bool = False,
        tracker: Optional[MlflowTracker] = None,
    ):
        super(Diamond, self).__init__()
        self.number_of_variables = dim_space + 1
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.problem = evaluation.BMP(sequence_info=sequence_info, dim_space=dim_space)

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = dim_space * [-1.0]
        # Probably would make gradient-based alg work, algs like OMOPSO would return 0 in gradient
        self.lower_bound.insert(0, 1e-2)
        self.upper_bound = dim_space * [1.0]
        self.upper_bound.insert(0, self.problem.t_upper_bound())
        self.enable_tracking = enable_tracking
        if enable_tracking:
            self.tracking_list = []
            self.tracker = tracker
            mlflow.log_dict(sequence_info, "sequence.json")

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        eval_results = self.problem.evaluate(solution_variables=np.array(solution.variables, dtype="float64"))
        solution.objectives[0] = eval_results[0]
        solution.objectives[1] = eval_results[1]
        if self.enable_tracking:
            self.tracker.log_step(
                variables=solution.variables,
                objectives=solution.objectives,
                eval_node_id=eval_results.node_id,
                diagonal_length=eval_results.diagonal_length,
                org_objectives=eval_results.unrotated_value,
            )
        return solution

    def get_name(self) -> str:
        return "diamond"


class NDiamond(FloatProblem):
    def __init__(
        self,
        dim_space: int,
        n_objectives: int,
        sequence_info: list[dict],
        enable_tracking: bool = False,
        rotate_t: bool = True,
        tracker: Optional[MlflowTracker] = None,
    ):
        """Initialize problem class

        Parameters
        ----------
        dim_space : Dimension of the design space X, should be greater than 1
        n_objectives : Number of objectives, should be greater than 2
        sequence_info : Information of input sequence
        enable_tracking : If tracking available
        tracker : tracker
        """
        super(NDiamond, self).__init__()

        self.problem_constructor_validator(dime_space=dim_space, n_objectives=n_objectives)

        # N of variables = dim_space of X + Number of t - 1, N of t = n_objectives - 1
        self.number_of_variables = dim_space + n_objectives - 1
        self.number_of_objectives = n_objectives
        self.number_of_constraints = 0

        # Problem instance for N-objectives BMP
        self.problem = n_objectives_problem.NBMP(
            sequence_info=sequence_info,
            dim_space=dim_space,
            n_objectives=n_objectives,
            t_rotate=rotate_t,
        )

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = [f"f_{x + 1}" for x in range(n_objectives)]

        # Set lower bound, variable format: [t_1, t_2, ... t_(n-1), x_1, x_2, ..., x_n]
        # Set lower bound of t first ([t_1, t_2, ... t_(n-1)])
        # Probably would make gradient-based alg work @ t_1, algs like OMOPSO would return 0 in gradient
        self.lower_bound = [1e-2] + [-1] * (n_objectives - 2)
        # Now add lower bound for x
        self.lower_bound += dim_space * [-1.0]

        # Set upper bound, variable format: [t_1, t_2, ... t_(n-1), x_1, x_2, ..., x_n]
        # Set upper bound of t first ([t_1, t_2, ... t_(n-1)])
        self.upper_bound = [self.problem.t_upper_bound()] + [1] * (n_objectives - 2)
        # Now add upper bound for x
        self.upper_bound += dim_space * [1.0]

        self.enable_tracking = enable_tracking
        if enable_tracking:
            self.tracking_list = []
            self.tracker = tracker
            mlflow.log_dict(sequence_info, "sequence.json")

    @staticmethod
    def problem_constructor_validator(dime_space: int, n_objectives: int):
        if type(dime_space) is not int:
            raise ValueError("dime_space should be an integer")
        if type(n_objectives) is not int:
            raise ValueError("n_objectives should be an integer")
        if dime_space < 1:
            raise ValueError("dime_space should at least be 1")
        if n_objectives < 2:
            raise ValueError("n_objectives should at least be 2")

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        eval_results = self.problem.n_evaluate(solution_variables=np.array(solution.variables, dtype="float64"))
        solution.objectives = eval_results.objective_values.tolist()
        if self.enable_tracking:
            self.tracker.log_step(
                variables=solution.variables,
                objectives=solution.objectives,
                eval_node_id=eval_results.node_id,
                diagonal_length=eval_results.diagonal_length,
                org_objectives=eval_results.unrotated_value,
            )
        return solution

    def get_name(self) -> str:
        return "n-diamond"
