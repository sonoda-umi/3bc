from collections import namedtuple

import numpy as np

from custom_benchmark_problems.diamon_problem.core.evaluation import (
    BMP,
)

NEvaluationResult = namedtuple(
    "NEvaluationResult",
    ["objective_values", "node_id", "diagonal_length", "unrotated_value"],
)


class NBMP(BMP):
    def __init__(
        self,
        sequence_info: list[dict],
        dim_space: int,
        n_objectives: int,
        rotate: bool = True,
        t_rotate: bool = True,
        degrees: float = -45,
        clockwise: bool = True,
    ):
        """

        Parameters
        ----------
        sequence_info
        dim_space : int, dimension of the design variables, referred to X in the paper
        n_objectives : int, number of objectives
        rotate : bool, if True the objective space is rotated
        t_rotate : bool, if True the time variable t_s are rotated
        degrees : float, the rotation angle commonly refer as theta
        clockwise : bool, if the positive rotation direction is clockwise
        """
        super().__init__(sequence_info=sequence_info, dim_space=dim_space, rotate=rotate)
        self.n_objectives = n_objectives
        self.rotate = rotate
        self.t_rotate = t_rotate
        self.degrees = degrees
        if t_rotate:
            self.t_rotation_matrix = self.create_t_rotation_matrix()
        if rotate:
            self.rot_matrix = self.create_rotation_matrix()

    @property
    def t_i_upper_bound(self) -> float:
        return 1.0

    @property
    def t_i_lower_bound(self) -> float:
        return -1.0

    def create_t_rotation_matrix(self):
        n_ts = self.n_objectives - 1
        if n_ts <= 1:
            raise ValueError(f"At least 3 objectives are required for t rotation operation")
        dim_list = []
        for i in range(n_ts):
            for j in range(i + 1, n_ts):
                dim_list.append((i, j))
        base_matrix = np.eye(n_ts)
        for dim_pair in dim_list:
            base_matrix = base_matrix @ self._rotation_matrix(n=n_ts, dim_pair=dim_pair, theta=self.degrees)
        return base_matrix

    def create_rotation_matrix(self):
        if self.n_objectives <= 1:
            raise ValueError("At least 2 objectives are required for rotation operation")
        return self._rotation_matrix(n=self.n_objectives, dim_pair=(0, 1), theta=self.degrees)

    @staticmethod
    def _rotation_matrix(n: int, dim_pair: tuple[int, int], theta: float) -> np.ndarray:
        """Creates an n-dimensional rotation matrix that rotates in the i-th and j-th dimensions by theta degrees.

        Parameters
        ----------
        n : int, total dimensions
        dim_pair : tuple[int, int], dimension pair to be rotated
        theta : float, degree of the angle of rotation

        Returns
        -------
        np.ndarray: the rotation matrix
        """

        # Rotation use radians
        theta_radians = np.radians(theta)

        # Initialize identity matrix
        rot_matrix = np.eye(n)

        # Set the elements for the 2D rotation in the i-th and j-th plane
        rot_matrix[dim_pair[0], dim_pair[0]] = np.cos(theta_radians)
        rot_matrix[dim_pair[1], dim_pair[1]] = np.cos(theta_radians)
        rot_matrix[dim_pair[0], dim_pair[1]] = np.sin(theta_radians)
        rot_matrix[dim_pair[1], dim_pair[0]] = -np.sin(theta_radians)
        return rot_matrix

    def n_evaluate(self, solution_variables: np.ndarray) -> NEvaluationResult:
        # Preprocessing solution variables
        bmp_solution_variables, t_s = self.parse_variables(solution_variables=solution_variables)
        bmp_eval_res = self.evaluate(solution_variables=bmp_solution_variables)
        f_hat = bmp_eval_res.t
        f_1 = f_hat + self.l1_dist(reg_variables=t_s[1:])
        objective_values = np.concatenate(([f_1], t_s))

        # Rotating objective values
        objective_values = self.parse_solutions(objective_values)

        return NEvaluationResult(
            objective_values=objective_values,
            node_id=bmp_eval_res.node_id,
            diagonal_length=bmp_eval_res.diagonal_length,
            unrotated_value=bmp_eval_res.unrotated_value,
        )

    @staticmethod
    def l1_dist(reg_variables: np.ndarray) -> float:
        return np.sum(np.abs(reg_variables))

    def parse_variables(self, solution_variables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Variable format: [t_1, t_2, ... t_(n-1), x_1, x_2, ..., x_n]
        # Takes t_1 and all Xs for original BMP evaluation
        bmp_variables = np.concatenate((solution_variables[:1], solution_variables[self.n_objectives - 1 :]))
        # Takes t_2 to t_(n-1) for the remaining computation
        t_s = solution_variables[: (self.n_objectives - 1)]
        if self.t_rotate:
            t_s = t_s @ self.t_rotation_matrix
        return bmp_variables, t_s

    def parse_solutions(self, objective_values: np.ndarray) -> np.ndarray:
        if self.rotate:
            objective_values = objective_values @ self.rot_matrix
        return objective_values


if __name__ == "__main__":
    n_bmp = NBMP(
        sequence_info=[],
        dim_space=1,
        n_objectives=3,
    )
    n_bmp.generate_rotation_matrix(2)
