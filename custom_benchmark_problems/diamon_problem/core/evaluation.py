import math as m
from collections import namedtuple

import numpy as np

NodeInfo = namedtuple("NodeInfo", ["symbol", "minima", "name"])
EvaluationResult = namedtuple("EvaluationResult", ["t", "y", "node_id", "diagonal_length", "unrotated_value"])
ParetoInfo = namedtuple(
    "ParetoInfo",
    ["symbol", "minima", "name", "minima_coordinates", "step_back_coordinates"],
)


class BMP:
    def __init__(self, sequence_info: list[dict], dim_space: int, rotate: bool = True):
        self.sequence_info = sequence_info
        self.s_lengths = self.s_lengths(sequence_info)
        self.dim_space = dim_space
        self.f_t_x_ = {}
        self.rotate = rotate
        # Define rotate degree here
        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        self.rotation_matrix = np.array(((c, -s), (s, c)))

    def t_upper_bound(self) -> float:
        return float(max(self.s_lengths)) + 2.0

    def evaluate(self, solution_variables: np.ndarray) -> EvaluationResult:
        """Main evaluation function, solution space is defined when the problem is constructed

        Parameters
        ----------
        solution_variables : np.ndarray
            Solution variables provided by the solver/user. Formatted in [t,x_i] as t and x are concatenated

        Returns
        -------

        """
        x = solution_variables[1:]
        t = solution_variables[0]
        y, node_id, diagonal_length = self.f_t_x(
            t=t,
            x=x,
            sequences=self.process_sequence(sequence_info=self.sequence_info),
        )
        t_org = t
        y_org = y
        if self.rotate:
            t, y = np.matmul(np.array([t, y]), self.rotation_matrix)
        return EvaluationResult(
            t=t,
            y=y,
            node_id=node_id,
            diagonal_length=diagonal_length,
            unrotated_value=[t_org, y_org],
        )

    def h_x(
        self,
        x: np.ndarray,
        x_s: np.ndarray,
        tau: int,
    ) -> np.ndarray:
        """Compute h(x) with solution variables and candidate coordinates and tau

        Parameters
        ----------
        x : np.ndarray
            Solution variables without t
        x_s: np.ndarray
            Coordinates of the local minima point
        tau : int
            tau value

        Returns
        -------
        np.ndarray
            A numpy array representing h(x) in each dimension
        """
        signs = self.compute_sign(
            x=x,
            x_s=x_s,
        )
        return signs / (4.0**tau)

    def compute_coordinates(self, symbol_sequence: list) -> np.ndarray:
        """Compute the coordinates for the given symbol sequence.

        Parameters
        ----------
        symbol_sequence : list
            List of ints representing the symbol sequence

        Returns
        -------
        np.ndarray
            The coordinates of the given symbol sequence, each element representing value in corresponding dimension
        """
        coordinates = np.zeros(self.dim_space, dtype="float64")
        # The origin in the x coordinates is an empty sequence.
        # The index in Python start from 0.
        # In the paper we start it from 1...
        for index, symbol in enumerate(symbol_sequence):
            index += 1  # The math index starts from 1
            if abs(symbol) > self.dim_space:
                raise ValueError(f"Dimension cannot be greater than axis. Got dimension: {self.dim_space}, axis: {symbol}")
            if symbol != 0:
                movement_length = np.sign(symbol) * 2.0 / (4.0**index)
                x = abs(symbol) - 1  # the 1st axis is x0 internally
                coordinates[x] += movement_length
        return coordinates

    @staticmethod
    def compute_sign(x: np.ndarray, x_s: np.ndarray) -> np.ndarray:
        """Compute the sign of x with regard to local minima point x_s, if x is the local minima point, then the sign
        will be set to positive (1 in practice)

        Parameters
        ----------
        x : np.ndarray
            Solution variables
        x_s : np.ndarray
            The coordinates of the local minima point

        Returns
        -------
        np.ndarray
            The sign in each dimension
        """
        diff = x - x_s
        return np.sign(diff) + (diff == 0)  # if x - x_s is 0, set the sign to 1 (by default it's 0).

    @staticmethod
    def get_tau(t: float) -> int:
        """Returns the corresponding tau for given t (tau<t<=tau+1, tau is int)

        Parameters
        ----------
        t : float
            t value

        Returns
        -------
        int
            tau value
        """
        if t < 0:
            raise Exception("Bad t")
        tau = m.floor(t)  # int type
        if float(tau) == t:
            tau -= 1
        return tau

    class ProcessedSequence:
        def __init__(self, symbol, minima, name):
            self.symbol = symbol
            self.minima = minima
            self.name = name

    @staticmethod
    def process_sequence(sequence_info: list) -> dict:
        """Pre-process the sequence and group symbol sequence with the same size togather

        Parameters
        ----------
        sequence_info : list
            Extract sequence from a standard JSON format

        Returns
        -------
        dict
            Dictionary with length as key and list of tuple(symbol, minima, name) as value
        """
        # TODO: Change naming here, don't want to do this right now
        s_dict = {}
        for item in sequence_info:
            minima = item["minima"]
            symbol = item["attrs"]["symbol"]
            name = item["name"]
            len_s = len(symbol)
            if len_s in s_dict:
                s_dict[len_s].append(NodeInfo(symbol, minima, name))
            else:
                s_dict[len_s] = [NodeInfo(symbol, minima, name)]
        return s_dict

    @staticmethod
    def s_lengths(sequence_info: list) -> list:
        """Get the available symbol length of the entire sequence tree

        Parameters
        ----------
        sequence_info : list
            List of dictionaries contains the tree information

        Returns
        -------
        list
            List of the available symbol lengths

        """
        # TODO: Naive implementation
        length_list = []
        for element in sequence_info:
            if len(element["attrs"]["symbol"]) not in length_list:
                length_list.append(len(element["attrs"]["symbol"]))
        return length_list

    @staticmethod
    def get_s_at_length(sequence_info: list, length: int) -> list:
        """Get all symbol sequence at given length

        Parameters
        ----------
        sequence_info : list
            List of dictionaries contains the tree information
        length : int
            The desired length

        Returns
        -------
        list
            List of sequence information at given length
        """
        s_list = []
        for element in sequence_info:
            if len(element["attrs"]["symbol"]) == length:
                s_list.append(element)
        return s_list

    def f_t_x(self, t: int, x: np.ndarray, sequences: dict) -> tuple[float, int, float]:
        if t == 0:
            return 0.0, 0, 0
        else:
            tau = self.get_tau(t)
            while tau >= 0:
                if tau not in sequences.keys():
                    return self.f_t_x(t=tau, x=x, sequences=sequences)
                else:
                    f_tau_x = self.f_t_x(t=tau, x=x, sequences=sequences)
                    sequences_tau = sequences[tau]
                    g_s_values = [f_tau_x[0]]
                    node_ids = [f_tau_x[1]]
                    diagonal_lengths = [f_tau_x[2]]
                    for s_tau in sequences_tau:
                        m_s = s_tau.minima
                        delta_t = t - tau
                        if not (0.0 <= delta_t <= 1.0):
                            raise Exception("sign error")

                        x_s = self.compute_coordinates(symbol_sequence=s_tau.symbol)
                        M_s = (1.0 - delta_t) * self.f_t_x(tau, x_s, sequences)[0] + delta_t * m_s
                        delta_x = x - x_s
                        h_x_ = self.h_x(x=x, x_s=x_s, tau=tau)
                        nabla_g = (self.f_t_x(tau, x_s + h_x_, sequences)[0] - m_s) / h_x_
                        diagonal_lengths.append(h_x_)
                        g_s_values.append(M_s + np.dot(nabla_g, delta_x.T))
                        node_ids.append(s_tau.name)
                    # self.f_t_x_[(t, tuple(x.tolist()))] = min(candidates)
                    g_s_values = np.array(g_s_values)
                    min_index = np.argmin(g_s_values)
                    minimal_value = g_s_values[min_index]
                    return (
                        minimal_value,
                        node_ids[min_index],
                        diagonal_lengths[min_index],
                    )


if __name__ == "__main__":
    bmp = BMP(
        sequence_info=[
            {
                "minima": -0.5,
                "attrs": {"symbol": [], "id": 0, "minima": -0.5},
                "name": 0,
            },
            {
                "minima": -0.4,
                "attrs": {"symbol": [1], "id": 1, "minima": -0.4},
                "name": 1,
            },
        ],
        dim_space=1,
        rotate=False,
    )
    x_ts = [
        [0, 0],
        [0, 0.5],
        [0, 1],
        [0, 1.5],
        [0, 2],
        [0.25, 0],
        [0.25, 0.5],
        [0.25, 1],
        [0.25, 1.5],
        [0.25, 2],
        [0.5, 0],
        [0.5, 0.5],
        [0.5, 1],
        [0.5, 1.5],
        [0.5, 2],
        [0.75, 0],
        [0.75, 0.5],
        [0.75, 1],
        [0.75, 1.5],
        [0.75, 2],
        [1.0, 0],
        [1.0, 0.5],
        [1.0, 1],
        [1.0, 1.5],
        [1.0, 2],
        [1.25, 0],
        [1.25, 0.5],
        [1.25, 1],
        [1.25, 1.5],
        [1.25, 2],
        [1.5, 0],
        [1.5, 0.5],
        [1.5, 1],
        [1.5, 1.5],
        [1.5, 2],
        [1.75, 0],
        [1.75, 0.5],
        [1.75, 1],
        [1.75, 1.5],
        [1.75, 2],
    ]
    for x_t in x_ts[-2:-1]:
        x_t.reverse()
        print(bmp.evaluate(solution_variables=np.array(x_t, dtype="float64")))
    #
    # bmp = BMP(
    #     sequence_info=[
    #         {"minima": 0.0, "attrs": {"symbol": [], "id": 0, "minima": 0.0}, "name": 0},
    #         {
    #             "minima": -1,
    #             "attrs": {"group": 0, "symbol": [1], "id": 1, "minima": -1},
    #             "name": 1,
    #         },
    #         {
    #             "minima": -20.78,
    #             "attrs": {"group": 0, "symbol": [1, -2, -1], "id": 2, "minima": -20.78},
    #             "name": 2,
    #         },
    #         {
    #             "minima": 33.75,
    #             "attrs": {"group": 0, "symbol": [1, -2, 1], "id": 3, "minima": 33.75},
    #             "name": 3,
    #         },
    #         {
    #             "minima": 31.75,
    #             "attrs": {"group": 0, "symbol": [1, -2], "id": 4, "minima": 31.75},
    #             "name": 4,
    #         },
    #     ],
    #     dim_space=2,
    # )
    # print(
    #     bmp.evaluate(
    #         solution_variables=np.array([1, 2, 3.9]),
    #     )
    # )
