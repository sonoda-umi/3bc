"""
Script Name: generate_MOEAD_weight.py
Author: Likun Liu
Date: 2024.10.21
Description:
    This script generates a Sobol sequence since jMetalPy is lacking MOEA/D's weight file.
    This sequence is a type of quasi-random low-discrepancy sequence commonly used in numerical simulations, integration, and optimization tasks.
    The generated sequence is then saved to a file in a user-specified format.

    The script offers two methods for generating the Sobol sequence:
    - **scipy**: Uses the `scipy.stats.qmc` module to generate the Sobol sequence.
    - **quasimc**: Uses the `quasimc` package's `Sobol` class to generate the sequence.

    The sequence starts with an identity matrix of size `n_dims` and appends the remaining Sobol sequence rows up to `n_rows`. The result is saved in a text file.

    Key functionalities:
    - Generate a Sobol sequence of a given dimension (`n_dims`) and row count (`n_rows`).
    - Choose between two methods for generating the sequence (`scipy` or `quasimc`).
    - Optionally set the `bits` parameter for the Sobol generator when using the `scipy` method.
    - Save the output to a specified file in space-delimited format.

    Usage:
    - To run the script, use the following command:
      python sobol_generator.py [n_dims] [n_rows] [file_path] [method] --bits [optional_bits]

    Example:
    - python sobol_generator.py 5 100 output.txt scipy --bits 32

    Notes:
    - The `n_dims` parameter specifies the dimensionality of the Sobol sequence.
    - The `n_rows` parameter defines the total number of rows to generate.
    - The `file_path` is where the generated sequence will be saved in a space-separated format.
    - The `method` can be "scipy" or "quasimc".
    - If using the "scipy" method, you can optionally provide a `bits` parameter to control the bit precision of the Sobol sequence.

"""

import argparse
import math

import numpy as np
from quasimc.sobol import Sobol
from scipy.stats import qmc


def generate(n_dims: int, n_rows: int, file_path: str, bits: int = None, method: str = "scipy"):
    if method == "scipy":
        n_power = math.ceil(math.log2(100))
        sampler = qmc.Sobol(d=n_dims, scramble=True, bits=bits)
        generated_arr = sampler.random_base2(n_power)[: n_rows - n_dims,]
    elif method == "quasimc":
        import time

        seed = int(time.time() * 10000000 % 10000)
        sobol = Sobol(dim=n_dims, seed=seed)
        generated_arr = sobol.generate(n_rows - n_dims).T
    else:
        raise NotImplementedError(f"Invalid method {method}")
    base_arr = np.eye(n_dims)
    generated_arr = np.vstack((base_arr, generated_arr))
    np.savetxt(file_path, generated_arr, fmt="%.16f", delimiter=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_dims", type=int)
    parser.add_argument("n_rows", type=int)
    parser.add_argument("file_path", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("--bits", type=int, required=False, default=None)
    args = parser.parse_args()
    generate(
        n_rows=args.n_rows,
        file_path=args.file_path,
        n_dims=args.n_dims,
        method=args.method,
    )
