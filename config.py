from pathlib import Path

base_path = Path(__file__).parent.absolute()
sample_file_path = base_path / "sample.json"

sql_engine_url = f"sqlite:///{base_path / 'db.sqlite3'}"
solver_info = {
    "multi_objective": {
        "evolutionary_algorithms": [
            {
                "name": "GDE3",
                "parameters": {
                    "population_size": 100,
                    "cr": 0.5,
                    "f": 0.5,
                    "max_evaluations": 25000,
                    "k": 0.5,
                },
            },
            {
                "name": "Dynamic GDE3",
                "parameters": {
                    "population_size": 100,
                    "cr": 0.5,
                    "f": 0.5,
                    "max_evaluations": 25000,
                    "k": 0.5,
                },
            },
        ],
        "PSO_algorithms": [
            {
                "name": "OMOPSO",
                "parameters": {
                    "swarm_size": 100,
                    "epsilon": 0.0075,
                    "mutation_probability": 0.8,
                    "max_evaluations": 25000,
                },
            },
            {
                "name": "SMPSO",
                "parameters": {
                    "swarm_size": 100,
                    "epsilon": 0.0075,
                    "mutation_probability": 0.8,
                    "max_evaluations": 25000,
                },
            },
        ],
    },
    "single_objective": [
        {
            "name": "GeneticAlgorithm",
            "parameters": {
                "population_size": 30,
                "offspring_population_size": 30,
                "mutation_probability": 0.8,
            },
        }
    ],
}

project_base = Path(__file__).parent.absolute()
data_dir = project_base / "data"
mlflow_tracking_uri = project_base / "data" / "mlflow"
