# 3BC: Benchmarking and Visualization Based on Basin Connectivity

The 3BC allows you to manipulate multi-objective multimodal landscapes through the means of basins of attraction, establishing a rigorous benchmarking having explicit local and global Pareto set and fronts.

This repository provides the code for both the **3BC benchmark generator** and the **web-based visualization tool**.
Our visualization projects the high-dimensional basins in a 2D plot, in which the individuals of benchmarked MOEAs for user-specified generations are visible.
It helps researchers configure 3BC landscapes and, more importantly, understand the behavior of different MOEAs.

This repository contains the official implementation for the papers:

1.  [**Towards Benchmarking Multi-Objective Optimization Algorithms Based on the Basin Connectivity**](https://dl.acm.org/doi/10.1145/3712255.3734279) (GECCO '25 Companion Proceedings)
2.  [**Visualization of Multiobjective Multimodal Benchmarking Based on Basin Connectivity**](https://dl.acm.org/doi/10.1145/3638530.3654190) (GECCO '24 Poster)

---
## Citation

If you find our work useful, please consider citing our paper as follows:

```
@inproceedings{10.1145/3638530.3654190,
author = {Liu, Likun and Ota, Ryosuke and Yamamoto, Takahiro and Hamada, Naoki and Sakurai, Daisuke},
title = {Visualization of Multiobjective Multimodal Benchmarking Based on Basin Connectivity},
year = {2024},
isbn = {9798400704956},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3638530.3654190},
doi = {10.1145/3638530.3654190},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
pages = {347–350},
numpages = {4},
keywords = {multiobjective optimization, visualization, benchmarking},
location = {Melbourne, VIC, Australia},
series = {GECCO '24 Companion}
}
```
as well as the original 3BC paper:
```
@inproceedings{10.1145/3712255.3734279,
author = {Ota, Ryosuke and Liu, Likun and Hamada, Naoki and Yamamoto, Takahiro and Tanaka, Shoichiro and Sakurai, Daisuke},
title = {Towards Benchmarking Multi-Objective Optimization Algorithms Based on the Basin Connectivity},
year = {2025},
isbn = {9798400714641},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712255.3734279},
doi = {10.1145/3712255.3734279},
abstract = {When applying evolutionary computation for multimodal multiobjective optimization, it is crucial to balance exploration and exploitation: exploring the design space to identify regions that contain Pareto optima and exploiting these regions to converge towards the Pareto optima. When it comes to benchmarking for multi-objective optimization, however, it has been difficult to ensure such a landscape with desired properties. We thus propose to pre-determine the landscape and implement it as a benchmark problem. This framework is named the Benchmarking Based on Basin Connectivity (3BC) and targets continuous optimization. The 3BC suite includes an instance with recursively nested basins, forming a funnel structure, and another instance without nesting. Our implementation is publicly available at https://github.com/dsakurai/benchmark-visualizer.},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
pages = {2294–2300},
numpages = {7},
keywords = {multi-objective optimization, benchmarking, multimodality},
location = {NH Malaga Hotel, Malaga, Spain},
series = {GECCO '25 Companion}
}
```

----

## Getting Started

### Prerequisites

A straightforward way is to use VSCode DevContainers, for which we have pre-configured the prerequisites.

To run the project without it, we recommend the following environment:
* Python 3.8+
* Node.js and Yarn (for the visualization tool's frontend)

## Getting started

- ``pip install -r requirements.txt``
(Already done for the DevContainer)

1. (Optional) You may want to install jupyter notebook in order to use jupyter notebook

- ``pip install jupyter``

### Run with CLI

1. With the activated environment and sample tree file ("sample.json"). If you do not wish to track
   the experiment with mlflow, append ``--disable_tracking`` flag as parameter to the command below:

- ``python cli_main.py -f sample.json --dim 2``

### Run with jupyter notebook

We keep a sample experiment in ``sample_experiment.ipynb``.

## Visualizer
This visualizer was built with vue.js and FastAPI. You can either run the application with detached front-end and back end,
or to build the front-end and serve with FastAPI. If the development/experiment server is your current computer, no change
is needed. If it is a remote server, change the proxy in ``front-end/vue.config.js`` to your server's address.

### Frontend configuration


#### Build the front-end

1. Install node.js.

2. Go to the front-ends directory and install dependencies via:
- ``yarn install``

(Make sure core pack is enabled, for more see: https://)

3. Build the front-end:
- ``yarn build``
4.Your front end should appear in ``front-end/dist``

### Run stand alone front-end application

1. Install node.js
2. Run the application using:
``npm run``

### Backend configuration
The backend can run with one simple command: ``uvicorn main:app --reload``
(To make sure you are in the right directory, it's better to set ``PYTHPNPATH`` variable by: ``export PYTHONPATH=.``)


1. Backend requires specification on the experiment result file directory to run.
By default, it will search for data in the current directory. To change this, run the startup command with positional argument.
For instance, if the experiment data is in ``data`` directory, run the backend with:
``uvicorn main:app data --reload``

2. By default, the backend only runs on localhost, to expose it on the network, use ``--host`` flag (not recommended).
It is highly recommended to use a reverse proxy application for deployment such as Apache2 or Nginx.

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
