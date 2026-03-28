#!/bin/bash

# # Define the number of times you want to run the command
NUM_TIMES=3


# # Generate the experiment configuration file
python utils/generate_experiment_config_n_obj.py --output_path experiment_config-N-obj.yaml


# # Loop through and run the command
for ((i=1; i<=NUM_TIMES; i++))
do
    echo "Running iteration $i"
    python yaml_main_parallel.py -f experiment_config-N-obj.yaml --n_objective
done

# # Generate the stats file
python utils/generate_stats_file.py --search_dir data


# Generate figures
python notebooks/rainplot_only_ploty_20260123.py --search_dir stats_output --gen 10