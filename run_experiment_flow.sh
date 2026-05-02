#!/bin/bash

# Define the number of times you want to run the command
# A single run generates one combination of all dimensions x algorithms
NUM_TIMES=3
PLOT_GENERATION_RANGE=90-100
ITERATIONS=10000
POPULATION_SIZE=100


IFS="-" read -r former latter <<< "$PLOT_GENERATION_RANGE"
threshold=$((latter * $POPULATION_SIZE))

if [ "$ITERATIONS" -lt "$threshold" ] ; then
    echo "Itrations: $ITERATIONS is lower than threshold ($threshold), exiting..."
    exit 0
fi

TIMESTAMP=$(date --iso-8601=seconds)



# Generate the experiment configuration file
python utils/generate_experiment_config_n_obj.py --output_path experiment_config-N-obj.yaml --iterations $ITERATIONS


# Loop through and run the command
for ((i=1; i<=NUM_TIMES; i++))
do
    echo "Running iteration $i"
    python yaml_main_parallel.py -f experiment_config-N-obj.yaml --n_objective --additional_path $TIMESTAMP
done

# Generate the stats file
python utils/generate_stats_file.py --search_dir data/$TIMESTAMP --gens $PLOT_GENERATION_RANGE --output_dir stats_output/$TIMESTAMP


# Generate figures
python notebooks/rainplot_only_ploty_20260123.py --search_dir stats_output/$TIMESTAMP --gens $PLOT_GENERATION_RANGE