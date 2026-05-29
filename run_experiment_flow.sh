#!/bin/bash

set -e

TIMESTAMP=$(date --iso-8601=seconds)

NUM_TIMES=100 # Number of times to run an algorithm

# Define the number of times you want to run the command
# A single run generates one combination of all dimensions x algorithms
# A single run generates one combination of all dimensions x algorithms, with where a population is evolved for a number of generations.
MIN_PLOT_GENERATION=180
MAX_PLOT_GENERATION=200
PLOT_GENERATION_RANGE=$MIN_PLOT_GENERATION-$MAX_PLOT_GENERATION

# ################################################
POPULATION_SIZE=100 # CANNOT BE CHANGED FROM 100
# ################################################

ITERATIONS=$((MAX_PLOT_GENERATION * POPULATION_SIZE + 100))


IFS="-" read -r former latter <<< "$PLOT_GENERATION_RANGE"
threshold=$((latter * $POPULATION_SIZE))

if [ "$ITERATIONS" -lt "$threshold" ] ; then
    echo "Itrations: $ITERATIONS is lower than threshold ($threshold), exiting..."
    exit 0
fi


# ==========================================
# Parameter Confirmation Block
# ==========================================
echo ""
echo "----------------------------------------"
echo "        PARAMETER CONFIRMATION          "
echo "----------------------------------------"
echo "TIMESTAMP:             $TIMESTAMP"
echo "NUM_TIMES:             $NUM_TIMES"
echo "MIN_PLOT_GENERATION:   $MIN_PLOT_GENERATION"
echo "MAX_PLOT_GENERATION:   $MAX_PLOT_GENERATION"
echo "PLOT_GENERATION_RANGE: $PLOT_GENERATION_RANGE"
echo "POPULATION_SIZE:       $POPULATION_SIZE"
echo "ITERATIONS:            $ITERATIONS"
echo "Threshold limit:       $threshold"
echo "----------------------------------------"
echo ""

echo -n "Waiting 3 seconds to proceed. Press ANY KEY to abort... "
if read -t 3 -n 1 -s -r; then
    echo -e "\nScript aborted by user."
    exit 1
fi


# Ensure directories exist
mkdir -p "data/$TIMESTAMP"
mkdir -p "stats_output/$TIMESTAMP"
mkdir -p "figures/$TIMESTAMP"


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
python notebooks/rainplot_only_ploty_20260123.py --search_dir stats_output/$TIMESTAMP --gens $PLOT_GENERATION_RANGE --output_dir figures/$TIMESTAMP