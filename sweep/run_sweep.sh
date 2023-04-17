#!/bin/bash

# File for running multiple wandb agents on multiple gpus. 
# Initiates multiple agents per GPU on a seperate screen for each one. Then runs the singularity container and the code
# TODO: Seperate out init_sweep and make this exclusively for running sweeps

# Set default value
default_config_loc="sweep_config.yaml"
number_agents="1"
project="lareqnn"
max_runs="20"
CONTAINER="/home/oalterkait/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif"
workdir="/home/oalterkait/lareqnn/sweep"

# Get the number of GPUs using nvidia-smi
number_gpus=$(nvidia-smi --list-gpus | wc -l)


# If no arguments are provided, use the default value for and config_loc. Make sure it's in the same location as train_lightning.py
if [ $# -eq 0 ]; then
    config_loc="$default_config_loc"
else
    config_loc="$1"
fi

# If sweep is already initialized and file is saved in sweep_id.txt, get id then run the init_sweep script with config_loc and wandb project name and save the id. Otherwise, take in predifined sweep_id

# Check if sweep_id.txt exists
if [ -e "sweep_id.txt" ]; then
    # Import the first line as variable sweep_id
    sweep_id=$(head -n 1 sweep_id.txt)
    echo "Using Existing Sweep ID: $sweep_id"
else
    # Run the init_sweep.py file and save the output to sweep_id.txt
    python3 ${workdir}/init_sweep.py "$config_lo\
c" "$project" | tail -n 1 > sweep_id.txt
    # Import the first line as variable sweep_id
    sweep_id=$(head -n 1 sweep_id.txt)
    echo "Using new Sweep ID: $sweep_id"
fi




# Run the second Python script number_agents times for each GPU with the last line as an argument
for gpu in `seq 0 $(( $number_gpus - 1))`; do

    for agent in `seq 0 $(( number_agents - 1))`; do
    
        echo "Running Agent $agent on GPU #$gpu"
        
        session_name="session_GPU${gpu}_AGENT${agent}"
        
        command_to_run_sweep="singularity exec --nv ${CONTAINER} bash -c \"CUDA_VISIBLE_DEVICES=${gpu} python3 ${workdir}/run_sweep.py ${sweep_id} ${max_runs} ${project} &> ${workdir}/GPU${gpu}_Agent${agent}_Log.txt\"\n"
        
        screen -dmS "$session_name" bash
        screen -S "$session_name" -X stuff "${command_to_run_sweep}"
        sleep 10
    done
done
