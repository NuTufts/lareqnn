# lareqnn
Equivariant Neural Networks applied to LAr neutrino data.
Data is currently on Trex and Goeppert machines.

## Setup:

 - Put singularity container in lareqnn.
can be found on goeppert and trex at `/home/oalterkait/lareqnn/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif`

 - Change the data location in `configs/default_config.yaml` or choose a different config file by changing `config_log` in `train_lightning.py`

## Run training file regularly:


 - run singularity file:
`singularity shell --nv /home/oalterkait/lareqnn/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif`

 - run training: 
`python3 train_lightning.py`


## Run wandb sweep
 - change workdir in `sweep/run_sweep.sh`, `sweep/sweep_config.yaml file`.

 - If you do not have `wandb.ai/nutufts` access, change `nutufts` to your wandb username in `sweep/run_sweep.py`.

 - You can choose how many agents to run in parallel on each gpu, and the maximum number of runs in `sweep/run_sweep.sh

 - Run the sweep by doing
`cd sweep`
`./run_sweep.sh`
