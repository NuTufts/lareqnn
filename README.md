# lareqnn
Equivariant Neural Networks applied to LAr neutrino data.
Data is currently on Trex and Goeppert machines.

## Setup:

 - Download Singularity. The container
can be found on goeppert and trex at `/home/oalterkait/mink-pytorch1.12.sif`. Can also be downloaded from the link at the bottom.

 - Change the data location in `configs/default_config.yaml` or choose a different config file by changing `config_log` in `train_lightning.py`

## Run training file regularly:


 - run singularity file:
`singularity shell --nv /home/oalterkait/mink-pytorch1.12.sif`

 - run training: 
`python3 train_lightning.py`

- jupyter notebook: `singularity exec --nv mink-pytorch1.12.sif jupyter lab --no-browser --ip=0.0.0.0 --port=8891` and then tunnel with `ssh -L 8891:localhost:8891 user@server`, where you can change the port and the server respectively.

## Run wandb sweep:
 - change workdir in `sweep/run_sweep.sh`, `sweep/sweep_config.yaml file`.

 - If you do not have `wandb.ai/nutufts` access, change `nutufts` to your wandb username in `sweep/run_sweep.py`.

 - You can choose how many agents to run in parallel on each gpu, and the maximum number of runs in `sweep/run_sweep.sh

 - Run the sweep by doing
`cd sweep`
`./run_sweep.sh`

## Dataset and Container:
The dataset and container can be found at 
https://tuftscloud-my.sharepoint.com/:f:/g/personal/oalter01_tufts_edu/Eh_ubyMCcDBEpU8NM270kiQBQm_Qsm2ds66EznRhtzHkwA?e=pifL7D