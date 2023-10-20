# lareqnn
Equivariant Neural Networks applied to LAr neutrino data.
Data is currently on Trex and Goeppert machines.

## Setup:

 - Download the needed libraries using `pip install -r requirements.txt`. make sure to have a gpu enabled for torchsparse to properly install.

 - Change the data location in `configs/default_config.yaml` or choose a different config file by changing `config_log` in `train_lightning.py`

## Run training file regularly:

 - run training: 
`python3 train_lightning.py`