import yaml
import wandb
import sys

def init_sweep(config_loc, project):
    with open(config_loc, 'r') as file:
        config = yaml.safe_load(file)


    sweep_id = wandb.sweep(sweep=config, project=project)
    print(sweep_id)
    return sweep_id

if __name__=='__main__':
    args = sys.argv[1:]
    init_sweep(args[0],args[1])
    
    
    

