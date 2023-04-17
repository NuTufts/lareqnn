import yaml
import wandb
import sys

def run_sweep(sweep_id, count):
    wandb.agent(sweep_id, count=count)
    
if __name__=='__main__':
    args = sys.argv[1:]
    sweep_id = args[0]
    count = int(args[1])
    project = args[2]
    sweep_loc = f"nutufts/{project}/{sweep_id}"
    run_sweep(sweep_loc,count)
    