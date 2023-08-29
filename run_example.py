import fire 
import torch
from circuits.auto_repressor.run_auto import run_auto
from circuits.mrna_srna.run_mrna import run_mrna

run_method = dict(
    autorepressor=run_auto,
    mrna=run_mrna
)

def run_simulations(circuit:str="autorepressor", end_time:float = 300, n_steps:int = 150, n_traj:int = 100, device=torch.device('cpu')):
    run_method[circuit](end_time=end_time, n_steps=n_steps, n_traj=n_traj, device=device)

if __name__ == '__main__':
    fire.Fire(run_simulations)