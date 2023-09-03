import fire
import torch

from circuits.auto_repressor.run_auto import run_auto
from circuits.mrna_srna.run_mrna import run_mrna

run_method = dict(
    auto=run_auto,
    mrna=run_mrna
)

choose_device = {
    "cpu": torch.device('cpu'),
    "cuda": torch.device('cuda:0')
}


def run_simulations(circuit: str = "mrna", end_time: float = 300, n_steps: int = 100, n_traj: int = 500, device="cuda"):
    device = choose_device[device]
    run_method[circuit](end_time=end_time, n_steps=n_steps,
                        n_traj=n_traj, device=device)


if __name__ == '__main__':
    fire.Fire(run_simulations)
