from circuits.auto_repressor.tetr_srna_incis import TetRsRNAInCis, cfg
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ssa_solvers.simulators import DeterministicSimulator
from ssa_solvers.simulators import StochasticSimulator

torch.set_default_tensor_type(torch.FloatTensor)


def run_auto(end_time: float = 300, n_steps: int = 150, n_traj: int = 100, device=torch.device('cpu')):
    time_grid = np.arange(0, end_time, end_time / n_steps)
    cfg['stochastic_sim_cfg'].update(
        dict(save_to_file=True, trajectories_per_file=50000))

    ode_simulator = DeterministicSimulator(
        reaction_system=TetRsRNAInCis(device=device),
        cfg=cfg
    )
    reaction_system_incis = TetRsRNAInCis(device=device)
    ssa_simulator_incis = StochasticSimulator(
        reaction_system=reaction_system_incis,
        cfg=cfg,
        device=device
    )
    reaction_system_incis.params = {'aTc': 100}  # setting aTc level
    init_pops = torch.zeros(
        (reaction_system_incis.n_species, ), dtype=torch.int64, device=device)
    print("starting simulation")
    ode_res_incis = ode_simulator.simulate(init_pops=np.zeros(
        (reaction_system_incis.n_species,)), time_grid=time_grid)
    ssa_simulator_incis.simulate(
        init_pops=init_pops, end_time=end_time, n_trajectories=n_traj)

    print("computing mean and std")
    means_incis, stds_incis = ssa_simulator_incis.data_set.mean_and_std(
        time_grid=time_grid)

    species_idx_incis = 1
    plt.figure()
    plt.plot(
        time_grid, means_incis[species_idx_incis, :], 'b', label='Mean SSA (in cis)')
    plt.fill_between(time_grid, means_incis[species_idx_incis, :]+stds_incis[species_idx_incis, :], means_incis[species_idx_incis, :]-stds_incis[species_idx_incis, :],
                     color='b', alpha=0.3)
    plt.xlim([0, end_time])
    plt.plot(
        time_grid, ode_res_incis[species_idx_incis, :], 'b--', label='ODE (in cis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of species')
    plt.legend()
    plt.savefig(Path(ssa_simulator_incis.log_path) / 'plot.png')
    print("done")
