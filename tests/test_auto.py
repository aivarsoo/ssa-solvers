from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from ssa_solvers.simulators import DeterministicSimulator
from ssa_solvers.simulators import StochasticSimulator

torch.set_default_tensor_type(torch.FloatTensor)

# device = torch.device('cuda:0') if torch.cuda.is_available else torch.device("cpu")


if __name__ == "__main__":
    device = torch.device("cpu")
    end_time = 300
    n_steps = 150
    n_traj = 5000
    time_grid = np.arange(0, end_time, end_time / n_steps)

    from circuits.auto_repressor.tetr_srna_incis import TetRsRNAInCis, cfg

    cfg['stochastic_sim_cfg']['save_to_file'] = True
    cfg['stochastic_sim_cfg']['trajectories_per_file'] = 50000

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
    # torch.randint(1, (reaction_system.n_species, ))
    init_pops = torch.zeros(
        (reaction_system_incis.n_species, ), dtype=torch.int64, device=device)
    print("starting simulation")
    ode_res_incis = ode_simulator.simulate(init_pops=np.zeros(
        (reaction_system_incis.n_species,)), time_grid=time_grid)
    ssa_simulator_incis.simulate(
        init_pops=init_pops, end_time=end_time, n_trajectories=n_traj)

    print("computing mean and std")
    # time_grid = np.array([time_grid[-1]])
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
    plt.savefig('my_plot.png')
