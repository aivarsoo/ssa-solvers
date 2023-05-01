from __future__ import annotations

import numpy as np
import torch

from ssa_solvers.data_class import SimulationData
from ssa_solvers.simulators import DeterministicSimulator
from ssa_solvers.simulators import StochasticSimulator

torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device(
    'cuda:0') if torch.cuda.is_available else torch.device("cpu")


if __name__ == "__main__":
    end_time = 300
    n_steps = 300
    n_traj = 10
    time_grid_np = np.arange(0, end_time, int(end_time / n_steps))
    time_grid_torch = torch.from_numpy(time_grid_np).to(device)
    n_species = 2

    from circuits.mrna_srna.mrna_srna_incis import mRNAsRNAInCis, cfg

    reaction_system_incis = mRNAsRNAInCis(device=torch.device("cpu"))
    ode_simulator = DeterministicSimulator(
        reaction_system=reaction_system_incis,
        cfg=cfg
    )
    reaction_system_incis = mRNAsRNAInCis(device=device)
    ssa_simulator = StochasticSimulator(
        reaction_system=reaction_system_incis,
        cfg=cfg,
        device=device
    )
    data_set_incis = SimulationData(
        n_species=reaction_system_incis.n_species, device=device)
    # increasing fmRNA production rate
    reaction_system_incis.params = {'beta_fmrna': 2}
    # torch.randint(1, (reaction_system.n_species, ), device=device)
    init_pops = torch.zeros(
        (reaction_system_incis.n_species, ), dtype=torch.int64, device=device)
    ssa_simulator.simulate(init_pops=init_pops,
                           end_time=end_time, n_trajectories=n_traj)
    ode_res_incis = ode_simulator.simulate(init_pops=np.zeros(
        (reaction_system_incis.n_species,)), time_grid=time_grid_np)

    from circuits.mrna_srna.mrna_srna_intrans import mRNAsRNAInTrans, cfg

    reaction_system_intrans = mRNAsRNAInTrans(device=torch.device("cpu"))
    ode_simulator = DeterministicSimulator(
        reaction_system=reaction_system_intrans,
        cfg=cfg
    )
    reaction_system_intrans = mRNAsRNAInTrans(device=device)
    ssa_simulator = StochasticSimulator(
        reaction_system=reaction_system_intrans,
        cfg=cfg,
        device=device
    )
    data_set_intrans = SimulationData(
        n_species=reaction_system_intrans.n_species, device=device)
    init_pops = torch.zeros(
        (reaction_system_intrans.n_species, ), dtype=torch.int64, device=device)
    ssa_simulator.simulate(init_pops=init_pops,
                           end_time=end_time, n_trajectories=n_traj)
    ode_res_intrans = ode_simulator.simulate(init_pops=np.zeros(
        (reaction_system_intrans.n_species,)), time_grid=time_grid_np)

    means_incis, stds_incis = data_set_incis.mean_and_std(
        time_grid=time_grid_torch)

    means_intrans, stds_intrans = data_set_intrans.mean_and_std(
        time_grid=time_grid_torch)
