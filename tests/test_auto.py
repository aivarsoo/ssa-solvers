import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as st
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device("cpu")

import os, sys 
from ssa_solvers.simulators import StochasticSimulator, DeterministicSimulator 
from ssa_solvers.data_class import SimulationData

if __name__ == "__main__":
    end_time = 400
    n_steps = 300
    n_traj = 10
    time_grid_np = np.arange(0, end_time, int(end_time / n_steps))
    time_grid_torch = torch.from_numpy(time_grid_np).to(device)

    from circuits.auto_repressor.tetr_srna_incis import TetRsRNAInCis, cfg

    data_set_incis = SimulationData(device=device)
    reaction_system_incis = TetRsRNAInCis(device=torch.device("cpu"))
    ode_simulator = DeterministicSimulator(
        reaction_system=reaction_system_incis,
        cfg=cfg
    )
    reaction_system_incis = TetRsRNAInCis(device=device)
    ssa_simulator_incis = StochasticSimulator(
        reaction_system=reaction_system_incis,
        data_set=data_set_incis,
        cfg=cfg,
        device=device
    )
    reaction_system_incis.params = {'aTc': 100}  # setting aTc level
    init_pops = torch.zeros((reaction_system_incis.n_species, ), dtype=torch.int64, device=device) #torch.randint(1, (reaction_system.n_species, ))
    ssa_simulator_incis.simulate(init_pops=init_pops, end_time=end_time, n_trajectories=n_traj)
    ode_res_incis = ode_simulator.simulate(init_pops=np.zeros((reaction_system_incis.n_species,)), time_grid=time_grid_np)

    from circuits.auto_repressor.tetr_srna_intrans import TetRsRNAInTrans, cfg

    data_set_intrans = SimulationData(device=device)
    reaction_system_intrans = TetRsRNAInTrans(device=torch.device("cpu"))
    ode_simulator = DeterministicSimulator(
        reaction_system=reaction_system_intrans,
        cfg=cfg
    )
    reaction_system_intrans = TetRsRNAInTrans(device=device)
    ssa_simulator_intrans = StochasticSimulator(
        reaction_system=reaction_system_intrans,
        data_set=data_set_intrans,
        cfg=cfg,
        device=device
    )
    reaction_system_intrans.params = {'aTc': 100}  # setting aTc level
    init_pops = torch.zeros((reaction_system_intrans.n_species, ), dtype=torch.int64, device=device) #torch.randint(1, (reaction_system.n_species, ))
    ssa_simulator_intrans.simulate(init_pops=init_pops, end_time=end_time, n_trajectories=n_traj)
    ode_res_intrans = ode_simulator.simulate(init_pops=np.zeros((reaction_system_intrans.n_species,)), time_grid=time_grid_np)

    means_incis = data_set_incis.mean(time_range=time_grid_torch).cpu().numpy()
    stds_incis = data_set_incis.std(time_range=time_grid_torch).cpu().numpy()
    cov_incis = data_set_incis.coefficient_of_variation(time_range=time_grid_torch).cpu().numpy()

    means_intrans = data_set_intrans.mean(time_range=time_grid_torch).cpu().numpy()
    stds_intrans = data_set_intrans.std(time_range=time_grid_torch).cpu().numpy()
    cov_intrans = data_set_intrans.coefficient_of_variation(time_range=time_grid_torch).cpu().numpy()