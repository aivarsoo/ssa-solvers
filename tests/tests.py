import copy
import os
import shutil
import unittest
import warnings

import numpy as np
import torch
from test_mrna.mrna_srna_incis import cfg
from test_mrna.mrna_srna_incis import mRNAsRNAInCis

from ssa_solvers.simulators import DeterministicSimulator
from ssa_solvers.simulators import StochasticSimulator

torch.set_default_tensor_type(torch.DoubleTensor)


class TestSSASolver(unittest.TestCase):
    def test_simulation_ssa_direct_memory(self):
        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        reaction_system = mRNAsRNAInCis(device=device)
        ssa_simulator = StochasticSimulator(
            reaction_system=reaction_system,
            cfg=cfg,
            device=device
        )
        end_time = 250
        n_traj = 500
        init_pops = torch.zeros(
            (reaction_system.n_species, ), dtype=torch.int64, device=device)
        ssa_simulator.simulate(init_pops=init_pops,
                               end_time=end_time, n_trajectories=n_traj)
        time_grid = [0, 10, end_time]
        means, _ = ssa_simulator.data_set.mean_and_std(time_grid=time_grid)
        if torch.abs(means[0, -1] - torch.Tensor([2.38])) > 0.2 or torch.abs(means[1, -1] - torch.Tensor([101.9])) > 2:
            shutil.rmtree(ssa_simulator.log_path)
            raise ValueError(
                "This may indicate that the stochastic simulation is not working properly. Increase the number of trajectories")

    def test_simulation_ssa_direct_csv(self):

        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['stochastic_sim_cfg']['save_to_file'] = True

        reaction_system = mRNAsRNAInCis(device=device)
        ssa_simulator = StochasticSimulator(
            reaction_system=reaction_system,
            cfg=cfg_copy,
            device=device
        )

        end_time = 250
        n_traj = 5
        init_pops = torch.zeros(
            (reaction_system.n_species, ), dtype=torch.int64, device=device)
        ssa_simulator.simulate(init_pops=init_pops,
                               end_time=end_time, n_trajectories=n_traj)
        if not os.path.exists(ssa_simulator.data_set.raw_data_path) or len(os.listdir(ssa_simulator.data_set.raw_data_path)) < 1:
            shutil.rmtree(ssa_simulator.log_path)
            raise ValueError

    def test_simulation_ssa_fr(self):

        device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['stochastic_sim_cfg']['solver'] = 'first_reaction'

        reaction_system = mRNAsRNAInCis(device=device)
        ssa_simulator = StochasticSimulator(
            reaction_system=reaction_system,
            cfg=cfg_copy,
            device=device
        )

        end_time = 250
        n_traj = 500
        init_pops = torch.zeros(
            (reaction_system.n_species, ), dtype=torch.int64, device=device)
        ssa_simulator.simulate(init_pops=init_pops,
                               end_time=end_time, n_trajectories=n_traj)
        time_grid = [0, 10, end_time]
        time_grid = [0, 10, end_time]
        means, _ = ssa_simulator.data_set.mean_and_std(time_grid=time_grid)
        if torch.abs(means[0, -1] - torch.Tensor([2.38])) > 0.2 or torch.abs(means[1, -1] - torch.Tensor([101.9])) > 2:
            shutil.rmtree(ssa_simulator.log_path)
            raise ValueError(
                "This may indicate that the stochastic simulation is not working properly. Increase the number of trajectories")

    def test_simulation_ode(self):

        device = torch.device("cpu")
        cfg_copy = copy.deepcopy(cfg)

        reaction_system = mRNAsRNAInCis(device=device)
        deterministic_simulator = DeterministicSimulator(
            reaction_system=reaction_system,
            cfg=cfg_copy,
        )
        end_time = 250
        time_grid = [0, end_time]
        ode_res = deterministic_simulator.simulate(init_pops=np.zeros(
            (reaction_system.n_species,)), time_grid=time_grid)

        assert np.allclose(
            ode_res[..., -1], np.array([2.63892445, 112.4354636]))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)
        unittest.main(warnings='ignore')
