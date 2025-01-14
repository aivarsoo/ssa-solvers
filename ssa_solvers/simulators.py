from typing import Dict
from typing import Tuple

import torch
import xitorch.integrate as integrate

from ssa_solvers.chemical_reaction_system import BaseChemicalReactionSystem
from ssa_solvers.data_class import SimulationDataInCSV
from ssa_solvers.data_class import SimulationDataInMemory


class DeterministicSimulator:
    "Wrapper around xitorch.integrate.solve_ivp for solving ODEs"

    def __init__(self,
                 reaction_system: BaseChemicalReactionSystem,
                 cfg: Dict):
        """
        :reaction_system: reaction system class
        :cfg: configuration dictionary
        """
        self.reaction_system = reaction_system
        self.atol = cfg['ode_sim_cfg']['atol']
        self.rtol = cfg['ode_sim_cfg']['rtol']
        self.solver = cfg['ode_sim_cfg']['solver']

    def simulate(self, init_pops: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        sol = integrate.solve_ivp(self.reaction_system.ode_fun(),
                                  ts=time_grid,
                                  y0=init_pops.double(),
                                  method=self.solver,
                                  atol=self.atol,
                                  rtol=self.rtol
                                  )
        return sol.T.cpu()


class BaseSimulateOneStepMixin:
    "Base class for mixing in a one step of a Gillespie simulation method"

    def sample_time(self, propensity_values: torch.Tensor) -> torch.Tensor:
        """
        Sample next reaction time using exponential distirbution
        :param propensity_values: propensities
        :return: a sample form  Exp(1 / propensity_values)
        """
        q = torch.clamp(torch.rand(*propensity_values.shape,
                        device=self.device), self.EPS, 1)
        return -q.log() / propensity_values


class SimulateOneStepDirectMixin(BaseSimulateOneStepMixin):
    "Mixes in a one step simulation using the direct Gillespie simulation method"

    def simulate_one_step(self, pops: torch.Tensor, times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates one step of the direct Gillespie simulation method
        :param pops:  current population values (updated in the function)
        :param times: current time values (update in the function)
        :return: next population values and next times values
        """
        # Get propensities
        cur_propensities = self.reaction_system.propensities(pops)
        # Sample next reaction times
        propensities_sum = cur_propensities.sum(dim=0)
        next_times = self.sample_time(propensities_sum)
        # Sample next reactions
        # Normalizing propensities
        cur_propensities /= propensities_sum
        next_reaction_ids = self.sample_reaction(cur_propensities)
        # Update pops
        next_pops = torch.index_select(
            self.reaction_system.stoichiometry_matrix.T, 0, next_reaction_ids)
        return pops + next_pops, times + next_times


class SimulateOneStepFirstReactionMixin(BaseSimulateOneStepMixin):
    "Mixes in a one step simulation using the first reaction Gillespie simulation method"

    def simulate_one_step(self, pops: torch.Tensor, times: torch.Tensor):
        """
        Simulates one step of the first reaction Gillespie simulation method
        :param pops:  current population values (updated in the function)
        :param times: current time values (updated in the function)
        """
        # Get propensities
        cur_propensities = self.reaction_system.propensities(pops)
        # Sample next reactions and reaction times
        possible_times = self.sample_time(cur_propensities)
        # Update time and pops
        next_times, next_reaction_ids = torch.min(possible_times, dim=0)
        next_pops = torch.index_select(
            self.reaction_system.stoichiometry_matrix.T, 0, next_reaction_ids)
        return pops + next_pops, times + next_times

    def sample_reaction(self, propensity_values: torch.Tensor) -> torch.Tensor:
        """
        Sample next reaction index
        :param propensities: normalized vector of propensities
        :return: next reaction indexes
        """
        n_traj = propensity_values.shape[-1]
        q = torch.rand(n_traj, device=self.device)
        propensities_cumsums = torch.cumsum(propensity_values, dim=0)
        flags = (propensities_cumsums < q).type(torch.int64)
        return torch.argmin(flags, dim=0)


class StochasticSimulator(SimulateOneStepDirectMixin, SimulateOneStepFirstReactionMixin):
    "Stochastic simulator class realizing Gillespie Stochastic Simulation algorithm"

    def __init__(self,
                 reaction_system: BaseChemicalReactionSystem,
                 cfg: Dict,
                 device=torch.device("cpu")):
        """
        :reaction_system: reaction system class
        :cfg: configuration dictionary
        :device: torch.device("cpu") or torch.device("cuda:0")
        """
        self.cfg = cfg
        self.device = device
        self.checkpoint_freq = self.cfg['stochastic_sim_cfg']['checkpoint_freq']
        self.reaction_system = reaction_system
        self.data_class = SimulationDataInCSV if cfg['stochastic_sim_cfg'][
            'save_to_file'] else SimulationDataInMemory
        self.data_set = self.data_class(
            n_species=self.reaction_system.n_species,
            device=device,
            cfg=self.cfg)
        self.log_path = self.data_set.log_path
        self.EPS = self.data_set.EPS
        self.solver = self.cfg['stochastic_sim_cfg'].get('solver', 'direct')
        if self.solver == 'first_reaction':
            SimulateOneStepFirstReactionMixin.__init__(self)
        elif self.solver == 'direct':
            SimulateOneStepDirectMixin.__init__(self)
        else:
            raise NotImplementedError

    def set_reaction_params(self, params: Dict):
        """
        Set new reaction parameters and re-initialize the data set
        :param params: parameters for the reaction network
        """
        self.reaction_system.params = params
        self.data_set = self.data_class(
            n_species=self.reaction_system.n_species,
            device=self.device,
            cfg=self.cfg,
        )

    def simulate(self, init_pops: torch.Tensor, end_time: int, n_trajectories: int):
        """
        Stochastic simulation loop with splitting into batches
        :param init_pops: initial population
        :param end_time: final time of sumulations
        :param n_trajectories: number of trajectories for simulations
        """
        self.data_set.end_time = end_time
        # making sure the size is correct
        init_pops = init_pops.flatten().view(1, -1)
        assert init_pops.shape[1] == self.reaction_system.n_species
        # reseting the dataset
        self.data_set.reset()
        # max trajectories per batch
        trajectories_per_batch = min(
            self.data_set.trajectories_per_batch, n_trajectories)
        # splitting simulation into batches and saving the results on disk
        for batch_idx in range(n_trajectories // trajectories_per_batch):
            pops = torch.repeat_interleave(
                init_pops, trajectories_per_batch, dim=0)
            times = torch.zeros((trajectories_per_batch, ), device=self.device)
            self.simulate_trajectories(pops, times, batch_idx=batch_idx)
        if n_trajectories % trajectories_per_batch > 0:
            n_trajs = min(trajectories_per_batch,
                          n_trajectories % trajectories_per_batch)
            self.simulate_trajectories(
                torch.repeat_interleave(init_pops, n_trajs, dim=0),
                torch.zeros((n_trajs, ), dtype=torch.int64,
                            device=self.device),
                batch_idx=n_trajectories // trajectories_per_batch)

    def simulate_trajectories(self, pops: torch.Tensor, times: torch.Tensor, batch_idx=0):
        """"
        Stochastic simulation loop of one batch
        :param init_pops: initial population
        :param end_time: final time of sumulations
        :param batch_idx: batch number
        """
        iter_idx = 0
        self.data_set.initialize(pops, times, batch_idx=batch_idx)
        while times.min() < self.data_set.end_time:
            iter_idx += 1
            try:  # pressing ctrl+c will add data to the data class and break prematurely
                pops, times = self.simulate_one_step(pops=pops, times=times)
                # saving a checkpoint
                if self.checkpoint_freq and iter_idx % self.checkpoint_freq == 0:
                    self.data_set.add(pops, times, batch_idx=batch_idx)
            except KeyboardInterrupt:
                break
        if self.checkpoint_freq and iter_idx % self.checkpoint_freq != 0:
            self.data_set.add(pops, times, batch_idx=batch_idx)
