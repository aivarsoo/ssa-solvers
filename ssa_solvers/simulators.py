import numpy as np
import torch 
from typing import Dict
import scipy.integrate as integrate
from .data_class import SimulationData

class Simulator:
    def simulate(self):
        raise NotImplementedError

class DeterministicSimulator(Simulator):
    def __init__(self, 
                reaction_system,
                cfg:Dict) -> None:
        self.reaction_system = reaction_system
        self.atol = cfg['ode_sim_cfg']['atol']
        self.rtol = cfg['ode_sim_cfg']['rtol']
        self.solver = cfg['ode_sim_cfg']['solver']

    def simulate(self, init_pops: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        sol = integrate.solve_ivp(self.reaction_system.ode_fun, 
                                  t_span=[time_grid[0], time_grid[-1]], y0=init_pops, 
                                  method=self.solver, t_eval=time_grid, 
                                  atol=self.atol, rtol=self.rtol
                )
        return sol.y

class StochasticSimulator(Simulator):
    def __init__(self, 
                reaction_system,
                cfg:Dict,
                device=torch.device("cpu")) -> None:
        self.cfg = cfg
        self.device=device
        self.checkpoint_freq = self.cfg['stochastic_sim_cfg']['checkpoint_freq']
        self.solver = self.cfg['stochastic_sim_cfg']['solver']
        self.reaction_system = reaction_system
        self.data_set = SimulationData(
            device=device, 
            save_to_file=self.cfg['stochastic_sim_cfg']['save_to_file'],
            trajectories_per_file=self.cfg['stochastic_sim_cfg']['trajectories_per_file'],
            path=self.cfg['stochastic_sim_cfg']['path']
            )

    def set_reaction_params(self, params:Dict):
        """
        Set new reaction parameters and re-initialize the data set
        :param params: parameters for the reaction network
        """
        self.reaction_system.params = params 
        self.data_set = SimulationData(
            device=self.device, 
            save_to_file=self.cfg['stochastic_sim_cfg']['save_to_file'],
            trajectories_per_file=self.cfg['stochastic_sim_cfg']['trajectories_per_file'],
            path=self.cfg['stochastic_sim_cfg']['path']
            )


    def simulate(self, init_pops: torch.Tensor, end_time:int, n_trajectories: int) -> None:
        """
        Stochastic simulation loop with splitting into batches
        :param init_pops: initial population 
        :param end_time: final time of sumulations
        :param n_trajectories: number of trajectories for simulations         
        """
        self.data_set.end_time = end_time
        init_pops = init_pops.flatten().view(1,-1) # making sure the size is correct
        assert init_pops.shape[1] == self.reaction_system.n_species
        if self.data_set.save_to_file:
            # splitting simulation into batches and saving the results on disk
            pops = torch.repeat_interleave(init_pops, self.data_set.trajectories_per_file, dim=0) 
            times = torch.zeros((self.data_set.trajectories_per_file, ), device=self.device)
            for batch_idx in range(n_trajectories // self.data_set.trajectories_per_file):
                self.simulate_trajectories(pops, times, batch_idx=batch_idx)
            if n_trajectories % self.data_set.trajectories_per_file > 0:    
                n_trajs = min(self.data_set.trajectories_per_file, n_trajectories % self.data_set.trajectories_per_file)
                self.simulate_trajectories(
                    torch.repeat_interleave(init_pops, n_trajs, dim=0), 
                    torch.zeros((n_trajs, ), device=self.device),
                    batch_idx=n_trajectories // self.data_set.trajectories_per_file)
        else:    
            # keeping everything in memmory (one batch) 
            pops = torch.repeat_interleave(init_pops, n_trajectories, dim=0) 
            times = torch.zeros((n_trajectories, ), device=self.device)
            self.simulate_trajectories(pops, times)

    def simulate_trajectories(self, pops:torch.Tensor, times:torch.Tensor, batch_idx=0):
        """"
        Stochastic simulation loop of one batch
        :param init_pops: initial population 
        :param end_time: final time of sumulations
        :param batch_idx: batch number            
        """
        self.data_set.add(pops, times, first_add=True, batch_idx=batch_idx)
        iter_idx = 0
        while times.min() < self.data_set.end_time:
            iter_idx += 1
            try: # pressing ctrl+c will add data to the data class and break prematurely 
                pops, times = self.simulate_one_step(pops=pops, times=times) 
                # saving a checkpoint 
                if self.checkpoint_freq and iter_idx % self.checkpoint_freq == 0:
                    self.data_set.add(pops, times, batch_idx=batch_idx)
            except KeyboardInterrupt:
                break
        if self.checkpoint_freq and iter_idx % self.checkpoint_freq != 0:
            self.data_set.add(pops, times, batch_idx=batch_idx)


    def simulate_one_step(self, pops: torch.Tensor, times: torch.Tensor) -> None:
        """
        Chooses the method for simulation  
        :param pops: current population 
        :param times: current simulation times for the population
        """
        if self.solver == 'direct':
            pops, times = self.simulate_one_step_direct(pops, times)
        elif self.solver == 'first_reaction':
            pops, times = self.simulate_one_step_first_reaction(pops, times)             
        else:
            raise NotImplementedError
        return pops, times

    def simulate_one_step_first_reaction(self, pops: torch.Tensor, times: torch.Tensor) -> None:
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
        next_pops = torch.index_select(self.reaction_system.stoichiometry_matrix.T, 0, next_reaction_ids)
        return pops + next_pops, times + next_times

    def simulate_one_step_direct(self, pops: torch.Tensor, times: torch.Tensor) -> None:
        """
        Simulates one step of the direct Gillespie simulation method
        :param pops:  current population values (updated in the function)
        :param times: current time values (update in the function)
        """
        # Get propensities
        cur_propensities = self.reaction_system.propensities(pops)
        # Sample next reaction times
        propensities_sum = cur_propensities.sum(dim=0)
        next_times = self.sample_time(propensities_sum)
        # Sample next reactions
        cur_propensities /= propensities_sum  # normalizing propensities      
        next_reaction_ids = self.sample_reaction(cur_propensities)
        # Update pops
        next_pops = torch.index_select(self.reaction_system.stoichiometry_matrix.T, 0, next_reaction_ids)
        return pops + next_pops, times + next_times

    def sample_reaction(self, propensity_values:torch.Tensor) -> torch.Tensor:
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

    def sample_time(self, propensity_values: torch.Tensor) -> torch.Tensor:
        """
        Sample next reaction time using exponential distirbution 
        :param propensity_values: propensities 
        :return: a sample form  Exp(1 / propensity_values)
        """
        q = torch.rand(*propensity_values.shape, device=self.device)
        return -q.log() / propensity_values 


if __name__ == "__main__":
    pass