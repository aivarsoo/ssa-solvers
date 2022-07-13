import numpy as np
import torch 
import copy 
from typing import Any, Dict
import torch.nn.functional as F
import scipy.integrate as integrate

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
                data_set,                 
                cfg:Dict,
                device=torch.device("cpu")) -> None:
        self.reaction_system = reaction_system
        self.data_set = data_set
        self.device=device
        self.checkpoint_freq = cfg['stochastic_sim_cfg']['checkpoint_freq']
        self.solver = cfg['stochastic_sim_cfg']['solver']

    def simulate(self, init_pops: torch.Tensor, end_time: int, n_trajectories: int) -> None:
        """
        Stochastic simulation loop 
        :param init_pops: initial population 
        :param end_time: final time of the sumulations
        :param n_trajectories: numner of trajectories for simulations         
        """
        pops = init_pops.flatten().view(1,-1)
        assert pops.shape[1] == self.reaction_system.n_species
        pops = torch.repeat_interleave(pops, n_trajectories, dim=0) 
        times = torch.zeros((n_trajectories, ), device=self.device)
        pops_evolution = [pops] #[copy.deepcopy(pops)]
        times_evolution = [times] #[copy.deepcopy(times)]
        iter_idx = 0
        while times.min() < end_time:
            iter_idx += 1
            try: # pressing ctrl+c will add data to the data class and break prematurely 
                pops, times = self.simulate_one_step(pops=pops, times=times) 
                pops_evolution.append(pops) #copy.deepcopy(pops))
                times_evolution.append(times) #copy.deepcopy(times))
                # saving a checkpoint 
                if self.checkpoint_freq and iter_idx % self.checkpoint_freq:
                    self.data_set.add(pops_evolution, times_evolution)
            except KeyboardInterrupt:
                break
        # self.data_set.add(pops_evolution, times_evolution)    
        

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
        next_times, next_reaction_ids = torch.min(possible_times, dim=0)
        # Update time and pops
        times += next_times
        pops += torch.index_select(self.reaction_system.stoichiometry_matrix.T, 0, next_reaction_ids)
        return pops, times

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
        times += self.sample_time(propensities_sum)
        # Sample next reactions
        cur_propensities /= propensities_sum  # normalizing propensities      
        next_reaction_ids = self.sample_reaction(cur_propensities)
        # Update pops
        pops += torch.index_select(self.reaction_system.stoichiometry_matrix.T, 0, next_reaction_ids)
        return pops, times

    def sample_reaction(self, propensity_values:torch.Tensor) -> torch.Tensor:
        """
        Sample next reaction index 
        :param propensities: normalized vector of propensities 
        :return: next reaction indexes
        """
        n_traj = propensity_values.shape[-1]
        q = torch.rand(n_traj, device=self.device)
        propensities_cumsums = torch.cumsum(propensity_values, dim=0)
        flags = (propensities_cumsums < q).type(self.reaction_system.int_type)
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