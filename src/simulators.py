import numpy as np
import torch 
import copy 
from typing import Any, Dict
import torch.nn.functional as F
import scipy.integrate as integrate

class DeterministicSimulator:
    def __init__(self, 
                reaction_system:Any,
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

class StochasticSimulator:
    def __init__(self, 
                data_set: Any, 
                reaction_system:Any,
                cfg:Dict) -> None:
        self.reaction_system = reaction_system
        self.data_set = data_set
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
        times = torch.zeros((n_trajectories, ))
        pops_evolution = [copy.deepcopy(pops)]
        times_evolution = [copy.deepcopy(times)]
        iter_idx = 0
        while times.min() < end_time:
            iter_idx += 1
            try: # pressing ctrl+c will add data to the data class and break prematurely 
                self.simulate_one_step(pops=pops, times=times) 
                pops_evolution.append(copy.deepcopy(pops.cpu()))
                times_evolution.append(copy.deepcopy(times.cpu()))
                # saving a checkpoint 
                if self.checkpoint_freq and iter_idx % self.checkpoint_freq:
                    self.data_set.add(pops_evolution, times_evolution)
            except KeyboardInterrupt:
                break
        self.data_set.add(pops_evolution, times_evolution)    

    def simulate_one_step(self, pops: torch.Tensor, times: torch.Tensor) -> None:
        """
        Chooses the method for simulation  
        :param pops: current population 
        :param times: current simulation times for the population
        """
        if self.solver == 'direct':
            self.simulate_one_step_direct(pops, times)
        else:
            raise NotImplementedError

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
        pops += torch.vstack([self.reaction_system.stoichiometry_matrix[:, idx] for idx in next_reaction_ids])

    def sample_reaction(self, propensity_values:torch.Tensor) -> torch.Tensor:
        """
        Sample next reaction index 
        :param propensities: normalized vector of propensities 
        :return: next reaction indexes
        """
        n_traj = propensity_values.shape[-1]
        q = torch.rand(n_traj)
        propensities_cumsums = torch.cumsum(propensity_values, dim=0)
        flags = (propensities_cumsums < q).type(self.reaction_system.int_type)
        return torch.argmin(flags, dim=0)

    def sample_time(self, propensity_values_sum: torch.Tensor) -> torch.Tensor:
        """
        Sample next reaction time using exponential distirbution 
        :param propensities_sum: sum of propensities 
        :return: a sample form  Exp(1 / propensity_values_sum)
        """
        q = torch.rand(propensity_values_sum.shape[0])
        return -q.log() / propensity_values_sum 


if __name__ == "__main__":
    pass