import torch 
import numpy as np
from ssa_solvers.chemical_reaction_system import BaseChemicalReactionSystem
from typing import List

cfg = {'stochastic_sim_cfg': {'checkpoint_freq': 0, 
                              'solver': 'direct'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4, 
                       'rtol': 1e-10}
    }

class mRNAsRNAInTrans(BaseChemicalReactionSystem):
    _params = {'volume': 0.6022, 'beta_m': 1.0, 'beta_s': 1.0, 'delta_m':  0.2476,  'delta_s':  0.0482, 'delta_p': 0.0234, \
                'k_t': 1.0, 'k_rep': 0.3} 
    _species = {'mRNA': 0, 'sRNA' : 1, 'Prot' : 2} 
    
    def __init__(self,  device=torch.device("cpu")):
        """
        :param device: torch device for simulations
        """
        self.stoichiometry_matrix = torch.tensor([
                                    [1, 0,  0, -1, -1,  0,  0], 
                                    [0, 1,  0, -1,  0, -1,  0],
                                    [0, 0 , 1,  0,  0,  0, -1] 
                                ], dtype=torch.int64, device=device)
        super(mRNAsRNAInTrans, self).__init__(device=device)      
                                
    def _propensities(self, pops: torch.Tensor) -> List[torch.Tensor]:
        """
        Composes a list of propensity functions  
        :params pops: current population 
        :return: list of propensities 
        """
        return [
            self.params['volume'] * self.params['beta_m'] * torch.ones(pops.shape[:-1], device=self.device), 
            self.params['volume'] * self.params['beta_s'] * torch.ones(pops.shape[:-1], device=self.device), 
            self.params['k_t'] * pops[..., self.species['mRNA']], 
            self.params['k_rep'] / self.params['volume'] * pops[..., self.species['mRNA']] * pops[..., self.species['sRNA']],     
            self.params['delta_m'] * pops[..., self.species['mRNA']], 
            self.params['delta_s'] * pops[..., self.species['sRNA']], 
            self.params['delta_p'] * pops[..., self.species['Prot']], 
        ]

    def _propensities_np(self, pops: np.ndarray) -> List[np.ndarray]:
        """
        Composes a list of propensity functions in numpy (hardcoded for speed)
        :params pops: current population
        :return: list of propensities
        """
        return [
            self.params['volume'] * self.params['beta_m'] * np.ones(pops.shape[:-1]), 
            self.params['volume'] * self.params['beta_s'] * np.ones(pops.shape[:-1]), 
            self.params['k_t'] * pops[..., self.species['mRNA']], 
            self.params['k_rep'] / self.params['volume'] * pops[..., self.species['mRNA']] * pops[..., self.species['sRNA']],     
            self.params['delta_m'] * pops[..., self.species['mRNA']], 
            self.params['delta_s'] * pops[..., self.species['sRNA']], 
            self.params['delta_p'] * pops[..., self.species['Prot']], 
        ]


if __name__ == "__main__":
    system = mRNAsRNAInTrans()
    n_species = len(system.species.species_names)
    n_trajs = 5
    pops = torch.rand((n_trajs, n_species))
    n_reactions = system.propensities(pops).shape[0]
    assert system.stoichiometry_matrix.shape == (n_species, n_reactions), \
        print(n_reactions, n_species, system.stoichiometry_matrix.shape)
    pass 