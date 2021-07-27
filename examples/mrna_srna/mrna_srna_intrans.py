import torch 
import numpy as np
from src.chemical_reaction_system import BaseChemicalReactionSystem, Array
from src.utils import is_torch_int_type
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
    
    def __init__(self, int_type=torch.int64):
        """
        :param int_type: specifies the integer type 
        """
        self.int_type = int_type  # chose to specify the type here, as for some networks it may be sufficient to use torch.int32
        assert is_torch_int_type(int_type), "Please specify a torch int type, e.g., torch.int64"
        self.stoichiometry_matrix = torch.tensor([
                                    [1, 0,  0, -1, -1,  0,  0], 
                                    [0, 1,  0, -1,  0, -1,  0],
                                    [0, 0 , 1,  0,  0,  0, -1] 
                                ], dtype=self.int_type)
        super(mRNAsRNAInTrans, self).__init__()      

    def _propensities(self, pops: Array) -> List[Array]:
        """
        Composes a list of propensity functions  
        :params pops: current population (either an np.ndarray or a torch.Tensor)
        :return: list of propensities (either an np.ndarray or a torch.Tensor)
        """
        return [
            self.params['volume'] * self.params['beta_m'] * torch.ones(pops.shape[:-1]), 
            self.params['volume'] * self.params['beta_s'] * torch.ones(pops.shape[:-1]), 
            self.params['k_t'] * pops[..., self.species['mRNA']], 
            self.params['k_rep'] / self.params['volume'] * pops[..., self.species['mRNA']] * pops[..., self.species['sRNA']],     
            self.params['delta_m'] * pops[..., self.species['mRNA']], 
            self.params['delta_s'] * pops[..., self.species['sRNA']], 
            self.params['delta_p'] * pops[..., self.species['Prot']], 
        ]

    def _jacobian(self, pops: Array) -> List[Array]:
        """
        Returns the Jacobian of the vector field for the ODE computations 
        :param pops: current population 
        """
        return NotImplemented


if __name__ == "__main__":
    system = mRNAsRNAInTrans()
    n_species = len(system.species.species_names)
    n_trajs = 5
    pops = torch.rand((n_trajs, n_species))
    n_reactions = system.propensities(pops).shape[0]
    assert system.stoichiometry_matrix.shape == (n_species, n_reactions), \
        print(n_reactions, n_species, system.stoichiometry_matrix.shape)
    pass 