import torch 
from src.chemical_reaction_system import BaseChemicalReactionSystem, Array
from src.utils import is_torch_int_type


cfg = {'stochastic_sim_cfg': {'checkpoint_freq': 0, 
                              'solver': 'direct'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4, 
                       'rtol': 1e-10}
    }

class TetRsRNAInCis(BaseChemicalReactionSystem):
    _params ={'volume': 0.6022, 
              'aTc': 0,  'K1': 1.054, 'K2': 18.46, 'KD':  0.1182, 'TX_ptet': 2.26/1.054, # promoter parameters
              'delta_fmrna': 0.0482,  'delta_tetr': 0.0234, 'k_t': 1, 'k_rep': 1}
    _species = {'fmRNA': 0, 'TetR' : 1} 
    
    def __init__(self, int_type=torch.int64):
        self.int_type = int_type  # chose to specify the type here, as for some networks it may be sufficient to use torch.int32
        assert is_torch_int_type(int_type), "Please specify a torch int type, e.g., torch.int64"        
        self.stoichiometry_matrix = torch.tensor([
                                    [1, 0, -1, -1,  0], 
                                    [0, 1,  0,  0, -1] 
                                ], dtype=self.int_type)
        super(TetRsRNAInCis, self).__init__()                        

    def _propensities(self, pops: Array) -> Array:
        param1 = self.params['K2'] / self.params['volume'] / (1 + self.params['aTc'] / self.params['KD'])
        ptet_tx_init = self.params['TX_ptet'] / (self.params['K1'] + (1 + pops[..., self.species['TetR']] * param1 ) ** 2)
        return [
            self.params['volume'] * ptet_tx_init, 
            self.params['k_t'] * pops[..., self.species['fmRNA']], 
            self.params['k_rep'] / self.params['volume'] * pops[..., self.species['fmRNA']] * (pops[..., self.species['fmRNA']] - 1) / 2.0,     
            self.params['delta_fmrna'] * pops[..., self.species['fmRNA']], 
            self.params['delta_tetr'] * pops[..., self.species['TetR']], 
        ]


if __name__ == "__main__":
    system = TetRsRNAInCis()
    n_species = len(system.species.species_names)
    n_trajs = 5
    pops = torch.rand((n_trajs, n_species))
    n_reactions = system.propensities(pops).shape[0]
    assert system.stoichiometry_matrix.shape == (n_species, n_reactions), \
        print(n_reactions, n_species, system.stoichiometry_matrix.shape)
    pass 