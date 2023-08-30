from __future__ import annotations

from typing import List

import numpy as np
import torch

from ssa_solvers.chemical_reaction_system import BaseChemicalReactionSystem


cfg = {'name': 'TetRsRNAInCis',
       'stochastic_sim_cfg': {'checkpoint_freq': 1,
                              'save_to_file': True,
                              'trajectories_per_batch': 50000,
                              'path': './logs/',
                              'solver': 'direct'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4,
                       'rtol': 1e-10}
       }


class TetRsRNAInCis(BaseChemicalReactionSystem):
    _params = {'volume': 0.6022,
               # promoter parameters
               'aTc': 0,  'K1': 1.054, 'K2': 18.46, 'KD':  0.1182, 'TX_ptet': 2.26/1.054,
               'delta_fmrna': 0.0482,  'delta_tetr': 0.0234, 'k_t': 1, 'k_rep': 1}
    _species = {'fmRNA': 0, 'TetR': 1}

    def __init__(self,  device=torch.device("cpu")):
        self.stoichiometry_matrix = torch.tensor([
            [1, 0, -1, -1,  0],
            [0, 1,  0,  0, -1]
        ], dtype=torch.int64, device=device)
        super(TetRsRNAInCis, self).__init__(device=device)

    def _propensities(self, pops: torch.Tensor) -> List[torch.Tensor]:
        param1 = self.params['K2'] / self.params['volume'] / \
            (1 + self.params['aTc'] / self.params['KD'])
        ptet_tx_init = self.params['TX_ptet'] / (self.params['K1'] + (
            1 + pops[..., self.species['TetR']] * param1) ** 2)
        return [
            self.params['volume'] * ptet_tx_init,
            self.params['k_t'] * pops[..., self.species['fmRNA']],
            self.params['k_rep'] / self.params['volume'] * pops[...,
                                                                self.species['fmRNA']] * (pops[..., self.species['fmRNA']] - 1) / 2.0,
            self.params['delta_fmrna'] * pops[..., self.species['fmRNA']],
            self.params['delta_tetr'] * pops[..., self.species['TetR']],
        ]

    def _propensities_np(self, pops: np.ndarray) -> List[np.ndarray]:
        return self._propensities(pops)


if __name__ == "__main__":
    system = TetRsRNAInCis()
    n_species = len(system.species.species_names)
    n_trajs = 5
    pops = torch.rand((n_trajs, n_species))
    n_reactions = system.propensities(pops).shape[0]
    assert system.stoichiometry_matrix.shape == (n_species, n_reactions), \
        print(n_reactions, n_species, system.stoichiometry_matrix.shape)
