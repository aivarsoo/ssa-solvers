from __future__ import annotations

from typing import List

import torch

from ssa_solvers.chemical_reaction_system import BaseChemicalReactionSystem

cfg = {'name': 'TetRsRNAInTrans',
       'stochastic_sim_cfg': {'checkpoint_freq': 1,
                              'save_to_file': True,
                              'trajectories_per_batch': 50000,
                              'path': './logs/',
                              'solver': 'direct'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4,
                       'rtol': 1e-10}
       }


class TetRsRNAInTrans(BaseChemicalReactionSystem):
    _params = {'volume': 0.6022,
               # promoter parameters
               'aTc': 0,  'K1': 1.054, 'K2': 18.46, 'KD':  0.1182, 'TX_ptet': 2.26/1.054,
               'delta_mrna': 0.2482, 'delta_srna': 0.0482, 'delta_tetr': 0.0234, 'k_t': 1, 'k_rep': 1}
    _species = {'mRNA': 0, 'sRNA': 1, 'TetR': 2}

    def __init__(self, device=torch.device("cpu")):
        self.stoichiometry_matrix = torch.tensor([
            [1, 0, 0, -1, -1,  0,  0],
            [0, 1, 0, -1,  0, -1,  0],
            [0, 0, 1,  0,  0,  0, -1]
        ], dtype=torch.int64, device=device)
        super(TetRsRNAInTrans, self).__init__(device=device)

    def propensities(self, pops: torch.Tensor) -> torch.Tensor:
        """
        Composes a vector of propensity functions
        :params pops: current population
        :return: vector of propensities
        """
        param1 = self.params['K2'] / self.params['volume'] / \
            (1 + self.params['aTc'] / self.params['KD'])
        ptet_tx_init = self.params['TX_ptet'] / (self.params['K1'] + (torch.ones(
            pops.shape[:-1],  device=self.device) + pops[..., self.species['TetR']] * param1) ** 2)
        return torch.vstack([
            self.params['volume'] * ptet_tx_init,
            self.params['volume'] * ptet_tx_init,
            self.params['k_t'] * pops[..., self.species['mRNA']],
            self.params['k_rep'] / self.params['volume'] * pops[...,
                                                                self.species['mRNA']] * pops[..., self.species['sRNA']],
            self.params['delta_mrna'] * pops[..., self.species['mRNA']],
            self.params['delta_srna'] * pops[..., self.species['sRNA']],
            self.params['delta_tetr'] * pops[..., self.species['TetR']],
        ])
