from __future__ import annotations

from typing import List

import numpy as np
import torch

from ssa_solvers.chemical_reaction_system import BaseChemicalReactionSystem

cfg = {'name': 'mRNAsRNAInCis',
       'stochastic_sim_cfg': {'checkpoint_freq': 1,
                              'save_to_file': False,
                              'trajectories_per_batch': 50000,
                              'path': './logs/',
                              'solver': 'direct'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4,
                       'rtol': 1e-10}
       }


class mRNAsRNAInCis(BaseChemicalReactionSystem):
    _species = {'fmRNA': 0, 'Prot': 1}
    _params = {'volume': 0.6022, 'beta_fmrna': 2.0,
               'delta_fmrna': 0.0482, 'delta_p': 0.0234, 'k_t': 1.0, 'k_rep': 0.3}

    def __init__(self, device=torch.device("cpu")):
        """
        :param int_type: specifies the integer type (sometimes it might be useful to use torch.int32 to safe memory)
        """
        self.stoichiometry_matrix = torch.tensor([
            [1,  0, -1, -1,  0],
            [0,  1,  0,  0, -1]
        ], dtype=torch.int64, device=device)
        super(mRNAsRNAInCis, self).__init__(device=device)

    def propensities(self, pops: torch.Tensor) -> List[torch.Tensor]:
        """
        Composes a list of propensity functions
        :params pops: current population
        :return: list of propensities
        """
        return torch.vstack([
            self.params['volume'] * self.params['beta_fmrna'] *
            torch.ones(pops.shape[:-1], device=self.device),
            self.params['k_t'] * pops[..., self.species['fmRNA']],
            self.params['k_rep'] / self.params['volume'] * pops[...,
                                                                self.species['fmRNA']] * (pops[..., self.species['fmRNA']] - 1) / 2.0,
            self.params['delta_fmrna'] * pops[..., self.species['fmRNA']],
            self.params['delta_p'] * pops[..., self.species['Prot']],
        ])
