from __future__ import annotations

from typing import List

import numpy as np
import torch

from ssa_solvers.chemical_reaction_system import BaseChemicalReactionSystem

cfg = {'name': 'mRNAsRNAInCis',
       'stochastic_sim_cfg': {'checkpoint_freq': 1,
                              'save_to_file': True,
                              'trajectories_per_file': 50000,
                              'path': './logs/',
                              'solver': 'direct'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4,
                       'rtol': 1e-10}
       }


class mRNAsRNAInCis(BaseChemicalReactionSystem):
    _species = {'fmRNA': 0, 'Prot': 1}
    _params = {'volume': 0.6022, 'beta_fmrna': 1.0,
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

    def _propensities(self, pops: torch.Tensor) -> List[torch.Tensor]:
        """
        Composes a list of propensity functions
        :params pops: current population
        :return: list of propensities
        """
        return [
            self.params['volume'] * self.params['beta_fmrna'] *
            torch.ones(pops.shape[:-1], device=self.device),
            self.params['k_t'] * pops[..., self.species['fmRNA']],
            self.params['k_rep'] / self.params['volume'] * pops[...,
                                                                self.species['fmRNA']] * (pops[..., self.species['fmRNA']] - 1) / 2.0,
            self.params['delta_fmrna'] * pops[..., self.species['fmRNA']],
            self.params['delta_p'] * pops[..., self.species['Prot']],
        ]

    def _propensities_np(self, pops: np.ndarray) -> List[np.ndarray]:
        """
        Composes a list of propensity functions
        :params pops: current population
        :return: list of propensities
        """
        return [
            self.params['volume'] * self.params['beta_fmrna'] *
            np.ones(pops.shape[:-1]),
            self.params['k_t'] * pops[..., self.species['fmRNA']],
            self.params['k_rep'] / self.params['volume'] * pops[...,
                                                                self.species['fmRNA']] * (pops[..., self.species['fmRNA']] - 1) / 2.0,
            self.params['delta_fmrna'] * pops[..., self.species['fmRNA']],
            self.params['delta_p'] * pops[..., self.species['Prot']],
        ]

    def _jacobian(self, pops: np.ndarray) -> List[np.ndarray]:
        """
        Returns the Jacobian of the vector field for the ODE computations
        :param pops: current population
        """
        return NotImplemented


if __name__ == "__main__":
    system = mRNAsRNAInCis()
    n_species = len(system.species.species_names)
    n_trajs = 5
    pops = torch.rand((n_trajs, n_species))
    n_reactions = system.propensities(pops).shape[0]
    assert system.stoichiometry_matrix.shape == (n_species, n_reactions), \
        print(n_reactions, n_species, system.stoichiometry_matrix.shape)
