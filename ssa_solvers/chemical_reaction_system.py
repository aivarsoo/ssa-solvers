from typing import List

import einops
import torch

from ssa_solvers.utils import is_matrix_int_type
from ssa_solvers.utils import is_tensor_int_type


class BaseChemicalReactionSystem:
    """
    Base class for chemical reaction systems
    """

    def __init__(self, device=torch.device("cpu")):
        self.device = device
        assert len(
            self._stoichiometry_matrix.shape) == 2, "Stoichiometry_matrix should be a matrix (2-d tensor)!"
        assert is_matrix_int_type(self._stoichiometry_matrix) or is_tensor_int_type(
            self._stoichiometry_matrix), "Stochimetry matrix is a matrix of integers!"
        assert self.n_reactions == self.propensities(torch.zeros(self.n_species, device=self.device)).shape[0], \
            "Propensity function dimension do not match stochiometry"

    def ode_fun(self):
        ":return: the function computing the vector field for the ODE simulation"
        return lambda time, pops: einops.einsum(self.stoichiometry_matrix.double(), self.propensities(pops), "m n, n k -> m k")

    def propensities(self, pops: torch.Tensor) -> torch.Tensor:
        """
        :param pops: current population number
        :return: the vector of propensities
        """
        raise NotImplementedError

    def ode_fun_jac(self, time: int, pops: torch.Tensor) -> torch.Tensor:
        """
        :param time: current time
        :param pops: current population number
        :return: the function computing the Jacobian of the vector field for the ODE simulation
        """
        return lambda time, pops: self._jacobian(pops)

    def _jacobian(self, pops: torch.Tensor):
        """
        :param pops: current population number
        :return: the Jacobian of the vector field.
        """
        raise NotImplementedError

    @property
    def params_names(self):
        return set(self._params.keys())

    @property
    def species_names(self):
        return set(self._species.keys())

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        for key in params.keys():
            assert key in self.params_names, "No such parameter in the system!"
            self._params[key] = params[key]

    @property
    def species(self):
        return self._species

    @property
    def n_species(self):
        return self.stoichiometry_matrix.shape[0]

    @n_species.setter
    def n_species(self):
        raise ValueError("Cannot set number of species directly!")

    @property
    def n_reactions(self):
        return self.stoichiometry_matrix.shape[1]

    @n_reactions.setter
    def n_reactions(self):
        raise AttributeError("Cannot set number of reactions directly!")

    @property
    def stoichiometry_matrix(self):
        "Stoichiometric matrix property"
        return self._stoichiometry_matrix

    @stoichiometry_matrix.setter
    def stoichiometry_matrix(self, stoichiometry_matrix: torch.Tensor):
        self._stoichiometry_matrix = stoichiometry_matrix
