import torch
import numpy as np
import copy
from ssa_solvers.utils import is_torch_int_type
from ssa_solvers.utils import is_matrix_int_type, is_tensor_int_type
from typing import Union
Array = Union[torch.Tensor, np.ndarray]

class BaseChemicalReactionSystem:
    """
    Base class for chemical reaction systems 
    """
    def __init__(self, int_type=torch.int64, device=torch.device("cpu")):
        self.int_type = int_type  # chose to specify the type here, as for some networks it may be sufficient to use torch.int32
        assert is_torch_int_type(int_type), "Please specify a torch int type, e.g., torch.int64"        
        self.device = device
        assert len(self._stoichiometry_matrix.shape) == 2, "Stoichiometry_matrix should be a matrix (2-d tensor)!"
        assert is_matrix_int_type(self._stoichiometry_matrix) or is_tensor_int_type(self._stoichiometry_matrix), "Stochimetry matrix is a matrix of integers!"
        assert self.n_reactions == self.propensities(torch.zeros(self.n_species, device=self.device)).shape[0], \
        "Propensity function dimension do not match stochiometry"

    def propensities(self, pops: torch.Tensor) -> torch.Tensor:
        return torch.vstack(self._propensities(pops))

    def ode_fun(self, time: int, pops: np.ndarray) -> np.ndarray:
        return self._stoichiometry_matrix_np @ np.vstack(self._propensities(pops)).squeeze()

    def ode_fun_jac(self, time: int, pops: np.ndarray) -> np.ndarray:
        return self._jacobian(pops)

    def _jacobian(self, pops: Array):
        "Returns the Jacobian of the vector field."
        raise NotImplementedError

    def _propensities(self, pops: Array):
        "Returns the vector of propensities."
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
            self._params[key] = copy.deepcopy(params[key])
    
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
        return self._stoichiometry_matrix    

    @stoichiometry_matrix.setter
    def stoichiometry_matrix(self, stoichiometry_matrix: torch.Tensor):
        self._stoichiometry_matrix = stoichiometry_matrix
        self._stoichiometry_matrix_np = self._stoichiometry_matrix.cpu().numpy()
