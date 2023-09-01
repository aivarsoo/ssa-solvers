import copy

import numpy as np
import scipy.interpolate as interpolate
import torch

EPS = 1e-13


def set_overrides(cfg, cfg_overrides):
    for key in cfg_overrides.keys():
        cfg[key] = copy.deepcopy(cfg_overrides[key])


def is_matrix_int_type(matrix):
    return is_np_int_type(matrix.dtype)


def is_tensor_int_type(tensor):
    return is_torch_int_type(tensor.dtype)


def is_np_int_type(_type):
    return _type is np.dtype('int64') or _type is np.dtype('int32') or \
        _type is np.dtype('int16') or _type is np.dtype('int8')


def is_torch_int_type(_type):
    return _type is torch.int64 or _type is torch.int32 or \
        _type is torch.int16 or _type is torch.int8


def scipy_interpolation(pops_, times_, t_grid):
    """
    Scipy interpolation of data in pops_ / times_ on the grid t_grid.
    We found it slow
    """
    n_traj = times_.shape[0]
    pops_2 = pops_.cpu().numpy()
    times_2 = times_.cpu().numpy()
    vals = []
    for traj_idx in range(n_traj):
        pop_fun = interpolate.interp1d(times_2[traj_idx, :], pops_2[traj_idx, :, :],
                                       kind='linear',
                                       axis=-1)
        vals.append(pop_fun(t_grid))
    return vals
