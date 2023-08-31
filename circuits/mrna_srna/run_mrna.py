import numpy as np
import torch

from circuits.mrna_srna.mrna_srna_incis import cfg
from circuits.mrna_srna.mrna_srna_incis import mRNAsRNAInCis
from circuits.mrna_srna.mrna_srna_intrans import cfg
from circuits.mrna_srna.mrna_srna_intrans import mRNAsRNAInTrans
from ssa_solvers.simulators import DeterministicSimulator
from ssa_solvers.simulators import StochasticSimulator

torch.set_default_tensor_type(torch.DoubleTensor)


def run_mrna(end_time: float = 300, n_steps: int = 100, n_traj: int = 2, device=torch.device('cpu')):
    time_grid = list(range(0, end_time, int(end_time / n_steps)))
    cfg['stochastic_sim_cfg'].update(
        dict(
            save_to_file=False,
            trajectories_per_batch=50000,
            solver='first_reaction'))

    reaction_system_incis = mRNAsRNAInCis(device=torch.device("cpu"))
    ode_simulator = DeterministicSimulator(
        reaction_system=reaction_system_incis,
        cfg=cfg
    )
    reaction_system_incis = mRNAsRNAInCis(device=device)
    ssa_simulator_incis = StochasticSimulator(
        reaction_system=reaction_system_incis,
        cfg=cfg,
        device=device
    )

    # increasing fmRNA production rate
    reaction_system_incis.params = {'beta_fmrna': 2}
    init_pops = torch.zeros(
        (reaction_system_incis.n_species, ), dtype=torch.int64, device=device)
    ssa_simulator_incis.simulate(init_pops=init_pops,
                                 end_time=end_time, n_trajectories=n_traj)
    ode_res_incis = ode_simulator.simulate(init_pops=np.zeros(
        (reaction_system_incis.n_species,)), time_grid=time_grid)
    means_incis, stds_incis = ssa_simulator_incis.data_set.mean_and_std(
        time_grid=time_grid)

    ode_simulator_intrans = DeterministicSimulator(
        reaction_system=mRNAsRNAInTrans(device=torch.device("cpu")),
        cfg=cfg
    )

    reaction_system_intrans = mRNAsRNAInTrans(device=device)
    ssa_simulator_intrans = StochasticSimulator(
        reaction_system=reaction_system_intrans,
        cfg=cfg,
        device=device
    )

    init_pops = torch.zeros(
        (reaction_system_intrans.n_species, ), dtype=torch.int64, device=device)
    ssa_simulator_intrans.simulate(init_pops=init_pops,
                                   end_time=end_time, n_trajectories=n_traj)
    ode_res_intrans = ode_simulator_intrans.simulate(init_pops=np.zeros(
        (reaction_system_intrans.n_species,)), time_grid=time_grid)
    means_intrans, stds_intrans = ssa_simulator_intrans.data_set.mean_and_std(
        time_grid=time_grid)
    print("done")
