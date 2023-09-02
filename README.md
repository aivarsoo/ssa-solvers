# Gillespie Stochastic Simulation implememntation in pytorch

This repo implements direct and first reaction methods of Gillespie [stochastic simulation algorithm](https://pubs.acs.org/doi/pdf/10.1021/j100540a008) for solving the Master equation. The code is based on PyTorch and can be used on CPU and GPU alike. The repo was developed and tested on Ubuntu 20.04 and Ubuntu 22.04 WSL2.

## Installation

### Using a conda environment
```bash
git clone git@github.com:aivarsoo/ssa-solvers.git
cd ssa-solvers
NAME=ssa_solvers
conda create --name $NAME -y python==3.10
conda activate $NAME
pip install -e .
```
### Using docker

Build the docker image

```
docker build . --tag ssa_solvers
```

Make sure the docker container can write in the directory `./logs` by running

```
chmod 775 -R logs
```

Connect to the container running `bash` while mounting the `./logs` directory
```
docker run -it --rm --net=host --gpus all -v ./logs:/home/docker_user/project/logs --group-add $(id -g) --user docker_user ssa_solvers bash
```
This command uses the Nvidia docker wrapper for GPU access, see [Nvidia docker installation notes](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for details.

To run notebooks
```
docker run -it --rm --net=host --gpus all -v ./logs:/home/docker_user/project/logs --group-add $(id -g) --user docker_user ssa_solvers jupyter lab
```

## Creating a new chemical reaction system class for simulation

The base reaction system is encoded in `BaseChemicalReactionSystem` class in `ssa_solvers.chemical_reaction_system`.
To perform simulations one needs to create a subclass of `BaseChemicalReactionSystem` with the following fields overloaded:

`self.stoichiometry_matrix` - Stoichiometric matrix of the chemical reaction system

`self._params` - Dictionary of parameters of the chemical reaction system with a string key for parameter name and a float value for parameter value

`self._species` - Dictionary of species of the chemical reaction system with a string key for species name and an integer value for species index in the species vector

The following method calculates the propensity vector based on the current population, e.g.:

```python
def propensities(self, pops: torch.Tensor) -> torch.Tensor:
    propensities = [
        ...,
        ...,
    ]
    return torch.vstack(propensities)
```
## Defining configuration file
```python
cfg = {
       'name': 'mRNAsRNAInTrans',
       'stochastic_sim_cfg': {'checkpoint_freq': 1,
                              'save_to_file': True,
                              'trajectories_per_batch': 50000,
                              'path': './logs/',
                              'solver': 'direct',
                              'precision': 'fp32'},
       'ode_sim_cfg': {'solver': 'RK23',
                       'atol': 1e-4,
                       'rtol': 1e-10}
}
```

`cfg['name']` - Reaction system name

`cfg['stochastic_sim_cfg']` - parameters for stochastic simulation:
* `checkpoint_freq` - frequency of checkpoints
* `save_to_file` - if `True` the results are saved to a CSV file, otherwise kept in memory
* `trajectories_per_batch` - number of trajectories to simulate at once. Set to `sys.maxsize` if `save_to_file = False`, i.e., `trajecories_per_batch = n_trajectories`
* `solver` - type of a solver: `direct` or `first_reaction`
* `path` - path to save the logs and data
* `precision` - float precision (`fp64` - double, `fp32` - single and `fp16` half precision, default: `fp16`)

`cfg['ode_sim_cfg']` - parameters for ODE simulation using `solve_ivp` method from `xitorch` package, see [xitorch documentation](https://xitorch.readthedocs.io/en/latest/api/xitorch_integrate/solve_ivp.html) for details

## Simulating a chemical reaction system
Define the classes
```python
device = torch.device("cpu")
reaction_system = MyReactionSystem(device=device)
ssa_simulator = StochasticSimulator(
        reaction_system=reaction_system,
        cfg=cfg,
        device=device
    )
```
Simulate
```python
init_pops = torch.zeros(
        (reaction_system_intrans.n_species, ), dtype=torch.int64, device=device)
end_time = 100
ssa_simulator.simulate(
    init_pops=init_pops,
    end_time=end_time,
    n_trajectories = 100)
```
Compute mean and variance on the specified grid
```python
n_steps = 100
time_grid = torch.arange(0, end_time, int(
        end_time / n_steps), device=device)
means, stds = ssa_simulator.data_set.mean_and_std(time_grid=time_grid)
```

## Testing pre-defined circuits

### Command line scripts

```bash
python run_example --circuit CIRCUIT --end_time END_TIME --n_steps N_STEPS --n_traj N_TRAJ --device DEVICE
```

`CIRCUIT` - Circuit name (currently implemented `mrna`, `auto`)

`END_TIME` - Simulation end time (positive `float`)

`N_STEPS` - Number of steps in the time grid (positive `int`)

`N_TRAJ` - Number of trajectories in the stochastic simulation (positive `int`)

`DEVICE` - Device for simulations (options: `cpu`, `cuda`)


### Notebooks

`notebooks/autorepressor.ipynb`

`notebooks/mrna_srna.ipynb`

`notebooks/speed_comparisons.ipynb`
