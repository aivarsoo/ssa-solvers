# Gillespie Stochastic Simulation implememntation in pytorch

This library implements direct and first reaction methods for stochastic simulations of the Master equation. The code is based on PyTorch and hence can be used on CPU and GPU alike.

## Installation


```bash
cd ssa-solvers
pip install -e .
```

# A small package for Stochastic Simulations

TBD

# Run on docker on a Linux system

Build the docker image 

```
docker build . --tag ssa_solvers
```

Make sure the docker container can write in the directory `./logs` while running 

```
chmod 775 -R logs
```

Run the container while mounting the `./logs` directory
```
docker run -it -rm --runtime nvidia --gpus all -v ./logs:/home/docker_user/project/logs --group-add $(id -g) --user docker_user ssa_solvers bash
```