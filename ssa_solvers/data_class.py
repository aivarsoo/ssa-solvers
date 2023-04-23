import torch 
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import os
import einops
from datetime import datetime
import json

class SimulationData:
    def __init__(self, n_species:int, device=torch.device("cpu"), cfg:Dict=None):
        self.device = device 
        self.end_time = None

        self.n_species = n_species
        self.species_idx = [str(idx) for idx in range(self.n_species)]
        self.save_to_file = cfg['stochastic_sim_cfg']['save_to_file']
        path=cfg['stochastic_sim_cfg']['path']
        if self.save_to_file:
            # saving to file
            self.trajectories_per_file = cfg['stochastic_sim_cfg']['trajectories_per_file']
            timestamp = datetime.now()
            if not os.path.exists(path):
                os.mkdir(path)
            self.path = os.path.join(path, str(timestamp))
            os.mkdir(self.path)
            self.raw_data_path = os.path.join(self.path, "raw")
            os.mkdir(self.raw_data_path)
            self.processed_data_path = os.path.join(self.path, "processed")
            os.mkdir(self.processed_data_path)
            self.raw_data_filename = "raw_data.csv"
            self.processed_data_filename = "processed_data.csv"
            with open(os.path.join(self.path, 'config.json'), 'w') as fp:
                json.dump(cfg, fp)
        else:
            # keeping in memory
            self.raw_times_trajectories = None
            self.raw_pops_trajectories = None
            self.processed_times_trajectories = None
            self.processed_pops_trajectories = None
        
    def add(self, pops: torch.Tensor, times: torch.Tensor, first_add:bool=False, batch_idx:int=0) -> None:
        """
        Adds raw data to the class.
        :param pops: population evolution
        :param times: time evolution
        """
        if self.save_to_file:
            n_runs = times.shape[0]
            start_idx = batch_idx*self.trajectories_per_file
            end_idx = batch_idx*self.trajectories_per_file + min(self.trajectories_per_file, n_runs)
            self._save_to_csv(
                pops=pops.cpu().numpy(),
                times=times.cpu().numpy(),
                run_ids=np.arange(start_idx, end_idx),
                filename=os.path.join(self.raw_data_path, str(batch_idx) + "_" + self.raw_data_filename),
                write_header=first_add)
        else:
            pops = torch.unsqueeze(pops, -1)
            times = torch.unsqueeze(times, -1)
            if first_add:
                self.raw_pops_trajectories = pops
                self.raw_times_trajectories = times
            else:
                self.raw_pops_trajectories = torch.cat([self.raw_pops_trajectories, pops], dim=2)
                self.raw_times_trajectories = torch.cat([self.raw_times_trajectories, times], dim=-1)
        
    def process_data(self, time_grid: Optional[List[int]] = None) -> None:
        """
        Processes raw data to interpolate it to the time_range. 
        :param time_grid: time grid for the interpolation if None an approriate equidistant grid is calculated
        """
        if time_grid is None:
            time_grid = np.arange(0, self.end_time)
        if self.save_to_file:
            assert not os.path.exists(os.path.join(self.path, self.raw_data_filename)), "Please provide data"
            files = os.listdir(self.raw_data_path)
            for file_idx, file_ in enumerate(files):
                filename = os.path.join(self.raw_data_path, file_)
                runs_ids = np.unique(pd.read_csv(filename, usecols=["run_id"])); runs_ids.sort()
                raw_times_trajectories = einops.rearrange(pd.read_csv(filename, usecols=["time"]).values, "(t h) m -> h (t m) ", h=runs_ids.shape[0])
                raw_pops_trajectories = einops.rearrange(pd.read_csv(filename, usecols=self.species_idx).values, "(t h) s -> h s t", h=runs_ids.shape[0], s=self.n_species)
                self._process_batch_trajectories(raw_times_trajectories, raw_pops_trajectories, time_grid, write_header=file_idx==0, runs_ids=runs_ids)
        else:
            assert self.raw_pops_trajectories is not None, "Please provide data"
            assert self.n_species == self.raw_pops_trajectories.shape[-2]
            time_grid = torch.Tensor(time_grid).to(device=self.device)
            self._process_batch_trajectories(self.raw_times_trajectories, self.raw_pops_trajectories, time_grid)

    def mean_and_std(self,  time_grid: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes means of the species of the species numbers 
        :param time_range: time indexes for which we compute the statistic, 
        :return: mean and variance of the pops
        """
        if self.save_to_file:
            if time_grid is not None: # if time grid is given re-processing data
                self.clear_processed_data()
                self.process_data(time_grid=time_grid)
            elif not os.listdir(self.processed_data_path): # check if the data was processed (check files?)
                self.process_data(time_grid=time_grid)
            else:
                print("Found existing processed data. Using these data")
            files = os.listdir(self.processed_data_path)
            time_length = len(files)
            _mean = np.zeros((self.n_species, time_length))
            _std = np.zeros((self.n_species, time_length))
            for file in files:
                t_idx = int(file.split("_")[0])
                df = pd.read_csv(os.path.join(self.processed_data_path, file),usecols=self.species_idx)
                _mean[:, t_idx] = df.mean().values
                _std[:, t_idx] = df.std().values
            return _mean, _std
        else:
            assert self.raw_pops_trajectories is not None, "Please get the data before computing statistics"
            assert not (time_grid is None and self.processed_pops_trajectories is None), "Specify time_range as no data was processed"
            if self.processed_pops_trajectories is None:
                self.process_data(time_grid=time_grid)
            float_data = self.processed_pops_trajectories.float()
            return torch.mean(float_data, dim=0).cpu().numpy(), torch.std(float_data, dim=0).cpu().numpy()

    def clear_processed_data(self):
        """
        Deleting all processed data
        """
        files = os.listdir(self.processed_data_path)
        for file_idx, file_ in enumerate(files):
            os.remove(os.path.join(self.processed_data_path, file_))

    # helper files
    def _process_batch_trajectories(self, raw_times_trajectories, raw_pops_trajectories, time_grid, write_header:bool=False, runs_ids:np.ndarray=None):
        """
        Process a batch of trajectories to get the values on a specified time grid
        """
        n_traj  = raw_times_trajectories.shape[0]
        if self.save_to_file:
            all_traj = np.arange(n_traj)
            time_idxs = np.zeros((n_traj, ), dtype=np.int64)
        else:
            self.processed_times_trajectories = time_grid
            self.processed_pops_trajectories = torch.zeros((n_traj, self.n_species, time_grid.shape[0]), dtype=torch.int64, device=self.device)
            time_idxs = torch.zeros((n_traj, ), dtype=torch.int64, device=self.device)
            all_traj = torch.arange(n_traj, device=self.device)
        cur_pops = raw_pops_trajectories[..., 0]
        for t_idx, cur_time in enumerate(time_grid):
            mask = (raw_times_trajectories[all_traj, time_idxs] <= cur_time)# figure out which times need updating
            while mask.any():
                cur_pops[mask, :] = raw_pops_trajectories[all_traj[mask], :, time_idxs[mask]]# update pops
                time_idxs += mask# update times
                mask = (raw_times_trajectories[all_traj, time_idxs] <= cur_time)# figure out which times need updating
            if self.save_to_file:
                self._save_to_csv(
                    pops=cur_pops,
                    times=cur_time*np.ones((n_traj,)),
                    run_ids=runs_ids,
                    filename=os.path.join(
                            self.processed_data_path,
                            str(t_idx) + "_" + self.processed_data_filename),
                    write_header=write_header)
            else:
                self.processed_pops_trajectories[all_traj, ..., t_idx] = cur_pops

    def _save_to_csv(self, pops:np.ndarray, times:np.ndarray, run_ids:np.ndarray, filename:str, write_header:bool=False) -> None:
        """
        Save data to csv
        :param pops: population evolution
        :param times: time evolution
        :param run_ids: run ids
        :param filename: filename to save csv
        :param write_header: write the first row (column indexes) in the file
        """
        df=pd.concat([
           pd.DataFrame(pops),
           pd.DataFrame({
                "time": times,
                "run_id": run_ids})
                ], axis=1)
        df.to_csv(filename, mode="a", header=write_header, index=False)
