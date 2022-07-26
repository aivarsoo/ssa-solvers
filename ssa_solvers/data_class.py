import torch 
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import os
import einops 
from datetime import datetime

class SimulationData:
    def __init__(self, device=torch.device("cpu"), save_to_file:bool=False, trajectories_per_file:int=20000, path:str=None):
        self.device = device 
        self.end_time = None
        self.save_to_file = save_to_file
        # saving to file
        if path is None:
            path = "./logs/"
        if save_to_file:
            self.trajectories_per_file = trajectories_per_file
            timestamp = datetime.now()
            if not os.path.exists(path):
                os.mkdir(path)
            self.path = os.path.join(path, str(timestamp)) #  + "_" + str(np.random.randint(1000))
            os.mkdir(self.path)
            self.raw_data_path = os.path.join(self.path, "raw")
            os.mkdir(self.raw_data_path)
            self.processed_data_path = os.path.join(self.path, "processed")
            os.mkdir(self.processed_data_path)
            self.raw_data_filename = "raw_data.csv"
            self.processed_data_filename = "processed_data.csv"
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
        :param time_range: time grid for the interpolation if None an approriate equidistant grid is calculated
        """
        if self.save_to_file:
            assert not os.path.exists(os.path.join(self.path, self.raw_data_filename)), "Please provide data"
            if time_grid is None:
                time_grid = np.arange(0, self.end_time)                
            files = os.listdir(self.raw_data_path)
            for file_idx in range(len(files)):
                self._process_one_file(time_grid, pd.read_csv(os.path.join(self.raw_data_path, files[file_idx])), write_header=file_idx==0)
        else:  
            assert self.raw_pops_trajectories is not None, "Please provide data"
            if time_grid is None:
                time_grid = torch.arange(0, self.end_time, device=self.device)
            else:
                time_grid = torch.Tensor(time_grid).to(device=self.device)                
            self.processed_times_trajectories = time_grid
            n_traj, n_species = self.raw_times_trajectories.shape[0], self.raw_pops_trajectories.shape[-2]
            self.processed_pops_trajectories = torch.zeros((n_traj, n_species, self.processed_times_trajectories.shape[0]), dtype=torch.int64, device=self.device)
            all_traj = torch.arange(n_traj, device=self.device)
            cur_pops = self.raw_pops_trajectories[..., 0] 
            time_idxs = torch.zeros((n_traj, ), dtype=torch.int64, device=self.device)
            for t_idx, cur_time in enumerate(self.processed_times_trajectories):
                while (self.raw_times_trajectories[all_traj, time_idxs] < cur_time).any():
                    mask = (self.raw_times_trajectories[all_traj, time_idxs] < cur_time)   # figure out which times need updating 
                    cur_pops[mask, :] = self.raw_pops_trajectories[all_traj[mask], :, time_idxs[mask]] # update pops 
                    time_idxs += mask   # update times 
                self.processed_pops_trajectories[all_traj, ..., t_idx] = cur_pops  

    def mean_and_std(self,  time_grid: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes means of the species of the species numbers 
        :param time_range: time indexes for which we compute the statistic, 
        :return: mean of the pops with species_idxs, if time_range is empty then returns the last processed 
        """
        if self.save_to_file:
            if not os.listdir(self.processed_data_path): # check if the data was processed (check files?)
                self.process_data(time_grid=time_grid)
            else:
                print("Found existing processed data. Using these data")
            return self._compute_stats_from_file()
        else:
            assert self.raw_pops_trajectories is not None, "Please get the data before computing statistics"
            assert not (time_grid is None and self.processed_pops_trajectories is None), "Specify time_range as no data was processed"
            if self.processed_pops_trajectories is None:
                self.process_data(time_grid=time_grid)
            float_data = self.processed_pops_trajectories.float()                
            return torch.mean(float_data, dim=0).cpu().numpy(), torch.std(float_data, dim=0).cpu().numpy()

    # helper files 
    def _process_one_file(self, time_range:np.ndarray, df:pd.DataFrame, write_header:bool) -> None:
        """"
        processing raw data from one file 
        :param time_range:
        :param df:
        :param write_header:
        """
        species_idx = df.columns[:-2]
        runs_ids = np.arange(df["run_id"].min(), df["run_id"].max()+1)
        n_traj, n_species = runs_ids.shape[0], len(species_idx)
        all_traj = np.arange(n_traj)
        raw_times_trajectories = einops.rearrange(df["time"].values, "(t h) -> h t", h=n_traj)
        raw_pops_trajectories = einops.rearrange(df[species_idx].values, "(t h) s -> h s t", h=n_traj, s=n_species)
        del df 
        cur_pops = raw_pops_trajectories[..., 0]
        time_idxs = np.zeros((n_traj, ), dtype=np.int64)             
        for t_idx, cur_time in enumerate(time_range):
            while (raw_times_trajectories[all_traj, time_idxs] < cur_time).any():
                mask = (raw_times_trajectories[all_traj, time_idxs] < cur_time)   # figure out which times need updating 
                cur_pops[mask, :] = raw_pops_trajectories[all_traj[mask], :, time_idxs[mask]] # update pops 
                time_idxs += mask   # update times 
            self._save_to_csv(
                pops=cur_pops,
                times=cur_time*np.ones((n_traj,)),
                run_ids=runs_ids,
                filename=os.path.join(
                        self.processed_data_path,
                        str(t_idx) + "_" + self.processed_data_filename),
                write_header=write_header)

    def _compute_stats_from_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute statistics of the data
        """
        files = os.listdir(self.processed_data_path)
        df = pd.read_csv(os.path.join(self.processed_data_path, files[0]))
        species_idx = df.columns[:-2]
        n_species, time_length = len(species_idx), len(files)
        _mean = np.zeros((n_species, time_length))
        _std = np.zeros((n_species, time_length))
        for file in files:
            t_idx = int(file.split("_")[0])
            df = pd.read_csv(os.path.join(self.processed_data_path, file))[species_idx]
            _mean[:, t_idx] = df.mean().values
            _std[:, t_idx] = df.std().values 
        return _mean, _std

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
