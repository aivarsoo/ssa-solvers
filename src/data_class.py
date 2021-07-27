import torch 
from typing import Optional,  List
import copy 
from src.utils import is_torch_int_type

class SimulationData:
    def __init__(self, int_type=torch.int64):
        self.int_type = int_type
        assert is_torch_int_type(int_type), "Please specify a torch int type, e.g., torch.int64"
        self.raw_times_trajectories = None 
        self.raw_pops_trajectories = None 
        self.processed_times_trajectories = None 
        self.processed_pops_trajectories = None 

    def save(self, filename: str) -> None:     
        """
        Save data to the disk
        :param filename: file name string
        """
        torch.save({'raw_times':self.raw_times_trajectories, 
                    'raw_pops':self.raw_pops_trajectories,
                    'processed_times':self.processed_times_trajectories, 
                    'processed_pops':self.processed_pops_trajectories}, filename)

    def load(self, filename:str) -> None:
        """
        Load data from the disk
        :param filename: file name string
        """
        self.raw_times, self.raw_pops, self.processed_times, self.processed_pops = torch.load(filename)

    def add(self, pops_evolution: List, times_evolution:List) -> None:
        """  
        Adds raw data to the class 
        :param pops_evolution: population evolution
        :param times_evolution: time evolution
        """
        self.raw_times_trajectories = torch.stack(times_evolution, dim=-1)
        self.raw_pops_trajectories = torch.stack(pops_evolution, dim=-1) 

    def process_data(self, time_range: Optional[List[int]] = None) -> None:
        """
        Processes raw data to make the time grid equidistance  
        """
        assert self.raw_pops_trajectories is not None, "Please provide data"
        if time_range is None:
            time_range = torch.arange(0, self.raw_times_trajectories[:, -1].min())
        n_traj, n_species = self.raw_times_trajectories.shape[0], self.raw_pops_trajectories.shape[-2]
        self.processed_times_trajectories = time_range
        self.processed_pops_trajectories = torch.zeros((n_traj, n_species, self.processed_times_trajectories.shape[0]), dtype=self.int_type)
        all_traj = torch.arange(n_traj)
        cur_pops = copy.deepcopy(self.raw_pops_trajectories[..., 0])
        time_idxs = torch.zeros((n_traj, ), dtype=self.int_type)
        for t_idx, cur_time in enumerate(self.processed_times_trajectories):
            while (self.raw_times_trajectories[all_traj, time_idxs] < cur_time).any():
                mask = (self.raw_times_trajectories[all_traj, time_idxs] < cur_time)   # figure out which times need updating 
                cur_pops[mask, :] = self.raw_pops_trajectories[all_traj[mask], :, time_idxs[mask]] # update pops 
                time_idxs += mask   # update times 
            self.processed_pops_trajectories[all_traj, ..., t_idx] = cur_pops  
          
    def mean(self, species_idxs: Optional[List[int]] = None, time_range: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes means of the species of the species numbers 
        :param species_idxs: species indexes for which we compute the statistic  
        :param time_range: time indexes for which we compute the statistic, 
        :return: mean of the pops with species_idxs, if time_range is empty then returns the last processed 
        """
        assert self.raw_pops_trajectories is not None, "Please get the data before computing statistics"
        assert not (time_range is None and self.processed_pops_trajectories is None), "Specify time_range as no data was processed"
        if time_range is not None or self.processed_pops_trajectories is None:
            self.process_data(time_range=time_range)
        if species_idxs is None:
            species_idxs = torch.arange(0, self.processed_pops_trajectories.shape[1]).type(torch.int64)
        return torch.mean(self.processed_pops_trajectories[:, species_idxs, :].type(torch.float64), dim=0)

    def std(self, species_idxs: Optional[List[int]] = None, time_range: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes standard deviations of the species numbers 
        :param species_idxs: species indexes for which we compute the statistic  
        :param time_range: time indexes for which we compute the statistic 
        :return: standard deviation of the pops with species_idxs, if time_range is empty then returns the last processed 
        """
        assert self.raw_pops_trajectories is not None, "Please get the data before computing statistics"
        assert not (time_range is None and self.processed_pops_trajectories is None), "Specify time_range as no data was processed"
        if time_range is not None or self.processed_pops_trajectories is None:
            self.process_data(time_range=time_range)
        if species_idxs is None:
            species_idxs = torch.arange(0, self.processed_pops_trajectories.shape[1]).type(torch.int64)
        return torch.std(self.processed_pops_trajectories[:, species_idxs, :].type(torch.float64), dim=0)

    def coefficient_of_variation(self, species_idxs: Optional[List[int]] = None, time_range: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes coefficient of variation of the species numbers 
        :param species_idxs: species indexes for which we compute the statistic  
        :param time_range: time indexes for which we compute the statistic 
        :return: coefficient of variations 
        """
        return  self.std(species_idxs, time_range) / self.mean(species_idxs, time_range)
