import pickle
import numpy as np
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms


class MujocoDataset(Dataset):
    """
    Dataset class for mujoco expert datasets
    """
    def __init__(self, pickle_file):
        """
        Args:
            pickle_file (string): path to dataset pickle file
        """
        # load the pickle file
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)

        # get actions and observations
        self.actions = dataset['actions'].astype(np.float32)
        self.observations = dataset['observations'].astype(np.float32)

        # calculate len
        self.data_len = self.actions.shape[0]

    def __getitem__(self, index):
        # return state, action pair
        return (self.observations[index,:], self.actions[index,0,:])

    def __len__(self):
        # return dataset length
        return self.data_len

class MujocoSeqDataset(Dataset):
    """
    Dataset class for mujoco expert datasets
    """
    def __init__(self, pickle_file, seq_size=250, start_ind=0):
        """
        Args:
            pickle_file (string): path to dataset pickle file
        """
        # set sequence size
        self.seq_size = seq_size

        # set the start index
        self.start_ind = start_ind

        # load the pickle file
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)

        # get actions and observations
        seq_tails = np.where(dataset['dones'])[0]+1
        observations = dataset['observations'].astype(np.float32)
        actions = np.squeeze(dataset['actions']).astype(np.float32)

        actions = np.split(actions, seq_tails)[:-1]
        observations = np.split(observations, seq_tails)[:-1]

        # assign class variables
        self.act_seqs = actions
        self.obs_seqs = observations

        # calculate len
        self.data_len = len(self.act_seqs)

    def __getitem__(self, index):
        # return state, action pair
        return (self.obs_seqs[index][self.start_ind:self.start_ind+self.seq_size,:], 
                self.act_seqs[index][self.start_ind:self.start_ind+self.seq_size,:])

    def __len__(self):
        # return dataset length
        return self.data_len

class MujocoDataLoader(BaseDataLoader):
    """
    Data loader for mujoco expert datasets
    """
    def __init__(self, pickle_file, batch_size, shuffle, 
                 validation_split, num_workers):
        self.dataset = MujocoDataset(pickle_file)

        super(MujocoDataLoader, self).__init__(self.dataset, batch_size, 
                                shuffle, validation_split, num_workers)

class MujocoSeqDataLoader(BaseDataLoader):
    """
    Data loader for mujoco expert datasets
    """
    def __init__(self, pickle_file, batch_size, shuffle, 
                 validation_split, num_workers, seq_size=250, start_ind=0):
        self.dataset = MujocoSeqDataset(pickle_file, seq_size, start_ind)

        super(MujocoSeqDataLoader, self).__init__(self.dataset, batch_size, 
                                    shuffle, validation_split, num_workers)
