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
    def __init__(self, pickle_file, seq_size=30):
        """
        Args:
            pickle_file (string): path to dataset pickle file
        """
        # set sequence size
        self.seq_size = seq_size

        # load the pickle file
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)

        # get actions and observations
        seq_tails = np.where(dataset['dones'])[0]+1
        observations = dataset['observations'].astype(np.float32)
        actions = np.squeeze(dataset['actions']).astype(np.float32)

        actions = np.split(actions, seq_tails)[:-1]
        observations = np.split(observations, seq_tails)[:-1]

        # split into small sequences
        for n in range(len(actions)):
            act_array = actions[n]
            obs_array = observations[n]
            n_seqs = int(act_array.shape[0]/self.seq_size)
            traj_len = act_array.shape[0]-act_array.shape[0]%self.seq_size

            act_array = np.split(act_array[:traj_len], n_seqs)
            obs_array = np.split(obs_array[:traj_len], n_seqs)
            if act_array[-1].shape[0] < self.seq_size:
                act_array = act_array[:-1]
                obs_array = obs_array[:-1]

            actions[n] = act_array
            observations[n] = obs_array

        # assign class variables
        self.act_seqs = np.concatenate(actions, axis=0)
        self.obs_seqs = np.concatenate(observations, axis=0)

        # calculate len
        self.data_len = self.act_seqs.shape[0]

    def __getitem__(self, index):
        # return state, action pair
        return (self.obs_seqs[index,:], self.act_seqs[index,:])

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
    def __init__(self, pickle_file, seq_size, batch_size, shuffle, 
                 validation_split, num_workers):
        self.dataset = MujocoSeqDataset(pickle_file, seq_size)

        super(MujocoSeqDataLoader, self).__init__(self.dataset, batch_size, 
                                    shuffle, validation_split, num_workers)
