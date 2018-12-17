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

class MujocoDataLoader(BaseDataLoader):
    """
    Data loader for mujoco expert datasets
    """
    def __init__(self, pickle_file, batch_size, shuffle, 
                 validation_split, num_workers):
        self.pickle_file = pickle_file
        self.dataset = MujocoDataset(self.pickle_file)

        super(MujocoDataLoader, self).__init__(self.dataset, batch_size, shuffle,
                                               validation_split, num_workers)