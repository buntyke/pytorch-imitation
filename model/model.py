import torch.nn as nn
from base import BaseModel
import torch.nn.functional as F


class MujocoPolicy(BaseModel):
    """
    Policy used for Mujoco environments
    3 fully connected layers with cont. output
    """
    def __init__(self, obs_dim, act_dim):
        super(MujocoPolicy, self).__init__()

        # initialize 3 fully connected layers
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)

    def forward(self, obs):
        # perform forward computation
        act = F.relu(self.fc1(obs))
        act = F.relu(self.fc2(act))
        act = self.fc3(act)
        return act