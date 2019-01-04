import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from base import BaseModel


class MujocoPolicy(BaseModel):
    """
    Policy used for Mujoco environments
    3 fully connected layers with cont. output
    """
    def __init__(self, obs_dim, act_dim, batch_size=1):
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

class MujocoLstmPolicy(BaseModel):
    """
    LSTM Policy used for Mujoco environments
    2 fully connected layers, lstm layer and cont. output
    """
    def __init__(self, obs_dim, act_dim, batch_size, 
                 hidden_dim=256, num_layers=1):
        super(MujocoLstmPolicy, self).__init__()

        # assign class variables
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        # initialize 3 layers
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, 
                            self.num_layers, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_dim, self.act_dim)

        # initialize hidden variable
        self.hidden = self.init_hidden()

    def init_hidden(self, epstarts=None):

        if epstarts is None:
            hx = torch.zeros(self.num_layers, 
                                self.batch_size, self.hidden_dim).cuda()
            cx = torch.zeros(self.num_layers,
                                self.batch_size, self.hidden_dim).cuda()
        else:
            hx = self.hidden[0]
            cx = self.hidden[1]

            inds = epstarts>0
            hx[:, inds, :] = 0.0
            cx[:, inds, :] = 0.0

        return (hx,cx)

    def forward(self, obs):
        # perform forward computation
        hidden = F.relu(self.fc1(obs))
        act, self.hidden = self.lstm(hidden, self.hidden)
        act = self.fc2(act)
        return act