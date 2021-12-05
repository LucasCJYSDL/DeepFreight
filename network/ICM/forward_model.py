import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class FwdModel(nn.Module):

    def __init__(self, args):
        super(FwdModel, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.state_shape + args.n_agents, args.icm_hidden_dim)
        self.fc2 = nn.Linear(args.icm_hidden_dim, args.icm_hidden_dim)
        # self.fc3 = nn.Linear(args.icm_hidden_dim, args.icm_hidden_dim)
        self.fc4 = nn.Linear(args.icm_hidden_dim, args.state_shape)

    def forward(self, input):

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        output = self.fc4(x)

        return output