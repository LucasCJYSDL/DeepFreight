import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    # input_shape = obs_shape + n_actions + n_agents
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.name = 'rnn'
        ## prediction part
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on the same device and with the same type
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q, h
