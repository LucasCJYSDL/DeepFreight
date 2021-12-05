import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module): ## TODO: struture fine-tune
    def __init__(self, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape()
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.critic_embed_dim)
        self.fc2 = nn.Linear(args.critic_embed_dim, args.critic_embed_dim)
        self.fc3 = nn.Linear(args.critic_embed_dim, self.n_actions)

    def forward(self, batch, max_seq_len=None, t=None):
        inputs = self._build_inputs(batch, max_seq_len, t=t)
        if self.args.cuda:
            inputs = inputs.cuda()
        bs, max_t, n_agents, vdim = inputs.shape
        if t is None:
            assert (max_seq_len is not None) and (max_t == max_seq_len)
        else:
            assert (max_seq_len) is None and (max_t == 1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # if self.rnn is not None:
        #     x = x.permute(0, 2, 1, 3).reshape(bs * n_agents, max_t, -1)
        #     x, h_out = self.rnn(x)  # h0 defaults to 0 if not provided, TODO: make explicit
        #     x = x.reshape(bs, n_agents, max_t, -1).permute(0, 2, 1, 3)
        q = self.fc3(x)
        # v = self.v_head(x)
        # q = adv - adv.mean(-1, keepdim=True).expand_as(adv) + v.expand_as(adv)
        return q

    def _build_inputs(self, batch, max_seq_len, t):
        bs = batch['o'].shape[0] ## check: batch should be torch tensor
        max_t = max_seq_len if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["s"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        # # observation
        # inputs.append(batch["o"][:, ts]) ## TODO: with ot without
        # actions (masked out by agent)
        actions = batch["onehot_a"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["onehot_a"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["onehot_a"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["onehot_a"][:, 0:1]), batch["onehot_a"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self):
        # state
        input_shape = self.args.state_shape
        # # observation
        # input_shape += self.args.obs_shape ## TODO: with ot without
        # actions and last actions
        input_shape += self.args.n_actions * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape