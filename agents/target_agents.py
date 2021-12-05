import torch as th
from agents.agents import Agents
from network.actors import REGISTRY as actor_REGISTRY

class Target_agents(Agents):

    def __init__(self, args):
        super().__init__(args, True)

    def parameters(self):
        return self.actor.parameters()

    def get_max_episode_len(self, batch):
        max_len = 0
        for episode in batch['padded']:
            length = episode.shape[0] - int(episode.sum()) ## check
            if length > max_len:
                max_len = length
        return int(max_len)

    def forward(self, batch, t, episode_num, epsilon, test_mode = False, noise = False):
        agent_inputs = self._build_inputs(batch, t, episode_num)
        if noise:
            assert self.args.alg == 'maven'
            noise_inputs = self._build_noise_input(batch)
        if self.args.cuda:
            agent_inputs = agent_inputs.cuda()
            if noise:
                noise_inputs = noise_inputs.cuda()
        if noise:
            agent_outs, self.hidden_states = self.actor(agent_inputs, self.hidden_states, noise_inputs)
        else:
            agent_outs, self.hidden_states = self.actor(agent_inputs, self.hidden_states)
        if self.args.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                agent_outs = ((1 - epsilon) * agent_outs + th.ones_like(agent_outs) * epsilon/epsilon_action_num)
        return agent_outs.view(episode_num, self.n_agents, -1)

    def _build_inputs(self, batch, t, episode_num):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = episode_num
        inputs = []
        inputs.append(batch["o"][:, t])  # b1av
        if self.args.last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["onehot_a"][:, t]))
            else:
                inputs.append(batch["onehot_a"][:, t-1])
        if self.args.reuse_network:
            inputs.append(th.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_noise_input(self, batch):
        agent_ids = th.eye(self.args.n_agents).repeat(batch["noise"].shape[0], 1)
        noise_repeated = batch["noise"].repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)
        noise_inputs = th.cat([noise_repeated, agent_ids], dim=-1)
        return noise_inputs

    def load_state(self, other_mac):
        self.actor.load_state_dict(other_mac.actor.state_dict())


class Target_agents_central(Agents):

    def __init__(self, args):

        self.args = args
        self.n_agents = args.n_agents
        self.target = True

        input_shape = self.args.obs_shape
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents
        self.actor = actor_REGISTRY[self.args.central_actor](input_shape, self.args) ## check
        if args.cuda:
            self.actor.cuda()
        self.hidden_states = None

    def parameters(self):
        return self.actor.parameters()

    def forward(self, batch, t, episode_num, test_mode=False):
        agent_inputs = self._build_inputs(batch, t, episode_num)
        if self.args.cuda:
            agent_inputs = agent_inputs.cuda()
        agent_outs, self.hidden_states = self.actor(agent_inputs, self.hidden_states)

        return agent_outs.view(episode_num, self.n_agents, self.args.n_actions, -1)

    def _build_inputs(self, batch, t, episode_num):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = episode_num
        inputs = []
        inputs.append(batch["o"][:, t])  # b1av
        if self.args.last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["onehot_a"][:, t]))
            else:
                inputs.append(batch["onehot_a"][:, t-1])
        if self.args.reuse_network:
            inputs.append(th.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def load_state(self, other_mac):
        self.actor.load_state_dict(other_mac.actor.state_dict())

    def save_models(self, path):
        th.save(self.actor.state_dict(), "{}/central_agent.th".format(path))

    def load_models(self, path):
        # self.actor.load_state_dict(th.load("{}/central_agent.th".format(path), map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(th.load("{}/central_agent.th".format(path)))

