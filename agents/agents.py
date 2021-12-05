# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
from network.actors import REGISTRY as actor_REGISTRY
from torch.distributions import Categorical

class Agents:
    def __init__(self, args, is_target):
        self.args = args
        self.n_agents = args.n_agents
        self.target = is_target

        input_shape = self.args.obs_shape
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents
        self.actor = actor_REGISTRY[self.args.actor](input_shape, self.args)
        if args.cuda:
            self.actor.cuda()
        self.hidden_states = None

    def is_target(self):
        return self.target

    def get_actor_name(self):
        return self.actor.name

    def init_hidden(self, batch_size):
        self.hidden_states = self.actor.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def save_models(self, path):
        torch.save(self.actor.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        # self.actor.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load("{}/agent.th".format(path)))

    def _get_output(self, obs, last_action, avail_actions_mask, epsilon, evaluate=False, noise=None):
        # agent index
        onehot_agent_idx = np.eye(self.n_agents)

        if self.args.last_action:
            obs = np.hstack((obs, last_action))
        if self.args.reuse_network:
            obs = np.hstack((obs, onehot_agent_idx))
        hidden_state = self.hidden_states
        obs = torch.Tensor(obs)  ## check
        avail_actions_mask = torch.Tensor(avail_actions_mask)  ## check

        if noise is not None:
            assert self.args.alg == 'maven'
            noise = np.array([noise]).repeat(self.args.n_agents, axis=0)
            noise = np.hstack((noise, onehot_agent_idx))
            noise = torch.Tensor(noise)

        if self.args.cuda:
            obs = obs.cuda()
            hidden_state = hidden_state.cuda() ## check: may be not required
            if noise is not None:
                noise = noise.cuda()

        if noise is not None:
            actor_output, self.hidden_states = self.actor(obs, hidden_state, noise)
        else:
            actor_output, self.hidden_states = self.actor(obs, hidden_state)

        if self.args.agent_output_type == "q_value":
            # mask out
            # qsa[avail_actions_mask == 1.0] = -float("inf")  ##check
            actor_output[avail_actions_mask == 1.0] = -99999999.0
            qsa_array = actor_output.clone().detach().cpu().numpy()

            ## convert the q_value to probability
            if evaluate:
                temperature = self.args.boltzmann_coe * self.args.min_epsilon
            else:
                temperature = self.args.boltzmann_coe * epsilon
            boltzmann = np.exp(qsa_array / temperature)
            prob = boltzmann / np.expand_dims(boltzmann.sum(axis=1), axis=1)

        elif self.args.agent_output_type == "pi_logits":
            actor_output = torch.nn.functional.softmax(actor_output, dim=-1) ## check
            if not evaluate: ## fine tune: with or without the noise
                epsilon_action_num = actor_output.size(-1)
                actor_output = ((1 - epsilon) * actor_output + torch.ones_like(actor_output) * epsilon / epsilon_action_num)
            actor_output[avail_actions_mask == 1.0] = 0.0
            prob = actor_output.clone().detach().cpu().numpy() ## check

        else:
            raise NotImplementedError

        return actor_output, prob

    def choose_action(self, obs, avail_actions_mask, last_action, epsilon, replay_buffer, evaluate=False, noise=None):

        # available actions

        actor_output, prob = self._get_output(obs, last_action, avail_actions_mask, epsilon, evaluate, noise)

        if evaluate:
            return np.argmax(actor_output.clone().detach().cpu().numpy(), axis=-1)

        if self.args.exploration == 'epsilon':
            assert self.args.agent_output_type == "q_value"
            action_output = actor_output.clone().detach().cpu().numpy()
            action_list = []
            for idx in range(self.args.n_agents):
                temp_action_output = action_output[idx]
                temp_action_mask = avail_actions_mask[idx]
                temp_avail_action = np.nonzero(1 - temp_action_mask)[0]
                if np.random.uniform() < epsilon:
                    action_list.append(np.random.choice(temp_avail_action))
                else:
                    action_list.append(np.argmax(temp_action_output))
            return np.array(action_list)

        elif self.args.exploration == 'ucb1':
            assert self.args.agent_output_type == "q_value"
            action_output = actor_output.clone().detach().cpu().numpy()
            ucb_term = np.array([self.args.ucb_coe * math.sqrt(2*math.log(replay_buffer.get_T())/replay_buffer.get_TA(i)) for i in range(self.args.n_actions)])
            temp_qsa = action_output + ucb_term
            return np.argmax(temp_qsa, axis=-1)

        elif self.args.exploration == 'boltzmann':
            assert self.args.agent_output_type == "q_value"
            cumProb_boltzmann = np.cumsum(prob, axis=1)

            action_list = []
            for cb in cumProb_boltzmann:
                try:
                    act = np.where(cb > np.random.rand(1))[0][0]
                except Exception:
                    # print("Index error occurs...")
                    # print(cb)
                    act = 0
                action_list.append(act)
            action_list = np.array(action_list)
            return action_list

        elif self.args.exploration == 'multinomial':
            assert self.args.agent_output_type == "pi_logits"
            action = Categorical(actor_output).sample()
            return action.clone().detach().cpu().numpy()

        else:
            raise NotImplementedError




