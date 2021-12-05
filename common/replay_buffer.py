# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.current_idx = 0
        self.size = 0
        self.T = 0
        self.Ta = np.zeros((self.args.n_actions, ))

        self.buffer = {
            'o': np.zeros([self.args.buffer_size, self.args.episode_limit+1, self.args.n_agents, self.args.obs_shape]),
            's': np.zeros([self.args.buffer_size, self.args.episode_limit+1, self.args.state_shape]),
            'a': np.zeros([self.args.buffer_size, self.args.episode_limit, self.args.n_agents, 1]),
            'onehot_a': np.zeros([self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'avail_a': np.zeros([self.args.buffer_size, self.args.episode_limit+1, self.args.n_agents, self.args.n_actions]),
            'r': np.zeros([self.args.buffer_size, self.args.episode_limit, 1]),
            'done': np.ones([self.args.buffer_size, self.args.episode_limit, 1]),
            'padded': np.ones([self.args.buffer_size, self.args.episode_limit, 1]),
            'gamma': np.zeros([self.args.buffer_size, self.args.episode_limit, 1]),
            'next_idx': np.zeros([self.args.buffer_size, self.args.episode_limit, 1])
        }
        if args.alg == 'maven':
            self.buffer['noise'] = np.zeros([self.args.buffer_size, self.args.noise_dim])

    def can_sample(self, batch_size):

        return self.size >= batch_size

    def sample(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        assert self.can_sample(batch_size)
        temp_buffer = {}
        # idxes = np.random.randint(0, self.size, batch_size) ## check
        idxes = np.random.choice(self.size, batch_size, replace=False)

        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idxes]

        return temp_buffer

    def store(self, episode_batch):

        num = episode_batch['o'].shape[0]
        idxes = self.get_idxes(num)

        for key in self.buffer.keys():
            self.buffer[key][idxes] = episode_batch[key]

        self.size = min(self.args.buffer_size, self.size + num)

        for k in range(num):
            for i in range(self.args.episode_limit):
                if episode_batch['padded'][k][i][0] > 0:
                    break
                for j in range(self.args.n_agents):
                    self.T += 1
                    action = episode_batch['a'][k][i][j][0]
                    # print("action: ", action)
                    self.Ta[int(action)] += 1

    def get_idxes(self, num):

        if self.current_idx + num <= self.args.buffer_size:
            idxes = np.arange(self.current_idx, self.current_idx + num)
            self.current_idx += num

        elif self.current_idx < self.args.buffer_size:
            overflow = num - (self.args.buffer_size - self.current_idx)
            idxes = np.concatenate([np.arange(self.current_idx, self.args.buffer_size),
                                    np.arange(0, overflow)])
            self.current_idx = overflow

        else:
            idxes = np.arange(0, num)
            self.current_idx = num

        return idxes

    def get_size(self):

        return self.size

    def get_T(self):

        if self.T > 0:
            return self.T
        else:
            return 1

    def get_TA(self, action):

        if self.Ta[action] > 0:
            return self.Ta[action]
        else:
            return 1

    def save(self, file_name):
        temp_buffer = self.buffer.copy()
        temp_buffer['T'] = self.T
        temp_buffer['Ta'] = self.Ta

        with open(file_name, 'wb') as f:
            pickle.dump(temp_buffer, f)

    def restore(self, file_name):
        with open(file_name, 'rb') as f:
            temp_buffer = pickle.load(f)
        for k in self.buffer.keys():
            self.buffer[k] = temp_buffer[k]
        self.size = temp_buffer['o'].shape[0]
        print("The size of the replay buffer is: ", self.size)
        self.T = temp_buffer['T']
        self.Ta = temp_buffer['Ta']


