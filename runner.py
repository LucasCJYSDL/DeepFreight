# -*- coding: utf-8 -*-

import numpy as np
from common.replay_buffer import ReplayBuffer
from agents.target_agents import Target_agents
from learners import REGISTRY as learner_REGISTRY
from network.bandits.uniform import Uniform
from network.bandits.hierarchial import EZ_agent
from utils import multi_step_TD
from environment.freight_env import Simulator
from network.ICM.ICM_learner import ICM_learner

class Runner:
    def __init__(self, env: Simulator, args):
        self.args = args
        self.env = env

        self.target_agents = Target_agents(args)
        self.replay_buffer = ReplayBuffer(self.args)
        if args.load_replay_buffer:
            self.replay_buffer.restore(args.replay_buffer_path)
            print("Loading buffers from {}".format(args.replay_buffer_path))
        self.learner = learner_REGISTRY[self.args.learner](self.target_agents, self.args)
        # self.icm_learner = ICM_learner(args)

        self.noise_generator = None
        if args.alg == 'maven':
            assert self.target_agents.get_actor_name() == 'noise_rnn'
            if args.noise_bandit:
                self.noise_generator = EZ_agent(args)
            else:
                self.noise_generator = Uniform(args)

        self.start_itr = 0
        if args.load_model:
            assert len(args.load_model_path) > 0
            print("Loading from model: {}!!!".format(args.load_model_path))
            self.learner.load_models(args.load_model_path)
            # self.icm_learner.load_models(args.load_model_path)
            if args.alg == 'maven':
                self.noise_generator.load_models(args.load_model_path)
            ## load_model_path: 'ckpt + '/' + args.alg + '/' + timestamp + '/' +idx
            alg = args.load_model_path.split('/')[-3]
            assert alg == args.alg, "Wrong model to load!"
            idx = int(args.load_model_path.split('/')[-1])
            self.start_itr = ((idx * args.save_model_period // args.train_steps)//7 + 1) * 7
            self.args.epsilon = self.args.epsilon - self.start_itr * self.args.epsilon_decay
            if self.args.epsilon < self.args.min_epsilon:
                self.args.epsilon = self.args.min_epsilon
        self.start_train_steps = self.start_itr * args.train_steps

    def generate_episode(self, episode_num, itr, evaluate=False):

        epsilon = 0 if evaluate else self.args.epsilon
        if self.args.epsilon_anneal_scale == 'episode' or (self.args.epsilon_anneal_scale == 'itr' and episode_num == 0):
            epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon
        if not evaluate:
            self.args.epsilon = epsilon

        episode_buffer = None
        if not evaluate:
            episode_buffer = {'o':            np.zeros([self.args.episode_limit + 1, self.args.n_agents, self.args.obs_shape]),
                              's':            np.zeros([self.args.episode_limit + 1, self.args.state_shape]),
                              'a':            np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a':     np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a':      np.zeros([self.args.episode_limit + 1, self.args.n_agents, self.args.n_actions]),
                              'r':            np.zeros([self.args.episode_limit, 1]),
                              'done':         np.ones([self.args.episode_limit, 1]),
                              'padded':       np.ones([self.args.episode_limit, 1]),
                              'gamma':        np.zeros([self.args.episode_limit, 1]),
                              'next_idx':     np.zeros([self.args.episode_limit, 1])}
        # roll out
        self.target_agents.init_hidden(1)
        target_last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.env.reset(episode=itr)
        state, obs_list, mask_list, unserved_packages = self.env.get_state_obs(0)

        target_noise = None
        if self.args.alg == 'maven':
            target_state = state.copy()
            target_noise = self.noise_generator.sample(target_state, test_mode=False)

        for episode in range(self.args.episode_limit):

            action_list = self.target_agents.choose_action(obs_list, mask_list, target_last_action, epsilon, \
                                                           self.replay_buffer, evaluate, noise=target_noise)
            assert len(action_list) == self.args.n_agents
            onehot_action_list = np.eye(self.args.n_actions)[action_list]
            target_last_action = np.eye(self.args.n_actions)[action_list]
            self.env.step(action_list, episode)
            # pre_next_state = self.icm_learner.sample(state, action_list)
            # state_t = state.copy()
            # action_list_t = action_list.copy()
            episode_buffer['o'][episode] = obs_list
            episode_buffer['s'][episode] = state
            episode_buffer['a'][episode] = np.reshape(action_list, [self.args.n_agents, 1])
            episode_buffer['onehot_a'][episode] = onehot_action_list
            episode_buffer['avail_a'][episode] = mask_list
            episode_buffer['padded'][episode] = [0.]

            state, obs_list, mask_list, unserved_packages = self.env.get_state_obs(episode+1)
            # cur_rwd = np.linalg.norm(state-pre_next_state) ** 2
            # print("1: ", state)
            # print("2: ", pre_next_state)
            # print("3: ", cur_rwd)
            # state_next_t = state
            # self.icm_learner.update_icm_buffer(state_t, action_list_t, state_next_t)
            # episode_buffer['r'][episode] = [0. + self.args.icm_factor * cur_rwd]  ## need further update
            episode_buffer['r'][episode] = [0.]
            done = self.env.is_done(mask_list, unserved_packages)
            episode_buffer['done'][episode] = [done]  ## need further update
            # print(mask_list)
            if done:
                break

        round_num = episode + 1
        print("The total round number is {}.".format(round_num))
        assert episode_buffer['done'][round_num-1][0]
        episode_buffer['o'][round_num] = episode_buffer['o'][round_num-1].copy()
        episode_buffer['s'][round_num] = episode_buffer['s'][round_num-1].copy()
        episode_buffer['avail_a'][round_num] = episode_buffer['avail_a'][round_num-1].copy()

        # print(episode_buffer['a'])
        print("Matching ......")
        self.env.match_delivery(is_final=True)
        for truck in self.env.truck_list:
            truck.show_info()
        print("Packing ......")
        self.env.final_pack()
        emp_ratio, rwd, req, fuel = self.env.get_reward()
        episode_buffer['r'][round_num-1][0] += rwd
        print('EMPTY: {%.3f}/ RWD: {%.3f}/ REQ_MISS: {%d}/ FUEL: {%.3f}' % (emp_ratio, rwd, req, fuel))

        if self.args.alg == 'maven':
            self.noise_generator.update_returns(target_state, target_noise, episode_buffer['r'][round_num - 1][0])
            episode_buffer['noise'] = np.array(target_noise)

        episode_buffer = multi_step_TD(self.args, episode_buffer, round_num)
        print(episode_buffer['r'])

        ## training ICM
        # self.icm_learner.update()

        return episode_buffer, emp_ratio, rwd, float(req), fuel

    def run(self):
        train_steps = self.start_train_steps
        print("train_steps: ", train_steps)
        for itr in range(self.start_itr, self.args.n_itr):
            print("##########################{}##########################".format(itr))

            empty_ratios, rwds, reqs, fuels = [], [], [], []
            episode_batch, empty_ratio, rwd, req, fuel = self.generate_episode(0, itr)
            empty_ratios.append(empty_ratio)
            rwds.append(rwd)
            reqs.append(req)
            fuels.append(fuel)
            for key in episode_batch.keys():
                episode_batch[key] = np.array([episode_batch[key]])
            if self.args.alg == 'coma':
                assert (self.args.n_episodes > 1) and (self.args.n_episodes == self.args.batch_size == self.args.buffer_size) \
                and self.args.train_steps == 1, "COMA should be online learning!!!"
            for e in range(1, self.args.n_episodes):
                episode, empty_ratio, rwd, req, fuel = self.generate_episode(e, itr)
                empty_ratios.append(empty_ratio)
                rwds.append(rwd)
                reqs.append(req)
                fuels.append(fuel)
                for key in episode.keys():
                    episode[key] = np.array([episode[key]])
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.replay_buffer.store(episode_batch)
            if not self.replay_buffer.can_sample(self.args.initial_buffer_size):
                print("No enough episodes!!!")
                continue

            log_dict = self.learner.get_log_dict()
            for _ in range(self.args.train_steps):
                batch = self.replay_buffer.sample(self.args.batch_size)
                max_episode_len = self.target_agents.get_max_episode_len(batch)
                for key in batch.keys():
                    if key == 'noise':
                        continue
                    if key in ['o', 's', 'avail_a']:
                        batch[key] = batch[key][:, :max_episode_len+1]
                    else:
                        batch[key] = batch[key][:, :max_episode_len]
                log_info = self.learner.train(batch, max_episode_len, train_steps)
                for key in log_dict.keys():
                    assert key in log_info, key
                    log_dict[key].append(log_info[key])
                if train_steps > 0 and train_steps % self.args.save_model_period == 0:
                    print("Saving the models!")
                    self.learner.save_models(train_steps)
                    save_dir = self.learner.get_save_dir()
                    # self.icm_learner.save_models(save_dir, train_steps)
                    if self.args.alg == 'maven':
                        self.noise_generator.save_models(save_dir, train_steps)
                train_steps += 1

            print("Log to the tensorboard!")
            self.learner.log_info(empty_ratios, rwds, reqs, fuels, log_dict, itr)

            if (itr > 0) and (itr%self.args.save_buffer_period == 0):
                print("Saving the replay buffer!")
                self.replay_buffer.save(self.args.replay_buffer_path)
