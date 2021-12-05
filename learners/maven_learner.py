import copy
from agents.target_agents import Target_agents
from network.mixers.noise_mix import NoiseQMixer
import torch as th
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os


class MAVENLearner:
    def __init__(self, mac: Target_agents, args):
        self.args = args
        self.mac = mac
        assert self.mac.is_target()
        self.params = list(mac.parameters())

        self.mixer = None
        assert args.mixer is not None
        if args.mixer == "noise_qmix":
            self.mixer = NoiseQMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        discrim_input = np.prod(self.args.state_shape) + self.args.n_agents * self.args.n_actions

        if self.args.rnn_discrim:
            self.rnn_agg = RNNAggregator(discrim_input, args)
            self.discrim = Discrim(args.rnn_agg_size, self.args.noise_dim, args)
            self.params += list(self.discrim.parameters())
            self.params += list(self.rnn_agg.parameters())
        else:
            self.discrim = Discrim(discrim_input, self.args.noise_dim, args)
            self.params += list(self.discrim.parameters())
        self.discrim_loss = th.nn.CrossEntropyLoss(reduction="none")

        if args.cuda:
            self.cuda()

        self.optimiser = th.optim.RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)

        assert args.alg == 'maven'
        temp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.model_dir = args.model_dir + '/' + args.alg + '/' + temp_time
        self.log_dir = args.result_dir + '/' + args.alg + '/' + temp_time
        self.writer = SummaryWriter(self.log_dir)
        print("Current learner is {}!!!".format(args.alg))

    def train(self, batch, max_seq_length, train_steps):
        # Get the relevant quantities
        episode_num = batch['o'].shape[0]
        for key in batch.keys():
            if key == 'a':
                batch[key] = th.LongTensor(batch[key])  ##check
            else:
                batch[key] = th.Tensor(batch[key])
        s, a, r, avail_a, done, gamma = batch['s'], batch['a'], batch['r'], batch['avail_a'], batch['done'], batch['gamma']
        mask = 1 - batch["padded"].float()
        noise = batch["noise"][:].unsqueeze(1).repeat(1, r.shape[1], 1)

        if self.args.cuda:
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            gamma = gamma.cuda()
            noise = noise.cuda()

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            agent_outs = self.mac.forward(batch, t, episode_num, self.args.epsilon, noise=True)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=a).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            target_agent_outs = self.target_mac.forward(batch, t, episode_num, self.args.epsilon, noise=True)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_a[:, 1:] == 1.0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            # mac_out[avail_a == 1.0] = -9999999
            # cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            # print("6: ", cur_max_actions.shape)
            # target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            double_mac_out_detach = mac_out.clone().detach()
            double_mac_out_detach[avail_a == 1.0] = -9999999
            cur_max_actions = double_mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals_tot = self.mixer(chosen_action_qvals, s[:, :-1], noise)
        target_max_qvals = self.target_mixer(target_max_qvals, s[:, 1:], noise)

        # Discriminator
        double_mac_out_detach = mac_out.clone().detach()
        double_mac_out_detach[avail_a == 1.0] = -9999999
        q_softmax_actions = th.nn.functional.softmax(double_mac_out_detach[:, :-1], dim=3)

        if self.args.hard_qs:
            maxs = th.max(mac_out[:, :-1], dim=3, keepdim=True)[1]
            zeros = th.zeros_like(q_softmax_actions)
            zeros.scatter_(dim=3, index=maxs, value=1)
            q_softmax_actions = zeros

        q_softmax_agents = q_softmax_actions.reshape(q_softmax_actions.shape[0], q_softmax_actions.shape[1], -1)
        # print("7: ", q_softmax_agents.shape)
        states = s[:, :-1]
        state_and_softactions = th.cat([q_softmax_agents, states], dim=2)

        if self.args.rnn_discrim:
            h_to_use = th.zeros(size=(batch.batch_size, self.args.rnn_agg_size)).to(states.device)
            hs = th.ones_like(h_to_use)
            for t in range(batch.max_seq_length - 1):
                hs = self.rnn_agg(state_and_softactions[:, t], hs)
                for b in range(batch.batch_size):
                    if t == batch.max_seq_length - 2 or (mask[b, t] == 1 and mask[b, t+1] == 0):
                        # This is the last timestep of the sequence
                        h_to_use[b] = hs[b]
            s_and_softa_reshaped = h_to_use
        else:
            s_and_softa_reshaped = state_and_softactions.reshape(-1, state_and_softactions.shape[-1])


        discrim_prediction = self.discrim(s_and_softa_reshaped)

        # Cross-Entropy
        target_repeats = 1
        if not self.args.rnn_discrim:
            target_repeats = q_softmax_actions.shape[1]
        discrim_target = batch["noise"][:].long().detach().max(dim=1)[1].unsqueeze(1).repeat(1, target_repeats).reshape(-1)
        # print("8: ", discrim_target.shape, " ", discrim_prediction.shape)
        if self.args.cuda:
            discrim_target = discrim_target.cuda()
        discrim_loss = self.discrim_loss(discrim_prediction, discrim_target)

        if self.args.rnn_discrim:
            averaged_discrim_loss = discrim_loss.mean()
        else:
            masked_discrim_loss = discrim_loss * mask.reshape(-1)
            averaged_discrim_loss = masked_discrim_loss.sum() / mask.sum()

        # Calculate 1-step Q-Learning targets
        # targets = r + self.args.gamma * (1 - done) * target_max_qvals

        n = self.args.step_num
        if n==1:
            # Calculate 1-step Q-Learning targets
            targets = r + gamma * (1 - done) * target_max_qvals
        else:
            # N step Q-Learning targets
            steps = batch['next_idx'].long()
            if self.args.cuda:
                steps = steps.cuda()
            indices = th.linspace(0, max_seq_length - 1, steps=max_seq_length, device=steps.device).unsqueeze(1).long()
            n_target_max_qvals = th.gather(target_max_qvals, dim=1, index=steps + indices - 1) ## check
            targets = r + gamma * n_target_max_qvals * (1 - done)

        # Td-error
        td_error = (chosen_action_qvals_tot - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        agent_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = agent_loss + self.args.mi_loss * averaged_discrim_loss
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.clip_norm)
        self.optimiser.step()

        if train_steps > 0 and train_steps % self.args.target_update_period == 0:
            self._update_targets()

        log_info = {}

        log_info["grad_norm"] = grad_norm
        mask_elems = mask.sum().item()
        log_info["target_mean"] = (targets * mask).sum().item()/mask_elems
        log_info["discrim_loss"] = averaged_discrim_loss.item()
        log_info["agent_loss"] = agent_loss.item()

        return log_info

    def get_log_dict(self):

        return {"grad_norm": [], "target_mean": [], "discrim_loss": [], "agent_loss": []}

    def _update_targets(self):
        print("Updated target network......")
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.discrim.cuda()
        if self.args.rnn_discrim:
            self.rnn_agg.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(self.model_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        self.mac.save_models(path)

        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.discrim.state_dict(), "{}/discrim.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)

        # self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
        self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path))) ## add line
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.discrim.load_state_dict(th.load("{}/discrim.th".format(path)))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path)))

    def log_info(self, empty_ratios, rwds, reqs, fuels, log_dict, itr):

        empty_ratios = th.tensor(empty_ratios)
        rwds = th.tensor(rwds)
        reqs = th.tensor(reqs)
        fuels = th.tensor(fuels)
        self.writer.add_scalar('Empty Ratio', empty_ratios.mean(), itr)
        self.writer.add_scalar('Reward', rwds.mean(), itr)
        self.writer.add_scalar('Unfinished Requests', reqs.mean(), itr)
        self.writer.add_scalar('Fuel Consumption', fuels.mean(), itr)
        for key in log_dict.keys():
            log_term = th.tensor(log_dict[key])
            self.writer.add_scalar(key, log_term.mean(), itr)
        self.writer.flush()

    def get_save_dir(self):

        return self.model_dir

class Discrim(th.nn.Module):

    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.args = args
        layers = [th.nn.Linear(input_size, self.args.discrim_size), th.nn.ReLU()]
        for _ in range(self.args.discrim_layers - 1):
            layers.append(th.nn.Linear(self.args.discrim_size, self.args.discrim_size))
            layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(self.args.discrim_size, output_size))
        self.model = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNAggregator(th.nn.Module):

    def __init__(self, input_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        output_size = args.rnn_agg_size
        self.rnn = th.nn.GRUCell(input_size, output_size)

    def forward(self, x, h):
        return self.rnn(x, h)
