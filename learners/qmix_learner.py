import copy
from agents.target_agents import Target_agents
from network.mixers.qmix import QMixer
import torch as th
import time
from torch.utils.tensorboard import SummaryWriter
import os

class Qmix_learner:
    def __init__(self, mac: Target_agents, args):
        self.args = args
        self.mac = mac
        assert self.mac.is_target()
        self.params = list(mac.parameters())

        assert args.mixer is not None
        if args.mixer == "qmix":
            self.mixer = QMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        if args.cuda:
            self.cuda()

        self.optimiser = th.optim.RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps) ## check

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        assert args.alg == 'qmix'
        temp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.model_dir = args.model_dir + '/' + args.alg + '/' + temp_time
        self.log_dir = args.result_dir + '/' + args.alg + '/' + temp_time
        self.writer = SummaryWriter(self.log_dir)
        print("Current learner is {}!!!".format(args.alg))

    def train(self, batch, max_seq_length, train_steps):

        episode_num = batch['o'].shape[0]
        for key in batch.keys():
            if key == 'a':
                batch[key] = th.LongTensor(batch[key]) ##check
            else:
                batch[key] = th.Tensor(batch[key])
        s, a, r, avail_a, done, gamma = batch['s'], batch['a'], batch['r'], batch['avail_a'], batch['done'], batch['gamma']
        mask = 1 - batch["padded"].float()
        if self.args.cuda:
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            gamma = gamma.cuda()

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            agent_outs = self.mac.forward(batch, t, episode_num, self.args.epsilon) ## check: epsilon
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=a).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            target_agent_outs = self.target_mac.forward(batch, t, episode_num, self.args.epsilon)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_a[:, 1:] == 1.0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            double_mac_out_detach = mac_out.clone().detach()
            double_mac_out_detach[avail_a == 1.0] = -9999999
            cur_max_actions = double_mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, s[:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, s[:, 1:])

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
            # print("Index: ", (steps + indices - 1)[0], '\n', mask[0])
            try:
                n_target_max_qvals = th.gather(target_max_qvals, dim=1, index=steps + indices - 1) ## check
            except Exception:
                print("Index Error: ", (steps + indices - 1), '\n', mask)
                n_target_max_qvals = target_max_qvals
                print("Replace with step_num 1...")
            targets = r + gamma * n_target_max_qvals * (1 - done)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error) ## check
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.clip_norm)
        self.optimiser.step()

        if train_steps > 0 and train_steps % self.args.target_update_period == 0:
            self._update_targets()

        log_targets = (targets.detach().clone() * mask).sum() / mask.sum()
        log_grad_norm = grad_norm
        log_loss = loss

        return {'Target Q-value': log_targets, 'Qmix Loss': log_loss, 'Grad Norm': log_grad_norm}

    def get_log_dict(self):

        return {'Target Q-value': [], 'Qmix Loss': [], 'Grad Norm': []}

    def _update_targets(self):
        print("Updated target network......")
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        # self.mac.cuda()
        # self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(self.model_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            # self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path))) ## add line
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
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