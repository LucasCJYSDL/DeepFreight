import copy
from agents.target_agents import Target_agents, Target_agents_central
from network.mixers.qmix import QMixer
from network.mixers.qmix_central_no_hyper import QMixerCentralFF
from network.mixers.qmix_central_attention import QMixerCentralAtten
import torch as th
import time
from torch.utils.tensorboard import SummaryWriter
import os
# from collections import deque


class WQmix_learner:
    def __init__(self, mac: Target_agents, args):
        self.args = args
        self.mac = mac
        assert self.mac.is_target()

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        assert args.mixer is not None
        if args.mixer == "qmix":
            self.mixer = QMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.mixer_params = list(self.mixer.parameters())
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        self.central_mac = None
        if self.args.central_mixer == "ff":
            self.central_mixer = QMixerCentralFF(args) # Feedforward network that takes state and agent utils as input
        elif self.args.central_mixer == "atten":
            self.central_mixer = QMixerCentralAtten(args)
        else:
            raise Exception("Error with central_mixer")

        assert args.central_agent == "basic_central_agent"
        self.central_mac = Target_agents_central(args)
        assert self.central_mac.is_target()
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params += list(self.central_mac.parameters())
        self.params += list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        if args.cuda:
            self.cuda()

        self.optimiser = th.optim.RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        assert 'wqmix' in args.alg
        temp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.model_dir = args.model_dir + '/' + args.alg + '/' + temp_time
        self.log_dir = args.result_dir + '/' + args.alg + '/' + temp_time
        self.writer = SummaryWriter(self.log_dir)
        print("Current learner is {}!!!".format(args.alg))
        # self.grad_norm = 1
        # self.mixer_norm = 1
        # self.mixer_norms = deque([1], maxlen=100)

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
            agent_outs = self.mac.forward(batch, t, episode_num, self.args.epsilon)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=a).squeeze(3)  # Remove the last dim
        # print("3: ", chosen_action_qvals_agents.shape)
        chosen_action_qvals = chosen_action_qvals_agents

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            target_agent_outs = self.target_mac.forward(batch, t, episode_num, self.args.epsilon)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_a[:, :] == 1.0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_a == 1.0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
            target_max_agent_qvals = th.gather(target_mac_out[:, :], 3, cur_max_actions[:, :]).squeeze(3)
        else:
            raise Exception("Use double q")

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            agent_outs = self.central_mac.forward(batch, t, episode_num)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        # print("4: ", central_mac_out.shape)
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, \
                                                       index=a.unsqueeze(4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)  # Remove the last dim
        # print("5: ", central_chosen_action_qvals_agents.shape)
        central_target_mac_out = []
        self.target_central_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            target_agent_outs = self.target_central_mac.forward(batch, t, episode_num)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        central_target_mac_out[avail_a[:, :] == 1.0] = -9999999  # From OG deepmarl
        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3, \
                                                   cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)
        # print("6: ", central_target_max_agent_qvals.shape)
        # ---
        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, s[:, :-1])
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals[:, 1:], s[:, 1:])

        # # Calculate 1-step Q-Learning targets
        # targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        n = self.args.step_num
        if n == 1:
            # Calculate 1-step Q-Learning targets
            targets = r + gamma * (1 - done) * target_max_qvals
        else:
            # N step Q-Learning targets
            steps = batch['next_idx'].long()
            if self.args.cuda:
                steps = steps.cuda()
            indices = th.linspace(0, max_seq_length - 1, steps=max_seq_length, device=steps.device).unsqueeze(1).long()
            # n_target_max_qvals = th.gather(target_max_qvals, dim=1, index=steps + indices - 1)
            try:
                n_target_max_qvals = th.gather(target_max_qvals, dim=1, index=steps + indices - 1) ## check
            except Exception:
                print("Index Error: ", (steps + indices - 1), '\n', mask)
                n_target_max_qvals = target_max_qvals
                print("Replace with step_num 1...")
            targets = r + gamma * n_target_max_qvals * (1 - done)

        # Td-error
        td_error = (chosen_action_qvals - (targets.detach()))

        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, s[:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # QMIX loss with weighting
        ws = th.ones_like(td_error) * self.args.w
        if self.args.hysteretic_qmix:  # OW-QMIX
            ws = th.where(td_error < 0, th.ones_like(td_error) * 1, ws)  # Target is greater than current max
            w_to_use = ws.mean().item()  # For logging
        else:  # CW-QMIX
            is_max_action = (a == cur_max_actions[:, :-1]).min(dim=2)[0] ## check
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], s[:, :-1]) ## different from the paper; maybe change to central_mixer
            qtot_larger = targets > max_action_qtot
            ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error) * 1, ws)  # Target is greater than current max
            w_to_use = ws.mean().item()  # Average of ws for logging

        qmix_loss = (ws.detach() * (masked_td_error ** 2)).sum() / mask.sum()

        # The weightings for the different losses aren't used (they are always set to 1)
        loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        # Logging
        # agent_norm = 0
        # for p in self.mac_params:
        #     param_norm = p.grad.data.norm(2)
        #     agent_norm += param_norm.item() ** 2
        # agent_norm = agent_norm ** (1. / 2)
        #
        # mixer_norm = 0
        # for p in self.mixer_params:
        #     param_norm = p.grad.data.norm(2)
        #     mixer_norm += param_norm.item() ** 2
        # mixer_norm = mixer_norm ** (1. / 2)
        # self.mixer_norm = mixer_norm
        # self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.clip_norm)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if train_steps > 0 and train_steps % self.args.target_update_period == 0:
            self._update_targets()

        log_targets = (targets.detach().clone() * mask).sum() / mask.sum()
        log_grad_norm = grad_norm
        log_central_loss = central_loss
        log_qmix_loss = qmix_loss

        return {'Target Q-value': log_targets, 'Central Loss': log_central_loss, 'Qmix Loss': log_qmix_loss, 'Grad Norm': log_grad_norm}


    def get_log_dict(self):

        return {'Target Q-value': [], 'Central Loss': [], 'Qmix Loss': [], 'Grad Norm': []}

    def _update_targets(self):
        print("Updated target network......")
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())

    def cuda(self):
        self.mixer.cuda()
        self.target_mixer.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    def save_models(self, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(self.model_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        self.mac.save_models(path)
        self.central_mac.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.central_mixer.state_dict(), "{}/central_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.central_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_central_mac.load_models(path)
        # self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
        self.central_mixer.load_state_dict(th.load("{}/central_mixer.th".format(path)))
        self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path)))  ## add line
        self.target_central_mixer.load_state_dict(th.load("{}/central_mixer.th".format(path)))
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