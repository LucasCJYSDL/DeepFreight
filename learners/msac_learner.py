import copy
from agents.target_agents import Target_agents, Target_agents_central
from network.mixers.qmix_central_no_hyper import QMixerCentralFF
import torch as th
from torch.distributions import Categorical
import time
from torch.utils.tensorboard import SummaryWriter
import os

class SACQLearner:
    def __init__(self, mac: Target_agents, args):
        self.args = args
        self.mac = mac
        assert self.mac.is_target()

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.mixer = None
        assert args.mixer is None

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        self.central_mac = None
        assert self.args.central_mixer == "ff"
        self.central_mixer = QMixerCentralFF(args)
        assert args.central_agent == "basic_central_agent"
        self.central_mac = Target_agents_central(args) # Groups aren't used in the CentralBasicController. Little hacky
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params += list(self.central_mac.parameters())
        self.params += list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

        if args.cuda:
            self.cuda()

        self.optimiser = th.optim.RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        assert args.alg == 'msac'
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
        if self.args.cuda:
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            gamma = gamma.cuda()

        # Current policies
        mac_out = []
        self.mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            agent_outs = self.mac.forward(batch, t, episode_num, self.args.epsilon)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_a == 1.0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_a == 1.0] = 0
        mac_out[(mac_out.sum(dim=-1, keepdim=True) == 0).expand_as(mac_out)] = 1 # Set any all 0 probability vectors to all 1s. They will be masked out later, but still need to be sampled.

        # Target policies
        target_mac_out = []
        self.target_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            target_agent_outs = self.target_mac.forward(batch, t, episode_num, self.args.epsilon)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions, renormalise (as in action selection)
        target_mac_out[avail_a == 1.0] = 0
        target_mac_out = target_mac_out/target_mac_out.sum(dim=-1, keepdim=True)
        target_mac_out[avail_a == 1.0] = 0
        target_mac_out[(target_mac_out.sum(dim=-1, keepdim=True) == 0).expand_as(target_mac_out)] = 1 # Set any all 0 probability vectors to all 1s. They will be masked out later, but still need to be sampled.

        # Sample actions
        sampled_actions = Categorical(mac_out).sample().long() # with noise
        sampled_target_actions = Categorical(target_mac_out).sample().long() # with noise

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            agent_outs = self.central_mac.forward(batch, t, episode_num)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        # Actions chosen from replay buffer
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, \
                                                       index=a.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(episode_num)
        for t in range(max_seq_length+1):
            target_agent_outs = self.target_central_mac.forward(batch, t, episode_num)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        central_target_action_qvals_agents = th.gather(central_target_mac_out[:,:], 3, \
                                                       sampled_target_actions[:,:].unsqueeze(3).unsqueeze(4)\
                                                        .repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # ---

        critic_bootstrap_qvals = self.target_central_mixer(central_target_action_qvals_agents[:,1:], s[:,1:])

        target_chosen_action_probs = th.gather(target_mac_out, dim=3, index=sampled_target_actions.unsqueeze(3)).squeeze(dim=3)
        target_policy_logs = th.log(target_chosen_action_probs).sum(dim=2, keepdim=True) # Sum across agents
        # Calculate 1-step Q-Learning targets
        # targets = r + self.args.gamma * (1 - done) * \
        #           (critic_bootstrap_qvals - self.args.entropy_temp * target_policy_logs[:,1:])

        n = self.args.step_num
        if n == 1:
            # Calculate 1-step Q-Learning targets
            targets = r + gamma * (1 - done) * (critic_bootstrap_qvals - self.args.entropy_temp * target_policy_logs[:,1:])
        else:
            # N step Q-Learning targets
            steps = batch['next_idx'].long()
            if self.args.cuda:
                steps = steps.cuda()
            indices = th.linspace(0, max_seq_length - 1, steps=max_seq_length, device=steps.device).unsqueeze(1).long()
            n_critic_bootstrap_qvals = th.gather(critic_bootstrap_qvals, dim=1, index=steps + indices - 1)
            n_target_policy_logs = th.gather(target_policy_logs[:,1:], dim=1, index=steps + indices - 1)
            targets = r + gamma * (n_critic_bootstrap_qvals - self.args.entropy_temp * n_target_policy_logs) * (1 - done)

        # Training Critic
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, s[:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum() ## why not central_mask

        # Actor Loss
        central_sampled_action_qvals_agents = th.gather(central_mac_out[:, :-1], 3, \
                                                        sampled_actions[:, :-1].unsqueeze(3).unsqueeze(4) \
                                                        .repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)
        central_sampled_action_qvals = self.central_mixer(central_sampled_action_qvals_agents, s[:,:-1]).repeat(1,1,self.args.n_agents)
        sampled_action_probs = th.gather(mac_out, dim=3, index=sampled_actions.unsqueeze(3)).squeeze(3)
        policy_logs = th.log(sampled_action_probs)[:,:-1]
        actor_mask = mask.expand_as(policy_logs)
        actor_loss = ((policy_logs * (self.args.entropy_temp * (policy_logs + 1) - central_sampled_action_qvals).detach()) * actor_mask).sum()/actor_mask.sum()

        loss = self.args.actor_loss * actor_loss + self.args.central_loss * central_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.clip_norm)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if train_steps > 0 and train_steps % self.args.target_update_period == 0:
            self._update_targets()

        log_info = {}
        log_info["actor_loss"] = actor_loss.item()
        log_info["grad_norm"] = grad_norm
        mask_elems = mask.sum().item()
        log_info["target_mean"] = (targets * mask).sum().item()/mask_elems
        log_info["central_loss"] = central_loss.item()

        return log_info

    def get_log_dict(self):

        return {"actor_loss": [], "grad_norm": [], "target_mean": [], "central_loss": []}

    def _update_targets(self):
        print("Updated target network......")
        self.target_mac.load_state(self.mac)
        self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())

    def cuda(self):
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    def save_models(self, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(self.model_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        self.mac.save_models(path)
        self.central_mac.save_models(path)
        th.save(self.central_mixer.state_dict(), "{}/central_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.central_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_central_mac.load_models(path)
        self.central_mixer.load_state_dict(th.load("{}/central_mixer.th".format(path)))
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