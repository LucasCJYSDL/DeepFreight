import copy
from agents.target_agents import Target_agents
from network.critics.coma import COMACritic
import torch as th
import time
from torch.utils.tensorboard import SummaryWriter
import os

def build_td_lambda_targets(rewards, terminated, mask, target_qs, gamma, td_lambda): ## TODO: check
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    # ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        # ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
        #             * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
        assert len(rewards[:, t].shape) == 2 and len(target_qs[:, t + 1].shape) == 2
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

class COMALearner:
    def __init__(self, mac: Target_agents, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        assert self.mac.is_target()

        self.critic = COMACritic(args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params
        self.critic_training_steps = 0

        if args.cuda:
            self.cuda()

        self.agent_optimiser = th.optim.RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = th.optim.RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        assert args.alg == 'coma'
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
        a, r, avail_a, done, gamma = batch['a'], batch['r'], batch['avail_a'][:, :-1], batch['done'], batch['gamma']
        mask = 1 - batch['padded'].float()
        if self.args.cuda:
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            gamma = gamma.cuda()

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        actions = th.cat([a[:, :], a[:, -1:]], dim=1)

        batch['onehot_a'] = th.cat([batch['onehot_a'][:, :], batch['onehot_a'][:, -1:]], dim=1)
        q_vals, critic_train_stats = self._train_critic(batch, r, done, actions, critic_mask, episode_num, max_seq_length)
        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(episode_num)
        for t in range(max_seq_length):
            agent_outs = self.mac.forward(batch, t, episode_num, self.args.epsilon)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_a == 1.0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_a == 1.0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.clip_norm)
        self.agent_optimiser.step()

        log_info = {}
        ts_logged = len(critic_train_stats["critic_loss"])
        for key in ["critic_loss", "critic_grad_norm", "target_mean"]:
            log_info[key] = sum(critic_train_stats[key])/ts_logged
        log_info["advantage_mean"] = (advantages * mask).sum().item() / mask.sum().item()
        log_info["coma_loss"] = coma_loss.item()
        log_info["agent_grad_norm"] = grad_norm

        return log_info

    def get_log_dict(self):

        return {"critic_loss": [], "critic_grad_norm": [], "target_mean": [], \
                "advantage_mean": [], "coma_loss": [], "agent_grad_norm": []}

    def _train_critic(self, batch, rewards, terminated, actions, mask, bs, max_seq_len):
        # Optimise critic
        target_q_vals = self.target_critic(batch, max_seq_len=max_seq_len+1)[:, :]

        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        assert self.args.step_num == 1
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:,:-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "target_mean": []
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t=t)

            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)

            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.clip_norm)
            self.critic_optimiser.step()

            if (self.critic_training_steps > 0) and (self.critic_training_steps % self.args.target_update_period == 0):
                self._update_targets()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        print("Updated target network......")
        self.target_critic.load_state_dict(self.critic.state_dict())

    def cuda(self):
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(self.model_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{}/critic.th".format(path)))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        # self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path)))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path)))

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