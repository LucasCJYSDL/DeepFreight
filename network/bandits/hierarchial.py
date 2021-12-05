# Categorical policy for discrete z
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import os

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(args.state_shape, 128)
        self.affine2 = nn.Linear(128, args.noise_dim)

    def forward(self, x):
        x = x.view(-1, self.args.state_shape)
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# Max entropy Z agent
class EZ_agent:
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.noise_dim = self.args.noise_dim
        # size of state vector
        self.state_hands_shape = self.args.state_shape
        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        if self.args.cuda:
            self.policy.cuda()
        # Scaling factor for entropy, would roughly be similar to MI scaling
        self.entropy_scaling = args.entropy_scaling
        self.uniform_distrib = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([1/self.args.noise_dim for _ in range(self.args.noise_dim)]).repeat(1, 1))

        self.buffer = deque(maxlen=self.args.bandit_buffer)

    def sample(self, state_hands, test_mode):
        # During testing we just sample uniformly
        if test_mode:
            action = self.uniform_distrib.sample().cpu()
        else:
            state_hands = torch.tensor(state_hands, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                state_hands = state_hands.cuda()
            probs = self.policy(state_hands)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            action = m.sample().cpu()

        return action.detach().clone().numpy()[0]

    def update_returns(self, state_hands, action, ret):

        s_h = torch.tensor(state_hands, dtype=torch.float32)
        a = torch.tensor(action, dtype=torch.float32)
        r = torch.tensor(ret, dtype=torch.float32) ## check the type
        self.buffer.append((s_h, a, r))

        if len(self.buffer) < self.args.bandit_batch:
            print("Not enough bandit buffer!!!")
            return

        print("Updating the hierarchial network......")
        for _ in range(self.args.bandit_iters):
            idxs = np.random.randint(0, len(self.buffer), size=self.args.bandit_batch)
            batch_elems = [self.buffer[i] for i in idxs]
            state_hands_ = torch.stack([x[0] for x in batch_elems])
            actions_ = torch.stack([x[1] for x in batch_elems])
            returns_ = torch.stack([x[2] for x in batch_elems])
            if self.args.cuda:
                state_hands_ = state_hands_.cuda()
                actions_ = actions_.cuda()
                returns_ = returns_.cuda()
            probs = self.policy(state_hands_)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            log_probs = m.log_prob(actions_.to(probs.device))
            # print("3: ", log_probs.shape)
            self.optimizer.zero_grad()
            policy_loss = -torch.dot(log_probs, torch.tensor(returns_, device=log_probs.device).float()) + self.entropy_scaling * log_probs.sum()
            policy_loss.backward()
            self.optimizer.step()

    def update_model(self, other_agent):
        self.policy.load_state_dict(other_agent.policy.state_dict())

    def save_models(self, save_dir, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(save_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), "{}/ez_bandit_policy.th".format(path))

    def load_models(self, path):

        self.policy.load_state_dict(torch.load("{}/ez_bandit_policy.th".format(path)))

