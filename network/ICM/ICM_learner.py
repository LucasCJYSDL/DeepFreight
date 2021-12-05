from network.ICM.forward_model import FwdModel
import torch.optim as optim
import numpy as np
from collections import deque
import torch
import os

class ICM_learner:
    def __init__(self, args):
        self.args = args
        self.lr = args.lr

        self.policy = FwdModel(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        if self.args.cuda:
            self.policy.cuda()

        self.buffer = deque(maxlen=self.args.icm_buffer_size)

    def sample(self, state, action_list):

        input = np.hstack((state, action_list))
        input = torch.Tensor(input).unsqueeze(0)
        if self.args.cuda:
            input = input.cuda()

        output = self.policy(input)

        return output.detach().clone().cpu().numpy()[0]

    def update_icm_buffer(self, state, action_list, next_state):

        s = torch.tensor(state, dtype=torch.float32)
        a = torch.tensor(action_list, dtype=torch.float32)
        n_s = torch.tensor(next_state, dtype=torch.float32) ## check the type
        self.buffer.append((s, a, n_s))

    def update(self):

        if len(self.buffer) < self.args.icm_batch_size:
            print("Not enough ICM buffer!!!")
            return

        print("Updating the forward network......")
        for _ in range(self.args.icm_iters):
            idxs = np.random.randint(0, len(self.buffer), size=self.args.icm_batch_size)
            batch_elems = [self.buffer[i] for i in idxs]
            states_ = torch.stack([x[0] for x in batch_elems])
            actions_ = torch.stack([x[1] for x in batch_elems])
            next_states_ = torch.stack([x[2] for x in batch_elems])
            inputs_ = torch.cat((states_, actions_), 1)
            if self.args.cuda:
                inputs_ = inputs_.cuda()
                next_states_ = next_states_.cuda()
            pre_next_states_ = self.policy(inputs_)

            self.optimizer.zero_grad()
            policy_loss = 0.5 * torch.norm(next_states_ - pre_next_states_) ** 2
            policy_loss.backward()
            self.optimizer.step()

    def save_models(self, save_dir, train_steps):
        idx = str(train_steps // self.args.save_model_period)
        path = os.path.join(save_dir, idx)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), "{}/fwd_model.th".format(path))

    def load_models(self, path):

        self.policy.load_state_dict(torch.load("{}/fwd_model.th".format(path)))