import torch as th

class Uniform:

    def __init__(self, args):
        self.args = args
        self.noise_distrib = th.distributions.one_hot_categorical.OneHotCategorical(th.tensor([1/self.args.noise_dim for _ in range(self.args.noise_dim)]).repeat(1, 1))

    def sample(self, state, test_mode):
        action = self.noise_distrib.sample().cpu()
        return action.detach().clone().numpy()[0]

    def update_returns(self, state, noise, returns):
        print("Uniform policy......")
        pass

    def save_models(self, save_dir, train_steps):
        pass

    def load_models(self, path):
        pass

    def update_model(self, other_agent):
        pass