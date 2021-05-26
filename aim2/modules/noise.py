import torch

class poisson_noise:
    def __init__(self, rate, device):
        self.rate =rate
        self.device= device
    def sample(self, shape):
        return torch.poisson(self.rate*torch.ones(shape).to(self.device))