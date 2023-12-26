import torch.nn as nn
import torch

"""
This is where the logic to add noise to an image lives
"""

class NoisyAccumulation(nn.Module): # updated the default value for input_shape
    def __init__(self, input_shape=112, budget_mean=4, sensitivity=None):
        super(NoisyAccumulation, self).__init__()
        self.h, self.w = input_shape, input_shape
        if sensitivity is None:
            sensitivity = torch.ones([3, self.h, self.w])
        self.sensitivity = sensitivity.reshape(3 * self.h * self.w)
        self.given_locs = torch.zeros((3, self.h, self.w))
        size = self.given_locs.shape
        self.budget = budget_mean * 3 * self.h * self.w
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs))
        self.rhos = nn.Parameter(torch.zeros(size))
        self.laplace = torch.distributions.laplace.Laplace(0, 1)
        self.rhos.requires_grad = True
        self.locs.requires_grad = True

    def scales(self):
        softmax = nn.Softmax()
        return (self.sensitivity / (softmax(self.rhos.reshape(3 * self.h * self.w))
                * self.budget)).reshape(3, self.h, self.w)

    def sample_noise(self):
        epsilon = self.laplace.sample(self.rhos.shape)
        return self.locs + self.scales() * epsilon

    def forward(self, input):
        noise = self.sample_noise()
        output = input + noise
        return output

    def aux_loss(self):
        scale = self.scales()
        loss = -1.0 * torch.log(scale.mean())
        return loss
