import torch
import torch.nn as nn
import math
import numpy as np
from transformers.activations import GELUActivation

def gelu_derivative(x):
	phi = 0.5 * (1.0 + torch.erf(torch.tensor(x / np.sqrt(2.0))))
	normal = 1/np.sqrt(2 * math.pi) * np.exp(-0.5 * x * x)
	return phi.item() + x * normal

def silu_derivative(x):
    x = torch.tensor(x)
    return torch.nn.functional.sigmoid(x) * (1 + x*(1 -  torch.nn.functional.sigmoid(x))).item()

class LeakySiLU(nn.Module):
    """
    Applies relu after a cutoff where the gradient of silu is close to 1, that way the gradient of the function always remains < 1.
    """

    def __init__(self, cutoff=1.2759):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, input):
        bool_ = (input > self.cutoff).to(input.dtype)
        input2 = input * bool_
        relu_applied = nn.functional.relu(input2)
        silu_applied = nn.SiLU()(input)
        return relu_applied * bool_ + (1-bool_) * silu_applied

class LeakyGeLU(nn.Module):
    """
    Applies relu after a cutoff where the gradient of silu is close to 1, that way the gradient of the function always remains < 1.
    """

    def __init__(self, cutoff=0.751):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, input):
        bool_ = (input > self.cutoff).to(input.dtype)
        input2 = input * bool_
        relu_applied = nn.functional.relu(input2)
        gelu_applied = GELUActivation()(input)
        return relu_applied * bool_ + (1-bool_) * gelu_applied
