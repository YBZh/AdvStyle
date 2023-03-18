import random
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

## Note that we keep the functional name identical to mixstyle. Therefore, the same training code is shared between EFDMix and MixStyle.
def random_mixstyle(m):
    pass


def crossdomain_mixstyle(m):
    pass


class IGN(nn.Module):
    """Instance Gaussian Normalization.
    """
    def __init__(self, num_features, fixed=False, affine=False):
        """
        Args:
          num_features (int): feature channels
          fixed (bool): whether adopted the same gaussian samples for all iterations
          affine (bool): a boolean value that when set to ``True``, this module has learnable affine parameters
        input: B*C*W*H --> output: B*C*W*H
        """
        super().__init__()
        self.num_features = num_features
        self.fixed = fixed
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        ############################# mixhist via mixsorting
        x_view = x.view(B,C, -1)
        x_gaussian = x_view.clone().normal_(0, 1)

        _, index_x = torch.sort(x_view)  # sort conduct a deep copy here.
        value_y, _ = torch.sort(x_gaussian)

        ########################### EFDM
        inverse_index = index_x.argsort(-1)
        x_view_copy = value_y.gather(-1, inverse_index)  # note gather conduct a deep copy here.
        new_x = x_view + (x_view_copy - x_view.detach())
        if self.affine:
            return new_x.view(B, C, W, H) * self.weight + self.bias
        else:
            return new_x.view(B, C, W, H)
