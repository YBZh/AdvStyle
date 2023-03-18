import random
from contextlib import contextmanager
import torch
import torch.nn as nn


## Note that we keep the functional name identical to mixstyle. Therefore, the same training code is shared between EFDMix and MixStyle.
def deactivate_mixstyle(m):
    if type(m) == EFDMix:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == EFDMix:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == EFDMix:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == EFDMix:
        m.update_mix_method('crossdomain')


def quicksort_mixstyle(m):
    if type(m) == EFDMix:
        m.update_sorting_method('quicksort')

def index_mixstyle(m):
    if type(m) == EFDMix:
        m.update_sorting_method('index')

def randomsort_mixstyle(m):
    if type(m) == EFDMix:
        m.update_sorting_method('random')

def neighbor_mixstyle(m):
    if type(m) == EFDMix:
        m.update_sorting_method('neighbor')

@contextmanager
def run_without_mixstyle(model):
    # Assume EFDMix was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


class SNR(nn.Module):
    """EFDMix.

    Reference:
      Exact Feature Distribution Matching for  Arbitrary Style Transfer and Domain Generalization
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', channel_num=0):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self.sorting = 'quicksort'
        self._activated = True
        self.IN = nn.InstanceNorm2d(channel_num, affine=True)
        self.style_reid_laye = ChannelGate_sub(channel_num, num_gates=channel_num, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def update_sorting_method(self, sorting='quicksort'):
        self.sorting = sorting

    def forward(self, x):
        # if not self.training or not self._activated:
        #     return x

        x_IN_1 = self.IN(x)
        x_style_1 = x - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye(x_style_1)
        x = x_IN_1 + x_style_1_reid_useful

        return x

