import random
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Any, Tuple

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


def deactivate_mixstyle(m):
    if type(m) == AdvStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == AdvStyle:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == AdvStyle:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == AdvStyle:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
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


class DSUStyle(nn.Module):
    """DSU.

    Reference:
      # UNCERTAINTY MODELING FOR OUT-OF-DISTRIBUTION GENERALIZATION
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', channel_num=0, **kwargs):
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
        self._activated = True
        self.channel_num = channel_num
        # self.adv_mean_std = torch.nn.Parameter(torch.FloatTensor(1,channel_num,1,1), requires_grad=True)
        # self.adv_mean_std.data.fill_(0.01)  ## initialization
        # self.adv_std_std = torch.nn.Parameter(torch.FloatTensor(1,channel_num,1,1), requires_grad=True)
        # self.adv_std_std.data.fill_(0.01)
        # self.grl_mean = GradientReverseLayer()  ## fix the backpropagation weight as 1.0.
        # self.grl_std = GradientReverseLayer()

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    # def forward(self, x):
    #     if not self.training or not self._activated:
    #         return x
    #
    #     if random.random() > self.p:
    #         return x
    #
    #     mean = x.mean(dim=[2, 3], keepdim=False)
    #     std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()
    #
    #     sqrtvar_mu = self.sqrtvar(mean)
    #     sqrtvar_std = self.sqrtvar(std)
    #
    #     beta = self._reparameterize(mean, sqrtvar_mu)
    #     gamma = self._reparameterize(std, sqrtvar_std)
    #
    #     x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
    #     x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
    #
    #     return x


    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x-mu) / sig

        var_mu = mu.var(0, keepdim=True)
        var_sig = sig.var(0, keepdim=True)

        sig_mu = (var_mu + self.eps).sqrt()
        sig_sig = (var_sig + self.eps).sqrt()
        # sig_mu = sig_mu.detach()
        # sig_sig = sig_sig.detach()

        initial_mean_std = torch.randn(mu.size()).cuda()
        initial_std_std = torch.randn(sig.size()).cuda()

        new_mu = initial_mean_std * sig_mu + mu
        new_sig = initial_std_std * sig_sig + sig

        return x_normed * new_sig + new_mu




