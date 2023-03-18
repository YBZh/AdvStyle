import random
from contextlib import contextmanager
import torch
import torch.nn as nn


def deactivate_mixstyle(m):
    if type(m) == MixStyle2w:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle2w:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixStyle2w:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == MixStyle2w:
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


class MixStyle2w(nn.Module):
    """MixStyle2w, extension of MixStyle.
    In MixStyle, the same mixing weight is adopted for both mean and standard deviation.
    In MixStyle2w, two different mixing weight are applied to mean and standard deviation, respectively. --> No obvious improvement.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
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

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig, var = mu.detach(), sig.detach(), var.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        lmda2 = self.beta.sample((B, 1, 1, 1))
        lmda2 = lmda2.to(x.device)
        if self.mix == 'random':
            # random shuffle
            perm1 = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm1 = torch.arange(B - 1, -1, -1) # inverse index
            perm_b1, perm_a1 = perm1.chunk(2)
            perm_b1 = perm_b1[torch.randperm(B // 2)]
            perm_a1 = perm_a1[torch.randperm(B // 2)]
            perm1 = torch.cat([perm_b1, perm_a1], 0)


        else:
            raise NotImplementedError

        mu2, var2 = mu[perm1], var[perm1]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        var_mix = var*lmda2 + var2 * (1-lmda2)
        sig_mix = (var_mix + self.eps).sqrt()

        return x_normed*sig_mix + mu_mix
