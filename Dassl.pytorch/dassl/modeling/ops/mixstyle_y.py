import random
from contextlib import contextmanager
import torch
import torch.nn as nn


def deactivate_mixstyle_y(m):
    if type(m) == MixStyleY:
        m.set_activation_status(False)


def activate_mixstyle_y(m):
    if type(m) == MixStyleY:
        m.set_activation_status(True)

def deactivate_training_weight_y(m):
    if type(m) == MixStyleY:
        m.set_weight_training_status(False)


def activate_training_weight_y(m):
    if type(m) == MixStyleY:
        m.set_weight_training_status(True)


def random_mixstyle_y(m):
    if type(m) == MixStyleY:
        m.update_mix_method('random')


def crossdomain_mixstyle_y(m):
    if type(m) == MixStyleY:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_mixstyle_y(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle_y)
        yield
    finally:
        model.apply(activate_mixstyle_y)


@contextmanager
def run_with_mixstyle_y(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle_y)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle_y)

    try:
        model.apply(activate_mixstyle_y)
        yield
    finally:
        model.apply(deactivate_mixstyle_y)


class MixStyleY(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', style_w=0):
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
        self.weight = torch.nn.Parameter(torch.FloatTensor([style_w]), requires_grad=True)
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix}, weight={self.weight.item()})'

    def set_activation_status(self, status=True):
        self._activated = status

    def set_weight_training_status(self, training=True):
        self.weight.requires_grad = training

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x, y=None):
        if not self.training or not self._activated:
            return x, y

        if random.random() > self.p:
            return x, y

        self.weight.data = torch.clamp(self.weight.data, 0, 1.0)

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)


        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        # MixStyle
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        if y == None:
            return x_normed*sig_mix + mu_mix, y
        else:
            y2 = y[perm]
            return x_normed * sig_mix + mu_mix, y * (1-self.weight + self.weight*lmda[:,:,0,0]) + y2 * (self.weight * (1-lmda[:,:,0,0]))

        #
        # # AdaIN
        # mu2, sig2 = mu[perm], sig[perm]
        # if y == None:
        #     return x_normed*sig2 + mu2, y
        # else:
        #     y2 = y[perm]
        #     return x_normed*sig2 + mu2, y * (1-self.weight) + y2 * (self.weight)