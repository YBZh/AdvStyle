import random
from contextlib import contextmanager
import torch
import torch.nn as nn


def first_order(m):
    if type(m) == MixOrders:
        m.update_order_method('one')

def second_order(m):
    if type(m) == MixOrders:
        m.update_order_method('two')

# def third_order(m):
#     if type(m) == MixOrders:
#         m.update_order_method('three')
#
# def fourth_order(m):
#     if type(m) == MixOrders:
#         m.update_order_method('four')
#
def first_second_order(m):
    if type(m) == MixOrders:
        m.update_order_method('one_two')
#
# def first_second_three_order(m):
#     if type(m) == MixOrders:
#         m.update_order_method('one_two_three')
#
# def first_second_three_four_order(m):
#     if type(m) == MixOrders:
#         m.update_order_method('one_two_three_four')

def all_order(m):
    if type(m) == MixOrders:
        m.update_order_method('all')



class MixOrders(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', orders='two'):
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
        self.orders= orders
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_order_method(self, orders='two'):
        self.orders = orders

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        if self.orders == 'all': ## i.e., EFDM
            lmda = self.beta.sample((B, 1, 1))
            lmda = lmda.to(x.device)
            x_view = x.view(B, C, -1)
            value_x, index_x = torch.sort(x_view)
            x_view_copy = x_view.detach().clone()
            inverse_index = index_x.argsort(-1)
            x_view_copy = x_view_copy * lmda + value_x[perm].gather(-1, inverse_index) * (1-lmda)
            new_x = x_view + (x_view_copy - x_view.detach())
            return new_x.view(B, C, W, H)

        elif self.orders == 'one': ## AdaMean
            mu = x.mean(dim=[2, 3], keepdim=True)
            mu = mu.detach()
            x_normed = x-mu
            mu2 = mu[perm]
            # mu_mix = mu*lmda + mu2 * (1-lmda)
            return x_normed + mu2

        elif self.orders == 'two': ## AdaStd in the paper.
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x-mu) / sig
            mu2, sig2 = mu[perm], sig[perm]
            # mu_mix = mu*lmda + mu2 * (1-lmda)
            # sig_mix = sig*lmda + sig2 * (1-lmda)
            return x_normed*sig2 + mu

        # It is impossible to only change the third and fourth moment while keep the mean and std unchanged.
        # elif self.orders == 'three':
        #     mu = x.mean(dim=[2, 3], keepdim=True)
        #     var = x.var(dim=[2, 3], keepdim=True)
        #     sig = (var + self.eps).sqrt()
        #     mu, sig = mu.detach(), sig.detach()
        #     x_normed = (x-mu) / sig
        #
        #     skewness = torch.pow(x_normed.detach(), 3).mean(dim=[2, 3], keepdim=True)
        #     skewness2 = skewness[perm]
        #     skewness_mix = skewness*lmda + skewness2 * (1-lmda)
        #     x_normed_three =  torch.pow(torch.pow(x_normed, 3) - skewness + skewness_mix , 1.0/3.0)
        #
        #     return x_normed_three*sig + mu
        #
        # elif self.orders == 'four':
        #     mu = x.mean(dim=[2, 3], keepdim=True)
        #     var = x.var(dim=[2, 3], keepdim=True)
        #     sig = (var + self.eps).sqrt()
        #     mu, sig = mu.detach(), sig.detach()
        #     x_normed = (x-mu) / sig
        #
        #     kurtosis = (torch.pow(x_normed.detach(), 4) + 1e-24).mean(dim=[2, 3], keepdim=True)
        #     kurtosis2 = kurtosis[perm]
        #     kurtosis_mix = kurtosis * lmda + kurtosis2 * (1 - lmda)
        #     x_normed_four =  torch.pow(torch.pow(x_normed, 4) / kurtosis * kurtosis_mix, 1.0/4.0)
        #
        #     return x_normed_four*sig + mu

        elif self.orders == 'one_two': ## AdaIN
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x-mu) / sig
            mu2, sig2 = mu[perm], sig[perm]
            return x_normed * sig2 + mu2
            # mu_mix = mu*lmda + mu2 * (1-lmda)
            # sig_mix = sig*lmda + sig2 * (1-lmda)
            # return x_normed*sig_mix + mu_mix


        else:
            raise NotImplementedError


