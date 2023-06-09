import random
from contextlib import contextmanager
import torch
import torch.nn as nn
from skimage.exposure import match_histograms
import numpy as np


def search_sorted(bin_locations, inputs, eps=-1e-6):
    """
    Searches for which bin an input belongs to (in a way that is parallelizable and amenable to autodiff)
    """
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

## Note that we keep the functional name identical to mixstyle. Therefore, the same training code is shared between MixHistogram and MixStyle.
def deactivate_mixstyle(m):
    if type(m) == MixHistogram:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixHistogram:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixHistogram:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == MixHistogram:
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


class MixHistogram(nn.Module):
    """MixStyle.

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

        # print('sorting method is: ', self.sorting)
        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        ############################# mixhist via mixsorting
        x_view = x.view(B,C, -1)
        value_x, index_x = torch.sort(x_view)  # sort conduct a deep copy here.
        lmda = self.beta.sample((B, 1, 1))
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

        # ########################## EFDMix, original, CVPR submission.
        # inverse_index = index_x.argsort(-1)
        # x_view_copy = value_x[perm].gather(-1, inverse_index)  # note gather conduct a deep copy here.
        # new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)   ## 这里其实是引入了另外一个的额外样本，这个样本可能不是同类的，这样为啥结果会变好呢？ 这不科学？
        # return new_x.view(B, C, W, H)
        ########################## EFDMix, output = lam x + (1-lam) y
        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index)
        new_x = x_view * lmda + x_view_copy * (1-lmda)
        return new_x.view(B, C, W, H)



    # def forward(self, x):
    #     if not self.training or not self._activated:
    #         return x
    #
    #     if random.random() > self.p:
    #         return x
    #     B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
    #     x_view = x.view(-1, W, H)
    #     lmda = self.beta.sample((B, 1, 1, 1))
    #     lmda = lmda.to(x.device)
    #
    #     if self.mix == 'random':
    #         # random shuffle
    #         perm = torch.randperm(B)
    #     elif self.mix == 'crossdomain':
    #         # split into two halves and swap the order
    #         perm = torch.arange(B - 1, -1, -1) # inverse index
    #         perm_b, perm_a = perm.chunk(2)
    #         perm_b = perm_b[torch.randperm(B // 2)]
    #         perm_a = perm_a[torch.randperm(B // 2)]
    #         perm = torch.cat([perm_b, perm_a], 0)
    #     else:
    #         raise NotImplementedError

        ######## histogram mix.
        # image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0,2)), np.array(x[perm].view(-1, W, H).detach().clone().cpu().float().transpose(0,2)), multichannel=True)
        # image1_temp = torch.from_numpy(image1_temp).float().to(x.device).transpose(0,2).view(B,C,W,H)
        # return x + (image1_temp - x).detach() * (1 - lmda)
        ## Histogram matching
        # return x + (image1_temp - x).detach()


        #######A Tensor implementation of Histogram matching, which is slower than the above implementation.
        # x_view_copy = x_view.detach().clone()
        # permed_x = x[perm].view(-1, W, H)
        # for i in range(x_view.size(0)):
        #     src_values, src_unique_indices, src_counts = torch.unique(x_view[i].ravel(),
        #                                                               return_inverse=True,
        #                                                               return_counts=True)
        #     tmpl_values, tmpl_counts = torch.unique(permed_x[i].ravel(), return_counts=True)
        #     src_quantiles = torch.cumsum(src_counts, 0) / src_unique_indices.size(0)
        #     tmpl_quantiles = torch.cumsum(tmpl_counts, 0) / src_unique_indices.size(0)
        #     x_view_copy[i] = tmpl_values[search_sorted(tmpl_quantiles, src_quantiles)][src_unique_indices].view(W, H)
        # x_view_copy = x_view_copy.view(B,C,W,H)
        # return x + (x_view_copy - x).detach() * (1 - lmda)
        ## Histogram matching
        # return x + (x_view_copy - x).detach()

