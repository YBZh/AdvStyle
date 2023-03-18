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


class AdvStyle_test(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', channel_num=0, adv_weight=1.0, mix_weight=1.0, **kwargs):
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
        self.mix_weight = mix_weight  # the weight control the strength of intensity perturbation.
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.channel_num = channel_num
        self.adv_mean_std = torch.nn.Parameter(torch.FloatTensor(1, channel_num, 1, 1), requires_grad=True)
        self.adv_mean_std.data.fill_(0.1)  ## initialization
        self.adv_std_std = torch.nn.Parameter(torch.FloatTensor(1, channel_num, 1, 1), requires_grad=True)
        self.adv_std_std.data.fill_(0.1)
        self.grl_mean = GradientReverseLayer()  ## fix the backpropagation weight as 1.0.
        self.grl_std = GradientReverseLayer()
        self.adv_weight = adv_weight
        self.first_flag = True
        self.init_flag = False
        print('adv weight is:', self.adv_weight)

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def reset_init_flag(self):
        self.init_flag = True

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if self.init_flag:
            raise NotImplementedError
            self.init_flag = False
            self.first_flag = False
            B = x.size(0)
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            x_normed = (x - mu) / sig
            var_mu = mu.var(0, keepdim=True)
            var_sig = sig.var(0, keepdim=True)
            sig_mu = (var_mu + self.eps).sqrt()
            sig_sig = (var_sig + self.eps).sqrt()

            self.adv_mean_std.data = sig_mu
            self.adv_std_std.data = sig_sig

            initial_mean_std = torch.randn(mu.size()).cuda()
            initial_std_std = torch.randn(sig.size()).cuda()
            new_mu = initial_mean_std * self.grl_mean(self.adv_mean_std, self.adv_weight) + mu
            new_sig = initial_std_std * self.grl_std(self.adv_std_std, self.adv_weight) + sig
            return x_normed * new_sig + new_mu
        else:
            random_value = random.random()
            if random_value < 1/2:
            # if random_value < 1 / 3:
                return x
            # elif random_value < 2/3:  ## AdvStyle
            else:
                B = x.size(0)

                mu = x.mean(dim=[2, 3], keepdim=True)
                var = x.var(dim=[2, 3], keepdim=True)
                sig = (var + self.eps).sqrt()
                x_normed = (x-mu) / sig

                ### initialization the mean_std and std_std
                if self.first_flag:
                    var_mu = mu.var(0, keepdim=True)
                    var_sig = sig.var(0, keepdim=True)
                    sig_mu = (var_mu + self.eps).sqrt()
                    sig_sig = (var_sig + self.eps).sqrt()
                    sig_mu = sig_mu.detach()
                    sig_sig = sig_sig.detach()
                    self.adv_mean_std.data = sig_mu
                    self.adv_std_std.data = sig_sig
                    self.first_flag = False
                    ####################################### initialize std with uniform distribution.
                    # self.adv_mean_std.data.uniform_(0, 1)
                    # self.adv_std_std.data.uniform_(0, 1)
                    # self.first_flag = False

                # initial_mean_std = torch.randn(mu.size()).cuda()
                # initial_std_std = torch.randn(sig.size()).cuda()
                # new_mu = initial_mean_std * self.grl_mean(self.adv_mean_std, self.adv_weight) + mu
                # new_sig = initial_std_std * self.grl_std(self.adv_std_std, self.adv_weight) + sig
                # return x_normed * new_sig + new_mu

                # 只用self.adv_mean_std 和 self.adv_std_std 的direction, and current batch std 的强度。
                var_mu = mu.var(0, keepdim=True) ## 1*C*1*1
                var_sig = sig.var(0, keepdim=True)
                sig_mu = (var_mu + self.eps).sqrt()  ## 这里提供强度
                sig_sig = (var_sig + self.eps).sqrt()

                qiangdu_sig_mu = self.grl_mean(self.adv_mean_std, self.adv_weight)  ## 这里提供方向
                qiangdu_sig_sig = self.grl_std(self.adv_std_std, self.adv_weight)

                used_sig_mu = qiangdu_sig_mu / torch.norm(qiangdu_sig_mu, p=2,dim=1,keepdim=True) * torch.norm(sig_mu, p=2,dim=1,keepdim=True) * self.mix_weight
                used_sig_sig = qiangdu_sig_sig / torch.norm(qiangdu_sig_sig, p=2, dim=1, keepdim=True) * torch.norm(sig_sig, p=2, dim=1, keepdim=True) * self.mix_weight

                initial_mean_std = torch.randn(mu.size()).cuda()
                initial_std_std = torch.randn(sig.size()).cuda()
                new_mu = initial_mean_std * used_sig_mu + mu
                new_sig = initial_std_std * used_sig_sig + sig
                return x_normed * new_sig + new_mu
            # else:  ### DSU
            #     B = x.size(0)
            #
            #     mu = x.mean(dim=[2, 3], keepdim=True)
            #     var = x.var(dim=[2, 3], keepdim=True)
            #     sig = (var + self.eps).sqrt()
            #     x_normed = (x - mu) / sig
            #
            #     var_mu = mu.var(0, keepdim=True)
            #     var_sig = sig.var(0, keepdim=True)
            #
            #     sig_mu = (var_mu + self.eps).sqrt()
            #     sig_sig = (var_sig + self.eps).sqrt()
            #     # sig_mu = sig_mu.detach()
            #     # sig_sig = sig_sig.detach()
            #     initial_mean_std = torch.randn(mu.size()).cuda()
            #     initial_std_std = torch.randn(sig.size()).cuda()
            #
            #     new_mu = initial_mean_std * sig_mu + mu
            #     new_sig = initial_std_std * sig_sig + sig
            #
            #     return x_normed * new_sig + new_mu
