import random
from contextlib import contextmanager
import torch
import torch.nn as nn

class RandStyle(nn.Module):
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
        self.mean = None  ## mean of C dimension
        self.sig = None   ## std of C dimension
        self.mean_std = None
        self.sig_std = None
        self.momentum = 0.1

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        if not self.training or not self._activated:
            return x_normed*self.sig + self.mean    ## use the training domain mean and var

        batch_mean = mu.mean(0, keepdim=True)
        batch_sig = sig.mean(0, keepdim=True)
        if self.mean == None:
            self.mean = batch_mean ## initialize
        else:
            self.mean = self.mean * (1 - self.momentum) + batch_mean * self.momentum
        if self.sig == None:
            self.sig = batch_sig ## initialize
        else:
            self.sig = self.sig * (1 - self.momentum) + batch_sig * self.momentum

        batch_mean_std = mu.std(0, keepdim=True)
        batch_sig_std = sig.std(0, keepdim=True)
        if self.mean_std == None:
            self.mean_std = batch_mean_std ## initialize
        else:
            self.mean_std = self.mean_std * (1 - self.momentum) + batch_mean_std * self.momentum
        if self.sig_std == None:
            self.sig_std = batch_sig_std ## initialize
        else:
            self.sig_std = self.sig_std * (1 - self.momentum) + batch_sig_std * self.momentum

        if random.random() > self.p:
            return x

        # B = x.size(0)
        generated_sig = torch.normal(self.sig.expand(mu.size()), self.sig_std.expand(mu.size())).to(x.device)
        generated_sig[generated_sig<0] = 0
        generated_mu = torch.normal(self.mean.expand(mu.size()), self.mean_std.expand(mu.size())).to(x.device)

        return x_normed*generated_sig + generated_mu
