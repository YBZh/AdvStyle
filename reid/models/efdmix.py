import random
from contextlib import contextmanager
import torch
import torch.nn as nn

# Inherit from Function
class Replace_with_grad(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, output):  ## input sorted x, sorted y, otherwise we also need to conduct sort here.
        ctx.save_for_backward(input, output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output): # B*C*D
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, output = ctx.saved_tensors  ## only operate on the last dimension.

        ## weight prior as ones, and bias prior as zeros.
        # grad_multiply = (input * output - input) / (torch.pow(input, 2) + 1) + 1


        # # weight prior as weight in AdaIn, and bias prior as bias in AdaIN， Lagrange_EFDMix_PACS_v2
        # ## IN the previous Lagrange_EFDMix_PACS_V1, we adopt one as weight prior, and zero as bias prior.
        # mu_a = input.mean(dim=[2], keepdim=True)
        # var_a = input.var(dim=[2], keepdim=True)
        # mu_b = output.mean(dim=[2], keepdim=True)
        # var_b = output.var(dim=[2], keepdim=True)
        # sig_a = (var_a + 1e-6).sqrt()
        # sig_b = (var_b + 1e-6).sqrt()
        # w_0 = sig_b / sig_a
        # b_0 = mu_b - (sig_b / sig_a) * mu_a # B*C*1
        # grad_multiply = (w_0 - b_0 * input + input * output) / (torch.pow(input, 2) + 1)


        # ### hongwei' method, it is easy to present inf/nan gradient.
        # y_plus_1 = torch.cat((output[:,:,1:], output[:,:,-1:]), -1)
        # y_minus_1 = torch.cat((output[:,:,:1], output[:,:,:-1]), -1)
        # x_plus_1 = torch.cat((input[:,:,1:], input[:,:,-1:]), -1)
        # x_minus_1 = torch.cat((input[:,:,1:], input[:,:,-1:]), -1)
        # grad_multiply = (y_plus_1 - y_minus_1) / (x_plus_1 - x_minus_1 + 1e-6)
        #
        # var_a = input.var(dim=[2], keepdim=True)
        # var_b = output.var(dim=[2], keepdim=True)
        # sig_a = (var_a + 1e-6).sqrt()
        # sig_b = (var_b + 1e-6).sqrt()
        # grad_multiply_adain = sig_b/sig_a
        # grad_multiply_adain = grad_multiply_adain.expand(grad_multiply.size())

        # # ## 用局部的scale 作为 0 位置的scale, 这个应该是合理的，先找到不为0 的位置，然后基于该位置向外扩展，然后将算出来的 scale 作为0 位置的scale.
        # # ## (input==0) | (output==0)  --> 全0位置的最长值， 取index, 然后index 乘以 1.33， 取值，然后减去初始值/ 对应的x 减去初始值--> BC1.
        # index = (((input==0) | (output==0)).sum(2, keepdim=True) * 1.33).long()
        # index[index >= input.size(2)] = input.size(2) - 1
        # local_scale = torch.gather(output, 2, index) / (torch.gather(input, 2, index) + 1e-6) ## BC1
        # local_scale = local_scale.expand(grad_multiply.size()) ## BCD
        # grad_multiply[(input == 0) & (output == 0)] = local_scale[(input == 0) & (output == 0)]

        # ## identity_grad_for_both_zero
        # grad_multiply[(input==0) & (output==0)] = 1
        ## adain_grad_for_both_zero,  用整体 scale 来作为局部的 scale, 这个勉强说的过去。
        # grad_multiply[(input==0) & (output==0)] = grad_multiply_adain[(input==0) & (output==0)]
        # ## set a min max threshold, try 5.
        # threshold = 5.0
        # grad_multiply[grad_multiply > grad_multiply_adain * threshold] = grad_multiply_adain[grad_multiply > grad_multiply_adain * threshold]
        # grad_multiply[grad_multiply < grad_multiply_adain / threshold] = grad_multiply_adain[grad_multiply < grad_multiply_adain / threshold]


        # ############ (left grad + right grad) / 2 + average pooling. ###########################################################################################
        # kernel_two_conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=1, bias=False, padding_mode='replicate').cuda()
        # kernel_two_conv.weight.fill_(0.5)
        # kernel_thr_conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False, padding_mode='replicate').cuda()
        # kernel_thr_conv.weight.fill_(1.0/3.0)
        #
        # y_shift_minum_y = output[:,:,1:] - output[:,:,:-1]
        # x_shift_minum_x = input[:,:,1:] - input[:,:,:-1]
        # grad_multiply = y_shift_minum_y / (x_shift_minum_x + 1e-8)
        # ## adain multiply.
        # var_a = input.var(dim=[2], keepdim=True)
        # var_b = output.var(dim=[2], keepdim=True)
        # sig_a = (var_a + 1e-8).sqrt()
        # sig_b = (var_b + 1e-8).sqrt()
        # grad_multiply_adain = sig_b/sig_a
        # grad_multiply_adain = grad_multiply_adain.expand(grad_multiply.size()) ## B*D*(C-1)
        # threshold = 10.0
        # grad_multiply[grad_multiply > grad_multiply_adain * threshold] = grad_multiply_adain[grad_multiply > grad_multiply_adain * threshold]
        # grad_multiply[grad_multiply < grad_multiply_adain / threshold] = grad_multiply_adain[grad_multiply < grad_multiply_adain / threshold]
        # grad_multiply = grad_multiply.view(grad_multiply.size(0)  * grad_multiply.size(1), 1, grad_multiply.size(2))
        # left_right_averaged_grad_multiply = kernel_two_conv(grad_multiply) ## B*D*C
        # smoothed_grad_multiply = kernel_thr_conv(left_right_averaged_grad_multiply) ## B*D*C
        # smoothed_grad_multiply = smoothed_grad_multiply.view(input.size())
        # del kernel_two_conv
        # del kernel_thr_conv
        ###################################################################################################################################################

        # ### last try, 跟之前的 identity 0 类似
        y_plus_1 = torch.cat((output[:,:,1:], output[:,:,-1:]), -1)
        y_minus_1 = torch.cat((output[:,:,:1], output[:,:,:-1]), -1)
        x_plus_1 = torch.cat((input[:,:,1:], input[:,:,-1:]), -1)
        x_minus_1 = torch.cat((input[:,:,1:], input[:,:,-1:]), -1)
        diff_y = y_plus_1 - y_minus_1
        diff_x = x_plus_1 - x_minus_1
        grad_multiply = (diff_y + 1e-6) / (diff_x + 1e-6)
        # adain multiply.
        var_a = input.var(dim=[2], keepdim=True)
        var_b = output.var(dim=[2], keepdim=True)
        sig_a = (var_a + 1e-6).sqrt()
        sig_b = (var_b + 1e-6).sqrt()
        grad_multiply_adain = sig_b/sig_a
        grad_multiply_adain = grad_multiply_adain.expand(grad_multiply.size())
        # set a min max threshold, try 5.
        threshold = 5.0
        grad_multiply[grad_multiply > grad_multiply_adain * threshold] = grad_multiply_adain[grad_multiply > grad_multiply_adain * threshold]
        grad_multiply[grad_multiply < grad_multiply_adain / threshold] = grad_multiply_adain[grad_multiply < grad_multiply_adain / threshold]

        return grad_output * grad_multiply, None

def deactivate_efdmix(m):
    if type(m) == EFDMix:
        m.set_activation_status(False)


def activate_efdmix(m):
    if type(m) == EFDMix:
        m.set_activation_status(True)


def random_efdmix(m):
    if type(m) == EFDMix:
        m.update_mix_method('random')


def crossdomain_efdmix(m):
    if type(m) == EFDMix:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_efdmix(model):
    # Assume EFDMix was initially activated
    try:
        model.apply(deactivate_efdmix)
        yield
    finally:
        model.apply(activate_efdmix)


@contextmanager
def run_with_efdmix(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_efdmix)

    elif mix == 'crossdomain':
        model.apply(crossdomain_efdmix)

    try:
        model.apply(activate_efdmix)
        yield
    finally:
        model.apply(deactivate_efdmix)


class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization
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
        self.replace = Replace_with_grad.apply

    def __repr__(self):
        return f'EFDMix(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        x_view = x.view(B,C, -1)
        ## sort input vectors.
        value_x, index_x = torch.sort(x_view)
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

        # x_view_copy = x_view.detach().clone()  ## one more copy.
        # inverse_index = index_x.argsort(-1)
        # x_view_copy = x_view_copy * lmda + value_x[perm].gather(-1, inverse_index) * (1-lmda)
        # new_x = x_view + (x_view_copy - x_view.detach())

        ## EFDMix, cvpr submission.
        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index) * (1-lmda)  # note gather conduct a deep copy here.
        new_x = x_view + (x_view_copy - x_view.detach() * (1-lmda))
        return new_x.view(B, C, W, H)

        # ## EFDMix, cvpr paper, no gradient to y. Bad result!
        # inverse_index = index_x.argsort(-1)
        # x_view_copy = value_x[perm].gather(-1, inverse_index) * (1-lmda)  # note gather conduct a deep copy here.
        # new_x = x_view + (x_view_copy - x_view).detach() * (1-lmda)

        # ### EFDMix, hongwei 5
        # inverse_index = index_x.argsort(-1)
        # output_ranked_no_grad = (value_x * lmda + value_x[perm] * (1-lmda)).detach()
        # output_with_grad = self.replace(value_x, output_ranked_no_grad)
        # new_x = output_with_grad.gather(-1, inverse_index)
        # return new_x.view(B, C, W, H)