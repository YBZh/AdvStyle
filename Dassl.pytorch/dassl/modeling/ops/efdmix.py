import random
from contextlib import contextmanager
import torch
import torch.nn as nn

# # Inherit from Function
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
        # # weight prior as weight in AdaIn, and bias prior as bias in AdaIN
        # mu_a = input.mean(dim=[2], keepdim=True)
        # var_a = input.var(dim=[2], keepdim=True)
        # mu_b = output.mean(dim=[2], keepdim=True)
        # var_b = output.var(dim=[2], keepdim=True)
        # sig_a = (var_a + 1e-6).sqrt()
        # sig_b = (var_b + 1e-6).sqrt()
        # w_0 = sig_b / sig_a
        # b_0 = mu_b - (sig_b / sig_a) * mu_a # B*C*1
        # grad_multiply = (w_0 - b_0 * input + input * output) / (torch.pow(input, 2) + 1)

        # ## hongwei' method, it is easy to present inf/nan gradient.
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
        #
        # # ## 用局部的scale 作为 0 位置的scale, 这个应该是合理的，先找到不为0 的位置，然后基于该位置向外扩展，然后将算出来的 scale 作为0 位置的scale.
        # # ## (input==0) | (output==0)  --> 全0位置的最长值， 取index, 然后index 乘以 1.33， 取值，然后减去初始值/ 对应的x 减去初始值--> BC1.
        # index = (((input==0) | (output==0)).sum(2, keepdim=True) * 1.33).long()
        # index[index >= input.size(2)] = input.size(2) - 1
        # local_scale = torch.gather(output, 2, index) / (torch.gather(input, 2, index) + 1e-6) ## BC1
        # local_scale = local_scale.expand(grad_multiply.size()) ## BCD
        # grad_multiply[(input == 0) & (output == 0)] = local_scale[(input == 0) & (output == 0)]
        #
        # ## set a min max threshold, try 5. hongwei_thr5
        # threshold = 5.0
        # grad_multiply[grad_multiply > grad_multiply_adain * threshold] = grad_multiply_adain[grad_multiply > grad_multiply_adain * threshold]
        # grad_multiply[grad_multiply < grad_multiply_adain / threshold] = grad_multiply_adain[grad_multiply < grad_multiply_adain / threshold]


        #################################################################################################################################################################
        # # ### if exist zero in fenzi or fenmu, turn it to adain multiply.
        # y_plus_1 = torch.cat((output[:,:,1:], output[:,:,-1:]), -1)
        # y_minus_1 = torch.cat((output[:,:,:1], output[:,:,:-1]), -1)
        # x_plus_1 = torch.cat((input[:,:,1:], input[:,:,-1:]), -1)
        # x_minus_1 = torch.cat((input[:,:,1:], input[:,:,-1:]), -1)
        # diff_y = y_plus_1 - y_minus_1
        # diff_x = x_plus_1 - x_minus_1
        # grad_multiply = (diff_y) / (diff_x + 1e-8)
        # ## adain multiply.
        # var_a = input.var(dim=[2], keepdim=True)
        # var_b = output.var(dim=[2], keepdim=True)
        # sig_a = (var_a + 1e-8).sqrt()
        # sig_b = (var_b + 1e-8).sqrt()
        # grad_multiply_adain = sig_b/sig_a
        # grad_multiply_adain = grad_multiply_adain.expand(grad_multiply.size())
        #
        # or_zero_mask = (diff_y == 0) | (diff_x == 0)
        # grad_multiply[or_zero_mask] = grad_multiply_adain[or_zero_mask]
        # ## set a min max threshold, try 5.
        # threshold = 5.0
        # grad_multiply[grad_multiply > grad_multiply_adain * threshold] = grad_multiply_adain[grad_multiply > grad_multiply_adain * threshold]
        # grad_multiply[grad_multiply < grad_multiply_adain / threshold] = grad_multiply_adain[grad_multiply < grad_multiply_adain / threshold]
        #######################################################################################################################################################################

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
        #
        # ##
        #
        # threshold = 10.0
        # grad_multiply[grad_multiply > grad_multiply_adain * threshold] = grad_multiply_adain[grad_multiply > grad_multiply_adain * threshold]
        # grad_multiply[grad_multiply < grad_multiply_adain / threshold] = grad_multiply_adain[grad_multiply < grad_multiply_adain / threshold]
        # grad_multiply = grad_multiply.view(grad_multiply.size(0)  * grad_multiply.size(1), 1, grad_multiply.size(2))
        # left_right_averaged_grad_multiply = kernel_two_conv(grad_multiply) ## B*D*C
        # smoothed_grad_multiply = kernel_thr_conv(left_right_averaged_grad_multiply) ## B*D*C
        # smoothed_grad_multiply = smoothed_grad_multiply.view(input.size())
        # del kernel_two_conv
        # del kernel_thr_conv
        # ####################################################################################################################################################


        # ### last try, 跟之前的 identity 0 的结果类似, 不过去掉了threshold.
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

        return grad_output * grad_multiply, grad_output



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


class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Exact Feature Distribution Matching for  Arbitrary Style Transfer and Domain Generalization
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
        self.sorting = 'quicksort'
        self._activated = True
        self.replace = Replace_with_grad.apply


    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def update_sorting_method(self, sorting='quicksort'):
        self.sorting = sorting

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        # print('sorting method is: ', self.sorting)
        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        ############################# mixhist via mixsorting
        x_view = x.view(B,C, -1)
        if self.sorting == 'quicksort':
            value_x, index_x = torch.sort(x_view)  # sort conduct a deep copy here.
        # elif self.sorting == 'index': ## i.e., preserving in the paper.
        #     value_x, index_x = torch.sort(x_view, stable=True)
        # elif self.sorting == 'random':
        #     value_x, index_x = torch.sort(x_view)
        #     value_x_rand, index_x_rand = torch.sort(x_view + torch.rand(x_view.size()).to(x.device) * 1e-32) ## random shuffle the index of identical value
        #     index_x = index_x_rand
        # elif self.sorting == 'neighbor': ## including local mean with a fixed kernel.
        #     value_x, index_x = torch.sort(x_view)
        #     temp_conv = torch.nn.Conv2d(C, C, 3, stride=1, padding=1, groups=C, bias=False, padding_mode='replicate')
        #     # print(temp_conv.weight.data.size())
        #     temp_conv.weight.data.fill_(1)
        #     temp_conv.weight.data[:,:,0,0] = 0
        #     temp_conv.weight.data[:, :, 2, 2] = 0
        #     temp_conv.weight.data[:, :, 0, 2] = 0
        #     temp_conv.weight.data[:, :, 2, 0] = 0
        #     temp_conv = temp_conv.cuda()
        #     for param in temp_conv.parameters():
        #         param.requires_grad = False
        #     neighbor = temp_conv(x.detach())
        #     value_x_rand, index_x_rand = torch.sort(x_view + neighbor.view(B,C, -1) * 1e-32)
        #     index_x = index_x_rand
        #     del temp_conv
        # else:
        #     raise NotImplementedError
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        ########################## calculate # percent of equivalent feature values
        # count = 0
        # count_0 = 0
        # count_not_0 = 0
        # for i in range(B):
        #     for j in range(C):
        #         value, counts = torch.unique(x_view[i][j].ravel(), return_counts=True)
        #         count += (counts[counts!=1]).sum().item() / x_view[i][j].ravel().size(0)
        #         counts_value_0 = counts[value==0]
        #         counts_value_not_0 = counts[value!=0]
        #         count_0 += (counts_value_0[counts_value_0!=1]).sum().item() / x_view[i][j].ravel().size(0)
        #         count_not_0 += (counts_value_not_0[counts_value_not_0!=1]).sum().item() / x_view[i][j].ravel().size(0)
        #         # if (counts[counts!=1]).sum().item() != 0:
        #         #     print(value[counts!=1], counts[counts!=1])
        #         # print(counts[counts!=1])
        # print(count / (B*C), W*H)  ##%% before ReLU: 5/1000,  after ReLU: 20/100 ~ 40/100 different layers
        # print(count_0 / (B * C), W * H)
        # print(count_not_0 / (B * C), W * H)

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

        # ########################### EFDM, original
        # inverse_index = index_x.argsort(-1)
        # x_view_copy = value_x[perm].gather(-1, inverse_index)  # note gather conduct a deep copy here.
        # new_x = x_view + (x_view_copy - x_view.detach())
        # return new_x.view(B, C, W, H)

        # ########################## EFDMix, original, CVPR submission.
        # inverse_index = index_x.argsort(-1)
        # x_view_copy = value_x[perm].gather(-1, inverse_index)  # note gather conduct a deep copy here.
        # new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)   ## 这里其实是引入了另外一个的额外样本，这个样本可能不是同类的，这样为啥结果会变好呢？ 这不科学？
        # return new_x.view(B, C, W, H)

        # ########################## EFDMix, original.  This is the ideal one, but its results are not very good.
        # inverse_index = index_x.argsort(-1)
        # x_view_copy = value_x[perm].gather(-1, inverse_index)  # note gather conduct a deep copy here.
        # new_x = x_view + (x_view_copy - x_view).detach() * (1-lmda)
        # return new_x.view(B, C, W, H)

        ########################### EFDMix, weight prior as weight in AdaIn, and bias prior as bias in AdaIN
        inverse_index = index_x.argsort(-1)
        output_ranked_no_grad = (value_x * lmda + value_x[perm] * (1-lmda))
        output_with_grad = self.replace(value_x, output_ranked_no_grad)
        new_x = output_with_grad.gather(-1, inverse_index)
        return new_x.view(B, C, W, H)