"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import warnings
import torch
import torch.nn as nn

from .radam import RAdam

AVAI_OPTIMS = ['adam', 'amsgrad', 'sgd', 'rmsprop', 'radam', 'adamw']


def build_optimizer(model, optim_cfg):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            'Unsupported optim: {}. Must be one of {}'.format(
                optim, AVAI_OPTIMS
            )
        )

    if staged_lr:
        if not isinstance(model, nn.Module):
            raise TypeError(
                'When staged_lr is True, model given to '
                'build_optimizer() must be an instance of nn.Module'
            )

        if isinstance(model, nn.DataParallel):
            model = model.module

        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn(
                    'new_layers is empty, therefore, staged_lr is useless'
                )
            new_layers = [new_layers]

        base_params = []
        base_layers = []
        new_params = []

        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        param_groups = [
            {
                'params': base_params,
                'lr': lr * base_lr_mult
            },
            {
                'params': new_params
            },
        ]

    else:
        if isinstance(model, nn.Module):
            param_groups = model.parameters()
        else:
            param_groups = model

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == 'radam':
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2)
        )

    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2)
        )

    return optimizer

def build_optimizer_advstle(model, optim_cfg):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            'Unsupported optim: {}. Must be one of {}'.format(
                optim, AVAI_OPTIMS
            )
        )

    if staged_lr:
        if not isinstance(model, nn.Module):
            raise TypeError(
                'When staged_lr is True, model given to '
                'build_optimizer() must be an instance of nn.Module'
            )

        if isinstance(model, nn.DataParallel):
            model = model.module

        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn(
                    'new_layers is empty, therefore, staged_lr is useless'
                )
            new_layers = [new_layers]

        base_params = []
        base_layers = []
        new_params = []
        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        param_groups = [
            {
                'params': base_params,
                'lr': lr * base_lr_mult
            },
            {
                'params': new_params
            },
        ]

    else:
        base_params = []
        base_layers = []
        new_params = []
        all_params = []
        for name, module in model.named_modules(remove_duplicate=False):
            # print(name)   ## backbone, classifier.
            # backbone.mixstyle2
            # backbone.mixstyle2.grl_mean
            # backbone.mixstyle2.grl_std
            if 'mixstyle' in name:
                new_params += [p for p in module.parameters()]
            elif name == 'backbone.conv1':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
            elif name == 'backbone.bn1':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
            elif name == 'backbone.layer1':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
            elif name == 'backbone.layer2':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
            elif name == 'backbone.layer3':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
            elif name == 'backbone.layer4':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
            elif name == 'classifier':
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        print(len(base_params))
        print(len(new_params))
        param_groups = [
            {
                'params': base_params,
            }
        ]
        param_groups_adv = [
            {
                'params': new_params
            }
        ]
        # if isinstance(model, nn.Module):
        #     param_groups = model.parameters()
        # else:
        #     param_groups = model

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
        optimizer_adv = torch.optim.Adam(
            param_groups_adv,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    # elif optim == 'amsgrad':
    #     optimizer = torch.optim.Adam(
    #         param_groups,
    #         lr=lr,
    #         weight_decay=weight_decay,
    #         betas=(adam_beta1, adam_beta2),
    #         amsgrad=True,
    #     )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

        optimizer_adv = torch.optim.SGD(
            param_groups_adv,
            lr=lr,
            momentum=0.0, ## do not use momentum when find adversarial direction.
            weight_decay=0.0,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    # elif optim == 'rmsprop':
    #     optimizer = torch.optim.RMSprop(
    #         param_groups,
    #         lr=lr,
    #         momentum=momentum,
    #         weight_decay=weight_decay,
    #         alpha=rmsprop_alpha,
    #     )
    #
    # elif optim == 'radam':
    #     optimizer = RAdam(
    #         param_groups,
    #         lr=lr,
    #         weight_decay=weight_decay,
    #         betas=(adam_beta1, adam_beta2)
    #     )
    #
    # elif optim == 'adamw':
    #     optimizer = torch.optim.AdamW(
    #         param_groups,
    #         lr=lr,
    #         weight_decay=weight_decay,
    #         betas=(adam_beta1, adam_beta2)
    #     )

    return optimizer, optimizer_adv
