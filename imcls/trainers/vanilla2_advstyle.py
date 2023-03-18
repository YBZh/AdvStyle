from torch.nn import functional as F
import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle, quicksort_mixstyle, index_mixstyle, randomsort_mixstyle, neighbor_mixstyle, reset_init_flag
import numpy as np
import os.path as osp
# import ipdb
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights
)
from dassl.optim import build_optimizer_advstle, build_lr_scheduler

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            before_relu=model_cfg.BACKBONE.BEFORE_RELU,
            adv_weight=model_cfg.BACKBONE.ADV_WEIGHT,
            mix_weight=model_cfg.BACKBONE.MIX_WEIGHT,
            **kwargs
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs
            )
            fdim = self.head.out_features


        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)
            # self.classifier = our_classifier(fdim, num_classes, self.backbone.layer4)
        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, y_input=None, return_feature=False):
        if y_input == None:
            f, _ = self.backbone(x)
            # return f
            if self.head is not None:
                f, _ = self.head(f)
            if self.classifier is None:
                return f
            y = self.classifier(f)
            if return_feature:
                return y, f
            return y
        else:  ## with label mix
            f,y_output = self.backbone(x, y_input)
            # return f
            if self.head is not None:
                f, y_output = self.head(f, y_output)
            if self.classifier is None:
                return f, y_output

            y = self.classifier(f)
            if return_feature:
                return y, f, y_output
            return y, y_output

@TRAINER_REGISTRY.register()
class Vanilla2_advstyle(TrainerX):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA2.MIX

        ### this only apply to EFDMix for analyses. we by default adopt the quicksort strategies.
        #sorting = cfg.TRAINER.VANILLA2.SORTING
        # if sorting == 'quicksort':
        #     self.model.apply(quicksort_mixstyle)
        #     print('MixHist sorting: quicksort')
        # elif sorting == 'index':
        #     self.model.apply(index_mixstyle)
        #     print('MixHist sorting: index')
        # elif sorting == 'random':
        #     self.model.apply(randomsort_mixstyle)
        #     print('MixHist sorting: random')
        # elif sorting == 'neighbor':
        #     self.model.apply(neighbor_mixstyle)
        #     print('MixHist sorting: random')
        # else:
        #     raise NotImplementedError

        if mix == 'random':
            self.model.apply(random_mixstyle)
            print('MixStyle: random mixing')

        elif mix == 'crossdomain':
            self.model.apply(crossdomain_mixstyle)
            print('MixStyle: cross-domain mixing')

        else:
            raise NotImplementedError

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print('Building model')
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim, self.optim_adv = build_optimizer_advstle(self.model, cfg.OPTIM)
        # print(self.optim)
        # print(self.optim_adv)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.sched_adv = build_lr_scheduler(self.optim_adv, cfg.OPTIM)
        self.register_model('adv', self.model, self.optim_adv, self.sched_adv)
        self.register_model('model', self.model, self.optim, self.sched)


    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)

        # self.model.apply(reset_init_flag)
        for i in range(3):
            output = self.model(input)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss, 'adv')
            # print('adversarial training')
            # for name, module in self.model.named_modules(remove_duplicate=False):
            #     # print(name)   ## backbone, classifier.
            #     # backbone.mixstyle2
            #     # backbone.mixstyle2.grl_mean
            #     # backbone.mixstyle2.grl_std
            #     # if name == 'backbone.mixstyle1':
            #     #     print('mixstyle1')
            #     #     print(module.adv_mean_std[0][0][0][0])
            #     # if name == 'backbone.mixstyle2':
            #     #     print('mixstyle2')
            #     #     print(module.adv_mean_std[0][0][0][0])
            #     # if name == 'backbone.mixstyle3':
            #     #     print('mixstyle3')
            #     #     print(module.adv_mean_std[0][0][0][0])
            #     # if name == 'classifier':
            #     #     print('classifier')
            #     #     print(module.weight[0][0])

        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss, 'model')
        # print('common training')
        # for name, module in self.model.named_modules(remove_duplicate=False):
        #     # print(name)   ## backbone, classifier.
        #     # backbone.mixstyle2
        #     # backbone.mixstyle2.grl_mean
        #     # backbone.mixstyle2.grl_std
        #     if name == 'backbone.mixstyle1':
        #         print('mixstyle1')
        #         print(module.adv_mean_std[0][0][0][0])
        #     if name == 'backbone.mixstyle2':
        #         print('mixstyle2')
        #         print(module.adv_mean_std[0][0][0][0])
        #     if name == 'backbone.mixstyle3':
        #         print('mixstyle3')
        #         print(module.adv_mean_std[0][0][0][0])
        #     if name == 'classifier':
        #         print('classifier')
        #         print(module.weight[0][0])

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary



    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        if loss.requires_grad:
            self.model_update(names)

    def model_update(self, names=None):
        names = self.get_model_names(names)
        # print(names) ## 'model'
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        if loss.requires_grad:
            loss.backward()

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
