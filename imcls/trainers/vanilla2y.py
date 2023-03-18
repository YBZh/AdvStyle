from torch.nn import functional as F
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import deactivate_training_weight_y, activate_training_weight_y, random_mixstyle_y, activate_mixstyle_y, run_with_mixstyle_y, deactivate_mixstyle_y, crossdomain_mixstyle_y, run_without_mixstyle_y
import numpy as np
import os.path as osp
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler
import time
import datetime
from dassl.modeling import build_head, build_backbone
import torch.nn as nn

class SimpleNet_Y(nn.Module):
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
            weight_list=cfg.MODEL.BACKBONE.WEIGHT_LIST,
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
            f, _,  x_res1, x_res2, x_res3, x_res4 = self.backbone(x)
            # return f
            if self.head is not None:
                f, _ = self.head(f)
            if self.classifier is None:
                return f
            y = self.classifier(f)
            if return_feature:
                return y, x_res1, x_res2, x_res3, x_res4
            return y
        else:  ## with label mix
            f,y_output, x_res1, x_res2, x_res3, x_res4 = self.backbone(x, y_input)
            # return f
            if self.head is not None:
                f, y_output = self.head(f, y_output)
            if self.classifier is None:
                return f, y_output

            y = self.classifier(f)
            if return_feature:
                return y, f, y_output, x_res1, x_res2, x_res3, x_res4
            return y, y_output



@TRAINER_REGISTRY.register()
class Vanilla2Y(TrainerX):  ###
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA2.MIX
        # stage = cfg.TRAINER.VANILLA2.STAGE
        # self.stage = stage


        if mix == 'random':
            self.model.apply(random_mixstyle_y)
            print('MixStyle: random mixing')

        elif mix == 'crossdomain':
            self.model.apply(crossdomain_mixstyle_y)
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
        self.model = SimpleNet_Y(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    def train(self):
        self.train_y(self.start_epoch, self.max_epoch)

    def train_y(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.before_train()

        # self.stage = 'one'
        # print('training stage one: training a baseline with the ERM ')
        # self.model.apply(deactivate_mixstyle_y)  ## no mixstyle.
        # self.model.apply(deactivate_training_weight_y) ## no training style weight
        # self.before_train()
        # for self.epoch in range(self.start_epoch, self.max_epoch):
        #     self.before_epoch()
        #     self.run_epoch()
        #     self.after_epoch()
        # self.after_train()
        #
        # self.stage = 'two'
        # print('training stage two: fixing the model weights, training the style weight with MixStyle.')
        # self.model.apply(activate_mixstyle_y)  ##  with mixstyle.
        # self.model.apply(activate_training_weight_y)  ##  learning style weight.
        # self.start_epoch = 0
        # for self.epoch in range(self.start_epoch, self.max_epoch):
        #     self.before_epoch()
        #     self.run_epoch()
        #     self.after_epoch()
        # self.after_train()

        self.stage = 'three'
        print('training stage three: fixing the style weight, training the model weights with MixStyle.')
        self.model.apply(activate_mixstyle_y)  ## with mixstyle (i.e., use style weight)
        self.model.apply(deactivate_training_weight_y) ## but fix the style weight
        self.start_epoch = 0
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()


    def set_model_mode(self, mode='train', names=None):
        names = self.get_model_names(names)  ## all model
        for name in names:
            if mode == 'train':
                if self.stage == 'two': ##
                    self._models[name].train()
                    for param in self.model.parameters():
                        if param.size(0) == 1:
                            ## only learn the style weight
                            param.requires_grad = True
                        else:
                            ## set other paramters
                            param.requires_grad = False
                else:
                    self._models[name].train()
            else:
                self._models[name].eval()

    def run_epoch(self):
        self.set_model_mode('train')

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            # names = self.get_model_names()
            # print(names) ## ['model']
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            # n_iter = self.epoch * self.num_batches + self.batch_idx
            # for name, meter in losses.meters.items():
            #     self.write_scalar('train/' + name, meter.avg, n_iter)
            # self.write_scalar('train/lr', self.get_current_lr(), n_iter)
            end = time.time()
        ### print the style weight in different residual block.
        for param in self.model.parameters():
            if param.size(0) == 1:
                print(param.data.item())

    ## only learn the style weight

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        label_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()

        output, label_mixed = self.model(input, label_onehot)
        loss = F.cross_entropy(output, label_mixed)
        self.model_backward_and_update(loss)

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
    
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        # x_res1, x_res2, x_res3, x_res4

        # out_embed = []
        # out_embed2 = []
        # out_domain = []
        # out_label = []
        #
        # out_embed_mean = []
        # out_embed_var = []
        # out_embed_third = []
        # out_embed_fourth = []
        # out_embed_infinity = []
        #
        # split = self.cfg.TEST.SPLIT
        # data_loader = self.val_loader if split == 'val' else self.test_loader
        #
        # print('Extracting style features')
        #
        # for batch_idx, batch in enumerate(data_loader):
        #     input = batch['img'].to(self.device)
        #     label = batch['label']
        #     domain = batch['domain']
        #     impath = batch['impath']
        #
        #     # model should directly output features or style statistics
        #     # raise NotImplementedError
        #     output = self.model(input) ## feature: N*C*W*H
        #
        #     ## 1. mean, variance
        #     mu = output.mean(dim=[2, 3])
        #     var = output.var(dim=[2, 3])
        #     sig = (var + 1e-8).sqrt()
        #     mu_var = torch.cat((mu, var), dim=1)
        #     mu_var = mu_var.cpu().numpy()
        #
        #     out_embed_mean.append(mu.cpu().clone().numpy())
        #     out_embed_var.append(sig.cpu().clone().numpy())
        #
        #     ## 0-1 normalized feature.
        #     mu = output.mean(dim=[2, 3], keepdim=True)
        #     var = output.var(dim=[2, 3], keepdim=True)
        #     sig = (var + 1e-8).sqrt()
        #     mu, sig = mu.detach(), sig.detach()
        #     x = (output - mu) / sig  ## N*C*W*H
        #
        #
        #     B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        #     x_view = x.view(B, C, -1)
        #     value_x = x_view
        #     # value_x, index_x = torch.sort(x_view) ## B*C*D
        #     value_x = value_x[:, 1:2, :] ## only one channel.
        #     value_x = value_x.view(B, -1).cpu().numpy()
        #     # print(value_x)
        #
        #     third_order = torch.pow(x_view, 3).mean(-1).cpu().numpy()
        #     fourth_order = torch.pow(x_view, 4).mean(-1).cpu().numpy()
        #     infinity_order = torch.max(x_view, -1)[0].cpu().numpy()
        #
        #     out_embed_infinity.append(infinity_order)
        #     out_embed_third.append(third_order)
        #     out_embed_fourth.append(fourth_order)
        #
        #     # output = output.cpu().numpy()
        #     # out_embed.append(output)
        #     out_embed.append(mu_var)
        #     out_embed2.append(value_x)
        #     out_domain.append(domain.numpy())
        #     out_label.append(label.numpy()) # CLASS LABEL
        #
        #     print('processed batch-{}'.format(batch_idx + 1))
        #
        # out_embed = np.concatenate(out_embed, axis=0)
        # out_embed2 = np.concatenate(out_embed2, axis=0)
        # out_embed_mean = np.concatenate(out_embed_mean, axis=0)
        # out_embed_var = np.concatenate(out_embed_var, axis=0)
        # out_embed_third = np.concatenate(out_embed_third, axis=0)
        # out_embed_fourth = np.concatenate(out_embed_fourth, axis=0)
        # out_embed_infinity = np.concatenate(out_embed_infinity, axis=0)
        # out_domain = np.concatenate(out_domain, axis=0)
        # out_label = np.concatenate(out_label, axis=0)
        # print('shape of feature matrix:', out_embed.shape)
        # out = {
        #     'embed': out_embed,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))
        #
        #
        # print('shape of feature matrix:', out_embed2.shape)
        # out = {
        #     'embed': out_embed2,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed_hist.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))
        #
        #
        # print('shape of feature matrix:', out_embed_mean.shape)
        # out = {
        #     'embed': out_embed_mean,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed_mean.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))
        #
        #
        # print('shape of feature matrix:', out_embed_var.shape)
        # out = {
        #     'embed': out_embed_var,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed_var.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))
        #
        #
        #
        #
        #
        # print('shape of feature matrix:', out_embed_third.shape)
        # out = {
        #     'embed': out_embed_third,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed_third.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))
        #
        #
        # print('shape of feature matrix:', out_embed_fourth.shape)
        # out = {
        #     'embed': out_embed_fourth,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed_fourth.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))
        #
        #
        # print('shape of feature matrix:', out_embed_infinity.shape)
        # out = {
        #     'embed': out_embed_infinity,
        #     'domain': out_domain,
        #     'dnames': source_domains,
        #     'label': out_label
        # }
        # out_path = osp.join(output_dir, 'embed_infinity.pt')
        # torch.save(out, out_path)
        # print('File saved to "{}"'.format(out_path))

