from torch.nn import functional as F
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle, quicksort_mixstyle, index_mixstyle, randomsort_mixstyle, neighbor_mixstyle
import numpy as np
import os.path as osp
import torch.nn as nn
import ipdb


@TRAINER_REGISTRY.register()
class Vanilla2(TrainerX):
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

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)

        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    # 尝试对 Advstyle 进行 可视化， 但是失败了。 最后的结论是我们的方法没有办法做可视化。
    # @torch.no_grad()
    # def vis(self):
    #     from torchvision.utils import save_image
    #     self.set_model_mode('train')
    #     output_dir = self.cfg.OUTPUT_DIR
    #     source_domains = self.cfg.DATASET.SOURCE_DOMAINS
    #     print('Source domains:', source_domains)
    #
    #     data_loader = self.train_loader_x
    #     decoder = nn.Sequential(
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(512, 256, (3, 3)),
    #         nn.ReLU(),
    #         nn.Upsample(scale_factor=2, mode='nearest'),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(256, 256, (3, 3)),
    #         nn.ReLU(),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(256, 256, (3, 3)),
    #         nn.ReLU(),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(256, 256, (3, 3)),
    #         nn.ReLU(),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(256, 128, (3, 3)),
    #         nn.ReLU(),
    #         nn.Upsample(scale_factor=2, mode='nearest'),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(128, 128, (3, 3)),
    #         nn.ReLU(),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(128, 64, (3, 3)),
    #         nn.ReLU(),
    #         nn.Upsample(scale_factor=2, mode='nearest'),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(64, 64, (3, 3)),
    #         nn.ReLU(),
    #         nn.ReflectionPad2d((1, 1, 1, 1)),
    #         nn.Conv2d(64, 3, (3, 3)),
    #     )
    #     pre_trained_decoder = '/home/yabin/syn_project/mixstyle-release-master/pytorch-AdaIN/models/decoder.pth'
    #     decoder.load_state_dict(torch.load(pre_trained_decoder))
    #
    #     decoder.to('cuda')
    #     # ipdb.set_trace()
    #     # print(list(self.model.children())[0])
    #     self.model = nn.Sequential(list(self.model.children())[0])
    #
    #
    #     print('Extracting style features')
    #
    #
    #     for batch_idx, batch in enumerate(data_loader):
    #         input = batch['img'].to(self.device)
    #
    #         label = batch['label']
    #         domain = batch['domain']
    #         impath = batch['impath']
    #
    #         # model should directly output features or style statistics
    #         # raise NotImplementedError
    #         # print(self.model)
    #         # print(input.size())
    #         feat = self.model(input)
    #         # print('i guess the error occurs before this')
    #         # print(feat.size())
    #         output = decoder(feat) ## feature: N*C*W*H
    #         # print(output.size())
    #         output = output.cpu()
    #         print(output.size())
    #         for i in range(output.size(0)):
    #             output_name = output_dir + '/' + str(i + batch_idx+output.size(0)) + '.jpg'
    #             save_image(output[0], str(output_name))

    ## calculating the A-distance; histogram of mean and standard deviation; and T-SNE visualization.
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        out_domain = []
        out_label = []

        out_embed_mean = []
        out_embed_var = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']
            # ipdb.set_trace()
            print(domain)
            print(impath)

            # model should directly output features or style statistics
            # raise NotImplementedError
            output = self.model(input) ## feature: N*C*W*H

            ## 1. mean, variance
            mu = output.mean(dim=[2, 3])
            var = output.var(dim=[2, 3])
            sig = (var + 1e-8).sqrt()
            mu_var = torch.cat((mu, var), dim=1)
            mu_var = mu_var.cpu().numpy()

            out_embed_mean.append(mu.cpu().clone().numpy())
            out_embed_var.append(sig.cpu().clone().numpy())

            out_domain.append(domain.clone().numpy())
            out_label.append(label.clone().numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embed_mean = np.concatenate(out_embed_mean, axis=0)
        out_embed_var = np.concatenate(out_embed_var, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)


        print('shape of feature matrix:', out_embed_mean.shape)
        out = {
            'embed': out_embed_mean,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_mean_test.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed_var.shape)
        out = {
            'embed': out_embed_var,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_var_test.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))
        print(out_domain)


        #### get feature statistics of training data.
        self.set_model_mode('train')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)


        out_domain = []
        out_label = []

        out_embed_mean = []
        out_embed_var = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.train_loader_x

        print('Extracting style features')
        for repeat in range(1): ## enlarge the training set.
            for batch_idx, batch in enumerate(data_loader):
                input = batch['img'].to(self.device)
                label = batch['label']
                domain = batch['domain']
                impath = batch['impath']

                # model should directly output features or style statistics
                # raise NotImplementedError
                output = self.model(input) ## feature: N*C*W*H

                ## 1. mean, variance
                mu = output.mean(dim=[2, 3])
                var = output.var(dim=[2, 3])
                sig = (var + 1e-8).sqrt()
                mu_var = torch.cat((mu, var), dim=1)
                mu_var = mu_var.cpu().numpy()

                out_embed_mean.append(mu.cpu().clone().numpy())
                out_embed_var.append(sig.cpu().clone().numpy())

                out_domain.append(domain.clone().numpy())
                out_label.append(label.clone().numpy()) # CLASS LABEL

                print('processed batch-{}'.format(batch_idx + 1))


        out_embed_mean = np.concatenate(out_embed_mean, axis=0)
        out_embed_var = np.concatenate(out_embed_var, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)
        print(out_domain)

        print('shape of feature matrix:', out_embed_mean.shape)
        out = {
            'embed': out_embed_mean,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_mean_train.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))


        print('shape of feature matrix:', out_embed_var.shape)
        out = {
            'embed': out_embed_var,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed_var_train.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))






    # # The vanilla vis, which present the different orders of feature statistics.
    # @torch.no_grad()
    # def vis(self):
    #     self.set_model_mode('eval')
    #     output_dir = self.cfg.OUTPUT_DIR
    #     source_domains = self.cfg.DATASET.SOURCE_DOMAINS
    #     print('Source domains:', source_domains)
    #
    #     out_embed = []
    #     out_embed2 = []
    #     out_domain = []
    #     out_label = []
    #
    #     out_embed_mean = []
    #     out_embed_var = []
    #     out_embed_third = []
    #     out_embed_fourth = []
    #     out_embed_infinity = []
    #
    #     split = self.cfg.TEST.SPLIT
    #     data_loader = self.val_loader if split == 'val' else self.test_loader
    #
    #     print('Extracting style features')
    #
    #     for batch_idx, batch in enumerate(data_loader):
    #         input = batch['img'].to(self.device)
    #         label = batch['label']
    #         domain = batch['domain']
    #         impath = batch['impath']
    #
    #         # model should directly output features or style statistics
    #         # raise NotImplementedError
    #         output = self.model(input) ## feature: N*C*W*H
    #
    #         ## 1. mean, variance
    #         mu = output.mean(dim=[2, 3])
    #         var = output.var(dim=[2, 3])
    #         sig = (var + 1e-8).sqrt()
    #         mu_var = torch.cat((mu, var), dim=1)
    #         mu_var = mu_var.cpu().numpy()
    #
    #         out_embed_mean.append(mu.cpu().clone().numpy())
    #         out_embed_var.append(sig.cpu().clone().numpy())
    #
    #         ## 0-1 normalized feature.
    #         mu = output.mean(dim=[2, 3], keepdim=True)
    #         var = output.var(dim=[2, 3], keepdim=True)
    #         sig = (var + 1e-8).sqrt()
    #         mu, sig = mu.detach(), sig.detach()
    #         x = (output - mu) / sig  ## N*C*W*H
    #
    #
    #         B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    #         x_view = x.view(B, C, -1)
    #         value_x = x_view
    #         # value_x, index_x = torch.sort(x_view) ## B*C*D
    #         value_x = value_x[:, 1:2, :] ## only one channel.
    #         value_x = value_x.view(B, -1).cpu().numpy()
    #         # print(value_x)
    #
    #         third_order = torch.pow(x_view, 3).mean(-1).cpu().numpy()
    #         fourth_order = torch.pow(x_view, 4).mean(-1).cpu().numpy()
    #         infinity_order = torch.max(x_view, -1)[0].cpu().numpy()
    #
    #         out_embed_infinity.append(infinity_order)
    #         out_embed_third.append(third_order)
    #         out_embed_fourth.append(fourth_order)
    #
    #         # output = output.cpu().numpy()
    #         # out_embed.append(output)
    #         out_embed.append(mu_var)
    #         out_embed2.append(value_x)
    #         out_domain.append(domain.numpy())
    #         out_label.append(label.numpy()) # CLASS LABEL
    #
    #         print('processed batch-{}'.format(batch_idx + 1))
    #
    #     out_embed = np.concatenate(out_embed, axis=0)
    #     out_embed2 = np.concatenate(out_embed2, axis=0)
    #     out_embed_mean = np.concatenate(out_embed_mean, axis=0)
    #     out_embed_var = np.concatenate(out_embed_var, axis=0)
    #     out_embed_third = np.concatenate(out_embed_third, axis=0)
    #     out_embed_fourth = np.concatenate(out_embed_fourth, axis=0)
    #     out_embed_infinity = np.concatenate(out_embed_infinity, axis=0)
    #     out_domain = np.concatenate(out_domain, axis=0)
    #     out_label = np.concatenate(out_label, axis=0)
    #     print('shape of feature matrix:', out_embed.shape)
    #     out = {
    #         'embed': out_embed,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
    #
    #     print('shape of feature matrix:', out_embed2.shape)
    #     out = {
    #         'embed': out_embed2,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed_hist.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
    #
    #     print('shape of feature matrix:', out_embed_mean.shape)
    #     out = {
    #         'embed': out_embed_mean,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed_mean.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
    #
    #     print('shape of feature matrix:', out_embed_var.shape)
    #     out = {
    #         'embed': out_embed_var,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed_var.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
    #
    #
    #
    #
    #     print('shape of feature matrix:', out_embed_third.shape)
    #     out = {
    #         'embed': out_embed_third,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed_third.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
    #
    #     print('shape of feature matrix:', out_embed_fourth.shape)
    #     out = {
    #         'embed': out_embed_fourth,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed_fourth.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
    #
    #     print('shape of feature matrix:', out_embed_infinity.shape)
    #     out = {
    #         'embed': out_embed_infinity,
    #         'domain': out_domain,
    #         'dnames': source_domains,
    #         'label': out_label
    #     }
    #     out_path = osp.join(output_dir, 'embed_infinity.pt')
    #     torch.save(out, out_path)
    #     print('File saved to "{}"'.format(out_path))
    #
