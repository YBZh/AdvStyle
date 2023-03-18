from torch.nn import functional as F
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle
import random

@TRAINER_REGISTRY.register()
class Vanilla3(TrainerX):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA2.MIX
        self.mix_or_swap = cfg.TRAINER.VANILLA3.mix_or_swap
        self.mix_alpha = cfg.TRAINER.VANILLA3.mix_alpha
        self.statistic_weight = cfg.TRAINER.VANILLA3.statistic_weight

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
        output, feature = self.model(input, return_feature=True)
        if random.random() > 0.5:
            loss = F.cross_entropy(output, label)
        else:
            ###########
            eps = 1e-6
            mu = feature.mean(dim=[2,3], keepdim=True)
            var = feature.var(dim=[2,3], keepdim=True)
            sig = (var + eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            feature_normed = (feature - mu) / sig
            B = feature.size(0)
            perm = torch.randperm(B)
            mu2, sig2 = mu[perm], sig[perm]
            if  self.mix_or_swap == 'swap':
                mu_new = mu2
                sig_new = sig2
                feature_new = feature_normed * sig_new + mu_new
                output_new = self.model.classifier(feature_new)
                loss = F.cross_entropy(output_new, label) * (1 - self.statistic_weight) + F.cross_entropy(output_new, label[perm]) *  self.statistic_weight
            elif  self.mix_or_swap == 'mix':
                self_beta = torch.distributions.Beta(self.mix_alpha, self.mix_alpha)
                lmda = self_beta.sample((B, 1, 1, 1))
                lmda = lmda.to(feature.device)
                mu_new = mu * lmda + mu2 * (1-lmda)
                sig_new = sig * lmda + sig2 * (1-lmda)
                feature_new = feature_normed * sig_new + mu_new
                output_new = self.model.classifier(feature_new)

                loss = (F.cross_entropy(output_new, label, reduction='none') * (1 - self.statistic_weight * (1-lmda[:,0,0,0]))).mean() + \
                       (F.cross_entropy(output_new, label[perm], reduction='none') * self.statistic_weight * (1-lmda[:,0,0,0])).mean()
            else:
                raise NotImplementedError

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
    
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        out_embed = []
        out_domain = []
        out_label = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']

            # model should directly output features or style statistics
            raise NotImplementedError
            output = self.model(input)
            output = output.cpu().numpy()
            out_embed.append(output)
            out_domain.append(domain.numpy())
            out_label.append(label.numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embed = np.concatenate(out_embed, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)
        print('shape of feature matrix:', out_embed.shape)
        out = {
            'embed': out_embed,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))
