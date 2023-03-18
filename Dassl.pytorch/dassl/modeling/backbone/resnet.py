import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, before_relu =False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.before_relu = before_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.before_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, before_relu =False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.before_relu = before_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.before_relu:
            out = self.relu(out)

        return out

class ResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,  ## default 0.1
        ins_layers=[], ## for instance normalization
        ign_layers=[], ## for instance gaussian normalization
        cnsn_layers = [],
        snr_layers=[],
        channel_layers = [],
        before_relu=False,
        adv_weight=1.0,
        mix_weight=1.0,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()
        self.before_relu = before_relu
        self.relu = nn.ReLU(inplace=False)
        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], before_relu = self.before_relu)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, before_relu =self.before_relu)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, before_relu =self.before_relu)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, before_relu =self.before_relu)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        assert len(ms_layers) == 0  or len(ins_layers) == 0 or len(ign_layers) == 0 or len(ins_layers) == 0 or len(snr_layers) == 0
        self.mixstyle = None
        if ms_layers:
            self.mixstyle1 = ms_class(p=ms_p, alpha=ms_a)
            self.mixstyle2 = ms_class(p=ms_p, alpha=ms_a)
            self.mixstyle3 = ms_class(p=ms_p, alpha=ms_a)
            self.mixstyle4 = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3', 'layer4']
            print(f'Insert MixStyle after {ms_layers}')
            self.ms_layers = ms_layers
        elif ins_layers:
            raise NotImplementedError ## layers[0] --> 64
            self.mixstyle1 = nn.InstanceNorm2d(layers[0])
            self.mixstyle2 = nn.InstanceNorm2d(layers[1])
            self.mixstyle3 = nn.InstanceNorm2d(layers[2])
            self.mixstyle4 = nn.InstanceNorm2d(layers[3])
            for layer_name in ins_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3', 'layer4']
            print(f'Insert Instance normalization after {ms_layers}')
            self.ms_layers = ins_layers
        elif ign_layers:
            from dassl.modeling.ops import IGN
            self.mixstyle1 = IGN(layers[0])
            self.mixstyle2 = IGN(layers[1])
            self.mixstyle3 = IGN(layers[2])
            self.mixstyle4 = IGN(layers[3])
            for layer_name in ins_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3', 'layer4']
            print(f'Insert IGN after {ms_layers}')
            self.ms_layers = ign_layers
        elif cnsn_layers:
            from dassl.modeling.ops import CNSN
            self.mixstyle1 = CNSN(chan_num=64 * block.expansion)
            self.mixstyle2 = CNSN(chan_num=128* block.expansion)
            self.mixstyle3 = CNSN(chan_num=256* block.expansion)
            self.mixstyle4 = CNSN(chan_num=512* block.expansion)
            for layer_name in ins_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3', 'layer4']
            print(f'Insert IGN after {ms_layers}')
            self.ms_layers = cnsn_layers
        elif snr_layers:
            from dassl.modeling.ops import SNR
            self.mixstyle1 = SNR(channel_num=64* block.expansion)
            self.mixstyle2 = SNR(channel_num=128* block.expansion)
            self.mixstyle3 = SNR(channel_num=256* block.expansion)
            self.mixstyle4 = SNR(channel_num=512* block.expansion)
            for layer_name in ins_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3', 'layer4']
            print(f'Insert IGN after {ms_layers}')
            self.ms_layers = snr_layers
        elif channel_layers:
            print('mix_weight is', mix_weight)
            self.mixstylec = ms_class(p=ms_p, alpha=ms_a, channel_num=64, adv_weight=adv_weight, mix_weight=mix_weight)
            self.mixstyle0 = ms_class(p=ms_p, alpha=ms_a, channel_num=64, adv_weight=adv_weight, mix_weight=mix_weight)
            self.mixstyle1 = ms_class(p=ms_p, alpha=ms_a, channel_num=64* block.expansion, adv_weight=adv_weight, mix_weight=mix_weight)
            self.mixstyle2 = ms_class(p=ms_p, alpha=ms_a, channel_num=128* block.expansion, adv_weight=adv_weight, mix_weight=mix_weight)
            self.mixstyle3 = ms_class(p=ms_p, alpha=ms_a, channel_num=256* block.expansion, adv_weight=adv_weight, mix_weight=mix_weight)
            self.mixstyle4 = ms_class(p=ms_p, alpha=ms_a, channel_num=512* block.expansion, adv_weight=adv_weight, mix_weight=mix_weight)
            for layer_name in ms_layers:
                assert layer_name in ['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4']
            print(f'Insert MixStyle after {ms_layers}')
            self.ms_layers = channel_layers
        else:
            self.ms_layers = ms_layers

        self._init_params()


    def _make_layer(self, block, planes, blocks, stride=1, before_relu = False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, before_relu=before_relu))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        if 'layerc' in self.ms_layers:
            x = self.mixstylec(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if 'layer0' in self.ms_layers:
            x = self.mixstyle0(x)
        x = self.layer1(x)
        if 'layer1' in self.ms_layers:
            x = self.mixstyle1(x)
        if self.before_relu:
            x = self.relu(x)
        x = self.layer2(x)
        if 'layer2' in self.ms_layers:
            x = self.mixstyle2(x)
        if self.before_relu:
            x = self.relu(x)
        x = self.layer3(x)
        if 'layer3' in self.ms_layers:
            x = self.mixstyle3(x)
        if self.before_relu:
            x = self.relu(x)
        x = self.layer4(x)
        if 'layer4' in self.ms_layers:
            x = self.mixstyle4(x)
        if self.before_relu:
            x = self.relu(x)
        return x




    def forward(self, x):
        f = self.featuremaps(x)
        return f, None
        # v = self.global_avgpool(f)
        # return v.view(v.size(0), -1), None

# _, counts = torch.unique(x.view(-1).ravel(), return_counts=True)
# print('input: ' , (counts[counts!=1]).sum().item() / x.view(-1).ravel().size(0))

def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


@BACKBONE_REGISTRY.register()
def resnet18(pretrained=True, before_relu=False, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], before_relu=before_relu)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet34(pretrained=True, before_relu=False, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], before_relu=before_relu)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50(pretrained=True, before_relu=False, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], before_relu=before_relu)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101(pretrained=True, before_relu=False, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], before_relu=before_relu)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


@BACKBONE_REGISTRY.register()
def resnet152(pretrained=True, before_relu=False, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], before_relu=before_relu)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])

    return model


"""
ms: Residual networks with mixstyle
efdmix: Residual networks with EFDMix
his: Residual networks with Histogram matching
order: Residual networks with AdaMean, AdaStd, AdaIN, EFDM
"""


@BACKBONE_REGISTRY.register()
def resnet18_ms_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_advs_l123(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=AdvStyle,
        channel_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
        )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_advs_lc01234(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=AdvStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_advs_l1234(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=AdvStyle,
        channel_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_l1234(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc01234(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc0123(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc012(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc01(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layerc', 'layer0', 'layer1'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc0(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layerc', 'layer0'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle,
        channel_layers=['layerc'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_advs_lc01234_test(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle_test

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=AdvStyle_test,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_advs_lc01234_test(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle_test

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=AdvStyle_test,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model



@BACKBONE_REGISTRY.register()
def resnet18_advs_l123_test(pretrained=True, before_relu=False, adv_weight=1.0, mix_weight=1.0, **kwargs):
    from dassl.modeling.ops import AdvStyle_test

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=AdvStyle_test,
        channel_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu, adv_weight=adv_weight, mix_weight=mix_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_dsus_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import DSUStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=DSUStyle,
        channel_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_dsus_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import DSUStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=DSUStyle,
        channel_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_dsus_lc01234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import DSUStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=DSUStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_dsus_lc01234(pretrained=True, before_relu=False, adv_weight=1.0, **kwargs):
    from dassl.modeling.ops import DSUStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=DSUStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_dsus_lc0123(pretrained=True, before_relu=False, adv_weight=1.0, **kwargs):
    from dassl.modeling.ops import DSUStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=DSUStyle,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3'], before_relu=before_relu, adv_weight=adv_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_dsus_lc01234_test(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import DSUStyle_test

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=DSUStyle_test,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_dsus_lc01234_test(pretrained=True, before_relu=False, adv_weight=1.0, **kwargs):
    from dassl.modeling.ops import DSUStyle_test

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=DSUStyle_test,
        channel_layers=['layerc', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu, adv_weight=adv_weight
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_order_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixOrders

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixOrders,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_his_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixHistogram

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixHistogram,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model



@BACKBONE_REGISTRY.register()
def resnet18_rs_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import RandStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=RandStyle,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_ins_l123(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ins_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ign_l123(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ign_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_cnsn_l123(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        cnsn_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_snr_l123(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        snr_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_rs_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import RandStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=RandStyle,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ins_l12(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ins_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ign_l12(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ign_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_ms_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms_l14(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms_l23(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms2_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle2w,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms2_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms2_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_msA_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyleA

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyleA,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms2_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms2_l23(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle2w,
        ms_layers=['layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_ms2_l14(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l14(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l23(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=['layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model




@BACKBONE_REGISTRY.register()
def resnet50_rs_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import RandStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=RandStyle,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_rs_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import RandStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=RandStyle,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_his_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixHistogram

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixHistogram,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model



@BACKBONE_REGISTRY.register()
def resnet50_order_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixOrders

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixOrders,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model




@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l14(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=['layer1', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l23(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=['layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ins_l123(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ins_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ins_l12(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ins_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ign_l12(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ign_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_cnsn_l12(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        cnsn_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_snr_l12(pretrained=True, before_relu=False, **kwargs):

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        snr_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l14(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l3(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l2(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model



@BACKBONE_REGISTRY.register()
def resnet50_ms_l23(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l4(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model



@BACKBONE_REGISTRY.register()
def resnet50_ms2_l1234(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer2', 'layer3', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms2_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms2_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_msA_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyleA

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyleA,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms2_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle2w,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms2_l14(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle2w,
        ms_layers=['layer1', 'layer4'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_ms2_l23(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle2w

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle2w,
        ms_layers=['layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

@BACKBONE_REGISTRY.register()
def resnet101_ms_l123(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l12(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l1(pretrained=True, before_relu=False, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1'], before_relu=before_relu
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model
