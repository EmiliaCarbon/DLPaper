import torch
from torchvision import models
from torch import nn

# for getting the output of specific maxpooling layer
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# the config of the VGG net feature extractor
net_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def get_features(model: str = 'vgg16', use_bn: bool = False) -> torch.nn.Sequential:
    assert model in net_cfg.keys()
    feature = []
    channel = 3
    for ele in net_cfg[model]:
        if ele == 'M':
            feature += torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            conv2d = torch.nn.Conv2d(in_channels=channel, out_channels=ele, kernel_size=3, stride=1, padding=1)
            if use_bn:
                feature += [conv2d, torch.nn.BatchNorm2d(channel), torch.nn.ReLU(True)]
            else:
                feature += [conv2d, torch.nn.ReLU(True)]
        channel = ele
    return torch.nn.Sequential(*feature)


class VGGNet(models.VGG):
    def __init__(self, model: str = 'vgg16', pre_trained: bool = True, rm_classifier: bool = True):
        super(VGGNet, self).__init__(features=get_features(model, True))
        assert model in ranges.keys()
        self.range = ranges[model]
        models.vgg16()
        if pre_trained:
            exec(f"self.load_state_dict(models.{model}().state_dict())")

        if rm_classifier:
            del self.classifier

    def forward(self, x: torch.Tensor):
        """
        :param x:
        :return:
        """
        output = []
        for index, (beg, end) in enumerate(self.range):
            for i in range(beg, end):
                x = self.features[i](x)
            output.append(x)
        assert len(output) == 5
        return output

class FCNet(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.vgg_feature = VGGNet('vgg16', True, True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 0)            # initiate the classifier layer weight as 0

    def forward(self, x):
        x3, x4, x5 = self.vgg_feature(x)[2: 5]

        merge = self.relu(self.conv6(x5))
        merge = self.relu(self.conv7(merge))        # 1 / 32
        merge = self.relu(self.deconv1(merge))         # 1 / 16
        merge = self.bn1(merge + x4)
        merge = self.relu(self.deconv2(merge))      # 1 / 8
        merge = self.bn2(merge + x3)
        merge = self.bn3(self.relu(self.deconv3(merge)))  # 1 / 4
        merge = self.bn4(self.relu(self.deconv4(merge)))  # 1 / 2
        merge = self.bn5(self.relu(self.deconv5(merge)))
        merge = self.classifier(merge)                   # channel = n_class
        return merge