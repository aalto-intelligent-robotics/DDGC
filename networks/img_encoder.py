import torch.nn as nn
from .networks import NetworkBase
import torchvision
import torch


class Network(NetworkBase):
    def __init__(self, output_dim=7):
        super(Network, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, output_dim)
        self.name = 'image_encoder'

        pretrained_conv1 = self.model.conv1.weight
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight[:, :3] = pretrained_conv1
        self.model.conv1.weight = torch.nn.Parameter(self.model.conv1.weight)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        img_representation = x.view(x.size(0), -1)
        return img_representation
