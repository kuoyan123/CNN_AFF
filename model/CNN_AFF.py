#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
from model.cbam import *
from model.AFF import *
class CNN_AFF(nn.Module):
    def __init__(self, pretrained=False, in_channel=2, out_channel=4):
        super(CNN_AFF, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.layer5 = nn.Sequential(
            AFF(256, group_num=4),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.cbam = CBAMBlock(16)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(1024, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.cbam(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
