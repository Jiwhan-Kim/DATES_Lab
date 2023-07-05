import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        def conv_normal(in_channels, in_size):
            return nn.Sequantial(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.ReLU()
            )
        
        def conv_halfthingy(in_channels, in_size):
            return nn.Sequantial(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size), # batch normalization occurs after conv, so size goryeo x
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.ReLU()
            )

        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),

            conv_normal(16, 64),
            conv_normal(16, 64),

            conv_halfthingy(8, 128),
            conv_normal(8, 128),

            conv_halfthingy(4, 256),
            conv_normal(4, 256),

            conv_halfthingy(2, 512),
            conv_normal(2, 512),

            nn.MaxPool2D(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(1*1*512, 10)
        )

    def forward(self):



        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


