import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        def conv_normal(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
        
        def conv_halfthingy(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), # batch normalization occurs after conv, so size goryeo x
                nn.ReLU(),

                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), # size = 32
            nn.MaxPool2d(2, 2),

            conv_normal(64), # size = 16
            conv_normal(64),

            conv_halfthingy(64, 128), # size = 8
            conv_normal(128),

            conv_halfthingy(128, 256), # size = 4
            conv_normal(256),

            conv_halfthingy(256, 512), # size = 2
            conv_normal(512),

            nn.MaxPool2d(2, 2)
            # size = 1
        )

        self.fc = nn.Sequential(
            nn.Linear(1*1*512, 10)
        )

    def forward(self, x):

        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


