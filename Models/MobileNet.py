'''
MobileNet v1
Implemented by KimJW
Please Do not Modify
'''

import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self, width=32, height=32):
        super(MobileNet, self).__init__()

        # Convolutional Layer at First
        def conv_bn(in_size, out_size, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size,
                          kernel_size=3, stride=stride, padding=1,
                          bias=False),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
        
        # Convolutional Layer with Depthwise Convolution
        def conv_dw(in_size, out_size, stride):
            return nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_size, out_channels=in_size,
                          kernel_size=3, stride=stride, padding=1,
                          groups=in_size, bias=False),
                nn.BatchNorm2d(in_size),
                nn.ReLU(),

                # Separable Filter
                nn.Conv2d(in_channels=in_size, out_channels=out_size,
                          kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
        
        self.layer = nn.Sequential( # Input Matrix
            conv_bn(3, 32, 1),      # 3   x w   x h
            conv_dw(32, 64, 2),     # 32  x w   x h
            conv_dw(64, 64, 1),     # 64  x w/2 x h/2
            conv_dw(64, 128, 2),    # 64  x w/2 x h/2
            conv_dw(128, 128, 1),   # 128 x w/4 x h/4
            conv_dw(128, 256, 2),   # 128 x w/4 x h/4
            conv_dw(256, 256, 1),   # 256 x w/8 x h/8
            conv_dw(256, 512, 2),   # 256 x w/8 x h/8
            nn.AdaptiveMaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * width * height, 10)
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

