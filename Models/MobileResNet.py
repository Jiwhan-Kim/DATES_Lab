'''
MobileResNet
Implemented by KimJW
Please Do not Modify (GitHub Rule)
'''

import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        
        # Convolutional Layer with Depthwise Convolution
        def conv_res_dw(in_size, out_size, stride):
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
        
        self.block = nn.Sequential(
            conv_res_dw(in_channels, out_channels, stride),
            conv_res_dw(out_channels, out_channels, 1),
        )

        self.shortcut = nn.Sequential() if stride == 1 else nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            ) 

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + self.shortcut(residual)
        return out


class MobileResNet(nn.Module):
    def __init__(self):
        super(MobileResNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.make_layer(64, 64, 1),
            self.make_layer(64, 128, 2),
            self.make_layer(128, 256, 2),
            self.make_layer(256, 512, 2),
            self.make_layer(512, 1024, 2),
            nn.AdaptiveMaxPool2d(1)
        )

        self.fc = nn.Linear(1024, 10)


    def make_layer(self, in_channels, out_channels, stride):
        layers=[]
        if stride == 1:
            layers.append(BasicBlock(in_channels, out_channels, 1))
        elif stride == 2:
            layers.append(BasicBlock(in_channels, out_channels, 2))
        layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
