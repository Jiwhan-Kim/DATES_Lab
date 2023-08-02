import torch.nn as nn
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # stride = 1 or 2
        self.dropout = nn.Dropout2d(0.3) # dropout ratio 0.3

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = nn.ReLU()(out)
        out = self.conv1(out)
        out = self.dropout(out)

        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)

        out = out + self.shortcut(residual)

        return out

class ResNet_better_dropout(nn.Module):
    def __init__(self):
        super(ResNet_better_dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        

        self.layer1 = self.make_layer(64, 64, 1) # first block stride 1
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        self.layer4 = self.make_layer(256, 512, 2)
        self.bn = nn.BatchNorm2d(512)

        self.fc = nn.Linear(2*2*512, 10)


    def make_layer(self, in_channels, out_channels, stride):
        layers=[]
        if stride == 1:
            layers.append(BasicBlock(in_channels, out_channels, 1))
            layers.append(BasicBlock(out_channels, out_channels, 1))
        elif stride == 2:
            layers.append(BasicBlock(in_channels, out_channels, 2))
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.MaxPool2d(2, 2)(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = nn.ReLU()(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

