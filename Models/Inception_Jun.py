import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out1x1, reduce3x3, out3x3, reduce5x5, out5x5, out1x1pool):
        super(BasicBlock, self).__init__()
        
        # 1x1 컨볼루션 브랜치
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1),
            nn.ReLU()
        )
        
        # 3x3 컨볼루션 브랜치
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce3x3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce3x3, out_channels=out3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 5x5 컨볼루션 브랜치
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce5x5, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce5x5, out_channels=out5x5, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # 폴링 브랜치
        self.branchpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1pool, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branchpool = self.branchpool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branchpool]
        return torch.cat(outputs, 1)
class Inception(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(64, 64, 1) # first block stride 1
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        self.layer4 = self.make_layer(256, 512, 2)

        self.fc = nn.Linear(1*1*512, 10)
        # Initialize the weights using Kaiming initialization for the fully connected layer
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='relu')


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
        
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = nn.MaxPool2d(2, 2)(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

