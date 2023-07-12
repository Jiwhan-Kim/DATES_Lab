import torch.nn as nn
import torch as torch

class BasicBlock(nn.Module):
    def __init__(self, in_channels, stride, out1x1, reduce3x3, out3x3, reducedouble3x3, outdouble3x3, out1x1pool):
        super(BasicBlock, self).__init__()
        
        # 1x1 컨볼루션 브랜치
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, stride=stride, out_channels=out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1),
            nn.ReLU()
        )
        
        # 3x3 컨볼루션 브랜치
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, stride=1, out_channels=reduce3x3, kernel_size=1),
            nn.BatchNorm2d(reduce3x3),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce3x3, stride=stride, out_channels=out3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out3x3),
            nn.ReLU()
        )
        
        # double 3x3 컨볼루션 브랜치
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, stride=1, out_channels=reducedouble3x3, kernel_size=1),
            nn.BatchNorm2d(reducedouble3x3),
            nn.ReLU(),
            nn.Conv2d(in_channels=reducedouble3x3, stride=1, out_channels=outdouble3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(reducedouble3x3),
            nn.ReLU(),
            nn.Conv2d(in_channels=outdouble3x3, stride=stride, out_channels=outdouble3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(outdouble3x3),
            nn.ReLU()
        )
        
        # 폴링 브랜치
        self.branchpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, stride=stride, out_channels=out1x1pool, kernel_size=1),
            nn.BatchNorm2d(out1x1pool),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branchpool = self.branchpool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branchpool]
        return torch.cat(outputs, 1)

class Inception_v1_better(nn.Module):
    def __init__(self):
        super(Inception_v1_better, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.layer3a = BasicBlock(in_channels=192, stride=1, out1x1=64, reduce3x3=96, out3x3=128, reducedouble3x3=16, outdouble3x3=32, out1x1pool=32) # 3a
        self.layer3b = BasicBlock(in_channels=256, stride=2, out1x1=128, reduce3x3=128, out3x3=192, reducedouble3x3=32, outdouble3x3=96, out1x1pool=64) # 3b

        self.layer4a = BasicBlock(in_channels=480, stride=1, out1x1=192, reduce3x3=96, out3x3=208, reducedouble3x3=16, outdouble3x3=48, out1x1pool=64) # 4a
        self.layer4b = BasicBlock(in_channels=512, stride=1, out1x1=160, reduce3x3=112, out3x3=224, reducedouble3x3=24, outdouble3x3=64, out1x1pool=64) # 4b
        self.layer4c = BasicBlock(in_channels=512, stride=1, out1x1=128, reduce3x3=128, out3x3=256, reducedouble3x3=24, outdouble3x3=64, out1x1pool=64) # 4c
        self.layer4d = BasicBlock(in_channels=512, stride=1, out1x1=112, reduce3x3=144, out3x3=288, reducedouble3x3=32, outdouble3x3=64, out1x1pool=64) # 4d
        self.layer4e = BasicBlock(in_channels=528, stride=2, out1x1=256, reduce3x3=160, out3x3=320, reducedouble3x3=32, outdouble3x3=128, out1x1pool=128) # 4e

        self.layer5a = BasicBlock(in_channels=832, stride=1, out1x1=256, reduce3x3=160, out3x3=320, reducedouble3x3=32, outdouble3x3=128, out1x1pool=128) # 5a
        self.layer5b = BasicBlock(in_channels=832, stride=2, out1x1=384, reduce3x3=192, out3x3=384, reducedouble3x3=48, outdouble3x3=128, out1x1pool=128) # 5b


        self.fc = nn.Linear(1*1*1024, 10)




    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm2d(64)(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d((3, 3), stride=2, padding=1)(x)

        x = self.conv2(x)
        x = nn.BatchNorm2d(64)(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.BatchNorm2d(192)(x)
        x = nn.ReLU()(x)

        x = self.layer3a(x)
        x = self.layer3b(x)

        x = self.layer4a(x)
        x = self.layer4b(x)
        x = self.layer4c(x)
        x = self.layer4d(x)
        x = self.layer4e(x)

        x = self.layer5a(x)
        x = self.layer5b(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

