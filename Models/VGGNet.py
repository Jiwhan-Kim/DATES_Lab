import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        def conv_2times(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        def conv_3times(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        
        self.layer = nn.Sequential(
          conv_2times(3, 64),
          nn.MaxPool2d(2, 2),

          conv_2times(64, 128), # size = 16
          nn.MaxPool2d(2, 2),

          conv_2times(128, 256), # size = 8
          nn.MaxPool2d(2, 2),

          conv_3times(256, 512), # size = 4
          nn.MaxPool2d(2, 2),

          conv_3times(512, 512), # size = 2
          nn.MaxPool2d(2, 2)

          # size = 1
        )

        self.fc = nn.Sequential(
          nn.Linear(1*1*512, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 10)
        )

        for m in self.modules():
             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                 nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


