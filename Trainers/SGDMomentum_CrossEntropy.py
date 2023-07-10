import torch
import torch.nn as nn
import torch.optim as optim


class SGDMC_Trainer_VGGNet:
    def __init__(self, lr, model, device):
        self.lr = lr
        self.model = model
        self.device = device
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99, weight_decay=0.0001)

    def step(self, image: torch.tensor, label: torch.tensor) -> float:
        x  = image.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
class SGDMC_Trainer_ResNet:
    def __init__(self, lr, model, device):
        self.lr = lr
        self.model = model
        self.device = device
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)

    def step(self, image: torch.tensor, label: torch.tensor) -> float:
        x  = image.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
class SGDMC_Trainer_Inception_Jun:
    def __init__(self, lr, model, device):
        self.lr = lr
        self.model = model
        self.device = device
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)

    def step(self, image: torch.tensor, label: torch.tensor) -> float:
        x  = image.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    