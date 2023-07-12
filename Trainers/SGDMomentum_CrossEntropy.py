import torch
import torch.nn as nn
import torch.optim as optim


class SGDMC_Trainer:
    def __init__(self, lr, momentum, weight_decay, model, device):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = model
        self.device = device
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def step(self, image: torch.tensor, label: torch.tensor) -> float:
        x  = image.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
