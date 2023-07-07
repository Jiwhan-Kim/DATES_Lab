import torch
import torch.nn as nn
import torch.optim as optim


class AC_Trainer:
    def __init__(self, lr, model, device):
        self.lr = lr
        self.model = model
        self.device = device
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)

    def step(self, image: torch.tensor, label: torch.tensor) -> float:
        x  = image.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
