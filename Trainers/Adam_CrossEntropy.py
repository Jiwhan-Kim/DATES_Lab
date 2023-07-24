import torch
import torch.nn as nn
import torch.optim as optim


class AC_Trainer:
    def __init__(self, model, device, max_lr, betas, weight_decay, epochs, train_load, grad_clip=None):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=max_lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=max_lr, epochs=epochs,
                                                       steps_per_epoch=len(train_load))

    def step(self, image: torch.tensor, label: torch.tensor) -> float:
        x  = image.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x)
        loss = self.lossF(output, y)
        loss.backward()

        if self.grad_clip:
            nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()
        return loss
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        pass
    
