'''
DATES Lab
Model Check by CIFAR10
Training
Made by KimJW (SSE21)
'''

import torch
import numpy as np
from tqdm import tqdm
from os import path

import Models   as M
import Trainers as T
import Datas    as D

# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu)
device = None
if   torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Global Data
train_size = 40_000
batch_size = 256
epoch      = 20


def train(loader, n_epoch):
    loss = 0
    model.train()

    pbar = tqdm(loader)
    for image, label in pbar:
        x = image.to(device)
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / std
        y = label.to(device)
        loss = trainer.step(x, y)
        pbar.set_description("Training Epoch %3d, %2.6f" % (n_epoch, loss))

def evaluate(loader, n_epoch):
    correct = 0
    model.eval()

    result_pbar = tqdm(loader)
    for image, label in result_pbar:
        x = image.to(device)
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / std
        y = label.to(device)
        output = model.forward(x)
        result = torch.argmax(output, dim=1)
        correct += batch_size - torch.count_nonzero(result - y)
    print("Epoch {}. Accuracy: {}".format(n_epoch, 100 * correct / 10000))



if __name__ == "__main__":
    print("running train.py")
    print("Device on Working: ", device)

    model   = M.ResNet().to(device)
    trainer = T.AC_Trainer(0.001, model, device)
    train_load, valid_load, test_load = D.Load_CIFAR10(train_size, batch_size)

    if path.exists("./model_params_ResNet.pth"):
        model.load_state_dict(torch.load("./model_params_ResNet.pth"))

    for i in range(epoch):
        train(train_load, i)
        evaluate(valid_load, i)

    with torch.no_grad():
        model.eval()
        val = np.zeros(10, dtype=int)
        correct = 0

        result_pbar = tqdm(test_load)
        for image, label in result_pbar:
            x = image.to(device)
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            x = (x - mean) / std
            y = label.to(device)
            output = model.forward(x)
            result = torch.argmax(output, dim=1)

            for res, ans in zip(result, y):
                if res == ans:
                    val[res] += 1
                    correct += 1
        
        print("Final Accuracy: {}\n\n".format(100 * correct / 10000))
        for i in range(10):
            print("class: {}: {} / 1000".format(i, val[i]))

        torch.save(model.state_dict(), "model_params_ResNet.pth")
    
