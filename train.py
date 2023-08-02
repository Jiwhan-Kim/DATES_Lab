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

model_numbering = {
    "0": "VGGNet",
    "1": "ResNet",
    "2": "ResNet_better",
    "3": "ResNet_better_dropout",
    "4": "Inception_v1",
    "5": "Inception_v1_better",
    "6": "MobileNet",
    "7": "MobileResNet",
    "8": "Transformer"
}

model_mapping = {
    "VGGNet": M.VGGNet,
    "ResNet": M.ResNet,
    "ResNet_better": M.ResNet_better,
    "ResNet_better_dropout": M.ResNet_better_dropout,
    "Inception_v1": M.Inception_v1,
    "Inception_v1_better": M.Inception_v1_better,
    "MobileNet": M.MobileNet,
    "MobileResNet": M.MobileResNet,
    "Transformer": M.Transformer
}


# Global Data
train_size = 40_000
batch_size = 128
epochs     = 20


def train(loader, n_epoch):
    loss = 0
    model.train()
    pbar = tqdm(loader)
    for image, label in pbar:
        x = image.to(device)
        y = label.to(device)
        loss = trainer.step(x, y)
        pbar.set_description(f"Training Epoch {n_epoch:3d}, {loss:2.6f}")


def evaluate(loader, n_epoch):
    with torch.no_grad():
        correct = 0
        model.eval()
        result_pbar = tqdm(loader)
        for image, label in result_pbar:
            x = image.to(device)
            y = label.to(device)
            output = model.forward(x)
            result = torch.argmax(output, dim=1)
            for res, ans in zip(result, y):
                if res == ans:
                    correct += 1
        print("Epoch {}. Accuracy: {}".format(n_epoch, 100 * correct / (50000 - train_size)))


if __name__ == "__main__":
    
    print("running train.py")
    print("Device on Working: ", device)
    train_load, valid_load, test_load = D.Load_CIFAR10_trans(train_size, batch_size)
    print("Choose Model:")
    print("VGGNet(0)   ResNet(1)   ResNet_better(2)   ResNet_better_dropout(3)   Inception_v1(4)   Inception_v1_better(5)   MobileNet(6)   MobileResNet(7)  Transformer(8)")
    InputString = input()
    if InputString.isdigit() == True:
      ChooseModel = model_numbering[InputString]
    else:
      ChooseModel = InputString
    
    model   = model_mapping[ChooseModel]().to(device)

    if ChooseModel == "VGGNet":
        trainer = T.SGDMC_Trainer(max_lr=0.01, momentum=0.99, weight_decay=0.0001, model=model, device=device, epochs=epochs, train_load=train_load)
    elif ChooseModel == "ResNet":
        trainer = T.SGDMC_Trainer(max_lr=0.01, momentum=0.92, weight_decay=0.000125, model=model, device=device, epochs=epochs, train_load=train_load)
    elif ChooseModel == "ResNet_better":
        trainer = T.SGDMC_Trainer(max_lr=0.01, momentum=0.92, weight_decay=0.000125, model=model, device=device, epochs=epochs, train_load=train_load)
    elif ChooseModel == "ResNet_better_dropout":
        trainer = T.SGDMC_Trainer(max_lr=0.01, momentum=0.92, weight_decay=0.00015, model=model, device=device, epochs=epochs, train_load=train_load)
    elif ChooseModel == "Inception_v1":
        trainer = T.SGDMC_Trainer(max_lr=0.015, momentum=0.9, weight_decay=0.00001, model=model, device=device, epochs=epochs, train_load=train_load)
    elif ChooseModel == "Inception_v1_better":
        trainer = T.SGDMC_Trainer(max_lr=0.015, momentum=0.9, weight_decay=0.00001, model=model, device=device, epochs=epochs, train_load=train_load)
    elif ChooseModel == "MobileNet":
        trainer = T.AC_Trainer(max_lr=0.001, betas=(0.9, 0.999), weight_decay=0.001, model=model, device=device, epochs=epochs, train_load=train_load, grad_clip=0.1)
    elif ChooseModel == "MobileResNet":
        trainer = T.AC_Trainer(max_lr=0.01, betas=(0.9, 0.999), weight_decay=0.0001, model=model, device=device, epochs=epochs, train_load=train_load, grad_clip=0.1)
    elif ChooseModel == "Transformer":
        trainer = T.SGDMC_Trainer(max_lr=0.005, momentum=0.9, weight_decay=0.00001, model=model, device=device, epochs=epochs, train_load=train_load, label_smoothing=0.01)
 
    
    model_params_file = f"./model_params_{ChooseModel}.pth"
    if path.exists(model_params_file):
        model.load_state_dict(torch.load(model_params_file))
        print("parameter exist")


    for i in range(epochs):
        train(train_load, i)
        evaluate(valid_load, i)
        print("Current Learning Rate", trainer.get_lr())

    with torch.no_grad():
        model.eval()
        val = np.zeros(10, dtype=int)
        correct = 0

        result_pbar = tqdm(test_load)
        for image, label in result_pbar:
            x = image.to(device)
            y = label.to(device)
            output = model.forward(x)
            result = torch.argmax(output, dim=1)

            for res, ans in zip(result, y):
                if res == ans:
                    val[res] += 1
                    correct += 1
        
        print("Final Accuracy: {}\n\n".format(100 * correct / 10000))
        for i in range(10):
            print("class {}: {} / 1000".format(i, val[i]))
    torch.save(model.state_dict(), model_params_file)