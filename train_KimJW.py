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
train_size = 49_000
batch_size = 128
epoch      = 100


def train(loader, n_epoch):
    loss = 0
    model.train()

    pbar = tqdm(loader)
    for image, label in pbar:
        x = image.to(device)
        y = label.to(device)
        loss = trainer.step(x, y)
        pbar.set_description("Training Epoch %3d, %2.6f" % (n_epoch, loss))


def evaluate(loader, n_epoch):
    with torch.no_grad():
        correct = 0
        model.eval()
        evaluatelosssum = 0
        result_pbar = tqdm(loader)
        for image, label in result_pbar:
            x = image.to(device)
            y = label.to(device)
            output = model.forward(x)

            evaluatelosssum = evaluatelosssum + torch.nn.CrossEntropyLoss()(output, y)
            result = torch.argmax(output, dim=1)
            for res, ans in zip(result, y):
                if res == ans:
                    correct += 1
        print("Epoch {}. Accuracy: {}".format(n_epoch, 100 * correct / (50000 - train_size)))
    return evaluatelosssum



if __name__ == "__main__":
    print("running train.py")
    print("Device on Working: ", device)

    model   = M.MobileResNet().to(device)
    trainer = T.AC_Trainer(lr=0.001, betas=(0.9, 0.999), weight_decay=0.001, model=model, device=device)

    train_load, valid_load, test_load = D.Load_CIFAR10(train_size, batch_size)
 
    patience = 3  # loss가 일정 에포크 동안 감소하지 않으면 lr decrease
    

    no_improvement_count = 0  # 개선이 없는 에포크 카운트

    if path.exists("./model_params_MobileResNet.pth"):
        model.load_state_dict(torch.load("./model_params_MobileResNet.pth"))

    prev = np.zeros(epoch, dtype=int)

    for i in range(epoch):
        train(train_load, i)
        loss_return = evaluate(valid_load, i)
        if i==0:
            no_improvement_count = 0
            best_eval_loss = loss_return
        else:
            if loss_return >= best_eval_loss:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                best_eval_loss = loss_return

        if no_improvement_count >= patience and trainer.lr >= 0.0001:
            no_improvement_count = 0
            trainer.lr = trainer.lr * 0.8
            print("LR decreased")
        elif no_improvement_count >= patience and trainer.lr < 0.001:
            no_improvement_count = 0
            break
            print("LR Increased")


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
            print("class: {}: {} / 1000".format(i, val[i]))

        torch.save(model.state_dict(), "model_params_MobileResNet.pth")
    
