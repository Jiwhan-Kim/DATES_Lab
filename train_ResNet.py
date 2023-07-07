'''
DATES Lab
Model Check by CIFAR10
Training
Made by KimJW (SSE21)
'''

import torch
import torch.nn
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
batch_size = 64
epoch      = 30


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
    correct = 0
    with torch.no_grad():
      model.eval()
      evaluatelosssum = 0
      result_pbar = tqdm(loader)
      for image, label in result_pbar:
          x = image.to(device)
          y = label.to(device)
          output = model.forward(x)
          evaluatelosssum = evaluatelosssum + torch.nn.CrossEntropyLoss()(output, y)
          result = torch.argmax(output, dim=1)
          correct += batch_size - torch.count_nonzero(result - y)
      print("Epoch {}. Accuracy: {}".format(n_epoch, 100 * correct / 10000))
    return evaluatelosssum



if __name__ == "__main__":
    print("running train.py")
    print("Device on Working: ", device)
    torch.cuda.empty_cache()
    model   = M.ResNet().to(device)
    trainer = T.SGDMC_Trainer(0.002, model, device)

    lr_patience = 5  # 검증 손실이 일정 에포크 동안 감소하지 않으면 lr decrease

    no_improvement_count = 0  # 개선이 없는 에포크 카운트

    train_load, valid_load, test_load = D.Load_CIFAR10(train_size, batch_size)

    if path.exists("./model_params_ResNet.pth"):
        model.load_state_dict(torch.load("./model_params_ResNet.pth"))

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
        if no_improvement_count >= lr_patience:
            trainer.lr = trainer.lr / 10
            no_improvement_count = 0
            print("LR decrease")

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

        torch.save(model.state_dict(), "model_params_ResNet.pth")
    
