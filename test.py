import torch
import numpy as np
from os import path
from tqdm import tqdm
import Models as M
import Datas as D

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
    "2": "M.ResNet_better",
    "3": "M.ResNet_better_dropout",
    "4": "M.Inception_v1",
    "5": "M.Inception_v1_better",
    "6": "M.MobileNet",
    "7": "M.MobileResNet"
}

model_mapping = {
    "VGGNet": M.VGGNet,
    "ResNet": M.ResNet,
    "ResNet_better": M.ResNet_better,
    "ResNet_better_dropout": M.ResNet_better_dropout,
    "Inception_v1": M.Inception_v1,
    "Inception_v1_better": M.Inception_v1_better,
    "MobileNet": M.MobileNet,
    "MobileResNet": M.MobileResNet
}

train_size = 40_000
batch_size = 64

if __name__ == "__main__":
    print("running test.py")
    print("Choose Model:")
    print("VGGNet   ResNet   ResNet_better   ResNet_better_dropout   Inception_v1   Inception_v1_better   MobileNet   MobileResNet")
    InputString = input()
    if InputString.isdigit() == True:
       ChooseModel = model_numbering[InputString]
    else:
       ChooseModel = InputString
    model   = model_mapping[ChooseModel]().to(device)
    _, _, test_load = D.Load_CIFAR10(train_size, batch_size)
    model_params_file = f"./model_params_{ChooseModel}.pth"
    if path.exists(model_params_file):
        model.load_state_dict(torch.load(model_params_file))
    else:
      print("run train first!")
      exit(0)


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