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

train_size = 40000
batch_size = 128

if __name__ == "__main__":
    print("running test.py")
    model   = M.MobileResNet().to(device)
    _, _, test_load = D.Load_CIFAR10(train_size, batch_size)
    if path.exists("./MobileResNet85.50.pth"):
        model.load_state_dict(torch.load("./MobileResNet85.50.pth"))
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