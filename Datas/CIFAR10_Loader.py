import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
     ]
)


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.x = torch.tensor(data / 256., dtype=torch.float).permute(0, 3, 1, 2)
        self.x -= 0.5
        self.x /= 0.5
        self.y = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def Load_CIFAR10(n_train=40000, batch_size=1):
    # Data Sets
    trainset = datasets.CIFAR10(root='./Datas', train=True, download=True)
    testset = datasets.CIFAR10(root='./Datas', train=False, download=True, transform=transform)

    datas = trainset.data
    labels = trainset.targets

    train_datas = datas[:n_train]
    valid_datas = datas[n_train:]

    train_labels = labels[:n_train]
    valid_labels = labels[n_train:]

    train_ds = MyDataset(data=train_datas, label=train_labels)
    valid_ds = MyDataset(data=valid_datas, label=valid_labels)

    train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_load = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_load = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print("Data Loading Completed")
    return train_load, valid_load, test_load


