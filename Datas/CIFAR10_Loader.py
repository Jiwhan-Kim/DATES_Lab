import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.x = torch.tensor(data, dtype=torch.float)
        self.y = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def Load_CIFAR10(n_train=40000, batch_size=1):
    # Data Sets
    trainset = datasets.CIFAR10(root='./Datas', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./Datas', train=False, download=True, transform=transform)
    n_eval = 50000 - n_train
    train_ds, valid_ds = random_split(trainset, [n_train, n_eval])

    train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_load = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_load = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print("Data Loading Completed")
    return train_load, valid_load, test_load


