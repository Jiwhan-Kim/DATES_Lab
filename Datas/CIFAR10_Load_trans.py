import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]
)



def Load_CIFAR10_trans(n_train, batch_size):
    # Data Sets
    trainset = datasets.CIFAR10(root='./Datas', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./Datas', train=False, download=True, transform=transform)
    n_eval = 50000 - n_train
    train_ds, valid_ds = random_split(trainset, [n_train, n_eval])

    train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_load = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_load = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Data Loading Completed")
    return train_load, valid_load, test_load
