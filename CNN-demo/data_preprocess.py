import os
import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64, download_enable=True):
    # 检查MNIST是否已下载，避免重复下载
    mnist_dir = "./data/MNIST"
    is_mnist_downloaded = os.path.exists(mnist_dir) and len(os.listdir(mnist_dir)) > 0
    download_enable = not is_mnist_downloaded
    print(f"MNIST是否已下载：{is_mnist_downloaded}，是否重新下载：{download_enable}")

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 单通道归一化
        ]
    )

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=download_enable
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=download_enable
    )
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, # num_workers>0时，我的电脑上多线程相关报错了
        pin_memory=True  # 锁页内存，加速数据从CPU到GPU
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader