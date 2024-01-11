import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms


# 定义一个简单的自定义数据集
class CustomDataset(Dataset):
    def __init__(self, size, transform=None):
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 随机生成图像和标签
        image = torch.randn(3, 128, 128)
        label = torch.randint(0, 2, (1,))
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    # 定义变换
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 实例化数据集
    dataset = CustomDataset(size=100, transform=transform)

    # 使用 DataLoader 加载数据集
    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 演示一次数据加载
    for images, labels in dataloader:
        print("Batch of images:", images.shape)  # 打印批次图像的形状
        print("Batch of labels:", labels.shape)  # 打印批次标签的形状
        break  # 仅演示一次，所以使用 break
