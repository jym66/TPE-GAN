import os

import torch
from PIL import Image
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


class RealDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [file for file in os.listdir(directory) if file.endswith('.jpg') or file.endswith('.png')]
        self.thu_images = [file for file in os.listdir(directory) if
                           (file.startswith("reduced_") and file.endswith('.jpg')) or (
                                   file.startswith("reduced_") and file.endswith('.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.images[idx])
        thu_image_path = os.path.join(self.directory, self.thu_images[idx])
        image = Image.open(image_path).convert('RGB')
        thu_image = Image.open(thu_image_path)
        if self.transform:
            image = self.transform(image)
            thu_image = self.transform(thu_image)
        return image, thu_image  # 返回图像和缩略图（或者您可以返回其他类型的标签）


if __name__ == '__main__':
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 实例化数据集
    dataset = CustomDataset(size=100, transform=transform)

    # 使用 DataLoader 加载数据集
    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 演示一次数据加载
    for images, labels in dataloader:
        print("Batch of images:", images)  # 打印批次图像的形状
        print("Batch of labels:", labels.shape)  # 打印批次标签的形状
        break  # 仅演示一次，所以使用 break
