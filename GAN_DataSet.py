import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, directory, thu_directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.thu_directory = thu_directory
        # 过滤出原始图像的文件名
        self.images = [file for file in os.listdir(self.directory) if file.endswith('.jpg')]

    def __len__(self):
        # return 1280
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.images[idx])
        thu_image_path = os.path.join(self.thu_directory, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        thu_image = Image.open(thu_image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            thu_image = self.transform(thu_image)
        # to_pil = ToPILImage()
        # pil_img = to_pil(image)
        # plt.imshow(pil_img)
        # plt.show()
        return image, thu_image


if __name__ == '__main__':
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 首先调整图像大小
        transforms.ToTensor(),  # 将 PIL 图像转换为浮点型张量并归一化像素值
    ])

    # 实例化数据集
    dataset = RealDataset('/root/coco_data/val2017/', "/root/coco_data/thumbnail/", transform=transform)

    # 使用 DataLoader 加载数据集
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 演示一次数据加载
    for images, labels in dataloader:
        print("Batch of images:", images)  # 打印批次图像的形状
        print("Batch of labels:", labels.shape)  # 打印批次标签的形状
        break  # 仅演示一次，所以使用 break
