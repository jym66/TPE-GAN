# 定义一个残差块的类，包括两个卷积层和跳跃连接
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, ToPILImage
from Loss import TPEGANLoss
from GAN_DataSet import RealDataset
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 下采样模块
        self.down_sample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 残差块 x9，每个残差块内部进行跳跃连接
        self.residuals = nn.Sequential(*[ResidualBlock(256) for _ in range(9)])

        # 上采样模块
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, out):
        out = self.down_sample(out)
        out = self.residuals(out)
        out = self.up_sample(out)
        return out


# 定义判别器结构
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, out):
        out = self.model(out)
        return out


def train_model(train_data_path, thu_data_path, transform, device, model_path="model.pth"):
    batch_size = 64
    lr = 0.01
    epochs = 100
    print(f"运行设备 {device}....")

    # 初始化自定义数据集和数据加载器
    train_dataset = RealDataset(train_data_path, thu_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 创建模型
    encryptor = Generator().to(device)
    decryptor = Generator().to(device)
    discriminator = Discriminator().to(device)
    # 创建优化器实例
    optimizer_e = Adam(encryptor.parameters(), lr=lr)
    optimizer_d = Adam(decryptor.parameters(), lr=lr)
    optimizer_dis = Adam(discriminator.parameters(), lr=0.0001)
    # 定义损失函数
    criterion = TPEGANLoss().to(device)
    # 尝试加载模型
    if os.path.isfile(model_path):
        print("加载模型.....")
        checkpoint = torch.load(model_path)
        encryptor.load_state_dict(checkpoint['encryptor_state_dict'])
        decryptor.load_state_dict(checkpoint['decryptor_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_e.load_state_dict(checkpoint['optimizer_E_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_Dis_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"继续训练 epoch {start_epoch}")
    else:
        print("模型加载失败，重新开始训练.....")

    for epoch in range(epochs):
        total_loss_dis, total_loss_enc, total_loss_dec, total_loss_thu = 0, 0, 0, 0
        for index, target in enumerate(train_loader):
            img, thu_image = target
            image = img.to(device)
            thu_image = thu_image.to(device)
            if index % 150 == 0:
                # 训练判别器
                optimizer_dis.zero_grad()
                enc_img = encryptor(image)
                # 生成器生成的图片经过判别器的输出
                d_enc_image = discriminator(enc_img)
                # 真实的图片经过判别器的输出
                d_real_image = discriminator(thu_image)
                # 计算判别器损失
                loss_dis = criterion.DLoss(d_enc_image, d_real_image)
                loss_dis.backward()
                optimizer_dis.step()
                total_loss_dis += loss_dis.item()
            else:
                # 训练加密和解密网络
                optimizer_e.zero_grad()
                optimizer_d.zero_grad()
                # 加密生成器输出
                enc_img = encryptor(image)
                # 解密生成器输出
                dec_image = decryptor(enc_img)
                # 生成器生成的图片经过判别器的输出
                d_enc_image = discriminator(enc_img.detach()).detach()
                loss_enc = criterion.EncLoss(d_enc_image)
                loss_dec = criterion.DecLoss(dec_image, image)
                loss_thu = criterion.LThuLoss(enc_img, image)
                total_loss = loss_enc + criterion.lambda_1 * loss_thu + criterion.lambda_2 * loss_dec
                total_loss.backward()
                optimizer_e.step()
                optimizer_d.step()
                total_loss_enc += loss_enc.item()
                total_loss_dec += loss_dec.item()
                total_loss_thu += loss_thu.item()

        # 打印每个周期的平均损失
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Dis Loss: {total_loss_dis / (len(train_loader) / 5):.4f}, "
              f"Enc Loss: {total_loss_enc / (len(train_loader) * 4 / 5):.4f}, "
              f"Dec Loss: {total_loss_dec / (len(train_loader) * 4 / 5):.4f}, "
              f"Thu Loss: {total_loss_thu / (len(train_loader) * 4 / 5):.4f}")
        # 保存模型
        torch.save({
            'epoch': epoch,
            'encryptor_state_dict': encryptor.state_dict(),
            'decryptor_state_dict': decryptor.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_E_state_dict': optimizer_e.state_dict(),
            'optimizer_D_state_dict': optimizer_d.state_dict(),
            'optimizer_Dis_state_dict': optimizer_dis.state_dict(),
            'loss': total_loss_dis,
        }, f"model.pth")


def train_model1(train_data_path, thu_data_path, transform, device, model_path="model.pth"):
    # 只训练判别器和加密网络
    batch_size = 64
    lr = 0.01
    epochs = 100
    print(f"运行设备 {device}....")

    # 初始化自定义数据集和数据加载器
    train_dataset = RealDataset(train_data_path, thu_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 创建模型
    encryptor = Generator().to(device)
    discriminator = Discriminator().to(device)
    # 创建优化器实例
    optimizer_e = Adam(encryptor.parameters(), lr=lr)
    optimizer_dis = Adam(discriminator.parameters(), lr=0.00002)
    # 定义损失函数
    criterion = TPEGANLoss().to(device)
    criterion = nn.BCELoss()
    # 尝试加载模型
    if os.path.isfile(model_path):
        print("加载模型.....")
        checkpoint = torch.load(model_path)
        encryptor.load_state_dict(checkpoint['encryptor_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_e.load_state_dict(checkpoint['optimizer_E_state_dict'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_Dis_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"继续训练 epoch {start_epoch}")
    else:
        print("模型加载失败，重新开始训练.....")

    for epoch in range(epochs):
        total_loss_dis, total_loss_enc = 0, 0
        for index, target in enumerate(train_loader):
            img, thu_image = target
            image = img.to(device)
            thu_image = thu_image.to(device)
            if index % 20 == 0:
                # 训练判别器
                optimizer_dis.zero_grad()
                enc_img = encryptor(image)
                # 生成器生成的图片经过判别器的输出
                d_enc_image = discriminator(enc_img)
                # 真实的图片经过判别器的输出
                d_real_image = discriminator(thu_image)
                # 计算判别器损失
                loss_dis = criterion.DLoss(d_enc_image, d_real_image)
                loss_dis.backward()
                optimizer_dis.step()
                total_loss_dis += loss_dis.item()
            else:
                # 训练加密网络
                optimizer_e.zero_grad()
                # 重新生成加密图片
                enc_img = encryptor(image)
                # 生成器生成的图片经过判别器的输出
                d_enc_image = discriminator(enc_img.detach())
                loss_enc = criterion.EncLoss(d_enc_image)
                loss_enc.backward()
                optimizer_e.step()
                total_loss_enc += loss_enc.item()

        # 打印每个周期的平均损失
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Dis Loss: {total_loss_dis / len(train_loader):.4f}, "
              f"Enc Loss: {total_loss_enc / len(train_loader):.4f}")
        # 保存模型
        torch.save({
            'epoch': epoch,
            'encryptor_state_dict': encryptor.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_E_state_dict': optimizer_e.state_dict(),
            'optimizer_Dis_state_dict': optimizer_dis.state_dict(),
            'loss': total_loss_dis,
        }, f"model.pth")


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def train_model2(train_data_path, thu_data_path, transform, device, model_path="model.pth"):
    # 只训练判别器和加密网络
    batch_size = 64
    lr = 0.01
    epochs = 100
    print(f"运行设备 {device}....")

    # 初始化自定义数据集和数据加载器
    train_dataset = RealDataset(train_data_path, thu_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 创建模型
    encryptor = Generator().to(device)
    discriminator = Discriminator().to(device)
    # 创建优化器实例
    optimizer_e = Adam(encryptor.parameters(), lr=lr)
    optimizer_dis = Adam(discriminator.parameters(), lr=0.00002)
    # 定义损失函数
    criterion = nn.BCELoss()

    # 尝试加载模型
    if os.path.isfile(model_path):
        print("加载模型.....")
        checkpoint = torch.load(model_path)
        encryptor.load_state_dict(checkpoint['encryptor_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_e.load_state_dict(checkpoint['optimizer_E_state_dict'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_Dis_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"继续训练 epoch {start_epoch}")
    else:
        print("模型加载失败，重新开始训练.....")

    for epoch in range(epochs):
        total_loss_dis, total_loss_enc = 0, 0
        for index, (img, thu_image) in enumerate(train_loader):
            image = img.to(device)
            thu_image = thu_image.to(device)

            # 训练判别器
            optimizer_dis.zero_grad()
            real_labels = torch.ones(thu_image.size(0), 1).to(device)
            fake_labels = torch.zeros(image.size(0), 1).to(device)

            outputs_real = discriminator(thu_image)
            loss_real = criterion(outputs_real, real_labels)

            fake_images = encryptor(image)
            outputs_fake = discriminator(fake_images.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_dis = loss_real + loss_fake
            loss_dis.backward()
            optimizer_dis.step()
            total_loss_dis += loss_dis.item()

            # 训练加密网络
            optimizer_e.zero_grad()
            outputs_fake_for_gen = discriminator(fake_images)
            loss_enc = criterion(outputs_fake_for_gen, real_labels)
            loss_enc.backward()
            optimizer_e.step()
            total_loss_enc += loss_enc.item()

        # 打印每个周期的平均损失
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Dis Loss: {total_loss_dis / len(train_loader):.4f}, "
              f"Enc Loss: {total_loss_enc / len(train_loader):.4f}")
        # 保存模型
        torch.save({
            'epoch': epoch,
            'encryptor_state_dict': encryptor.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_E_state_dict': optimizer_e.state_dict(),
            'optimizer_Dis_state_dict': optimizer_dis.state_dict(),
            'loss': total_loss_dis,
        }, f"model.pth")


if __name__ == "__main__":
    if os.path.exists("/kaggle/working"):
        train_data_path = "/kaggle/input/tpe-gan/val2017"
        thu_data_path = "/kaggle/input/tpe-gan/thu"
    else:
        train_data_path = "/root/coco_data/val2017"
        thu_data_path = "/root/coco_data/thumbnail"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "train"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 首先调整图像大小
        transforms.ToTensor(),  # 将 PIL 图像转换为浮点型张量并归一化像素值
    ])

    if mode == "train":
        train_model2(train_data_path, thu_data_path, device=device, transform=transform)
    else:
        # 初始化自定义数据集和数据加载器
        test_dataset = RealDataset(train_data_path, thu_data_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        checkpoint = torch.load("model.pth", map_location=device)

        discriminator = Discriminator().to(device)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        encryptor = Generator().to(device)
        encryptor.load_state_dict(checkpoint['encryptor_state_dict'])

        for img, thu in test_loader:
            img = img.to(device)
            thu = thu.to(device)
            pred = encryptor(img)
            pred = pred.detach()

            to_pil = ToPILImage()
            img_pil = to_pil(thu[0])
            pred_pil = to_pil(pred[0])

            # 创建一个带有两个子图的绘图窗口
            plt.figure(figsize=(10, 5))
            # 显示原始图片
            plt.subplot(1, 2, 1)
            plt.imshow(img_pil)
            plt.title("Original Image")
            plt.axis('off')

            # 显示预测图片
            plt.subplot(1, 2, 2)
            plt.imshow(pred_pil)
            plt.title("Predicted Image")
            plt.axis('off')

            # 显示绘图窗口
            plt.show()
            break
