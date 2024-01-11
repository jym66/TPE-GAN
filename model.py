# 定义一个残差块的类，包括两个卷积层和跳跃连接
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Loss import TPEGANLoss
from GAN_DataSet import CustomDataset


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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 超参数设置
    batch_size = 1
    lr = 0.0002
    beta1 = 0.5
    epochs = 100

    # 数据变换设置
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 初始化自定义数据集和数据加载器
    train_dataset = CustomDataset(size=10, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    encryptor = Generator().to(device)
    decryptor = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 定义优化器
    optimizer_E = Adam(encryptor.parameters())
    optimizer_D = Adam(decryptor.parameters())
    optimizer_Dis = Adam(discriminator.parameters())
    # 定义损失函数
    criterion = TPEGANLoss().to(device)

    for epoch in range(epochs):
        total_loss_dis, total_loss_enc, total_loss_dec, total_loss_thu = 0, 0, 0, 0
        for index, (img, thu_image) in enumerate(train_loader):
            image = img.to(device)
            if index % 5 == 0:
                # 训练判别器
                optimizer_Dis.zero_grad()
                enc_img = encryptor(image)
                # 生成器生成的图片经过判别器的输出
                d_enc_image = discriminator(enc_img)
                # 真实的图片经过判别器的输出
                d_real_image = discriminator(thu_image)
                # 计算判别器损失
                loss_dis = criterion.DLoss(d_enc_image, d_real_image)
                loss_dis.backward()
                optimizer_Dis.step()
                total_loss_dis += loss_dis.item()
            else:
                # 训练加密和解密网络
                optimizer_E.zero_grad()
                optimizer_D.zero_grad()
                # 加密生成器输出
                enc_img = encryptor(image)
                # 解密生成器输出
                dec_image = decryptor(enc_img)
                # 生成器生成的图片经过判别器的输出
                d_enc_image = discriminator(enc_img)
                loss_enc = criterion.EncLoss(d_enc_image)
                loss_dec = criterion.DecLoss(dec_image, image)
                loss_thu = criterion.LThuLoss(enc_img, image)
                total_loss = loss_enc + criterion.lambda_1 * loss_thu + criterion.lambda_2 * loss_dec
                total_loss.backward()
                optimizer_E.step()
                optimizer_D.step()
                total_loss_enc += loss_enc.item()
                total_loss_dec += loss_dec.item()
                total_loss_thu += loss_thu.item()

        # 打印每个周期的平均损失
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Dis Loss: {total_loss_dis / (len(train_loader) / 5):.4f}, "
              f"Enc Loss: {total_loss_enc / (len(train_loader) * 4 / 5):.4f}, "
              f"Dec Loss: {total_loss_dec / (len(train_loader) * 4 / 5):.4f}, "
              f"Thu Loss: {total_loss_thu / (len(train_loader) * 4 / 5):.4f}")
