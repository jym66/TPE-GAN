from torch import nn
import torch
import torch.nn.functional as F
from ignite.metrics import SSIM


class SSIMLoss(torch.nn.Module):
    def __init__(self, data_range):
        super().__init__()
        self.ssim = SSIM(data_range=data_range)

    def forward(self, y_pred, y):
        # SSIM 度量计算的是相似度，所以1 - SSIM 度量将会给我们损失
        # Ignite的SSIM实现需要数据在[0, data_range]范围内
        self.ssim.update((y_pred, y))
        loss = 1 - self.ssim.compute()
        self.ssim.reset()  # 重置度量的内部状态以备下次计算使用
        return loss


class LDecLoss(torch.nn.Module):
    def __init__(self, lambda_3):
        super(LDecLoss, self).__init__()
        self.lambda_3 = lambda_3
        self.ssim = SSIMLoss(1.0)

    def forward(self, dec_img, real_img):
        return self.lambda_3 * self.ssim(dec_img, real_img) + (1 - self.lambda_3) * torch.norm(dec_img - real_img, p=1,
                                                                                               dim=1).mean()


class LEncLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_fake_img):
        return torch.norm(1 - d_fake_img, p=2, dim=1).mean()


class LDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_fake_img, d_real_img):
        # 对于假图片的损失，我们计算D(G(x))的L2范数
        loss_fake = torch.norm(d_fake_img, p=2, dim=1).mean()
        # 对于真实图片的损失，我们计算1 - D(y)的L2范数
        loss_real = torch.norm(1 - d_real_img, p=2, dim=1).mean()
        # 总损失是假图片和真实图片损失的和
        return loss_fake + loss_real


class LThuLoss(nn.Module):
    def __init__(self, kernel_size=128):
        super(LThuLoss, self).__init__()
        self.kernel_size = kernel_size
        # 初始化卷积核（在forward方法中移动到正确的设备上）
        self.kernel = torch.full((1, 1, kernel_size, kernel_size), 1.0 / (kernel_size ** 2))

    def thumbnail_extraction(self, img):
        # 在每次调用时将卷积核移动到img所在的设备
        device = img.device
        kernel = self.kernel.to(device)
        thumbnail = F.conv2d(img, kernel, stride=self.kernel_size)
        return thumbnail

    def forward(self, fake_img, real_img):
        real_thumbnail = self.thumbnail_extraction(real_img)
        fake_thumbnail = self.thumbnail_extraction(fake_img)
        return torch.norm(fake_thumbnail - real_thumbnail, p=1, dim=[1, 2, 3]).mean()



class TPEGANLoss(nn.Module):
    def __init__(self, lambda_1=10, lambda_2=20, lambda_3=0.9):
        super(TPEGANLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.EncLoss = LEncLoss()
        self.LThuLoss = LThuLoss()
        self.DLoss = LDLoss()
        self.DecLoss = LDecLoss(self.lambda_3)

    # def forward(self, d_fake_img, d_real_img, fake_img, real_img, dec_img):
    #     return self.EncLoss(d_fake_img) + self.lambda_1 * self.LThuLoss(fake_img,
    #                                                                     real_img) + self.lambda_2 * self.DecLoss(
    #         dec_img, real_img) + self.DLoss(d_fake_img, d_real_img)


def test_tpe_gan_loss():
    # 初始化模拟的网络输出
    batch_size = 4
    image_size = (1, 28, 28)  # 假设图像大小为28x28，单通道

    # 模拟真实图像和生成图像
    real_img = torch.rand(batch_size, *image_size)
    fake_img = torch.rand(batch_size, *image_size)
    dec_img = torch.rand(batch_size, *image_size)

    # 模拟判别器对假图像和真图像的输出
    d_fake_img = torch.rand(batch_size, 1)
    d_real_img = torch.rand(batch_size, 1)

    loss_fn = TPEGANLoss()

    # 计算损失
    loss = loss_fn(d_fake_img, d_real_img, fake_img, real_img, dec_img)

    print(f"Calculated TPEGAN Loss: {loss.item()}")


if __name__ == "__main__":
    test_tpe_gan_loss()
