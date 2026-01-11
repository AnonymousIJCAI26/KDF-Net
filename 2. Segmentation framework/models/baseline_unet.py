import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, out_channels // 2), out_channels),  # 根据通道数调整group数
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, out_channels // 2), out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Baseline_UNet(nn.Module):
    """最小通道版UNet: [8, 16, 24, 32, 48]"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        # 最小通道配置
        # channels = [8, 16, 24, 32, 48]
        channels = [8, 16, 32, 64, 64]
        print(f"创建最小通道版UNet: channels={channels}, input_channels={input_channels}, num_classes={num_classes}")
        
        # 编码器
        self.inc = DoubleConv(input_channels, channels[0])
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels[0], channels[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels[1], channels[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels[2], channels[3])
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels[3], channels[4])
        )
        
        # 上采样
        self.up1 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.conv1 = DoubleConv(channels[3] * 2, channels[3])
        
        self.up2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.conv2 = DoubleConv(channels[2] * 2, channels[2])
        
        self.up3 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.conv3 = DoubleConv(channels[1] * 2, channels[1])
        
        self.up4 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.conv4 = DoubleConv(channels[0] * 2, channels[0])
        
        # 输出
        self.outc = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        
        # 统计参数量
        self._print_model_info()

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,} ({total_params/1e6:.3f}M)")

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器
        x = self.up1(x5)
        if x.shape != x4.shape:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x3.shape:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape != x2.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        # 输出
        logits = self.outc(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    # 测试模型
    model = Baseline_UNet(num_classes=1)
    input_tensor = torch.randn(2, 3, 256, 256)
    output = model(input_tensor)
    
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")