import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(4, F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(4, F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        # 简化版本：使用双线性插值上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.use_attention = use_attention
        
        if use_attention:
            # 注意力机制
            self.att = AttentionBlock(F_g=in_channels, F_l=out_channels, F_int=in_channels // 2)
        
        # DoubleConv 的输入通道数：in_channels（上采样后） + out_channels（跳跃连接）
        self.conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 来自解码器的特征（in_channels）
        # x2: 来自编码器的跳跃连接特征（out_channels）
        
        # 上采样 x1
        x1 = self.up(x1)
        
        # 调整尺寸以确保匹配
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        # 应用注意力机制（如果启用）
        if self.use_attention:
            x2 = self.att(x1, x2)
        
        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class Baseline_Attention_UNet(nn.Module):
    """标准Attention UNet"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        print(f"创建标准Attention UNet: input_channels={input_channels}, num_classes={num_classes}")
        
        # 使用与baseline_unet.py相同的通道数配置
        # 编码器通道数: [64, 128, 256, 512, 1024]
        
        # 编码器
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 上采样（带注意力机制）
        # 使用双线性插值上采样，避免 ConvTranspose2d 的通道数问题
        self.up1 = Up(in_channels=1024, out_channels=512, use_attention=True)
        self.up2 = Up(in_channels=512, out_channels=256, use_attention=True)
        self.up3 = Up(in_channels=256, out_channels=128, use_attention=True)
        self.up4 = Up(in_channels=128, out_channels=64, use_attention=True)
        
        # 输出
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels
        
        # 解码路径（带注意力机制）
        x = self.up1(x5, x4)  # 输出: 512 channels
        x = self.up2(x, x3)   # 输出: 256 channels
        x = self.up3(x, x2)   # 输出: 128 channels
        x = self.up4(x, x1)   # 输出: 64 channels
        
        # 输出
        logits = self.outc(x)
        return torch.sigmoid(logits)