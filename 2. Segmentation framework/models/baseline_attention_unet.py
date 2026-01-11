import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(max(1, F_int // 2), F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(max(1, F_int // 2), F_int)
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
            nn.GroupNorm(max(1, out_channels // 2), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, out_channels // 2), out_channels),
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
    """最小通道版Attention UNet: [8, 16, 24, 32, 48]"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        # 最小通道配置
        channels = [8, 16, 24, 32, 48]
        print(f"创建最小通道版Attention UNet: channels={channels}, input_channels={input_channels}, num_classes={num_classes}")
        
        # 编码器
        self.inc = DoubleConv(input_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])
        
        # 上采样（带注意力机制）
        self.up1 = Up(in_channels=channels[4], out_channels=channels[3], use_attention=True)
        self.up2 = Up(in_channels=channels[3], out_channels=channels[2], use_attention=True)
        self.up3 = Up(in_channels=channels[2], out_channels=channels[1], use_attention=True)
        self.up4 = Up(in_channels=channels[1], out_channels=channels[0], use_attention=True)
        
        # 输出
        self.outc = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        
        self._init_weights()
        self._print_model_info()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,} ({total_params/1e6:.3f}M)")

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)      # 8 channels
        x2 = self.down1(x1)   # 16 channels
        x3 = self.down2(x2)   # 24 channels
        x4 = self.down3(x3)   # 32 channels
        x5 = self.down4(x4)   # 48 channels
        
        # 解码路径（带注意力机制）
        x = self.up1(x5, x4)  # 输出: 32 channels
        x = self.up2(x, x3)   # 输出: 24 channels
        x = self.up3(x, x2)   # 输出: 16 channels
        x = self.up4(x, x1)   # 输出: 8 channels
        
        # 输出
        logits = self.outc(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    # 测试模型
    model = Baseline_Attention_UNet(num_classes=1)
    input_tensor = torch.randn(2, 3, 256, 256)
    output = model(input_tensor)
    
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")