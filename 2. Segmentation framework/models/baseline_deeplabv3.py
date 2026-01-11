import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 32  # 调整为32以匹配小通道配置
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.ReLU()
        ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(low_level_channels, 12, 1, bias=False),  # 调整为12
            nn.GroupNorm(min(4, 12), 12),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + 12, 16, 3, padding=1, bias=False),  # 304改为44, 256改为16
            nn.GroupNorm(min(4, 16), 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, 16), 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Conv2d(16, num_classes, 1)  # 256改为16

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        # 使用双线性插值，确保对齐模式一致
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Baseline_DeeplabV3(nn.Module):
    """轻量版DeepLabV3+ - 使用小通道配置：[8, 16, 24, 32, 48, 64]"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        print(f"创建轻量版DeepLabV3+: input_channels={input_channels}, num_classes={num_classes}")
        print(f"通道配置: {[8, 16, 24, 32, 48, 64]}")
        
        # 使用小通道配置：[8, 16, 24, 32, 48, 64]
        channels = [8, 16, 24, 32, 48, 64]
        
        # 编码器（简化ResNet）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 3, stride=2, padding=1),
            nn.GroupNorm(min(4, channels[0]), channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.GroupNorm(min(4, channels[1]), channels[1]),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.GroupNorm(min(4, channels[2]), channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[3], 3, padding=1),
            nn.GroupNorm(min(4, channels[3]), channels[3]),
            nn.ReLU()
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, stride=2, padding=1),
            nn.GroupNorm(min(4, channels[4]), channels[4]),
            nn.ReLU(),
            nn.Conv2d(channels[4], channels[5], 3, padding=1),
            nn.GroupNorm(min(4, channels[5]), channels[5]),
            nn.ReLU()
        )
        
        # ASPP模块
        self.aspp = ASPP(channels[5], atrous_rates=[12, 24, 36])
        
        # 解码器
        self.decoder = Decoder(channels[3], num_classes)
        
        # 初始化
        self._init_weights()
        
        # 打印参数量信息
        self._print_model_info()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        """打印模型信息用于调试"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,} ({total_params/1e6:.3f}M)")

    def forward(self, x):
        # 保存输入尺寸
        input_size = x.shape[-2:]
        
        # 编码器
        low_level_feat = self.encoder1(x)  # 用于解码器的低层特征
        x = self.encoder2(low_level_feat)
        mid_level_feat = x  # 保存中层特征
        x = self.encoder3(x)
        
        # ASPP
        x = self.aspp(x)
        
        # 解码器
        x = self.decoder(x, mid_level_feat)
        
        # 上采样到原始输入大小
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return torch.sigmoid(x)


# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = Baseline_DeeplabV3(num_classes=1)
    
    # 测试前向传播
    input_tensor = torch.randn(2, 3, 256, 256)
    output = model(input_tensor)
    
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,} ({total_params/1e6:.3f}M)")