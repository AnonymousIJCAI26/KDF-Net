import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(max(1, middle_channels // 2), middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(max(1, out_channels // 2), out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out


class Baseline_UNetPlusPlus(nn.Module):
    """最小通道版UNet++: [8, 16, 24, 32, 48]"""
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        # 最小通道配置
        nb_filter = [8, 16, 24, 32, 48]
        
        print(f"创建最小通道版UNet++: channels={nb_filter}, input_channels={input_channels}, num_classes={num_classes}")
        
        self.deep_supervision = deep_supervision
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 编码器部分
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        # 第一层密集连接
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        
        # 第二层密集连接
        self.conv0_2 = VGGBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], nb_filter[2])
        
        # 第三层密集连接
        self.conv0_3 = VGGBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], nb_filter[1])
        
        # 第四层密集连接
        self.conv0_4 = VGGBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], nb_filter[0])
        
        # 输出层
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
        # 统计参数量
        self._print_model_info()

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,} ({total_params/1e6:.3f}M)")

    def forward(self, input):
        # 编码路径
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = torch.sigmoid(self.final1(x0_1))
            output2 = torch.sigmoid(self.final2(x0_2))
            output3 = torch.sigmoid(self.final3(x0_3))
            output4 = torch.sigmoid(self.final4(x0_4))
            return (output1 + output2 + output3 + output4) / 4
        else:
            output = self.final(x0_4)
            return torch.sigmoid(output)


if __name__ == "__main__":
    # 测试模型
    model = Baseline_UNetPlusPlus(num_classes=1)
    input_tensor = torch.randn(2, 3, 256, 256)
    output = model(input_tensor)
    
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")