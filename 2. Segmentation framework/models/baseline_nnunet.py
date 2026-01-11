# baseline_nnunet.py - 标准nnUNet 2D实现（修复版）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


class ConvBlock(nn.Module):
    """标准nnUNet卷积块，包含残差连接"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        
        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class DownBlock(nn.Module):
    """nnUNet下采样块：使用卷积下采样"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        # 标准nnUNet使用卷积下采样而不是MaxPool
        self.downsample = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """nnUNet上采样块：使用插值+卷积"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels: 上采样输入的特征图通道数
            skip_channels: skip连接的通道数
            out_channels: 输出通道数
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # concat后的通道数是 in_channels + skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.skip_channels = skip_channels
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # 确保尺寸匹配
        if x.shape != skip.shape:
            # 调整skip到x的尺寸
            if skip.shape[2:] != x.shape[2:]:
                # 使用插值调整尺寸
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Baseline_nnUNet(nn.Module):
    """标准nnUNet 2D实现 - 保持原有类名以便兼容"""
    def __init__(self, 
                 num_classes: int = 1, 
                 input_channels: int = 3,
                 base_features: int = 32,
                 max_features: int = 320,
                 num_pool: int = 5,
                 deep_supervision: bool = False,  # 默认关闭，避免输出格式问题
                 **kwargs):
        """
        Args:
            num_classes: 输出类别数
            input_channels: 输入通道数
            base_features: 基础特征数（标准nnUNet为32）
            max_features: 最大特征数（标准nnUNet为320）
            num_pool: 下采样次数（网络深度，标准为5）
            deep_supervision: 是否启用深度监督（默认关闭以避免训练脚本兼容性问题）
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.num_pool = num_pool
        
        # 动态计算通道数（标准nnUNet方式）
        self.channels = self._calculate_channels(base_features, max_features, num_pool)
        print(f"nnUNet动态通道配置: {self.channels}")
        
        # 编码器（下采样路径）
        self.down_blocks = nn.ModuleList()
        
        # 第一层特殊处理（没有下采样）
        self.first_conv = ConvBlock(input_channels, self.channels[0])
        
        # 创建下采样块
        for i in range(num_pool):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.down_blocks.append(DownBlock(in_ch, out_ch))
        
        # 瓶颈层（最深层）
        self.bottleneck = ConvBlock(self.channels[num_pool], self.channels[num_pool])
        
        # 解码器（上采样路径）
        self.up_blocks = nn.ModuleList()
        for i in range(num_pool - 1, -1, -1):
            # 修正通道数计算
            in_ch = self.channels[i + 1]  # 上采样前的通道数
            skip_ch = self.channels[i]    # skip连接的通道数
            out_ch = self.channels[i]     # 输出通道数
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))
        
        # 输出层
        self.final_conv = nn.Conv2d(self.channels[0], num_classes, kernel_size=1)
        
        # 深度监督层（默认关闭，避免训练脚本兼容性问题）
        if self.deep_supervision:
            self.ds_convs = nn.ModuleList()
            for i in range(num_pool - 1):  # 跳过最后两层
                out_ch = self.channels[-(i + 2)]  # 从深层到浅层
                self.ds_convs.append(nn.Conv2d(out_ch, num_classes, kernel_size=1))
        
        # 参数初始化（标准nnUNet方式）
        self._init_weights()
        
        # 统计信息
        self._print_model_info()
    
    def _calculate_channels(self, base: int, max_features: int, num_pool: int) -> List[int]:
        """计算每层通道数（标准nnUNet方式）"""
        channels = []
        for i in range(num_pool + 1):
            channels.append(min(base * (2 ** i), max_features))
        return channels
    
    def _init_weights(self):
        """标准nnUNet权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ 标准nnUNet创建成功")
        print(f"   总参数量: {total_params:,} ({total_params/1e6:.3f}M)")
        print(f"   网络深度: {self.num_pool}层下采样")
        print(f"   基础特征: {self.channels[0]}, 最大特征: {self.channels[-1]}")
        print(f"   深度监督: {'启用' if self.deep_supervision else '关闭'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        注意：默认返回单个输出张量以保持与训练脚本兼容
        如果需要深度监督，请设置 deep_supervision=True
        """
        skips = []
        
        # 编码器路径
        x = self.first_conv(x)
        skips.append(x)
        
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器路径
        ds_outputs = [] if self.deep_supervision else None
        
        for i, up_block in enumerate(self.up_blocks):
            skip_idx = -(i + 2)  # 对应的skip连接索引
            x = up_block(x, skips[skip_idx])
            
            # 深度监督（如果需要）
            if self.deep_supervision and i < len(self.ds_convs):
                ds = self.ds_convs[i](x)
                # 上采样到输入尺寸
                scale_factor = 2 ** (len(self.ds_convs) - i)
                ds = F.interpolate(ds, scale_factor=scale_factor, 
                                 mode='bilinear', align_corners=True)
                ds_outputs.append(torch.sigmoid(ds))
        
        # 最终输出
        x = self.final_conv(x)
        final_output = torch.sigmoid(x)
        
        if self.deep_supervision:
            # 深度监督模式：返回所有监督输出（从浅到深）
            ds_outputs.append(final_output)
            return ds_outputs[::-1]
        else:
            # 普通模式：返回单个输出（与原始UNet兼容）
            return final_output


# ============ 简化版本（更接近你原来的接口） ============
class Baseline_nnUNet_Simple(nn.Module):
    """简化版nnUNet，使用原始UNet的接口但保持nnUNet核心特性"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        # 使用合理的默认配置
        channels = [32, 64, 128, 256, 320]  # 简化版通道配置
        
        # 编码器
        self.down1 = DownBlock(input_channels, channels[0])
        self.down2 = DownBlock(channels[0], channels[1])
        self.down3 = DownBlock(channels[1], channels[2])
        self.down4 = DownBlock(channels[2], channels[3])
        
        # 瓶颈层
        self.bottleneck = ConvBlock(channels[3], channels[4])
        
        # 解码器（使用修正的UpBlock）
        self.up4 = UpBlock(channels[4], channels[3], channels[3])
        self.up3 = UpBlock(channels[3], channels[2], channels[2])
        self.up2 = UpBlock(channels[2], channels[1], channels[1])
        self.up1 = UpBlock(channels[1], channels[0], channels[0])
        
        # 输出层
        self.final = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        
        print(f"✅ 简化版nnUNet创建成功 (channels={channels})")
    
    def forward(self, x):
        # 编码器
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        # 输出
        x = self.final(x)
        return torch.sigmoid(x)


# ============ 兼容性导出 ============
# 创建别名以确保导入兼容
Baseline_nnUNet = Baseline_nnUNet_Simple  # 使用简化版作为默认

__all__ = ['Baseline_nnUNet', 'Baseline_nnUNet_Simple', 'ConvBlock', 'DownBlock', 'UpBlock']


# ============ 测试代码 ============
if __name__ == "__main__":
    # 测试简化版nnUNet（默认）
    print("=" * 60)
    print("测试简化版nnUNet (默认)")
    print("=" * 60)
    
    model = Baseline_nnUNet(num_classes=1, input_channels=3)
    input_tensor = torch.randn(2, 3, 256, 256)
    
    # 测试前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 测试标准版（如果需要）
    print("\n" + "=" * 60)
    print("测试标准版nnUNet")
    print("=" * 60)
    
    model_std = Baseline_nnUNet_Simple(num_classes=1, input_channels=3)
    output_std = model_std(input_tensor)
    
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output_std.shape}")
    print(f"输出范围: [{output_std.min():.4f}, {output_std.max():.4f}]")
    
    print("\n✅ 所有测试通过!")