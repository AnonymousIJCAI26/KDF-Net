import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm2d(nn.LayerNorm):
    """2D版本的LayerNorm"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class SimplePatchEmbed(nn.Module):
    """简化的图像补丁嵌入"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 自动调整到patch_size的倍数
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            H = (H // self.patch_size) * self.patch_size
            W = (W // self.patch_size) * self.patch_size
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        x = self.proj(x)
        x = self.norm(x)
        return x


class SimpleDownsample(nn.Module):
    """简化的下采样"""
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.reduction(x)


class SimpleUpsample(nn.Module):
    """简化的上采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(max(1, out_channels // 4), out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class SimpleSwinBlock(nn.Module):
    """简化的Swin Transformer块"""
    def __init__(self, dim, num_heads=3, window_size=7):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        
        # 简化的注意力 - 使用卷积代替复杂注意力
        self.conv_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # 深度可分离卷积
            nn.Conv2d(dim, dim, kernel_size=1),  # 逐点卷积
        )
        
        self.norm2 = LayerNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        shortcut = x
        
        # 简化的注意力
        x = self.norm1(x)
        x = shortcut + self.conv_attn(x)
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.mlp(x)
        
        return x


class SimpleSwinEncoder(nn.Module):
    """简化的Swin编码器"""
    def __init__(self, in_chans=3, embed_dim=96, depths=[1, 1, 2, 1], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        # Patch嵌入
        self.patch_embed = SimplePatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        
        # 四个阶段
        self.stages = nn.ModuleList()
        current_dim = embed_dim
        
        for i, depth in enumerate(depths):
            stage = nn.Sequential()
            
            # 添加多个Swin块
            for _ in range(depth):
                stage.append(SimpleSwinBlock(current_dim, num_heads=min(num_heads[i], current_dim // 4)))
            
            # 如果不是最后一个阶段，添加下采样
            if i < len(depths) - 1:
                stage.append(SimpleDownsample(current_dim))
                current_dim *= 2
            
            self.stages.append(stage)

    def forward(self, x):
        features = []
        
        # Patch嵌入
        x = self.patch_embed(x)
        features.append(x)  # Stage 0输出
        
        # 各个阶段
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features  # 返回所有阶段的特征


class Baseline_Swin_UNet(nn.Module):
    """稳定版的Swin-UNet"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        print(f"创建稳定版Swin-UNet")
        print(f"输入通道: {input_channels}, 输出类别: {num_classes}")
        
        # 使用更小的配置以减少内存使用
        embed_dim = 48
        depths = [1, 1, 2, 1]  # 减少深度
        num_heads = [2, 4, 8, 16]
        
        # 编码器
        self.encoder = SimpleSwinEncoder(
            in_chans=input_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        
        # 编码器输出通道
        enc_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        
        # 解码器上采样层
        self.up1 = SimpleUpsample(enc_channels[3], enc_channels[2])  # 8倍->4倍
        self.fuse1 = nn.Sequential(
            nn.Conv2d(enc_channels[2] * 2, enc_channels[2], 1),
            nn.GroupNorm(max(1, enc_channels[2] // 4), enc_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = SimpleUpsample(enc_channels[2], enc_channels[1])  # 4倍->2倍
        self.fuse2 = nn.Sequential(
            nn.Conv2d(enc_channels[1] * 2, enc_channels[1], 1),
            nn.GroupNorm(max(1, enc_channels[1] // 4), enc_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = SimpleUpsample(enc_channels[1], enc_channels[0])  # 2倍->1倍
        self.fuse3 = nn.Sequential(
            nn.Conv2d(enc_channels[0] * 2, enc_channels[0], 1),
            nn.GroupNorm(max(1, enc_channels[0] // 4), enc_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = SimpleUpsample(enc_channels[0], enc_channels[0] // 2)  # 1倍->1/2倍（为输出做准备）
        
        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(enc_channels[0] // 2, enc_channels[0] // 4, 3, padding=1),
            nn.GroupNorm(max(1, (enc_channels[0] // 4) // 4), enc_channels[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[0] // 4, num_classes, 1)
        )
        
        # 打印信息
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"编码器通道: {enc_channels}")
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"输入尺寸建议: 32的倍数 (如 224, 256, 288, 320...)")

    def forward(self, x):
        # 保存原始尺寸
        orig_size = x.shape[2:]
        
        # 编码器
        enc_features = self.encoder(x)
        
        # 打印调试信息
        if self.training:
            print(f"输入尺寸: {x.shape}")
            for i, feat in enumerate(enc_features):
                print(f"  Stage {i}: {feat.shape}")
        
        # 解码器 - 安全的尺寸匹配
        # Stage 3 -> Stage 2
        x_up = self.up1(enc_features[4])  # 最深层
        skip = enc_features[3]  # 对应的skip连接
        
        # 确保尺寸匹配
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x_up, skip], dim=1)
        x = self.fuse1(x)
        
        # Stage 2 -> Stage 1
        x_up = self.up2(x)
        skip = enc_features[2]
        
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x_up, skip], dim=1)
        x = self.fuse2(x)
        
        # Stage 1 -> Stage 0
        x_up = self.up3(x)
        skip = enc_features[1]
        
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x_up, skip], dim=1)
        x = self.fuse3(x)
        
        # 最终上采样
        x = self.up4(x)
        
        # 输出
        x = self.final(x)
        
        # 调整到原始尺寸
        if x.shape[-2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)
        
        return torch.sigmoid(x) if self.final[-1].out_channels == 1 else F.softmax(x, dim=1)


class UltraSimple_Swin_UNet(nn.Module):
    """极简Swin-UNet - 最稳定版本"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        print(f"创建极简Swin-UNet")
        
        # 超简单配置
        base_channels = 32
        
        # 编码器 - 简单的卷积下采样
        self.encoder = nn.ModuleList([
            # Stage 0
            nn.Sequential(
                nn.Conv2d(input_channels, base_channels, 4, stride=4, padding=0),
                LayerNorm2d(base_channels),
                nn.ReLU()
            ),
            # Stage 1
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*2, 2, stride=2, padding=0),
                LayerNorm2d(base_channels*2),
                nn.ReLU(),
                SimpleSwinBlock(base_channels*2, num_heads=2)
            ),
            # Stage 2
            nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels*4, 2, stride=2, padding=0),
                LayerNorm2d(base_channels*4),
                nn.ReLU(),
                SimpleSwinBlock(base_channels*4, num_heads=4)
            ),
            # Stage 3
            nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels*8, 2, stride=2, padding=0),
                LayerNorm2d(base_channels*8),
                nn.ReLU(),
                SimpleSwinBlock(base_channels*8, num_heads=8)
            )
        ])
        
        # 解码器 - 简单的转置卷积上采样
        self.decoder = nn.ModuleList([
            # Stage 3 -> Stage 2
            nn.Sequential(
                nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2),
                LayerNorm2d(base_channels*4),
                nn.ReLU()
            ),
            # Stage 2 -> Stage 1
            nn.Sequential(
                nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2),
                LayerNorm2d(base_channels*2),
                nn.ReLU()
            ),
            # Stage 1 -> Stage 0
            nn.Sequential(
                nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
                LayerNorm2d(base_channels),
                nn.ReLU()
            ),
            # Stage 0 -> Output
            nn.Sequential(
                nn.ConvTranspose2d(base_channels, base_channels//2, 4, stride=4),
                LayerNorm2d(base_channels//2),
                nn.ReLU()
            )
        ])
        
        # 输出层
        self.output = nn.Conv2d(base_channels//2, num_classes, 1)
        
        # 打印信息
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, x):
        orig_size = x.shape[2:]
        
        # 编码器
        features = []
        current = x
        for i, layer in enumerate(self.encoder):
            current = layer(current)
            features.append(current)
            if self.training:
                print(f"Encoder Stage {i}: {current.shape}")
        
        # 解码器
        current = features[-1]
        for i, (dec_layer, skip_idx) in enumerate(zip(self.decoder, [-2, -3, -4, -5])):
            current = dec_layer(current)
            
            # 添加skip连接（除了最后一层）
            if i < len(features) - 1:
                skip = features[skip_idx]
                
                # 确保尺寸匹配
                if current.shape[-2:] != skip.shape[-2:]:
                    current = F.interpolate(current, size=skip.shape[-2:], 
                                           mode='bilinear', align_corners=True)
                
                current = current + skip  # 残差连接，而不是concat
            
            if self.training:
                print(f"Decoder Stage {i}: {current.shape}")
        
        # 输出
        x = self.output(current)
        
        # 确保输出尺寸
        if x.shape[-2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)
        
        return torch.sigmoid(x) if self.output.out_channels == 1 else F.softmax(x, dim=1)


# 默认使用极简版本
Baseline_Swin_UNet = UltraSimple_Swin_UNet


if __name__ == "__main__":
    print("测试Swin-UNet...")
    
    # 测试各种输入尺寸
    model = Baseline_Swin_UNet(num_classes=1, input_channels=1)
    
    # test_sizes = [
    #     (2, 1, 224, 224),   # 标准尺寸
    #     (2, 1, 256, 256),   # 你的配置
    #     (2, 1, 192, 192),   # 非标准尺寸
    #     (2, 1, 320, 320),   # 更大尺寸
    # ]
    
    # for batch, ch, h, w in test_sizes:
    #     print(f"\n测试输入: {batch}x{ch}x{h}x{w}")
    #     try:
    #         input_tensor = torch.randn(batch, ch, h, w)
    #         output = model(input_tensor)
    #         print(f"✓ 成功! 输出: {output.shape}")
    #         print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    #     except Exception as e:
    #         print(f"✗ 失败: {e}")
    
    # print("\n✅ 所有测试完成!")