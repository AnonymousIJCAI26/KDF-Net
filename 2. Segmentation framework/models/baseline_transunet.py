import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    """简化版Transformer块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x):
        # LayerNorm + Attention
        shortcut = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = shortcut + attn_output
        
        # LayerNorm + MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x


class Baseline_TransUNet(nn.Module):
    """修复版TransUNet - 保证通道数匹配"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        # 使用合理的通道数配置
        channels = [32, 64, 128, 256]  # 更小的通道数，更容易训练
        
        # 编码器层 - 分离定义以便获取skip连接
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 7, stride=2, padding=3),
            nn.GroupNorm(max(1, channels[0] // 4), channels[0]),
            nn.ReLU(inplace=True)
        )
        self.enc_pool1 = nn.MaxPool2d(2)
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.GroupNorm(max(1, channels[1] // 4), channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.GroupNorm(max(1, channels[2] // 4), channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Transformer编码器
        self.transformer_embed = nn.Conv2d(channels[2], channels[3], 1)
        self.transformer = TransformerBlock(channels[3], num_heads=4)
        
        # 解码器层 - 修正通道数
        self.dec_conv3 = nn.Sequential(
            # 处理concat后的特征：256 + 128 = 384 -> 128
            nn.Conv2d(channels[3] + channels[2], channels[2], 3, padding=1),
            nn.GroupNorm(max(1, channels[2] // 4), channels[2]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)  # 128 -> 64
        )
        
        self.dec_conv2 = nn.Sequential(
            # 64 + 64 = 128 -> 64
            nn.Conv2d(channels[1] * 2, channels[1], 3, padding=1),
            nn.GroupNorm(max(1, channels[1] // 4), channels[1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)  # 64 -> 32
        )
        
        self.dec_conv1 = nn.Sequential(
            # 32 + 32 = 64 -> 32
            nn.Conv2d(channels[0] * 2, channels[0], 3, padding=1),
            nn.GroupNorm(max(1, channels[0] // 4), channels[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], channels[0], 2, stride=2)  # 32 -> 32
        )
        
        # 最终上采样（如果需要恢复到原图尺寸）
        self.final_upsample = nn.ConvTranspose2d(channels[0], channels[0], 2, stride=2)
        
        # 输出层
        self.final = nn.Conv2d(channels[0], num_classes, 1)
        
        # 打印配置信息
        print(f"TransUNet通道配置: {channels}")
        print(f"输入通道: {input_channels}, 输出类别: {num_classes}")
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 保存原始尺寸
        original_size = x.shape[2:]
        
        # 编码器路径（保存skip连接）
        enc1 = self.enc_conv1(x)      # [B, 32, H/2, W/2]
        pool1 = self.enc_pool1(enc1)  # [B, 32, H/4, W/4]
        
        enc2 = self.enc_conv2(pool1)  # [B, 64, H/8, W/8]
        enc3 = self.enc_conv3(enc2)   # [B, 128, H/16, W/16]
        
        # Transformer路径
        B, C, H, W = enc3.shape
        x_trans = self.transformer_embed(enc3)  # [B, 256, H/16, W/16]
        
        # 转换为序列格式
        x_seq = x_trans.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        x_seq = self.transformer(x_seq)
        
        # 转换回空间格式
        x_trans = x_seq.transpose(1, 2).view(B, -1, H, W)  # [B, 256, H/16, W/16]
        
        # 解码器路径（带skip连接）
        # 第一级解码
        x = torch.cat([x_trans, enc3], dim=1)  # [B, 384, H/16, W/16]
        x = self.dec_conv3(x)                  # [B, 64, H/8, W/8]
        
        # 第二级解码
        x = torch.cat([x, enc2], dim=1)        # [B, 128, H/8, W/8]
        x = self.dec_conv2(x)                  # [B, 32, H/4, W/4]
        
        # 第三级解码
        x = torch.cat([x, pool1], dim=1)       # [B, 64, H/4, W/4]
        x = self.dec_conv1(x)                  # [B, 32, H/2, W/2]
        
        # 最终上采样到原图尺寸
        if x.shape[2:] != original_size:
            x = self.final_upsample(x)         # [B, 32, H, W]
        
        # 输出
        x = self.final(x)
        
        # 确保输出尺寸匹配输入
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        
        return torch.sigmoid(x)


if __name__ == "__main__":
    # 测试模型
    print("测试Baseline_TransUNet...")
    model = Baseline_TransUNet(num_classes=1, input_channels=3)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,} ({total_params/1e6:.3f}M)")
    
    # 测试不同尺寸的输入
    test_sizes = [(1, 3, 256, 256), (2, 3, 512, 512), (4, 3, 128, 128)]
    
    for batch, channels, height, width in test_sizes:
        input_tensor = torch.randn(batch, channels, height, width)
        output = model(input_tensor)
        
        print(f"\n测试尺寸: {input_tensor.shape}")
        print(f"输出尺寸: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n✅ TransUNet测试通过!")