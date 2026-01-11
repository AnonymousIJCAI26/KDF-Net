import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline_FCN(nn.Module):
    """FCN-8s简化版 - 标准配置"""
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        
        # FCN标准VGG通道数：[64, 128, 256, 512, 512, 512]
        channels = [64, 128, 256, 512, 512, 512]
        
        # 编码器（VGG风格）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 3, padding=1),
            nn.GroupNorm(4, channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.GroupNorm(4, channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.GroupNorm(4, channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], 3, padding=1),
            nn.GroupNorm(4, channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.GroupNorm(4, channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[2], 3, padding=1),
            nn.GroupNorm(4, channels[2]),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[3], 3, padding=1),
            nn.GroupNorm(4, channels[3]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, padding=1),
            nn.GroupNorm(4, channels[4]),
            nn.ReLU(),
            nn.Conv2d(channels[4], channels[4], 3, padding=1),
            nn.GroupNorm(4, channels[4]),
            nn.ReLU(),
            nn.Conv2d(channels[4], channels[4], 3, padding=1),
            nn.GroupNorm(4, channels[4]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        
        self.encoder5 = nn.Sequential(
            nn.Conv2d(channels[4], channels[5], 3, padding=1),
            nn.GroupNorm(4, channels[5]),
            nn.ReLU(),
            nn.Conv2d(channels[5], channels[5], 3, padding=1),
            nn.GroupNorm(4, channels[5]),
            nn.ReLU(),
            nn.Conv2d(channels[5], channels[5], 3, padding=1),
            nn.GroupNorm(4, channels[5]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(channels[5], channels[4], 7, padding=3),
            nn.GroupNorm(4, channels[4]),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            nn.Conv2d(channels[4], channels[4], 1),
            nn.GroupNorm(4, channels[4]),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            nn.Conv2d(channels[4], num_classes, 1)
        )
        
        # 跳跃连接层
        self.score_pool4 = nn.Conv2d(channels[4], num_classes, 1)
        self.score_pool3 = nn.Conv2d(channels[3], num_classes, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器
        pool1 = self.encoder1(x)
        pool2 = self.encoder2(pool1)
        pool3 = self.encoder3(pool2)
        pool4 = self.encoder4(pool3)
        pool5 = self.encoder5(pool4)
        
        # 主分类器
        score = self.classifier(pool5)
        
        # 跳跃连接（FCN-8s风格）
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)
        
        # 上采样和融合
        score = F.interpolate(score, size=score_pool4.shape[2:], mode='bilinear', align_corners=True)
        score += score_pool4
        
        score = F.interpolate(score, size=score_pool3.shape[2:], mode='bilinear', align_corners=True)
        score += score_pool3
        
        # 上采样到原始大小
        score = F.interpolate(score, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return torch.sigmoid(score)