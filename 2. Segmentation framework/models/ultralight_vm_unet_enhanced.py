"""
增强版UltraLight VM-UNet with Dynamic Modal Fusion - 改进版
"""
import torch
import torch.nn as nn
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from models.dynamic_modal_fusion import DynamicModalFusion, FusionVisualizer
from typing import Tuple, Optional, Dict, Any
import os

class EnhancedUltraLightVMUNet(nn.Module):
    """
    增强版UltraLight VM-UNet - 改进版
    """
    
    def __init__(self, 
                 num_classes: int = 1,
                 input_channels: int = 3,
                 c_list: list = [8, 16, 24, 32, 48, 64],
                 split_att: str = 'fc',
                 bridge: bool = True,
                 enable_fusion: bool = True,
                 fusion_verbose: bool = False,
                 test_weight_method: str = 'historical_mean'):
        """
        Args:
            num_classes: 输出类别数
            input_channels: 输入通道数
            c_list: 通道数列表
            split_att: 注意力分割类型
            bridge: 是否使用桥接
            enable_fusion: 是否启用动态融合
            fusion_verbose: 是否输出融合调试信息
            test_weight_method: 测试时权重选择方法
        """
        super().__init__()
        
        # 验证输入通道数
        if enable_fusion and input_channels != 3:
            print(f"⚠️ Warning: Dynamic fusion requires 3 input channels, got {input_channels}")
            print("  Disabling fusion module.")
            enable_fusion = False
        
        # ==================== 动态融合模块（使用改进版） ====================
        self.dynamic_fusion = DynamicModalFusion(
            enabled=enable_fusion,
            verbose=fusion_verbose,
            test_weight_method=test_weight_method
        )
        
        # ==================== 原始backbone ====================
        self.backbone = UltraLight_VM_UNet(
            num_classes=num_classes,
            input_channels=input_channels,
            c_list=c_list,
            split_att=split_att,
            bridge=bridge
        )
        
        # ==================== 训练状态标记 ====================
        self.enable_fusion = enable_fusion
        self.fusion_verbose = fusion_verbose
        self.test_weight_method = test_weight_method
        
        # 用于存储当前融合权重
        self._current_fusion_weights = None
        
        # 打印配置信息
        self._print_config()
    
    def _print_config(self):
        """打印模型配置"""
        print("\n" + "=" * 50)
        print("ENHANCED ULTRALIGHT VM-UNET CONFIGURATION")
        print("=" * 50)
        print(f"Dynamic Fusion: {'✅ ENABLED' if self.enable_fusion else '❌ DISABLED'}")
        if self.enable_fusion:
            print(f"Fusion Verbose: {'✅ ON' if self.fusion_verbose else '❌ OFF'}")
            print(f"Test Weight Method: {self.test_weight_method}")
            print(f"   - current: Use current model weights")
            print(f"   - historical_mean: Use mean of training history (recommended)")
            print(f"   - historical_median: Use median of training history")
            print(f"   - last: Use last training weights")
        print("=" * 50 + "\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # Step 1: 动态模态融合
        if self.enable_fusion:
            # 动态融合模块在训练时返回 (features, weights)，测试时返回 features
            fusion_output = self.dynamic_fusion(x)
            
            # 处理不同的返回值
            if isinstance(fusion_output, tuple):
                # 训练模式：返回 (features, weights)
                fused_features = fusion_output[0]
                # 保存权重用于后续分析（如果需要）
                if self.training and len(fusion_output) > 1:
                    self._current_fusion_weights = fusion_output[1].detach()
            else:
                # 测试模式：只返回 features
                fused_features = fusion_output
        else:
            fused_features = x
        
        # Step 2: 原始backbone处理
        output = self.backbone(fused_features)
        
        return output
    
    def get_current_fusion_weights(self) -> Optional[torch.Tensor]:
        """获取当前融合权重（如果有）"""
        return self._current_fusion_weights
    
    def analyze_fusion(self) -> Dict[str, Any]:
        """分析融合效果"""
        if not self.enable_fusion:
            return {"error": "Fusion not enabled"}
        
        return self.dynamic_fusion.get_fusion_analysis()
    
    def visualize_fusion(self, output_dir: str = "./fusion_analysis"):
        """可视化融合分析"""
        if not self.enable_fusion:
            print("❌ Fusion module is not enabled.")
            return
        
        FusionVisualizer.generate_fusion_report(self.dynamic_fusion, output_dir)
    
    def reset_fusion_history(self):
        """重置融合历史记录"""
        if self.enable_fusion:
            self.dynamic_fusion.reset_history()
    
    def get_test_weights_info(self) -> Optional[Dict[str, Any]]:
        """获取测试权重信息"""
        if self.enable_fusion:
            return self.dynamic_fusion.get_test_weights_info()
        return None
    
    @property
    def fusion_enabled(self) -> bool:
        return self.enable_fusion


def create_ultralight_model(config, enable_fusion: bool = True, 
                           fusion_verbose: bool = False,
                           test_weight_method: str = 'historical_mean'):
    """
    创建增强版UltraLight VM-UNet - 改进版
    """
    model = EnhancedUltraLightVMUNet(
        num_classes=config.model_config['num_classes'],
        input_channels=config.model_config['input_channels'],
        c_list=config.model_config['c_list'],
        split_att=config.model_config['split_att'],
        bridge=config.model_config['bridge'],
        enable_fusion=enable_fusion,
        fusion_verbose=fusion_verbose,
        test_weight_method=test_weight_method
    )
    return model