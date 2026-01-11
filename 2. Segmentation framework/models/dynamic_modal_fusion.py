"""
轻量级动态模态融合模块 - 改进版
测试时使用训练历史均值权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
import os
from datetime import datetime

class DynamicModalFusion(nn.Module):
    """
    动态模态融合模块 - 改进版
    
    改进点：
    1. 测试时使用训练历史均值权重（更稳健）
    2. 提供多种权重选择策略
    """
    
    def __init__(self, enabled: bool = True, verbose: bool = False, 
                 test_weight_method: str = 'historical_mean'):
        """
        Args:
            enabled: 是否启用动态融合
            verbose: 是否输出详细调试信息
            test_weight_method: 测试时权重选择方法
                - 'current': 使用当前权重（原始实现）
                - 'historical_mean': 使用训练历史均值（推荐）
                - 'historical_median': 使用训练历史中位数
                - 'last': 使用最后一次训练权重
        """
        super().__init__()
        self.enabled = enabled
        self.verbose = verbose
        self.test_weight_method = test_weight_method
        
        if not enabled:
            if verbose:
                print("⚠️ Dynamic modal fusion is DISABLED. Using direct 3-channel input.")
            return
        
        print(f"🎯 DynamicModalFusion Initializing (test method: {test_weight_method})...")
        
        # ==================== 极轻量融合组件 ====================
        self.conv_t1 = nn.Conv2d(1, 1, 1, bias=False)  # T1映射
        self.conv_ser = nn.Conv2d(1, 1, 1, bias=False)  # SER映射
        self.conv_pe = nn.Conv2d(1, 1, 1, bias=False)   # PE映射
        
        # 可学习的模态权重 [3]
        self.modal_weights = nn.Parameter(torch.ones(3) / 3.0)
        
        # ==================== 融合后调整 ====================
        self.fusion_adjust = nn.Conv2d(3, 3, 1, bias=False)
        
        # ==================== 【新增】训练历史存储 ====================
        # 存储训练过程中的归一化权重（用于测试时计算均值）
        self._train_normalized_history = []  # 存储训练时的归一化权重
        self._train_raw_history = []  # 存储训练时的原始权重
        self._train_sample_count = 0  # 训练样本计数
        
        # 可解释性存储
        self.modal_weights_history = []  # 存储权重历史（用于分析）
        self.modal_statistics_history = []  # 存储统计特征历史
        
        # 初始化权重
        self._init_weights()
        
        # 计算并打印增加的参数量
        self._print_parameter_info()
        
        if verbose:
            print("✅ DynamicModalFusion initialized successfully")
    
    def _init_weights(self):
        """初始化权重"""
        if not self.enabled:
            return
        
        # 简单的权重初始化
        def init_simple(conv_layer):
            nn.init.normal_(conv_layer.weight, mean=1.0, std=0.01)
        
        init_simple(self.conv_t1)
        init_simple(self.conv_ser)
        init_simple(self.conv_pe)
        
        # 模态权重初始化为平均分配
        with torch.no_grad():
            self.modal_weights.copy_(torch.ones(3) / 3.0)
        
        # 融合调整初始化为接近单位矩阵
        nn.init.normal_(self.fusion_adjust.weight, mean=1.0, std=0.01)
    
    def _print_parameter_info(self):
        """打印参数量信息"""
        if not self.enabled:
            return
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🎯 DynamicModalFusion Parameters: {total_params:,} ({total_params/1e6:.6f}M)")
    
    def compute_modal_statistics(self, t1: torch.Tensor, ser: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """计算模态统计特征"""
        B = t1.shape[0]
        stats_list = []
        
        for modal in [t1, ser, pe]:
            modal_flat = modal.view(B, -1)
            mean_val = modal_flat.mean(dim=1, keepdim=True)
            std_val = modal_flat.std(dim=1, keepdim=True)
            modal_stats = torch.cat([mean_val, std_val], dim=1)
            stats_list.append(modal_stats)
        
        return torch.cat(stats_list, dim=1)  # [B, 6]
    
    def _get_normalized_weights(self) -> torch.Tensor:
        """获取归一化权重（softmax确保和为1）"""
        return torch.softmax(self.modal_weights, dim=0)  # [3]
    
    def _get_test_weights(self) -> torch.Tensor:
        """
        获取测试时使用的权重
        
        根据test_weight_method选择：
        - 'current': 当前权重
        - 'historical_mean': 训练历史均值（推荐）
        - 'historical_median': 训练历史中位数
        - 'last': 最后一次训练权重
        """
        if self.test_weight_method == 'current':
            # 原始实现：使用当前权重
            return self._get_normalized_weights()
        
        elif self.test_weight_method in ['historical_mean', 'historical_median', 'last']:
            # 需要训练历史数据
            if not self._train_normalized_history:
                if self.verbose:
                    print("⚠️ No training history, using current weights")
                return self._get_normalized_weights()
            
            # 将历史数据转换为tensor
            history_tensor = torch.stack(self._train_normalized_history)  # [N, 3]
            
            if self.test_weight_method == 'historical_mean':
                return history_tensor.mean(dim=0)
            elif self.test_weight_method == 'historical_median':
                return history_tensor.median(dim=0).values
            elif self.test_weight_method == 'last':
                return history_tensor[-1]
        
        else:
            # 未知方法，使用当前权重
            print(f"⚠️ Unknown test_weight_method: {self.test_weight_method}, using 'current'")
            return self._get_normalized_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] 三通道输入
            
        Returns:
            训练模式: (fused_features, weight_matrix)
            测试模式: fused_features
        """
        if not self.enabled:
            dummy_weights = torch.ones(x.shape[0], 3, device=x.device) / 3.0
            return (x, dummy_weights) if self.training else x
        
        B, C, H, W = x.shape
        if C != 3:
            raise ValueError(f"Expected 3-channel input, got {C} channels")
        
        # 分离模态
        t1 = x[:, 0:1, :, :]
        ser = x[:, 1:2, :, :]
        pe = x[:, 2:3, :, :]
        
        # 独立特征提取
        f_t1 = self.conv_t1(t1)
        f_ser = self.conv_ser(ser)
        f_pe = self.conv_pe(pe)
        
        # 【关键改进】根据训练/测试模式选择权重
        if self.training:
            # 训练模式：使用当前权重，并记录历史
            current_normalized = self._get_normalized_weights()
            weights_to_use = current_normalized
            
            # 记录训练历史（每100个样本记录一次，避免内存过大）
            self._train_sample_count += B
            if self._train_sample_count >= 100:
                self._train_normalized_history.append(current_normalized.detach().clone())
                self._train_raw_history.append(self.modal_weights.detach().clone())
                self._train_sample_count = 0
                
                # 限制历史长度（最多保存1000个记录）
                if len(self._train_normalized_history) > 1000:
                    self._train_normalized_history.pop(0)
                    self._train_raw_history.pop(0)
        else:
            # 测试模式：使用指定的历史统计方法
            weights_to_use = self._get_test_weights()
        
        W1, W2, W3 = weights_to_use[0], weights_to_use[1], weights_to_use[2]
        
        # 动态加权融合
        fused_weighted = W1 * f_t1 + W2 * f_ser + W3 * f_pe
        
        # 调整融合特征
        fused_repeated = fused_weighted.repeat(1, 3, 1, 1)
        fused_adjusted = self.fusion_adjust(fused_repeated)
        
        # 保存可解释性数据（训练和测试都保存）
        with torch.no_grad():
            # 保存权重用于分析
            weights_cpu = weights_to_use.detach().cpu().numpy()
            expanded_weights = np.tile(weights_cpu, (B, 1))
            self.modal_weights_history.append(expanded_weights)
            
            # 保存统计特征
            stats = self.compute_modal_statistics(t1, ser, pe)
            self.modal_statistics_history.append(stats.detach().cpu().numpy())
        
        if self.verbose and not self.training:
            print(f"\n🔍 Dynamic Fusion (test mode):")
            print(f"   Method: {self.test_weight_method}")
            print(f"   Weights: T1={W1:.3f}, SER={W2:.3f}, PE={W3:.3f}")
        
        # 创建权重矩阵用于返回
        weight_matrix = weights_to_use.unsqueeze(0).repeat(B, 1)
        
        if self.training:
            return fused_adjusted, weight_matrix
        else:
            return fused_adjusted
    
    def get_weight_history(self) -> np.ndarray:
        """获取权重历史数据"""
        if not self.modal_weights_history:
            return np.array([])
        try:
            return np.concatenate(self.modal_weights_history, axis=0)
        except Exception as e:
            print(f"⚠️ Failed to get weight history: {e}")
            return np.array([])
    
    def get_fusion_analysis(self) -> Dict:
        """获取融合分析结果"""
        if not self.enabled:
            return {"status": "Fusion not enabled"}
        
        try:
            analysis = {
                "status": "success",
                "test_weight_method": self.test_weight_method,
                "train_history_size": len(self._train_normalized_history),
                "num_samples": len(self.modal_weights_history) * (100 if self._train_sample_count > 0 else 0),
                "current_weights": {
                    "T1": float(self.modal_weights[0].item()),
                    "SER": float(self.modal_weights[1].item()),
                    "PE": float(self.modal_weights[2].item()),
                    "normalized_T1": float(self._get_normalized_weights()[0].item()),
                    "normalized_SER": float(self._get_normalized_weights()[1].item()),
                    "normalized_PE": float(self._get_normalized_weights()[2].item()),
                }
            }
            
            # 如果有训练历史，计算历史统计
            if self._train_normalized_history:
                history_tensor = torch.stack(self._train_normalized_history)
                analysis["historical_statistics"] = {
                    "mean_T1": float(history_tensor[:, 0].mean().item()),
                    "mean_SER": float(history_tensor[:, 1].mean().item()),
                    "mean_PE": float(history_tensor[:, 2].mean().item()),
                    "std_T1": float(history_tensor[:, 0].std().item()),
                    "std_SER": float(history_tensor[:, 1].std().item()),
                    "std_PE": float(history_tensor[:, 2].std().item()),
                    "median_T1": float(history_tensor[:, 0].median().item()),
                    "median_SER": float(history_tensor[:, 1].median().item()),
                    "median_PE": float(history_tensor[:, 2].median().item()),
                }
                
                # 当前测试权重
                test_weights = self._get_test_weights()
                analysis["test_weights"] = {
                    "T1": float(test_weights[0].item()),
                    "SER": float(test_weights[1].item()),
                    "PE": float(test_weights[2].item()),
                    "method": self.test_weight_method
                }
            
            # 如果有可解释性数据
            if self.modal_weights_history:
                all_weights = self.get_weight_history()
                if len(all_weights) > 0:
                    analysis["modal_weights"] = {
                        "num_samples": len(all_weights),
                        "T1_mean": float(all_weights[:, 0].mean()),
                        "T1_std": float(all_weights[:, 0].std()),
                        "SER_mean": float(all_weights[:, 1].mean()),
                        "SER_std": float(all_weights[:, 1].std()),
                        "PE_mean": float(all_weights[:, 2].mean()),
                        "PE_std": float(all_weights[:, 2].std()),
                    }
            
            return analysis
            
        except Exception as e:
            return {"status": f"error: {str(e)}"}
    
    def reset_history(self):
        """重置历史记录"""
        self._train_normalized_history = []
        self._train_raw_history = []
        self._train_sample_count = 0
        self.modal_weights_history = []
        self.modal_statistics_history = []
    
    def get_current_weights(self) -> torch.Tensor:
        """获取当前归一化的模态权重"""
        return self._get_normalized_weights()
    
    def get_test_weights_info(self) -> Dict:
        """获取测试权重信息"""
        test_weights = self._get_test_weights()
        return {
            "method": self.test_weight_method,
            "weights": test_weights.detach().cpu().numpy(),
            "has_history": len(self._train_normalized_history) > 0
        }


class FusionVisualizer:
    """融合可视化工具"""
    
    @staticmethod
    def plot_modal_weights(weights_data: np.ndarray, save_path: Optional[str] = None):
        """
        绘制模态权重分布
        
        Args:
            weights_data: [N, 3] 权重数据
            save_path: 保存路径（可选）
        """
        if len(weights_data) == 0:
            print("⚠️ No weight data to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        modal_names = ['T1', 'SER', 'PE']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (ax, name, color) in enumerate(zip(axes, modal_names, colors)):
            ax.hist(weights_data[:, idx], bins=20, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'{name} Weight Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Weight Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = weights_data[:, idx].mean()
            std_val = weights_data[:, idx].std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}')
            ax.legend(loc='upper right')
        
        plt.suptitle('Dynamic Modal Weight Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Weight distribution plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_weight_evolution(weight_history: List[np.ndarray], save_path: Optional[str] = None):
        """
        绘制权重随训练的变化
        
        Args:
            weight_history: 权重历史列表
            save_path: 保存路径（可选）
        """
        if not weight_history:
            print("⚠️ No weight history to visualize")
            return
        
        # 将历史数据转换为数组
        all_weights = np.concatenate(weight_history, axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制三条曲线
        modal_names = ['T1', 'SER', 'PE']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i in range(3):
            ax.plot(all_weights[:, i], label=modal_names[i], color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Weight Value', fontsize=12)
        ax.set_title('Modal Weight Evolution During Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加移动平均线
        window_size = min(50, len(all_weights) // 10)
        if window_size > 1:
            for i in range(3):
                moving_avg = np.convolve(all_weights[:, i], np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(window_size-1, len(all_weights)), moving_avg, 
                       color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Weight evolution plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def generate_fusion_report(fusion_module: DynamicModalFusion, 
                               output_dir: str = "./fusion_analysis"):
        """
        生成完整的融合分析报告
        
        Args:
            fusion_module: 融合模块实例
            output_dir: 输出目录
        """
        if not fusion_module.enabled:
            print("⚠️ Fusion module is not enabled. No report generated.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取分析数据
        analysis = fusion_module.get_fusion_analysis()
        
        if analysis["status"] != "success":
            print(f"❌ Failed to get fusion analysis: {analysis}")
            return
        
        # 1. 保存分析结果为JSON
        import json
        report_path = os.path.join(output_dir, "fusion_report.json")
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📄 Fusion report saved to: {report_path}")
        
        # 2. 绘制权重分布
        weight_data = fusion_module.get_weight_history()
        if len(weight_data) > 0:
            weight_plot_path = os.path.join(output_dir, "weight_distribution.png")
            FusionVisualizer.plot_modal_weights(weight_data, weight_plot_path)
        
        # 3. 绘制权重演化
        if fusion_module.modal_weights_history:
            evolution_path = os.path.join(output_dir, "weight_evolution.png")
            FusionVisualizer.plot_weight_evolution(fusion_module.modal_weights_history, evolution_path)
        
        # 4. 生成文本总结
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("DYNAMIC MODAL FUSION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Weight Method: {analysis.get('test_weight_method', 'N/A')}\n")
            f.write(f"Train History Size: {analysis.get('train_history_size', 0)}\n\n")
            
            # 当前权重
            current_weights = analysis.get('current_weights', {})
            if current_weights:
                f.write("Current Weights:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Raw weights:\n")
                f.write(f"  T1: {current_weights.get('T1', 0):.4f}\n")
                f.write(f"  SER: {current_weights.get('SER', 0):.4f}\n")
                f.write(f"  PE: {current_weights.get('PE', 0):.4f}\n\n")
                
                f.write(f"Normalized weights:\n")
                f.write(f"  T1: {current_weights.get('normalized_T1', 0):.4f}\n")
                f.write(f"  SER: {current_weights.get('normalized_SER', 0):.4f}\n")
                f.write(f"  PE: {current_weights.get('normalized_PE', 0):.4f}\n\n")
            
            # 测试权重
            test_weights = analysis.get('test_weights', {})
            if test_weights:
                f.write("Test Weights (for inference):\n")
                f.write("-" * 30 + "\n")
                f.write(f"Method: {test_weights.get('method', 'N/A')}\n")
                f.write(f"T1: {test_weights.get('T1', 0):.4f}\n")
                f.write(f"SER: {test_weights.get('SER', 0):.4f}\n")
                f.write(f"PE: {test_weights.get('PE', 0):.4f}\n\n")
            
            # 模态权重统计
            modal_weights = analysis.get('modal_weights', {})
            if modal_weights:
                f.write("Modal Weight Statistics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Number of samples: {modal_weights.get('num_samples', 0)}\n")
                f.write(f"T1: Mean = {modal_weights.get('T1_mean', 0):.4f}, Std = {modal_weights.get('T1_std', 0):.4f}\n")
                f.write(f"SER: Mean = {modal_weights.get('SER_mean', 0):.4f}, Std = {modal_weights.get('SER_std', 0):.4f}\n")
                f.write(f"PE: Mean = {modal_weights.get('PE_mean', 0):.4f}, Std = {modal_weights.get('PE_std', 0):.4f}\n\n")
            
            # 历史统计
            historical_stats = analysis.get('historical_statistics', {})
            if historical_stats:
                f.write("Historical Statistics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean: T1={historical_stats.get('mean_T1', 0):.4f}, "
                       f"SER={historical_stats.get('mean_SER', 0):.4f}, "
                       f"PE={historical_stats.get('mean_PE', 0):.4f}\n")
                f.write(f"Std: T1={historical_stats.get('std_T1', 0):.4f}, "
                       f"SER={historical_stats.get('std_SER', 0):.4f}, "
                       f"PE={historical_stats.get('std_PE', 0):.4f}\n\n")
            
            # 判断主导模态
            if current_weights:
                normalized_weights = [
                    current_weights.get('normalized_T1', 0),
                    current_weights.get('normalized_SER', 0),
                    current_weights.get('normalized_PE', 0)
                ]
                if sum(normalized_weights) > 0:
                    dominant_idx = np.argmax(normalized_weights)
                    modal_names = ['T1', 'SER', 'PE']
                    
                    f.write(f"Dominant Modal: {modal_names[dominant_idx]} ")
                    f.write(f"({normalized_weights[dominant_idx]:.3f})\n\n")
                    
                    # 医学解释
                    f.write("Medical Interpretation:\n")
                    f.write("-" * 30 + "\n")
                    if normalized_weights[0] > 0.4:
                        f.write("• T1 dominant: Strong anatomical structure information\n")
                    if normalized_weights[1] > 0.4:
                        f.write("• SER dominant: Strong hemodynamic information\n")
                    if normalized_weights[2] > 0.4:
                        f.write("• PE dominant: Strong perfusion heterogeneity information\n")
                    if sum(normalized_weights) > 0.9:
                        f.write("• Good weight normalization (sum close to 1.0)\n")
        
        print(f"📋 Fusion analysis complete. Results saved to: {output_dir}")


# ==================== 测试函数 ====================
def test_dynamic_fusion():
    """测试动态融合模块"""
    print("🧪 Testing DynamicModalFusion...")
    
    # 创建模块
    fusion = DynamicModalFusion(enabled=True, verbose=True)
    
    # 测试输入 [batch_size=4, channels=3, height=256, width=256]
    test_input = torch.randn(4, 3, 256, 256)
    print(f"\nTest input shape: {test_input.shape}")
    
    # 测试训练模式
    print("\n--- Testing Training Mode ---")
    fusion.train()
    output, weights = fusion(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights (first sample): {weights[0].detach().numpy()}")
    
    # 测试推理模式
    print("\n--- Testing Inference Mode ---")
    fusion.eval()
    with torch.no_grad():
        output_inference = fusion(test_input)
    print(f"Inference output shape: {output_inference.shape}")
    
    # 测试获取当前权重
    current_weights = fusion.get_current_weights()
    print(f"\nCurrent normalized weights: {current_weights.detach().numpy()}")
    
    # 测试分析功能
    analysis = fusion.get_fusion_analysis()
    print(f"\nFusion analysis status: {analysis['status']}")
    
    # 测试权重历史
    weight_history = fusion.get_weight_history()
    print(f"\nWeight history shape: {weight_history.shape}")
    
    print("\n✅ DynamicModalFusion test completed successfully!")
    return fusion


if __name__ == "__main__":
    # 运行测试
    test_dynamic_fusion()