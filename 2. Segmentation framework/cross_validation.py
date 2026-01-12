"""
五折交叉验证脚本
"""

import os
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from collections import defaultdict
import shutil
from datetime import datetime
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore', message='upsample_bilinear2d_backward_out_cuda')

from utils import set_seed, get_logger, seed_data_loader, BceDiceLoss, get_optimizer, get_scheduler
from config_setting_mama_mia import MamaMiaConfig
from mama_mia_loader import MAMAMIADataLoader
from engine import train_one_epoch, val_one_epoch, test_one_epoch

try:
    from models.ultralight_vm_unet_enhanced import create_ultralight_model
    USE_ENHANCED_MODEL = True
    print("✅ Enhanced model module found")
except ImportError:
    # 如果增强版模型不存在，使用原始模型（向后兼容）
    USE_ENHANCED_MODEL = False
    print("⚠️ Enhanced model module not found")

# 导入thop用于计算FLOPs
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ thop module not found, cannot calculate FLOPs")

from mama_mia_dataset import MAMAMIADataset2D, MAMAMIAMultiModalAugmentation


def parse_args():
    parser = argparse.ArgumentParser(description='MAMA-MIA 五折交叉验证')
    
    parser.add_argument('--model', type=str, default='ultralight', 
                       choices=['ultralight', 'ultralight_enhanced', 'unet', 'attention_unet', 
                               'unet_plusplus', 'deeplabv3', 'swin_unet', 'nnunet', 
                               'transunet', 'fcn'],
                       help='选择模型类型: ultralight(你的原始模型), ultralight_enhanced(增强版), unet等')
    
    # 实验配置
    parser.add_argument('--name', type=str, default='cv_experiment', help='实验名称')
    parser.add_argument('--k_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 数据集配置
    parser.add_argument('--datasets', nargs='+', default=['DUKE', 'NACT', 'ISPY1', 'ISPY2'],
                       help='使用的数据集列表')
    
    # 模型配置
    parser.add_argument('--multimodal', action='store_true', 
                       help='使用多模态输入 (T1 + SER + PE)')
    parser.add_argument('--input_channels', type=int, default=1, 
                       help='输入通道数')
    
    # 动态融合参数
    parser.add_argument('--enable_fusion', action='store_true', 
                       help='Enable dynamic modal fusion (requires multimodal)')
    parser.add_argument('--fusion_verbose', action='store_true',
                       help='Enable verbose output for fusion module')
    parser.add_argument('--test_weight_method', type=str, default='historical_mean',
                       choices=['current', 'historical_mean', 'historical_median', 'last'],
                       help='Test weight selection method for dynamic fusion')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100, help='每折训练的epoch数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 数据增强
    parser.add_argument('--use_augmentation', action='store_true',
                       help='使用数据增强')
    parser.add_argument('--balanced_sampling', action='store_true',
                       help='使用平衡采样')
    
    # 优化器配置
    parser.add_argument('--opt', type=str, default='AdamW', help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 学习率调度器
    parser.add_argument('--sch', type=str, default='CosineAnnealingLR', help='学习率调度器')
    parser.add_argument('--T_max', type=int, default=50, help='CosineAnnealingLR的T_max')
    
    # 测试配置
    parser.add_argument('--cross_dataset_test', action='store_true',
                       help='是否进行跨数据集测试')
    parser.add_argument('--test_datasets', nargs='+', 
                       help='跨数据集测试时使用的测试数据集')
    
    parser.add_argument('--fold_indices', type=int, nargs='+', default=None,
                       help='指定要训练的折数索引（0-based），例如：0 或 0 2 4。不指定则训练所有折')
    
    
    return parser.parse_args()


class MAMAMIADatasetCV(MAMAMIADataset2D):
    """专门用于交叉验证的数据集，使用所有数据"""
    
    def __init__(self, **kwargs):
        # 保存原始参数用于调试
        original_kwargs = kwargs.copy()
        
        # 移除MAMAMIADataset2D不支持的参数
        if 'train_ratio' in kwargs:
            kwargs.pop('train_ratio')
        if 'val_ratio' in kwargs:
            kwargs.pop('val_ratio')
        
        # 设置mode为'train'以获取所有数据
        kwargs['mode'] = 'train'
        kwargs['cross_dataset_test'] = False
        
        print(f"创建交叉验证数据集...")
        
        super().__init__(**kwargs)
        
        # 验证我们确实使用了所有患者
        print(f"交叉验证数据集: 使用所有 {len(self.original_dataset.patient_ids)} 个患者")
        print(f"切片总数: {len(self)}")


class KFoldSplitter:
    """五折交叉验证数据分割器"""
    
    def __init__(self, dataset, n_splits=5, seed=42, show_progress=True):
        self.dataset = dataset
        self.n_splits = n_splits
        self.seed = seed
        self._show_progress = show_progress
        
        print(f"\n{'='*50}")
        print(f"创建 {n_splits} 折交叉验证分割器")
        print(f"{'='*50}")
        
        # 获取所有患者ID
        self.patient_ids = list(dataset.original_dataset.patient_data.keys())
        self.patient_ids.sort()  # 排序以确保可重复性
        self.total_patients = len(self.patient_ids)
        
        print(f"患者总数: {self.total_patients}")
        print(f"切片总数: {len(self.dataset)}")
        
        # 快速创建映射
        self._create_patient_to_slices_mapping()
        
        # 生成折数划分
        self.folds = self._generate_folds()
        
        # 验证划分
        self._validate_folds()
        
        print(f"\n✅ KFold Splitter 创建完成")
        print(f"{'='*50}")
    
    def _create_patient_to_slices_mapping(self):
        """创建患者ID到所有切片索引的映射"""
        print(f"\n[步骤1] 创建患者到切片映射...")
        
        self.patient_slices = defaultdict(list)
        self.patient_slice_counts = {}
        
        # 使用原始数据集的slice_indices
        original_patient_ids = self.dataset.original_dataset.patient_ids
        total_slices = len(self.dataset)
        
        if self._show_progress:
            print(f"  正在处理 {total_slices} 个切片...")
        
        start_time = time.time()
        
        # 快速遍历
        for slice_idx in range(total_slices):
            patient_idx, _ = self.dataset.slice_indices[slice_idx]
            
            # 显示进度
            if self._show_progress and slice_idx % 1000 == 0 and slice_idx > 0:
                elapsed = time.time() - start_time
                progress = (slice_idx + 1) / total_slices * 100
                print(f"  进度: {slice_idx+1}/{total_slices} ({progress:.1f}%) - 已用 {elapsed:.1f}秒")
            
            if patient_idx < len(original_patient_ids):
                patient_id = original_patient_ids[patient_idx]
                self.patient_slices[patient_id].append(slice_idx)
            else:
                print(f"  警告: patient_idx {patient_idx} 超出范围")
        
        # 统计
        for patient_id, slices in self.patient_slices.items():
            self.patient_slice_counts[patient_id] = len(slices)
        
        elapsed = time.time() - start_time
        print(f"  完成: {len(self.patient_slices)} 个患者，{total_slices} 个切片")
        print(f"  耗时: {elapsed:.2f} 秒")
    
    def _generate_folds(self):
        """生成五折划分"""
        print(f"\n[步骤2] 生成 {self.n_splits} 折划分...")
        
        # 只使用有切片的患者
        patients_with_slices = [pid for pid in self.patient_ids 
                               if pid in self.patient_slices and len(self.patient_slices[pid]) > 0]
        
        print(f"  有切片的患者: {len(patients_with_slices)}/{self.total_patients}")
        
        if len(patients_with_slices) == 0:
            raise ValueError("没有找到有切片的患者！")
        
        # 设置随机种子
        np.random.seed(self.seed)
        
        # 随机打乱患者ID
        shuffled_patient_ids = patients_with_slices.copy()
        np.random.shuffle(shuffled_patient_ids)
        
        # 计算每折的患者数量
        patients_per_fold = len(shuffled_patient_ids) // self.n_splits
        remainder = len(shuffled_patient_ids) % self.n_splits
        
        folds = []
        start_idx = 0
        
        for fold_idx in range(self.n_splits):
            # 计算该折的患者数量
            fold_patient_count = patients_per_fold + (1 if fold_idx < remainder else 0)
            
            # 获取该折的患者ID
            fold_patient_ids = shuffled_patient_ids[start_idx:start_idx + fold_patient_count]
            
            # 获取这些患者的所有切片索引
            fold_slice_indices = []
            for patient_id in fold_patient_ids:
                fold_slice_indices.extend(self.patient_slices[patient_id])
            
            folds.append({
                'fold_idx': fold_idx,
                'patient_ids': fold_patient_ids,
                'slice_indices': fold_slice_indices,
                'num_patients': len(fold_patient_ids),
                'num_slices': len(fold_slice_indices)
            })
            
            start_idx += fold_patient_count
        
        return folds
    
    def _validate_folds(self):
        """验证划分结果"""
        print(f"\n[步骤3] 验证划分结果...")
        
        total_slices_in_folds = 0
        
        for fold in self.folds:
            print(f"  折 {fold['fold_idx']}: {fold['num_patients']} 患者, {fold['num_slices']} 切片")
            total_slices_in_folds += fold['num_slices']
        
        print(f"\n  总切片数: {len(self.dataset)}")
        print(f"  已分配切片: {total_slices_in_folds}")
        
        if total_slices_in_folds != len(self.dataset):
            print(f"  警告: 有 {len(self.dataset) - total_slices_in_folds} 个切片未分配!")
        else:
            print(f"  ✅ 所有切片已正确分配")
    
    def get_fold_data(self, fold_idx):
        """获取指定折的训练集和验证集索引"""
        if fold_idx >= self.n_splits:
            raise ValueError(f"Fold index {fold_idx} out of range (0-{self.n_splits-1})")
        
        # 验证集是当前折
        val_slice_indices = self.folds[fold_idx]['slice_indices']
        
        if len(val_slice_indices) == 0:
            print(f"警告: 折 {fold_idx} 的验证集有0个切片!")
        
        # 训练集是所有其他折
        train_slice_indices = []
        for i, fold in enumerate(self.folds):
            if i != fold_idx:
                train_slice_indices.extend(fold['slice_indices'])
        
        if len(train_slice_indices) == 0:
            print(f"警告: 折 {fold_idx} 的训练集有0个切片!")
        
        return train_slice_indices, val_slice_indices


def create_cross_validation_datasets(config, fold_splitter, fold_idx):
    """创建交叉验证的数据集和加载器"""
    # 获取当前折的训练/验证切片索引
    train_indices, val_indices = fold_splitter.get_fold_data(fold_idx)
    
    print(f"\nFold {fold_idx} Dataset Info:")
    print(f"  Train slices: {len(train_indices)}")
    print(f"  Val slices: {len(val_indices)}")
    
    # 检查索引是否有效
    if len(train_indices) == 0:
        print(f"错误: 折 {fold_idx} 的训练集为空!")
        return None, None
    if len(val_indices) == 0:
        print(f"错误: 折 {fold_idx} 的验证集为空!")
        return None, None
    
    # 创建Subset数据集
    train_subset = Subset(fold_splitter.dataset, train_indices)
    val_subset = Subset(fold_splitter.dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,  # 训练集需要shuffle
        num_workers=min(config.num_workers, 2),  # 限制线程数
        pin_memory=True,
        drop_last=True if len(train_indices) > config.batch_size else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=1,  # 验证时batch_size=1
        shuffle=False,
        num_workers=min(config.num_workers, 2),
        pin_memory=True,
        drop_last=False
    )
    
    # 应用随机种子
    train_loader = seed_data_loader(train_loader, config.seed + fold_idx)
    val_loader = seed_data_loader(val_loader, config.seed + fold_idx)
    
    return train_loader, val_loader


def create_model_by_type(model_type, config):
    """根据模型类型创建模型"""
    print(f"正在创建 {model_type.upper()} 模型...")
    
    # 导入需要的模块
    import sys
    import os
    
    # 获取models目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    
    # 确保models目录在Python路径中
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    
    try:
        # ==================== 【你的模型 - 需要c_list参数】 ====================
        if model_type in ['ultralight', 'ultralight_enhanced']:
            if model_type == 'ultralight':
                from UltraLight_VM_UNet import UltraLight_VM_UNet
                model = UltraLight_VM_UNet(
                    num_classes=config.model_config['num_classes'],
                    input_channels=config.model_config['input_channels'],
                    c_list=config.model_config['c_list'],
                    split_att=config.model_config['split_att'],
                    bridge=config.model_config['bridge'],
                )
                model_type_display = "UltraLight VM-UNet"
                print(f"  ✅ 你的模型: c_list={config.model_config['c_list']}")
                
            elif model_type == 'ultralight_enhanced':
                if USE_ENHANCED_MODEL:
                    from ultralight_vm_unet_enhanced import create_ultralight_model
                    model = create_ultralight_model(config)
                    model_type_display = "Enhanced UltraLight VM-UNet"
                else:
                    from UltraLight_VM_UNet import UltraLight_VM_UNet
                    model = UltraLight_VM_UNet(
                        num_classes=config.model_config['num_classes'],
                        input_channels=config.model_config['input_channels'],
                        c_list=config.model_config['c_list'],
                        split_att=config.model_config['split_att'],
                        bridge=config.model_config['bridge'],
                    )
                    model_type_display = "UltraLight VM-UNet (增强版不可用)"
                print(f"  ✅ 你的增强模型")
        
        elif model_type == 'unet':
            # 标准UNet - 使用默认配置
            from baseline_unet import Baseline_UNet
            
            print(f"  ✅ 标准UNet: 使用默认标准配置")
            
            # 关键：标准UNet应该使用自己的默认参数，不传递c_list
            model = Baseline_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数！让模型使用自己的默认值
            )
            model_type_display = "Standard UNet"
            
        elif model_type == 'attention_unet':
            from baseline_attention_unet import Baseline_Attention_UNet
            print(f"  ✅ 标准Attention UNet: 使用默认标准配置")
            model = Baseline_Attention_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数
            )
            model_type_display = "Standard Attention UNet"
            
        elif model_type == 'unet_plusplus':
            from baseline_unet_plusplus import Baseline_UNetPlusPlus
            print(f"  ✅ 标准UNet++: 使用默认标准配置")
            model = Baseline_UNetPlusPlus(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                deep_supervision=False,  # 可以根据需要调整
                # 不传递c_list参数
            )
            model_type_display = "Standard UNet++"
            
        elif model_type == 'deeplabv3':
            from baseline_deeplabv3 import Baseline_DeeplabV3
            print(f"  ✅ 标准DeeplabV3: 使用默认标准配置")
            model = Baseline_DeeplabV3(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数
            )
            model_type_display = "Standard DeeplabV3"
            
        elif model_type == 'swin_unet':
            from baseline_swin_unet import Baseline_Swin_UNet
            print(f"  ✅ 标准Swin UNet: 使用默认标准配置")
            model = Baseline_Swin_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数
            )
            model_type_display = "Standard Swin UNet"
            
        elif model_type == 'nnunet':
            from baseline_nnunet import Baseline_nnUNet
            print(f"  ✅ 标准nnUNet: 使用默认标准配置")
            model = Baseline_nnUNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数
            )
            model_type_display = "Standard nnUNet"
            
        elif model_type == 'transunet':
            from baseline_transunet import Baseline_TransUNet
            print(f"  ✅ 标准TransUNet: 使用默认标准配置")
            model = Baseline_TransUNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数
            )
            model_type_display = "Standard TransUNet"
            
        elif model_type == 'fcn':
            from baseline_fcn import Baseline_FCN
            print(f"  ✅ 标准FCN: 使用默认标准配置")
            model = Baseline_FCN(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                # 不传递c_list参数
            )
            model_type_display = "Standard FCN"
            
        else:
            print(f"⚠️ 未知模型类型 '{model_type}'，使用默认UltraLight VM-UNet")
            from UltraLight_VM_UNet import UltraLight_VM_UNet
            model = UltraLight_VM_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.model_config['input_channels'],
                c_list=config.model_config['c_list'],
                split_att=config.model_config['split_att'],
                bridge=config.model_config['bridge'],
            )
            model_type_display = "UltraLight VM-UNet (默认)"
            print(f"  ✅ 默认使用你的模型: c_list={config.model_config['c_list']}")
    
    except ImportError as e:
        print(f"❌ 导入模型失败: {e}")
        print("使用UltraLight VM-UNet作为替代")
        from UltraLight_VM_UNet import UltraLight_VM_UNet
        model = UltraLight_VM_UNet(
            num_classes=config.model_config['num_classes'],
            input_channels=config.model_config['input_channels'],
            c_list=config.model_config['c_list'],
            split_att=config.model_config['split_att'],
            bridge=config.model_config['bridge'],
        )
        model_type_display = f"UltraLight VM-UNet ({model_type}替代)"
        print(f"  ✅ 替代使用你的模型: c_list={config.model_config['c_list']}")
    
    print(f"✅ {model_type_display} 模型创建成功")
    
    # 融合配置显示
    if hasattr(config, 'multimodal') and config.multimodal and hasattr(config, 'enable_fusion') and config.enable_fusion:
        print(f"🎯 Dynamic Fusion: ✅ ENABLED")
    
    return model, model_type_display


def calculate_model_complexity(model, input_channels=3, image_size=256):
    """计算模型的参数量 - 格式化显示版本"""
    print("\n#----------Model Complexity Analysis----------#")
    
    try:
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 格式化显示
        def format_number(num):
            if num >= 1_000_000:
                return f"{num / 1_000_000:.2f}M ({num:,})"
            elif num >= 1_000:
                return f"{num / 1_000:.2f}K ({num:,})"
            else:
                return f"{num:,}"
        
        print(f"Total Parameters: {format_number(total_params)}")
        print(f"Trainable Parameters: {format_number(trainable_params)}")
        
        # 计算非训练参数
        non_trainable_params = total_params - trainable_params
        if non_trainable_params > 0:
            print(f"Non-trainable Parameters: {format_number(non_trainable_params)}")
        
        # 计算百分比
        if total_params > 0:
            trainable_percent = (trainable_params / total_params) * 100
            print(f"Trainable Percentage: {trainable_percent:.1f}%")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_formatted': format_number(total_params),
            'trainable_params_formatted': format_number(trainable_params)
        }
        
    except Exception as e:
        print(f"⚠️ Model complexity analysis failed: {e}")
        return None

def train_fold(config, train_loader, val_loader, fold_idx, k_folds, work_dir):
    """训练单个折"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{k_folds}")
    print(f"{'='*60}")
    
    # 创建当前折的目录
    fold_dir = os.path.join(work_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    log_dir = os.path.join(fold_dir, 'log')
    checkpoint_dir = os.path.join(fold_dir, 'checkpoints')
    outputs_dir = os.path.join(fold_dir, 'outputs')
    
    for dir_path in [log_dir, checkpoint_dir, outputs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 设置日志
    logger = get_logger(f'fold_{fold_idx + 1}', log_dir)
    
    # 初始化模型
    print('#----------Initializing Model----------#')
    print(f"🔧 使用模型类型: {config.model_type.upper()}")
    
    # 根据模型类型创建模型
    model, model_type_display = create_model_by_type(config.model_type, config)
    
    # 将模型移到GPU
    model = model.cuda()
    
    # 计算模型复杂度
    complexity_info = calculate_model_complexity(
        model, 
        input_channels=config.input_channels,
        image_size=256  # 假设图像大小为256x256
    )
    
    # 使用DataParallel
    model = torch.nn.DataParallel(model, device_ids=[0], output_device=0)
    
    # 损失函数
    criterion = BceDiceLoss().cuda()
    
    # 优化器
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # 训练循环
    best_val_dice = 0.0  # 【修改】基于Dice分数保存最佳模型
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        # 训练
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            epoch, logger, config
        )
        
        # 验证
        val_loss, val_dice = val_one_epoch(  # 【修改】接收dice分数
            val_loader, model, criterion, epoch, logger, config
        )
        
        if hasattr(config, 'enable_fusion') and config.enable_fusion and USE_ENHANCED_MODEL and epoch % 10 == 0:
            try:
                if hasattr(model.module, 'analyze_fusion'):
                    analysis = model.module.analyze_fusion()
                    if analysis and analysis.get("status") == "success":
                        print(f"\n🔍 Fold {fold_idx + 1} - Fusion Analysis Epoch {epoch}:")
                        weights = analysis["modal_weights"]
                        if 'T1_mean' in weights:
                            print(f"  T1 weight: {weights['T1_mean']:.3f} ± {weights['T1_std']:.3f}")
                        if 'SER_mean' in weights:
                            print(f"  SER weight: {weights['SER_mean']:.3f} ± {weights['SER_std']:.3f}")
                        if 'PE_mean' in weights:
                            print(f"  PE weight: {weights['PE_mean']:.3f} ± {weights['PE_std']:.3f}")
            except Exception as e:
                print(f"⚠️ Fusion analysis failed: {e}")
        
        # 【修改】保存最佳模型 - 基于Dice分数
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch
            
            # 保存模型
            model_path = os.path.join(checkpoint_dir, f'best-epoch{epoch}-dice{val_dice:.4f}.pth')  
            
            # 清理state_dict - 与train_mama_mia_ultralight.py保持一致
            def clean_state_dict(state_dict):
                """清理state_dict，移除thop添加的额外参数"""
                cleaned_state_dict = {}
                removed_keys = []
                for key, value in state_dict.items():
                    if 'total_ops' not in key and 'total_params' not in key:
                        cleaned_state_dict[key] = value
                    else:
                        removed_keys.append(key)
                
                if removed_keys:
                    print(f"Cleaned {len(removed_keys)} extra parameters from state_dict")
                return cleaned_state_dict
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': clean_state_dict(model.module.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice, 
                'train_loss': train_loss,
                'model_type': config.model_type,
                'model_type_display': model_type_display,  
                'complexity_info': complexity_info
            }, model_path)
            
            # 删除旧的best模型
            old_models = [f for f in os.listdir(checkpoint_dir) if f.startswith('best-epoch') and f != f'best-epoch{epoch}-dice{val_dice:.4f}.pth']
            for old_model in old_models:
                try:
                    os.remove(os.path.join(checkpoint_dir, old_model))
                except:
                    pass
            
            print(f"✅ Saved best model for fold {fold_idx + 1} at epoch {epoch} (val_dice: {val_dice:.4f}, val_loss: {val_loss:.4f})")
    
    print(f"\n🎯 Fold {fold_idx + 1} completed. Best epoch: {best_epoch}, Best val_dice: {best_val_dice:.4f}, Best val_loss: {best_val_loss:.4f}")
    
    return {
        'fold_idx': fold_idx + 1,
        'best_epoch': best_epoch,
        'best_val_dice': best_val_dice,  
        'best_val_loss': best_val_loss,
        'fold_dir': fold_dir,
        'best_model_path': os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-dice{best_val_dice:.4f}.pth'),  
        'model_type': model_type_display,
        'complexity_info': complexity_info
    }


def evaluate_fold(fold_info, test_loader, config, dataset_name):
    """评估单个折的模型"""
    print(f"\nEvaluating Fold {fold_info['fold_idx']} on {dataset_name}...")
    
    # 加载检查点以获取模型类型
    checkpoint = torch.load(fold_info['best_model_path'], map_location=torch.device('cpu'))
    
    # 获取保存的模型类型
    saved_model_type = checkpoint.get('model_type', 'ultralight')
    
    # 创建相同类型的模型 - 使用create_model_by_type确保正确配置
    model, _ = create_model_by_type(saved_model_type, config)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0], output_device=0)
    
    # 处理状态字典 - 过滤掉thop添加的参数
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 过滤掉不必要的键
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'total_ops' not in key and 'total_params' not in key:
            filtered_state_dict[key] = value
    
    model.module.load_state_dict(filtered_state_dict, strict=True)
    model.eval()
    
    # 创建临时logger
    class DummyLogger:
        def info(self, msg):
            print(f"[LOG] {msg}")
    
    dummy_logger = DummyLogger()
    
    # 在测试集上评估
    test_loss = test_one_epoch(
        test_loader, model, BceDiceLoss().cuda(), 
        dummy_logger, config, test_data_name=f"fold_{fold_info['fold_idx']}_{dataset_name}"
    )
    
    return test_loss


def main():
    args = parse_args()
    
    print("="*60)
    print("MAMA-MIA 五折交叉验证")
    print("="*60)
    print(f"实验名称: {args.name}")
    print(f"数据集: {args.datasets}")
    print(f"折数: {args.k_folds}")
    print(f"随机种子: {args.seed}")
    print(f"多模态: {args.multimodal}")
    print(f"模型类型: {args.model.upper()}")
    
    if args.multimodal and args.enable_fusion:
        print("🎯 Dynamic Modal Fusion: ✅ ENABLED")
        print(f"   - Test weight method: {args.test_weight_method}")
        print(f"   - Verbose mode: {'✅ ON' if args.fusion_verbose else '❌ OFF'}")
    elif args.multimodal:
        print("🎯 Dynamic Modal Fusion: ❌ DISABLED")

    
    print(f"训练epoch: {args.epochs}")
    if args.cross_dataset_test and args.test_datasets:
        print(f"跨数据集测试: {args.test_datasets}")
    print("="*60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    config = MamaMiaConfig(
        model_type=args.model,  # 这里传递 args.model
        multimodal=args.multimodal,
        datasets_list=args.datasets,
        input_channels=args.input_channels,
        num_workers=args.num_workers,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.opt,
        weight_decay=args.weight_decay,
        sch=args.sch,
        T_max=args.T_max,
        threshold=0.5
    )
    
    # 双重确保 model_type 被正确设置
    config.model_type = args.model
    print(f"配置中的模型类型: {config.model_type.upper()}")
    
    # 确保输入通道正确
    if args.multimodal:
        config.input_channels = 3
        print(f"多模态输入，设置 input_channels=3")
    else:
        print(f"单模态输入，使用 input_channels={config.input_channels}")
    
    # 更新配置
    config.use_augmentation = args.use_augmentation
    config.balanced_sampling = args.balanced_sampling
    
    config.enable_fusion = args.enable_fusion
    config.fusion_verbose = args.fusion_verbose
    config.test_weight_method = args.test_weight_method
    
    # 设置工作目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.work_dir = f'results/IJCAI_experiments/{args.name}_{timestamp}'
    os.makedirs(config.work_dir, exist_ok=True)
    
    # 保存配置
    config_dict = {k: v for k, v in vars(args).items()}
    config_path = os.path.join(config.work_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"\n配置已保存到: {config_path}")
    
    # 加载完整数据集
    print("\n#----------Loading Complete Dataset for Cross-Validation----------#")
    
    # 配置数据增强
    transform = None
    if args.use_augmentation and args.multimodal:
        transform = MAMAMIAMultiModalAugmentation(p=0.5)
        print("启用多模态数据增强")
    
    # 创建专门用于交叉验证的数据集
    print(f"\n创建交叉验证数据集...")
    
    full_dataset = MAMAMIADatasetCV(
        data_dir=config.data_dir,
        seg_dir=config.seg_dir,
        datasets=config.datasets_list,
        mode='train',
        input_channels=config.input_channels,
        multimodal=config.multimodal,
        ser_dir=config.ser_dir,
        pe_dir=config.pe_dir,
        transform=transform,
        seed=config.seed,
        balanced_sampling=args.balanced_sampling,
        cross_dataset_test=False
    )
    
    print(f"\n完整数据集信息:")
    print(f"  切片总数: {len(full_dataset)}")
    print(f"  患者数量: {len(full_dataset.original_dataset.patient_data)}")
    
    # 创建五折分割器
    print("\n#----------Creating K-Fold Splits----------#")
    start_time = time.time()
    fold_splitter = KFoldSplitter(full_dataset, n_splits=args.k_folds, seed=args.seed)
    end_time = time.time()
    print(f"创建五折划分耗时: {end_time - start_time:.2f} 秒")
    
    # 存储结果
    fold_results = []

    if args.fold_indices is None:
        # 如果没有指定，训练所有折
        folds_to_train = list(range(args.k_folds))
    else:
        # 过滤有效的折数索引
        folds_to_train = []
        for idx in args.fold_indices:
            if 0 <= idx < args.k_folds:
                folds_to_train.append(idx)
            else:
                print(f"警告: 折索引 {idx} 超出范围 (0-{args.k_folds-1})，将被忽略")
    
    if not folds_to_train:
        print("错误: 没有有效的折数索引可训练！")
        return
    
    print(f"将训练以下折: {folds_to_train}")
    
    # 进行五折交叉验证（仅训练指定的折）
    for fold_idx in folds_to_train:
        print(f"\n{'='*60}")
        print(f"开始处理第 {fold_idx + 1}/{args.k_folds} 折")
        print(f"{'='*60}")
        
        # 创建当前折的数据加载器
        train_loader, val_loader = create_cross_validation_datasets(
            config, fold_splitter, fold_idx
        )
        
        # 检查数据加载器是否为空
        if train_loader is None or val_loader is None:
            print(f"跳过折 {fold_idx}，因为数据加载器创建失败")
            continue
        
        # 训练当前折
        fold_info = train_fold(
            config, train_loader, val_loader, 
            fold_idx, args.k_folds, config.work_dir
        )
        
        # 保存结果
        fold_results.append(fold_info)
    
    if not fold_results:
        print("错误: 没有成功训练任何折!")
        return
    
    # 如果有测试数据集，进行测试
    if args.cross_dataset_test and args.test_datasets:
        print(f"\n{'='*60}")
        print(f"进行跨数据集测试")
        print(f"{'='*60}")
        
        # 创建测试配置
        test_config = MamaMiaConfig(
            multimodal=args.multimodal,
            datasets_list=args.test_datasets,
            input_channels=args.input_channels,
            cross_dataset_test=True,
            num_workers=args.num_workers,
            enable_fusion=args.enable_fusion,
            fusion_verbose=args.fusion_verbose,
            test_weight_method=args.test_weight_method
        )
        
        # 确保测试配置也有正确的模型类型
        test_config.model_type = config.model_type
        
        # 创建测试数据加载器
        test_data_loader = MAMAMIADataLoader(test_config)
        test_loader = test_data_loader.get_test_loader()
        
        dataset_name = '_'.join(args.test_datasets)
        
        # 评估每个折的模型
        for fold_info in fold_results:
            try:
                test_loss = evaluate_fold(fold_info, test_loader, config, dataset_name)
                fold_info[f'test_loss_{dataset_name}'] = test_loss
                print(f"折 {fold_info['fold_idx']} 在 {dataset_name} 上的测试损失: {test_loss:.4f}")
            except Exception as e:
                print(f"评估折 {fold_info['fold_idx']} 时出错: {e}")
                fold_info[f'test_loss_{dataset_name}'] = None
    
    # 保存交叉验证结果
    print("\n#----------Saving Cross-Validation Results----------#")
    
    # 准备结果数据
    results_data = []
    for r in fold_results:
        result_row = {
            'fold': r['fold_idx'],
            'best_epoch': r['best_epoch'],
            'best_val_dice': r['best_val_dice'],  # 【修改】改为best_val_dice
            'best_val_loss': r['best_val_loss'],
            'model_path': r['best_model_path'],
            'model_type': r.get('model_type', 'Unknown'),  # 添加模型类型
        }
        
        # 添加模型复杂度信息
        if r.get('complexity_info'):
            result_row['total_params'] = r['complexity_info'].get('total_params', 'N/A')
            result_row['trainable_params'] = r['complexity_info'].get('trainable_params', 'N/A')
            result_row['total_params_formatted'] = r['complexity_info'].get('total_params_formatted', 'N/A')
            result_row['trainable_params_formatted'] = r['complexity_info'].get('trainable_params_formatted', 'N/A')
        
        # 添加测试结果
        for key in r.keys():
            if key.startswith('test_loss_'):
                result_row[key] = r[key]
        
        results_data.append(result_row)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results_data)
    
    # 计算统计信息
    stats = {
        'mean_val_dice': results_df['best_val_dice'].mean(),  # 【修改】统计Dice
        'std_val_dice': results_df['best_val_dice'].std(),
        'min_val_dice': results_df['best_val_dice'].min(),
        'max_val_dice': results_df['best_val_dice'].max(),
        'mean_val_loss': results_df['best_val_loss'].mean(),
        'std_val_loss': results_df['best_val_loss'].std(),
        'model_type': args.model.upper(),  # 使用命令行参数中的模型类型
        'enable_fusion': args.enable_fusion if args.multimodal else False
    }
    
    # 添加模型复杂度统计
    if 'total_params' in results_df.columns:
        try:
            # 只处理数值型参数
            total_params_numeric = pd.to_numeric(results_df['total_params'], errors='coerce').dropna()
            if len(total_params_numeric) > 0:
                stats['mean_total_params'] = total_params_numeric.mean()
                stats['std_total_params'] = total_params_numeric.std()
        except:
            pass
    
    # 添加测试统计
    test_columns = [col for col in results_df.columns if col.startswith('test_loss_')]
    for test_col in test_columns:
        dataset_name = test_col.replace('test_loss_', '')
        test_values = results_df[test_col].dropna()
        if len(test_values) > 0:
            stats.update({
                f'mean_test_loss_{dataset_name}': test_values.mean(),
                f'std_test_loss_{dataset_name}': test_values.std(),
                f'min_test_loss_{dataset_name}': test_values.min(),
                f'max_test_loss_{dataset_name}': test_values.max(),
            })
    
    # 保存结果
    results_path = os.path.join(config.work_dir, 'cv_results.csv')
    stats_path = os.path.join(config.work_dir, 'cv_statistics.json')
    
    results_df.to_csv(results_path, index=False)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("🎉 五折交叉验证完成!")
    print(f"{'='*60}")
    print(f"验证Dice统计:")  # 【修改】显示Dice统计
    print(f"  平均: {stats['mean_val_dice']:.4f} ± {stats['std_val_dice']:.4f}")
    print(f"  最小: {stats['min_val_dice']:.4f}")
    print(f"  最大: {stats['max_val_dice']:.4f}")
    print(f"\n验证损失统计:")
    print(f"  平均: {stats['mean_val_loss']:.4f} ± {stats['std_val_loss']:.4f}")
    
    # 打印模型信息
    print(f"\n模型信息:")
    print(f"  模型类型: {stats['model_type']}")
    if args.multimodal:
        print(f"  动态融合: {'✅ ENABLED' if stats['enable_fusion'] else '❌ DISABLED'}")
    
    # 打印模型复杂度信息
    if 'mean_total_params' in stats:
        print(f"\n模型复杂度:")
        print(f"  平均参数量: {stats['mean_total_params']:,.0f} ± {stats['std_total_params']:,.0f}")
    
    # 打印测试结果
    for test_col in test_columns:
        dataset_name = test_col.replace('test_loss_', '')
        if f'mean_test_loss_{dataset_name}' in stats:
            print(f"\n在 {dataset_name} 上的测试结果:")
            print(f"  平均: {stats[f'mean_test_loss_{dataset_name}']:.4f} ± {stats[f'std_test_loss_{dataset_name}']:.4f}")
    
    print(f"\n结果已保存到: {config.work_dir}")
    print(f"  - 详细结果: {results_path}")
    print(f"  - 统计信息: {stats_path}")
    
    # 保存最佳模型
    if len(results_df) > 0:
        best_fold_idx = results_df['best_val_dice'].idxmax()  # 【修改】基于Dice选择最佳模型
        best_fold_info = fold_results[best_fold_idx]
        
        best_model_dest = os.path.join(config.work_dir, 'best_model.pth')
        try:
            shutil.copy(best_fold_info['best_model_path'], best_model_dest)
            print(f"  - 最佳模型已复制到: {best_model_dest} (来自折 {best_fold_idx}, Dice: {best_fold_info['best_val_dice']:.4f})")
        except Exception as e:
            print(f"  - 复制最佳模型时出错: {e}")
    
    # 创建汇总报告
    report_path = os.path.join(config.work_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MAMA-MIA 五折交叉验证汇总报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"实验名称: {args.name}\n")
        f.write(f"数据集: {args.datasets}\n")
        f.write(f"折数: {args.k_folds}\n")
        f.write(f"实际训练的折数: {len(folds_to_train)} (索引: {folds_to_train})\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"多模态: {args.multimodal}\n")
        f.write(f"模型类型: {args.model.upper()}\n")
        if args.multimodal and args.enable_fusion:
            f.write(f"动态融合: ✅ ENABLED (method: {args.test_weight_method})\n")
        f.write("\n")
        
        f.write("验证Dice统计:\n")
        f.write(f"  平均: {stats['mean_val_dice']:.4f} ± {stats['std_val_dice']:.4f}\n")
        f.write(f"  最小: {stats['min_val_dice']:.4f}\n")
        f.write(f"  最大: {stats['max_val_dice']:.4f}\n\n")
        
        f.write("验证损失统计:\n")
        f.write(f"  平均: {stats['mean_val_loss']:.4f} ± {stats['std_val_loss']:.4f}\n\n")
        
        if 'mean_total_params' in stats:
            f.write("模型复杂度统计:\n")
            f.write(f"  平均参数量: {stats['mean_total_params']:,.0f} ± {stats['std_total_params']:,.0f}\n\n")
        
        if test_columns:
            f.write("跨数据集测试结果:\n")
            for test_col in test_columns:
                dataset_name = test_col.replace('test_loss_', '')
                if f'mean_test_loss_{dataset_name}' in stats:
                    f.write(f"  {dataset_name}:\n")
                    f.write(f"    平均: {stats[f'mean_test_loss_{dataset_name}']:.4f} ± {stats[f'std_test_loss_{dataset_name}']:.4f}\n")
        
        if len(results_df) > 0:
            best_fold_idx = results_df['best_val_dice'].idxmax()  # 【修改】基于Dice
            f.write(f"\n最佳模型来自: 折 {best_fold_idx}\n")
            f.write(f"最佳Dice分数: {results_df.loc[best_fold_idx, 'best_val_dice']:.4f}\n")
            best_model_type = results_df.loc[best_fold_idx, 'model_type'] if 'model_type' in results_df.columns else args.model.upper()
            f.write(f"最佳模型类型: {best_model_type}\n")
    
    print(f"  - 汇总报告: {report_path}")


if __name__ == '__main__':

    main()
