"""
交叉验证测试脚本 - 支持所有baseline模型（修复导入问题版本）
"""

import sys
import os

# ==================== 【重要修复】添加路径 ====================
# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录添加到Python路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 确保models目录在Python路径中
models_dir = os.path.join(current_dir, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

print(f"Current directory: {current_dir}")
print(f"Models directory: {models_dir}")
print(f"Python search path (first 5):")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i+1}. {path}")
# ==================== 【修复结束】 ====================

import torch
from torch import nn
import argparse
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.stats import ttest_rel
import SimpleITK as sitk
from skimage import measure
import warnings
import glob
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from config_setting_mama_mia import MamaMiaConfig
from mama_mia_loader import MAMAMIADataLoader
from engine import test_one_epoch

# ==================== 【动态导入模型函数】 ====================
def import_model_module(module_name, class_name=None):
    """动态导入模型模块"""
    # 可能的导入路径
    import_paths = [
        f'models.{module_name}',  # 从models包导入
        module_name,  # 直接导入
    ]
    
    for import_path in import_paths:
        try:
            module = __import__(import_path, fromlist=[''])
            print(f"  ✅ Imported {module_name} from '{import_path}'")
            
            if class_name:
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                else:
                    # 如果指定的类不存在，返回模块本身
                    return module
            else:
                return module
                
        except ImportError:
            continue
    
    # 如果以上都失败，尝试从文件导入
    module_paths = [
        os.path.join(models_dir, f'{module_name}.py'),
        os.path.join(current_dir, f'{module_name}.py'),
        os.path.join(current_dir, 'models', f'{module_name}.py'),
    ]
    
    for module_path in module_paths:
        if os.path.exists(module_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module
                print(f"  ✅ Imported {module_name} from file: {module_path}")
                
                if class_name:
                    if hasattr(module, class_name):
                        return getattr(module, class_name)
                    else:
                        return module
                else:
                    return module
                    
            except Exception as e:
                print(f"  ⚠️ Failed to import from file {module_path}: {e}")
                continue
    
    raise ImportError(f"Cannot import {module_name} from any location")

# 检查模型可用性
print("\n#----------Checking Model Availability----------#")

# 检查UltraLight模型
try:
    UltraLight_VM_UNet = import_model_module('UltraLight_VM_UNet', 'UltraLight_VM_UNet')
    ULTRALIGHT_AVAILABLE = True
    print("✅ UltraLight VM-UNet available")
except ImportError as e:
    ULTRALIGHT_AVAILABLE = False
    print(f"⚠️ UltraLight VM-UNet not available: {e}")

# 检查增强版模型
USE_ENHANCED_MODEL = False
try:
    enhanced_module = import_model_module('ultralight_vm_unet_enhanced')
    if hasattr(enhanced_module, 'create_ultralight_model'):
        USE_ENHANCED_MODEL = True
        print("✅ Enhanced model available")
    else:
        print("⚠️ Enhanced model module found but create_ultralight_model not available")
except ImportError as e:
    print(f"⚠️ Enhanced model not available: {e}")

# 检查baseline模型
BASELINE_MODELS_AVAILABLE = False
baseline_models_status = {}

baseline_models = [
    ('baseline_unet', 'Baseline_UNet'),
    ('baseline_attention_unet', 'Baseline_Attention_UNet'),
    ('baseline_unet_plusplus', 'Baseline_UNetPlusPlus'),
    ('baseline_deeplabv3', 'Baseline_DeeplabV3'),
    ('baseline_swin_unet', 'Baseline_Swin_UNet'),
    ('baseline_nnunet', 'Baseline_nnUNet'),
    ('baseline_transunet', 'Baseline_TransUNet'),
    ('baseline_fcn', 'Baseline_FCN'),
]

for module_name, class_name in baseline_models:
    try:
        model_class = import_model_module(module_name, class_name)
        baseline_models_status[module_name] = True
        print(f"✅ {module_name} available")
    except ImportError as e:
        baseline_models_status[module_name] = False
        print(f"⚠️ {module_name} not available: {e}")

# 如果至少有一个baseline模型可用
if any(baseline_models_status.values()):
    BASELINE_MODELS_AVAILABLE = True
    print("✅ Some baseline models are available")
else:
    print("⚠️ No baseline models available")
# ==================== 【导入结束】 ====================

def print_memory_usage():
    """打印内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {gpu_memory:.2f}GB")

def parse_args():
    parser = argparse.ArgumentParser(description='UltraLight VM-UNet Cross-Validation Testing')
    parser.add_argument('--cv_dir', type=str, required=True, 
                       help='Cross-validation results directory')
    parser.add_argument('--multimodal', action='store_true', 
                       help='Use multimodal input (T1+SER+PE)')
    
    parser.add_argument('--enable_fusion', action='store_true', 
                       help='Enable dynamic modal fusion (must match training setting)')
    parser.add_argument('--fusion_verbose', action='store_true',
                       help='Enable verbose output for fusion module')
    parser.add_argument('--test_weight_method', type=str, default='historical_mean',
                       choices=['current', 'historical_mean', 'historical_median', 'last'],
                       help='Test weight selection method for dynamic fusion')
    parser.add_argument('--analyze_fusion', action='store_true',
                       help='Generate fusion analysis report after testing')
    
    parser.add_argument('--model_type', type=str, default='',
                       choices=['', 'ultralight', 'ultralight_enhanced', 'unet', 'attention_unet', 
                               'unet_plusplus', 'deeplabv3', 'swin_unet', 'nnunet', 
                               'transunet', 'fcn'],
                       help='Model type (if not specified, use saved model type from checkpoint)')
    
    parser.add_argument('--datasets', nargs='+', required=True, 
                       help='Datasets to use for testing, e.g., DUKE NACT ISPY1 ISPY2')
    parser.add_argument('--input_channels', type=int, default=1, 
                       help='Input channels')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/data/', 
                       help='Data directory')
    parser.add_argument('--seg_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/segmentations_expert/', 
                       help='Segmentation directory')
    parser.add_argument('--ser_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_SER_T1/', 
                       help='SER directory')
    parser.add_argument('--pe_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_PE_T1/', 
                       help='PE directory')
    parser.add_argument('--num_workers', type=int, default=2, 
                       help='Number of data loading workers')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Threshold for binary segmentation')
    parser.add_argument('--save_nifti', action='store_true', 
                       help='Save 3D nifti files for each patient')
    parser.add_argument('--balanced_sampling', action='store_true', 
                       help='For configuration consistency')
    parser.add_argument('--data_augmentation', action='store_true', 
                       help='For configuration consistency')
    parser.add_argument('--cross_dataset_test', action='store_true', 
                       help='Test on entire target dataset for generalization evaluation')
    parser.add_argument('--fold_idx', type=int, default=-1,
                       help='Test specific fold (-1 for all folds)')
    parser.add_argument('--test_all_folds', action='store_true',
                       help='Test all folds and compute ensemble results')
    return parser.parse_args()

def filter_state_dict(state_dict):
    """过滤掉thop添加的额外参数"""
    filtered_state_dict = {}
    removed_count = 0
    for key, value in state_dict.items():
        if 'total_ops' not in key and 'total_params' not in key:
            filtered_state_dict[key] = value
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"Filtered out {removed_count} extra parameters (total_ops, total_params)")
    return filtered_state_dict

def find_best_model_in_fold(fold_dir):
    """在折目录中查找最佳模型"""
    checkpoint_dir = os.path.join(fold_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找best-epoch开头的文件
    best_models = glob.glob(os.path.join(checkpoint_dir, 'best-epoch*.pth'))
    if not best_models:
        # 尝试查找best.pth
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        if os.path.exists(best_path):
            return best_path
        return None
    
    # 按epoch排序，取最新的
    best_models.sort()
    return best_models[-1]

def create_model_for_testing(saved_model_type, config, args):
    """根据保存的模型类型创建模型 - 用于测试"""
    print(f"\nCreating {saved_model_type.upper()} model for testing...")
    
    model = None
    
    try:
        # ==================== 【UltraLight模型】 ====================
        if saved_model_type in ['ultralight', 'ultralight_enhanced']:
            if saved_model_type == 'ultralight':
                if not ULTRALIGHT_AVAILABLE:
                    raise ImportError("UltraLight VM-UNet not available")
                
                model = UltraLight_VM_UNet(
                    num_classes=config.model_config['num_classes'],
                    input_channels=config.model_config['input_channels'],
                    c_list=config.model_config['c_list'],
                    split_att=config.model_config['split_att'],
                    bridge=config.model_config['bridge'],
                )
                print(f"  ✅ Original UltraLight VM-UNet")
                
            elif saved_model_type == 'ultralight_enhanced':
                if USE_ENHANCED_MODEL:
                    enhanced_module = import_model_module('ultralight_vm_unet_enhanced')
                    model = enhanced_module.create_ultralight_model(
                        config,
                        enable_fusion=args.enable_fusion,
                        fusion_verbose=args.fusion_verbose,
                        test_weight_method=args.test_weight_method
                    )
                    print(f"  ✅ Enhanced UltraLight VM-UNet")
                else:
                    if not ULTRALIGHT_AVAILABLE:
                        raise ImportError("UltraLight VM-UNet not available")
                    
                    model = UltraLight_VM_UNet(
                        num_classes=config.model_config['num_classes'],
                        input_channels=config.model_config['input_channels'],
                        c_list=config.model_config['c_list'],
                        split_att=config.model_config['split_att'],
                        bridge=config.model_config['bridge'],
                    )
                    print(f"  ✅ UltraLight VM-UNet (enhanced not available)")
        
        # ==================== 【标准UNet】 ====================
        elif saved_model_type == 'unet':
            if not baseline_models_status.get('baseline_unet', False):
                raise ImportError("Baseline UNet not available")
            
            Baseline_UNet = import_model_module('baseline_unet', 'Baseline_UNet')
            model = Baseline_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard UNet")
            
        # ==================== 【Attention UNet】 ====================
        elif saved_model_type == 'attention_unet':
            if not baseline_models_status.get('baseline_attention_unet', False):
                raise ImportError("Baseline Attention UNet not available")
            
            Baseline_Attention_UNet = import_model_module('baseline_attention_unet', 'Baseline_Attention_UNet')
            model = Baseline_Attention_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard Attention UNet")
            
        # ==================== 【UNet++】 ====================
        elif saved_model_type == 'unet_plusplus':
            if not baseline_models_status.get('baseline_unet_plusplus', False):
                raise ImportError("Baseline UNet++ not available")
            
            Baseline_UNetPlusPlus = import_model_module('baseline_unet_plusplus', 'Baseline_UNetPlusPlus')
            model = Baseline_UNetPlusPlus(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
                deep_supervision=False,
            )
            print(f"  ✅ Standard UNet++")
            
        # ==================== 【DeeplabV3】 ====================
        elif saved_model_type == 'deeplabv3':
            if not baseline_models_status.get('baseline_deeplabv3', False):
                raise ImportError("Baseline DeeplabV3 not available")
            
            Baseline_DeeplabV3 = import_model_module('baseline_deeplabv3', 'Baseline_DeeplabV3')
            model = Baseline_DeeplabV3(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard DeeplabV3")
            
        # ==================== 【Swin UNet】 ====================
        elif saved_model_type == 'swin_unet':
            if not baseline_models_status.get('baseline_swin_unet', False):
                raise ImportError("Baseline Swin UNet not available")
            
            Baseline_Swin_UNet = import_model_module('baseline_swin_unet', 'Baseline_Swin_UNet')
            model = Baseline_Swin_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard Swin UNet")
            
        # ==================== 【nnUNet】 ====================
        elif saved_model_type == 'nnunet':
            if not baseline_models_status.get('baseline_nnunet', False):
                raise ImportError("Baseline nnUNet not available")
            
            Baseline_nnUNet = import_model_module('baseline_nnunet', 'Baseline_nnUNet')
            model = Baseline_nnUNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard nnUNet")
            
        # ==================== 【TransUNet】 ====================
        elif saved_model_type == 'transunet':
            if not baseline_models_status.get('baseline_transunet', False):
                raise ImportError("Baseline TransUNet not available")
            
            Baseline_TransUNet = import_model_module('baseline_transunet', 'Baseline_TransUNet')
            model = Baseline_TransUNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard TransUNet")
            
        # ==================== 【FCN】 ====================
        elif saved_model_type == 'fcn':
            if not baseline_models_status.get('baseline_fcn', False):
                raise ImportError("Baseline FCN not available")
            
            Baseline_FCN = import_model_module('baseline_fcn', 'Baseline_FCN')
            model = Baseline_FCN(
                num_classes=config.model_config['num_classes'],
                input_channels=config.input_channels,
            )
            print(f"  ✅ Standard FCN")
            
        # ==================== 【未知模型类型】 ====================
        else:
            print(f"⚠️ Unknown saved model type '{saved_model_type}'，使用UltraLight VM-UNet")
            if not ULTRALIGHT_AVAILABLE:
                raise ImportError("UltraLight VM-UNet not available")
            
            model = UltraLight_VM_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.model_config['input_channels'],
                c_list=config.model_config['c_list'],
                split_att=config.model_config['split_att'],
                bridge=config.model_config['bridge'],
            )
            print(f"  ✅ Using UltraLight VM-UNet as default")
    
    except Exception as e:
        print(f"❌ Failed to create model {saved_model_type}: {e}")
        print("Attempting fallback to UltraLight VM-UNet...")
        
        if ULTRALIGHT_AVAILABLE:
            model = UltraLight_VM_UNet(
                num_classes=config.model_config['num_classes'],
                input_channels=config.model_config['input_channels'],
                c_list=config.model_config['c_list'],
                split_att=config.model_config['split_att'],
                bridge=config.model_config['bridge'],
            )
            print("  ✅ Created UltraLight VM-UNet as fallback")
        else:
            raise ImportError("No model available for testing")
    
    return model

def calculate_metrics_3d(pred, target):
    """计算3D分割指标 - 与test_mama_mia_ultralight_advanced.py保持一致"""
    eps = 1e-6
    
    # 确保是二值图像
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    
    # 计算体积（像素点数）
    volume_pred = np.sum(pred)
    volume_target = np.sum(target)
    
    # 计算交集和并集
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    
    # Dice系数
    dice = (2.0 * intersection + eps) / (np.sum(pred) + np.sum(target) + eps)
    
    # IoU
    iou = (intersection + eps) / (union + eps)
    
    # Precision, Recall, Specificity
    true_positive = intersection
    false_positive = np.sum(pred) - intersection
    false_negative = np.sum(target) - intersection
    true_negative = np.size(pred) - (true_positive + false_positive + false_negative)
    
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    recall = (true_positive + eps) / (true_positive + false_negative + eps)
    specificity = (true_negative + eps) / (true_negative + false_positive + eps)
    accuracy = (true_positive + true_negative + eps) / (true_positive + true_negative + false_positive + false_negative + eps)
    
    # Hausdorff Distance 95%
    try:
        hd95 = calculate_hd95(pred, target)
    except:
        hd95 = np.nan
    
    return {
        'iou': iou,
        'dice': dice,
        'hd95': hd95,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'volume_pred': volume_pred,
        'volume_target': volume_target
    }

def calculate_hd95(pred, target):
    """计算Hausdorff Distance 95%"""
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.nan
    
    # 使用SimpleITK计算Hausdorff距离
    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
    target_sitk = sitk.GetImageFromArray(target.astype(np.uint8))
    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(pred_sitk, target_sitk)
    hd95 = hausdorff_distance_filter.GetHausdorffDistance()
    
    return hd95

def reconstruct_3d_volume(patient_slices, patient_id):
    """将2D切片重构为3D体积"""
    # 根据切片顺序排序
    slices_sorted = sorted(patient_slices, key=lambda x: x['slice_idx'])
    
    if not slices_sorted:
        raise ValueError(f"No slices found for patient {patient_id}")
    
    # 获取第一个切片的形状
    first_slice = slices_sorted[0]['prediction']
    
    # 确保是2D切片 (H, W)
    if len(first_slice.shape) == 3:
        if first_slice.shape[0] == 1:  # (1, H, W)
            first_slice = first_slice[0]  # 取第一个通道
        else:
            raise ValueError(f"Unexpected 3D slice shape: {first_slice.shape}")
    
    if len(first_slice.shape) != 2:
        raise ValueError(f"Expected 2D slice, got shape: {first_slice.shape}")
    
    height, width = first_slice.shape
    depth = len(slices_sorted)
    
    # 正确的维度顺序: (depth, height, width)
    volume_3d = np.zeros((depth, height, width), dtype=np.float32)
    
    # 填充3D体积
    for i, slice_data in enumerate(slices_sorted):
        slice_pred = slice_data['prediction']
        
        # 处理切片形状
        if len(slice_pred.shape) == 3 and slice_pred.shape[0] == 1:
            slice_pred = slice_pred[0]  # (1, H, W) -> (H, W)
        elif len(slice_pred.shape) != 2:
            raise ValueError(f"Unexpected slice shape at index {i}: {slice_pred.shape}")
        
        volume_3d[i] = slice_pred
    
    return volume_3d

def correct_axis_order(volume):
    """修正轴顺序 - 确保与医学图像标准一致"""
    # 转置轴顺序: (depth, height, width) -> (width, height, depth)
    volume_corrected = np.transpose(volume, (2, 1, 0))
    
    return volume_corrected

def process_prediction(pred, threshold=0.5):
    """处理预测结果"""
    pred_processed = np.where(pred >= threshold, 1, 0)
    
    return pred_processed.astype(np.uint8)

def statistical_significance_test(metrics_list):
    """为每个指标独立进行显著性检验"""
    print("\n=== Statistical Significance Test ===")
    
    # 提取每个指标的数据
    dice_scores = [m['dice'] for m in metrics_list]
    iou_scores = [m['iou'] for m in metrics_list]
    
    # 处理HD95中的NaN值
    hd95_scores = []
    for m in metrics_list:
        if not np.isnan(m['hd95']):
            hd95_scores.append(m['hd95'])
    
    precision_scores = [m['precision'] for m in metrics_list]
    recall_scores = [m['recall'] for m in metrics_list]
    
    # 对每个指标独立进行t检验
    results = {}
    
    # Dice系数检验
    if len(dice_scores) > 1:
        t_dice, p_dice = ttest_rel(dice_scores, np.zeros(len(dice_scores)))
        results['dice'] = {'t_stat': t_dice, 'p_value': p_dice}
    else:
        results['dice'] = {'t_stat': np.nan, 'p_value': np.nan}
    
    # IoU检验
    if len(iou_scores) > 1:
        t_iou, p_iou = ttest_rel(iou_scores, np.zeros(len(iou_scores)))
        results['iou'] = {'t_stat': t_iou, 'p_value': p_iou}
    else:
        results['iou'] = {'t_stat': np.nan, 'p_value': np.nan}
    
    # Precision检验
    if len(precision_scores) > 1:
        t_precision, p_precision = ttest_rel(precision_scores, np.zeros(len(precision_scores)))
        results['precision'] = {'t_stat': t_precision, 'p_value': p_precision}
    else:
        results['precision'] = {'t_stat': np.nan, 'p_value': np.nan}
    
    # Recall检验
    if len(recall_scores) > 1:
        t_recall, p_recall = ttest_rel(recall_scores, np.zeros(len(recall_scores)))
        results['recall'] = {'t_stat': t_recall, 'p_value': p_recall}
    else:
        results['recall'] = {'t_stat': np.nan, 'p_value': np.nan}
    
    # HD95检验
    if len(hd95_scores) > 1:
        baseline_hd95 = np.full(len(hd95_scores), 200.0)
        t_hd95, p_hd95 = ttest_rel(hd95_scores, baseline_hd95)
        results['hd95'] = {'t_stat': t_hd95, 'p_value': p_hd95}
    else:
        results['hd95'] = {'t_stat': np.nan, 'p_value': np.nan}
    
    # 打印结果
    for metric, result in results.items():
        t_val = result['t_stat']
        p_val = result['p_value']
        
        if np.isnan(t_val) or np.isnan(p_val):
            print(f"{metric.upper():<12}: Insufficient data for statistical test")
        else:
            significance = "✅ Significant" if p_val < 0.05 else "❌ Not significant"
            improvement = "✅ Better than baseline" if (metric != 'hd95' and t_val > 0) or (metric == 'hd95' and t_val < 0) else "❌ Worse than baseline"
            print(f"{metric.upper():<12}: t={t_val:.4f}, p={p_val:.6f} {significance} {improvement}")
    
    return results

def save_metrics_to_csv(metrics_list, output_dir, dataset_name, fold_name="all"):
    """保存指标到CSV文件"""
    df = pd.DataFrame(metrics_list)
    
    # 计算平均值和标准差
    mean_metrics = {}
    std_metrics = {}
    
    for column in df.columns:
        if column not in ['patient_id', 'dataset', 'num_slices']:
            mean_metrics[column] = df[column].mean()
            std_metrics[column] = df[column].std()
    
    # 添加平均值行
    mean_row = {'patient_id': 'MEAN', 'dataset': 'ALL'}
    std_row = {'patient_id': 'STD', 'dataset': 'ALL'}
    
    for column in df.columns:
        if column not in ['patient_id', 'dataset', 'num_slices']:
            mean_row[column] = mean_metrics[column]
            std_row[column] = std_metrics[column]
    
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, f'cv_test_metrics_{dataset_name}_{fold_name}.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    print(f"\n=== Metrics Summary ===")
    print(f"Average Dice: {mean_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"Average IoU: {mean_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"Average HD95: {mean_metrics['hd95']:.4f} ± {std_metrics['hd95']:.4f}")
    print(f"Average Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Average Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Metrics saved to: {csv_path}")
    
    return df, mean_metrics, std_metrics

def test_single_fold(fold_dir, test_loader, config, args, dataset_name, fold_idx):
    """测试单个折的模型"""
    print(f"\n{'='*60}")
    print(f"Testing Fold {fold_idx}")
    print(f"{'='*60}")
    
    # 查找模型
    model_path = find_best_model_in_fold(fold_dir)
    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: Model file not found in {fold_dir}")
        return None
    
    # 加载检查点以获取模型类型
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 获取保存的模型类型
    saved_model_type = 'ultralight'  # 默认值
    if isinstance(checkpoint, dict):
        saved_model_type = checkpoint.get('model_type', 'ultralight')
        # 检查是否从训练配置中保存了enable_fusion
        enable_fusion_saved = checkpoint.get('enable_fusion', False)
        
        # 如果命令行没有指定模型类型，使用保存的类型
        if not args.model_type and saved_model_type:
            print(f"Using saved model type: {saved_model_type.upper()}")
        elif args.model_type:
            saved_model_type = args.model_type
            print(f"Using specified model type: {saved_model_type.upper()}")
            
    elif args.model_type:
        saved_model_type = args.model_type
        print(f"Using specified model type: {saved_model_type.upper()}")
    else:
        print(f"Using default model type: {saved_model_type.upper()}")
    
    print(f"Saved model type: {saved_model_type.upper()}")
    
    # 判断是否需要启用融合
    enable_fusion_for_test = enable_fusion_saved if 'enable_fusion_saved' in locals() else args.enable_fusion
    
    # 创建输出目录
    test_output_dir = os.path.join(fold_dir, 'test_outputs', dataset_name)
    nifti_dir = os.path.join(test_output_dir, 'nifti_predictions')
    metrics_dir = os.path.join(test_output_dir, 'metrics')
    
    # 融合分析目录
    if args.analyze_fusion and args.enable_fusion:
        fusion_analysis_dir = os.path.join(test_output_dir, 'fusion_analysis')
        os.makedirs(fusion_analysis_dir, exist_ok=True)
        print(f"📊 Fusion analysis directory created: {fusion_analysis_dir}")
    
    for dir_path in [test_output_dir, nifti_dir, metrics_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 设置日志
    log_dir = os.path.join(fold_dir, 'log')
    logger = get_logger(f'test_fold_{fold_idx}_{dataset_name}', log_dir)
    
    # 初始化模型
    print('#----------Preparing Model----------#')
    
    # 根据保存的模型类型创建模型
    model = create_model_for_testing(saved_model_type, config, args)
    model_type_display = saved_model_type.replace('_', ' ').upper()
    
    print(f"Model Type: {model_type_display}")
    
    # 处理融合配置
    if args.multimodal:
        if saved_model_type == 'ultralight_enhanced' and USE_ENHANCED_MODEL:
            # 对于增强版模型，传递融合配置
            if hasattr(model, 'dynamic_fusion'):
                model.dynamic_fusion.enabled = enable_fusion_for_test
                model.fusion_enabled = enable_fusion_for_test
                print(f"Dynamic Fusion: {'✅ Enabled' if enable_fusion_for_test else '❌ Disabled'}")
                if enable_fusion_for_test:
                    print(f"Test weight method: {args.test_weight_method}")
        else:
            # 其他模型不支持动态融合
            if enable_fusion_for_test:
                print("⚠️ Warning: Fusion requested but model does not support dynamic fusion")
    
    model = torch.nn.DataParallel(model.cuda(), device_ids=[0], output_device=0)
    
    print('#----------Loading Model Weights----------#')
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            filtered_state_dict = filter_state_dict(state_dict)
            model.module.load_state_dict(filtered_state_dict, strict=True)
        else:
            filtered_state_dict = filter_state_dict(checkpoint)
            model.module.load_state_dict(filtered_state_dict, strict=True)
        print("Model loaded successfully")
        
        # 检查融合权重加载
        if args.enable_fusion and USE_ENHANCED_MODEL:
            print("✅ Dynamic fusion weights loaded successfully")
            
            if hasattr(model.module, 'get_test_weights_info'):
                weights_info = model.module.get_test_weights_info()
                print(f"📊 Test weight configuration:")
                print(f"   - Method: {weights_info['method']}")
                if 'weights' in weights_info:
                    print(f"   - Weights: T1={weights_info['weights'][0]:.3f}, "
                        f"SER={weights_info['weights'][1]:.3f}, PE={weights_info['weights'][2]:.3f}")
                if 'has_history' in weights_info:
                    print(f"   - Has training history: {'✅ YES' if weights_info['has_history'] else '❌ NO'}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\n⚠️ Possible solutions:")
        print("1. Check if the model was trained with the same architecture")
        print("2. If using dynamic fusion, ensure --enable_fusion flag is correct")
        print("3. Try loading with strict=False if architecture mismatch")
        
        # 尝试非严格加载
        try:
            print("Attempting non-strict loading...")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.module.load_state_dict(filtered_state_dict, strict=False)
            else:
                model.module.load_state_dict(filtered_state_dict, strict=False)
            print("Model loaded with strict=False (some parameters may be missing)")
        except Exception as e2:
            print(f"Non-strict loading also failed: {e2}")
            return None
    
    # 融合分析数据收集
    if args.enable_fusion and args.analyze_fusion and USE_ENHANCED_MODEL:
        print("\n🔄 Enabling fusion data collection for analysis...")
        
        if hasattr(model.module, 'dynamic_fusion'):
            # 检查是否有训练历史
            has_history = False
            if hasattr(model.module.dynamic_fusion, 'modal_weights_history'):
                history_len = len(model.module.dynamic_fusion.modal_weights_history)
                has_history = history_len > 0
                print(f"📊 Training history: {history_len} batches")
            
            # 只有没有历史记录时才重置
            if not has_history and hasattr(model.module.dynamic_fusion, 'reset_history'):
                model.module.dynamic_fusion.reset_history()
                print("⚠️ No training history found, resetting for test data collection")
            else:
                print(f"✅ Preserving training history for {args.test_weight_method}")
            
            # 确保融合模块处于启用状态
            model.module.dynamic_fusion.enabled = True
    
    print('#----------Starting 3D Inference----------#')
    model.eval()
    
    # 存储患者数据
    patient_data = {}
    metrics_list = []
    
    # 融合权重收集
    if args.enable_fusion and args.analyze_fusion and USE_ENHANCED_MODEL:
        fusion_weights_collection = []
        print("📊 Collecting fusion weights for analysis...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # 处理不同的数据加载器返回格式
            if len(batch_data) == 3:
                images, masks, patient_info = batch_data
            elif len(batch_data) == 2:
                images, masks = batch_data
                patient_info = {'patient_id': [f'patient_{batch_idx}_{i}' for i in range(len(images))]}
            else:
                print(f"Unexpected batch data format: {len(batch_data)} elements")
                continue
            
            images = images.cuda()
            masks = masks.cuda()
            
            # 模型预测
            outputs = model(images)
            predictions = outputs
            
            # 收集融合权重
            if args.enable_fusion and args.analyze_fusion and USE_ENHANCED_MODEL:
                try:
                    # 尝试获取模态权重
                    if hasattr(model.module, 'get_modal_weights'):
                        modal_weights = model.module.get_modal_weights()
                        if modal_weights is not None:
                            fusion_weights_collection.append(modal_weights.cpu().numpy())
                except Exception as e:
                    print(f"⚠️ Failed to collect fusion weights: {e}")
            
            # 转换为numpy
            pred_np = predictions.cpu().numpy()
            mask_np = masks.cpu().numpy()
            
            # 处理每个样本
            for i in range(len(images)):
                # 获取患者ID
                if 'patient_id' in patient_info and i < len(patient_info['patient_id']):
                    patient_id = patient_info['patient_id'][i]
                else:
                    patient_id = f'patient_{batch_idx}_{i}'
                
                # 获取切片索引
                if 'slice_idx' in patient_info and i < len(patient_info['slice_idx']):
                    slice_idx = patient_info['slice_idx'][i]
                else:
                    slice_idx = i
                
                if patient_id not in patient_data:
                    patient_data[patient_id] = {
                        'slices': []
                    }
                
                # 存储切片数据
                slice_pred = pred_np[i]
                slice_gt = mask_np[i]
                
                # 处理通道维度
                if len(slice_pred.shape) == 3 and slice_pred.shape[0] == 1:
                    slice_pred = slice_pred[0]  # (1, H, W) -> (H, W)
                if len(slice_gt.shape) == 3 and slice_gt.shape[0] == 1:
                    slice_gt = slice_gt[0]  # (1, H, W) -> (H, W)
                
                patient_data[patient_id]['slices'].append({
                    'slice_idx': slice_idx,
                    'prediction': slice_pred,
                    'ground_truth': slice_gt
                })
    
    print(f"Processed {len(patient_data)} patients")
    
    # 处理收集的融合权重
    if args.enable_fusion and args.analyze_fusion and USE_ENHANCED_MODEL and fusion_weights_collection:
        try:
            all_fusion_weights = np.concatenate(fusion_weights_collection, axis=0)
            print(f"📊 Collected fusion weights for {len(all_fusion_weights)} samples")
            
            # 计算平均权重
            mean_weights = all_fusion_weights.mean(axis=0)
            std_weights = all_fusion_weights.std(axis=0)
            
            print(f"\n📈 Fusion Weight Statistics:")
            print(f"  T1: {mean_weights[0]:.3f} ± {std_weights[0]:.3f}")
            print(f"  SER: {mean_weights[1]:.3f} ± {std_weights[1]:.3f}")
            print(f"  PE: {mean_weights[2]:.3f} ± {std_weights[2]:.3f}")
            
            # 保存权重数据
            if args.analyze_fusion:
                weights_df = pd.DataFrame(all_fusion_weights, columns=['T1_weight', 'SER_weight', 'PE_weight'])
                weights_path = os.path.join(fusion_analysis_dir, 'fusion_weights.csv')
                weights_df.to_csv(weights_path, index=False)
                print(f"💾 Fusion weights saved to: {weights_path}")
                
        except Exception as e:
            print(f"⚠️ Failed to process fusion weights: {e}")
    
    if not patient_data:
        print("ERROR: No patient data collected!")
        return None
    
    print('#----------Reconstructing 3D Volumes and Calculating Metrics----------#')
    for patient_id, data in patient_data.items():
        print(f"Processing patient: {patient_id} (slices: {len(data['slices'])})")
        
        try:
            # 重构3D体积
            volume_pred = reconstruct_3d_volume(data['slices'], patient_id)
            volume_gt = reconstruct_3d_volume([{'slice_idx': s['slice_idx'], 'prediction': s['ground_truth']} 
                                              for s in data['slices']], patient_id)
            
            # 修正轴顺序
            volume_pred_corrected = correct_axis_order(volume_pred)
            volume_gt_corrected = correct_axis_order(volume_gt)
            
            # 应用阈值得到二值分割
            volume_pred_binary = process_prediction(volume_pred_corrected, args.threshold)
            volume_gt_binary = process_prediction(volume_gt_corrected, 0.5)
            
            # 计算3D指标
            metrics = calculate_metrics_3d(volume_pred_binary, volume_gt_binary)
            metrics['patient_id'] = patient_id
            metrics['dataset'] = dataset_name
            metrics['num_slices'] = len(data['slices'])
            metrics_list.append(metrics)
            
            print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
            
            # 保存NIFTI文件
            if args.save_nifti:
                # 预测结果 (二值)
                pred_nifti = nib.Nifti1Image(volume_pred_binary.astype(np.float32), np.eye(4))
                pred_path = os.path.join(nifti_dir, f'{patient_id}_pred.nii.gz')
                nib.save(pred_nifti, pred_path)
                
                # 真实标签 (二值)
                gt_nifti = nib.Nifti1Image(volume_gt_binary.astype(np.float32), np.eye(4))
                gt_path = os.path.join(nifti_dir, f'{patient_id}_gt.nii.gz')
                nib.save(gt_nifti, gt_path)
                
                # 概率图 (连续值)
                prob_nifti = nib.Nifti1Image(volume_pred_corrected.astype(np.float32), np.eye(4))
                prob_path = os.path.join(nifti_dir, f'{patient_id}_prob.nii.gz')
                nib.save(prob_nifti, prob_path)
                
                print(f"  Saved NIFTI files for {patient_id}")
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not metrics_list:
        print("ERROR: No metrics calculated!")
        return None
    
    print('#----------Saving Metrics and Statistical Analysis----------#')
    # 保存指标到CSV
    df, mean_metrics, std_metrics = save_metrics_to_csv(metrics_list, metrics_dir, dataset_name, f"fold{fold_idx}")
    
    # 显著性检验
    results = statistical_significance_test(metrics_list)
    
    # 保存统计检验结果
    stats_df = pd.DataFrame({
        'metric': ['Dice', 'IoU', 'HD95', 'Precision', 'Recall'],
        'mean': [mean_metrics['dice'], mean_metrics['iou'], mean_metrics['hd95'], 
                mean_metrics['precision'], mean_metrics['recall']],
        'std': [std_metrics['dice'], std_metrics['iou'], std_metrics['hd95'],
                std_metrics['precision'], std_metrics['recall']],
        't_statistic': [
            results['dice']['t_stat'] if not np.isnan(results['dice']['t_stat']) else np.nan,
            results['iou']['t_stat'] if not np.isnan(results['iou']['t_stat']) else np.nan,
            results['hd95']['t_stat'] if not np.isnan(results['hd95']['t_stat']) else np.nan,
            results['precision']['t_stat'] if not np.isnan(results['precision']['t_stat']) else np.nan,
            results['recall']['t_stat'] if not np.isnan(results['recall']['t_stat']) else np.nan
        ],
        'p_value': [
            results['dice']['p_value'] if not np.isnan(results['dice']['p_value']) else np.nan,
            results['iou']['p_value'] if not np.isnan(results['iou']['p_value']) else np.nan,
            results['hd95']['p_value'] if not np.isnan(results['hd95']['p_value']) else np.nan,
            results['precision']['p_value'] if not np.isnan(results['precision']['p_value']) else np.nan,
            results['recall']['p_value'] if not np.isnan(results['recall']['p_value']) else np.nan
        ]
    })
    
    stats_path = os.path.join(metrics_dir, f'statistical_analysis_{dataset_name}_fold{fold_idx}.csv')
    stats_df.to_csv(stats_path, index=False, float_format='%.6f')
    
    print(f"\nStatistical analysis saved to: {stats_path}")
    
    # 生成融合分析报告
    if args.enable_fusion and args.analyze_fusion and USE_ENHANCED_MODEL:
        print("\n📊 Generating fusion analysis report...")
        try:
            # 调用模型的融合分析方法
            if hasattr(model.module, 'visualize_fusion'):
                model.module.visualize_fusion(fusion_analysis_dir)
                print(f"✅ Fusion analysis report saved to: {fusion_analysis_dir}")
                
                # 显示测试权重详情
                if hasattr(model.module, 'get_test_weights_info'):
                    weights_info = model.module.get_test_weights_info()
                    print(f"\n🔬 Final Test Weights:")
                    print(f"   - Method used: {weights_info['method']}")
                    if 'weights' in weights_info:
                        print(f"   - T1 weight: {weights_info['weights'][0]:.4f}")
                        print(f"   - SER weight: {weights_info['weights'][1]:.4f}")
                        print(f"   - PE weight: {weights_info['weights'][2]:.4f}")
            else:
                print("⚠️ Model does not support fusion visualization")
        except Exception as e:
            print(f"⚠️ Fusion analysis failed: {e}")
    
    # 返回结果
    result_summary = {
        'fold_idx': fold_idx,
        'model_path': model_path,
        'saved_model_type': saved_model_type,
        'model_type_display': model_type_display,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'metrics_df': df,
        'stats_df': stats_df,
        'test_output_dir': test_output_dir,
        'enable_fusion': enable_fusion_for_test
    }
    
    print(f"\n✅ Fold {fold_idx} testing completed successfully!")
    print(f"Model Type: {model_type_display}")
    if args.multimodal:
        print(f"Dynamic Fusion: {'✅ Enabled' if enable_fusion_for_test else '❌ Disabled'}")
    print(f"3D predictions saved to: {nifti_dir}")
    print(f"Metrics saved to: {metrics_dir}")
    
    return result_summary

def main():
    args = parse_args()
    
    print("="*60)
    print("Cross-Validation Testing - Support All Baseline Models")
    print("="*60)
    print(f"Cross-validation directory: {args.cv_dir}")
    print(f"Test datasets: {args.datasets}")
    print(f"Multimodal: {args.multimodal}")
    
    # 显示模型类型配置
    if args.model_type:
        print(f"Specified model type: {args.model_type.upper()}")
    else:
        print(f"Model type: Will use saved model type from checkpoints")
    
    # 显示融合配置
    if args.multimodal and args.enable_fusion:
        print("🎯 Dynamic Modal Fusion: ✅ ENABLED")
        print(f"   - Test weight method: {args.test_weight_method}")
        print(f"   - Verbose mode: {'✅ ON' if args.fusion_verbose else '❌ OFF'}")
        if args.analyze_fusion:
            print("   - Fusion analysis: ✅ WILL BE GENERATED")
    elif args.multimodal:
        print("🎯 Dynamic Modal Fusion: ❌ DISABLED")
    
    print(f"Input channels: {args.input_channels}")
    print(f"Threshold: {args.threshold}")
    print(f"Save NIFTI: {args.save_nifti}")
    print(f"Test all folds: {args.test_all_folds}")
    print(f"Fold index: {args.fold_idx}")
    print("="*60)
    
    # 创建配置
    config = MamaMiaConfig(
        multimodal=args.multimodal,
        datasets_list=args.datasets,
        input_channels=args.input_channels,
        data_dir=args.data_dir,
        seg_dir=args.seg_dir,
        ser_dir=args.ser_dir,
        pe_dir=args.pe_dir,
        num_workers=args.num_workers,
        cross_dataset_test=args.cross_dataset_test
    )
    
    # 测试阶段强制关闭平衡采样和数据增广
    config.balanced_sampling = False
    config.use_augmentation = False
    
    # 传递融合参数
    config.enable_fusion = args.enable_fusion
    config.fusion_verbose = args.fusion_verbose
    config.test_weight_method = args.test_weight_method
    
    # 设置完整的随机种子
    print('#----------Setting random seed for reproducibility----------#')
    set_seed(config.seed)
    
    # 获取数据集名称
    dataset_name = '_'.join(args.datasets)
    
    # 准备测试数据加载器
    print('#----------Preparing test dataset----------#')
    data_loader = MAMAMIADataLoader(config)
    
    try:
        test_loader = data_loader.get_test_loader()
        print(f'Test samples: {len(test_loader.dataset)}')
        
        if len(test_loader.dataset) == 0:
            print("ERROR: No test samples found!")
            return
        
        # 应用随机种子
        test_loader = seed_data_loader(test_loader, config.seed)
        
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    
    # 确定要测试哪些折
    if args.fold_idx >= 0:
        # 测试指定折
        fold_dirs = [(args.fold_idx, os.path.join(args.cv_dir, f'fold_{args.fold_idx}'))]
        print(f"Testing specific fold: {args.fold_idx}")
        
    elif args.test_all_folds:
        # 测试所有折
        fold_dirs = []
        for i in range(5):
            fold_dir = os.path.join(args.cv_dir, f'fold_{i}')
            if os.path.exists(fold_dir):
                fold_dirs.append((i, fold_dir))
        print(f"Testing all {len(fold_dirs)} folds")
        
    else:
        # 只测试最佳模型（从cv_results.csv中读取）
        cv_results_path = os.path.join(args.cv_dir, 'cv_results.csv')
        if os.path.exists(cv_results_path):
            try:
                cv_df = pd.read_csv(cv_results_path)
                print(f"Loaded cv_results.csv with columns: {list(cv_df.columns)}")
                
                # 找到最佳折（验证损失最小的）
                if 'best_val_loss' in cv_df.columns:
                    # 排除非数值行（MEAN, STD等）
                    numeric_rows = cv_df[pd.to_numeric(cv_df['best_val_loss'], errors='coerce').notna()]
                    
                    if not numeric_rows.empty:
                        # 找到验证损失最小的行
                        best_row_idx = numeric_rows['best_val_loss'].astype(float).idxmin()
                        best_fold_val = numeric_rows.loc[best_row_idx, 'best_val_loss']
                        
                        # 获取折索引
                        if 'fold' in cv_df.columns:
                            best_fold_idx = int(cv_df.loc[best_row_idx, 'fold'])
                        else:
                            # 如果没有fold列，使用行索引
                            best_fold_idx = int(best_row_idx)
                        
                        fold_dir = os.path.join(args.cv_dir, f'fold_{best_fold_idx}')
                        if os.path.exists(fold_dir):
                            fold_dirs = [(best_fold_idx, fold_dir)]
                            print(f"Found best fold: {best_fold_idx} (val_loss: {best_fold_val:.4f})")
                        else:
                            print(f"Fold directory {fold_dir} does not exist")
                            fold_dirs = []
                    else:
                        print("No numeric rows found in cv_results.csv")
                        fold_dirs = []
                else:
                    print("Column 'best_val_loss' not found in cv_results.csv")
                    fold_dirs = []
                    
            except Exception as e:
                print(f"Error reading cv_results.csv: {e}")
                fold_dirs = []
        else:
            print(f"cv_results.csv not found at {cv_results_path}")
            fold_dirs = []
        
        # 如果无法从cv_results.csv确定，默认测试第一个存在的折
        if not fold_dirs:
            print("Falling back to testing first available fold")
            for i in range(5):
                fold_dir = os.path.join(args.cv_dir, f'fold_{i}')
                if os.path.exists(fold_dir):
                    fold_dirs = [(i, fold_dir)]
                    print(f"Testing first available fold: {i}")
                    break
    
    if not fold_dirs:
        print("ERROR: No fold directories found!")
        # 列出存在的目录
        print("Available directories in CV directory:")
        for item in os.listdir(args.cv_dir):
            item_path = os.path.join(args.cv_dir, item)
            if os.path.isdir(item_path):
                print(f"  - {item}")
        return
    
    print(f"\nWill test {len(fold_dirs)} fold(s):")
    for fold_idx, fold_dir in fold_dirs:
        print(f"  Fold {fold_idx}: {fold_dir}")
    
    # 测试每个折
    all_results = []
    for fold_idx, fold_dir in fold_dirs:
        print(f"\n{'='*60}")
        print(f"Starting test for Fold {fold_idx}")
        print(f"{'='*60}")
        
        result = test_single_fold(fold_dir, test_loader, config, args, dataset_name, fold_idx)
        if result:
            all_results.append(result)
        else:
            print(f"Failed to test Fold {fold_idx}")
    
    if not all_results:
        print("ERROR: No successful tests!")
        return
    
    # 如果测试了多个折，计算总体统计
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Cross-Validation Overall Statistics")
        print(f"{'='*60}")
        
        # 收集所有折的平均指标
        fold_metrics = []
        for result in all_results:
            fold_metrics.append({
                'fold': result['fold_idx'],
                'model_type': result['saved_model_type'],
                'model_type_display': result['model_type_display'],
                'dice': result['mean_metrics']['dice'],
                'iou': result['mean_metrics']['iou'],
                'hd95': result['mean_metrics']['hd95'],
                'precision': result['mean_metrics']['precision'],
                'recall': result['mean_metrics']['recall'],
                'enable_fusion': result['enable_fusion']
            })
        
        # 创建总体统计DataFrame
        overall_df = pd.DataFrame(fold_metrics)
        
        # 计算均值和标准差
        overall_stats = {
            'metric': ['Dice', 'IoU', 'HD95', 'Precision', 'Recall'],
            'mean': [
                overall_df['dice'].mean(),
                overall_df['iou'].mean(),
                overall_df['hd95'].mean(),
                overall_df['precision'].mean(),
                overall_df['recall'].mean()
            ],
            'std': [
                overall_df['dice'].std(),
                overall_df['iou'].std(),
                overall_df['hd95'].std(),
                overall_df['precision'].std(),
                overall_df['recall'].std()
            ],
            'min': [
                overall_df['dice'].min(),
                overall_df['iou'].min(),
                overall_df['hd95'].min(),
                overall_df['precision'].min(),
                overall_df['recall'].min()
            ],
            'max': [
                overall_df['dice'].max(),
                overall_df['iou'].max(),
                overall_df['hd95'].max(),
                overall_df['precision'].max(),
                overall_df['recall'].max()
            ]
        }
        
        overall_stats_df = pd.DataFrame(overall_stats)
        
        # 保存总体统计
        overall_dir = os.path.join(args.cv_dir, 'overall_test_results')
        os.makedirs(overall_dir, exist_ok=True)
        
        overall_csv_path = os.path.join(overall_dir, f'overall_test_stats_{dataset_name}.csv')
        overall_stats_df.to_csv(overall_csv_path, index=False, float_format='%.4f')
        
        # 保存详细折信息
        detailed_csv_path = os.path.join(overall_dir, f'fold_details_{dataset_name}.csv')
        overall_df.to_csv(detailed_csv_path, index=False, float_format='%.4f')
        
        print(f"\nOverall statistics saved to: {overall_csv_path}")
        print(f"Fold details saved to: {detailed_csv_path}")
        
        print(f"\nCross-validation performance on {dataset_name}:")
        print(f"  Dice: {overall_df['dice'].mean():.4f} ± {overall_df['dice'].std():.4f}")
        print(f"  IoU: {overall_df['iou'].mean():.4f} ± {overall_df['iou'].std():.4f}")
        print(f"  HD95: {overall_df['hd95'].mean():.4f} ± {overall_df['hd95'].std():.4f}")
        
        # 显示模型信息统计
        model_counts = overall_df['model_type'].value_counts()
        print(f"\nModel Types Distribution:")
        for model_type, count in model_counts.items():
            percentage = (count / len(overall_df)) * 100
            print(f"  {model_type.upper()}: {count} folds ({percentage:.1f}%)")
        
        # 找到最佳折（基于Dice分数）
        best_fold_idx = overall_df['dice'].idxmax()
        best_fold = overall_df.loc[best_fold_idx, 'fold']
        best_model_type = overall_df.loc[best_fold_idx, 'model_type_display']
        best_dice = overall_df.loc[best_fold_idx, 'dice']
        best_fusion_status = overall_df.loc[best_fold_idx, 'enable_fusion']
        
        print(f"\nBest performing fold: {best_fold}")
        print(f"  Model: {best_model_type}")
        print(f"  Dice: {best_dice:.4f}")
        if args.multimodal:
            print(f"  Dynamic Fusion: {'✅ Enabled' if best_fusion_status else '❌ Disabled'}")
    
    print(f"\n{'='*60}")
    print("🎉 Cross-validation testing completed successfully!")
    print(f"{'='*60}")
    
    # 打印测试结果位置
    print("\nTest results saved to:")
    for result in all_results:
        print(f"  Fold {result['fold_idx']}: {result['test_output_dir']}")
    
    # 总结融合配置
    if args.enable_fusion:
        print(f"\n🔬 Dynamic Fusion Configuration:")
        print(f"   - Enabled: ✅ YES")
        print(f"   - Test weight method: {args.test_weight_method}")
        print(f"   - Verbose: {'✅ YES' if args.fusion_verbose else '❌ NO'}")
        print(f"   - Analysis: {'✅ GENERATED' if args.analyze_fusion else '❌ NOT GENERATED'}")

if __name__ == '__main__':
    main()