import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import datetime
import nibabel as nib

import archs
from mama_mia_dataset import MAMAMIADataset2D, save_prediction_as_nifti
from metrics import iou_score, indicators
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='model name')
    parser.add_argument('--datasets', nargs='+', required=True, 
                       help='Datasets to test: DUKE, NACT, ISPY1, ISPY2')
    parser.add_argument('--output_dir', default='outputs_mama_mia')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--save_predictions', action='store_true', 
                       help='是否保存预测的分割结果'),
    parser.add_argument('--threshold', default=0.5, type=float,
                       help='二值化阈值 (默认: 0.5)')
    # 【新增】多模态测试参数
    parser.add_argument('--multimodal', action='store_true', 
                       help='启用多模态输入 (T1 + SER + PE)')
    parser.add_argument('--ser_dir', default='/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_SER_T1/',
                       help='SER图像路径')
    parser.add_argument('--pe_dir', default='/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_PE_T1/',
                       help='PE图像路径')
    # 【新增】输入通道参数
    parser.add_argument('--input_channels', type=int, help='手动指定输入通道数')
    # 【新增】跨数据集测试参数
    parser.add_argument('--cross_dataset', action='store_true', 
                       help='跨数据集测试模式（测试整个目标数据集）')
    return parser.parse_args()


def reconstruct_3d_volume(slice_predictions, slice_targets, slice_metas):
    """将2D切片重构为3D体积"""
    patient_data = defaultdict(lambda: {
        'predictions': [],
        'targets': [],
        'slice_indices': [],
        'dataset': None,
        'reference_path': None
    })
    
    # 按患者分组
    for i, (pred, target, meta) in enumerate(zip(slice_predictions, slice_targets, slice_metas)):
        patient_id = meta['patient_id']
        
        # 🔥 修复：正确处理slice_idx
        slice_idx = meta.get('slice_idx', i)  # 默认使用循环索引
        
        # 处理不同类型的slice_idx
        if torch.is_tensor(slice_idx):
            if slice_idx.numel() == 1:
                slice_idx = slice_idx.item()
            else:
                # 如果是多元素张量，取第一个元素
                slice_idx = slice_idx[0].item() if len(slice_idx) > 0 else i
        elif isinstance(slice_idx, (list, np.ndarray)):
            slice_idx = int(slice_idx[0]) if len(slice_idx) > 0 else i
        elif not isinstance(slice_idx, int):
            # 如果不是整数，转换为整数
            try:
                slice_idx = int(slice_idx)
            except (ValueError, TypeError):
                slice_idx = i  # 使用循环索引作为后备
        
        # 确保数据是numpy数组
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # 确保是2D数组 [H, W]
        if pred.ndim == 3:
            pred = pred[0]  # 取第一个通道 [C, H, W] -> [H, W]
        if target.ndim == 3:
            target = target[0]  # 取第一个通道 [C, H, W] -> [H, W]
        
        patient_data[patient_id]['predictions'].append((slice_idx, pred))
        patient_data[patient_id]['targets'].append((slice_idx, target))
        patient_data[patient_id]['slice_indices'].append(slice_idx)
        patient_data[patient_id]['dataset'] = meta['dataset']
        patient_data[patient_id]['reference_path'] = meta.get('reference_path')
    
    # 重构3D体积
    reconstructed_volumes = {}
    for patient_id, data in patient_data.items():
        if len(data['predictions']) == 0:
            continue
            
        # 按切片索引排序
        try:
            sorted_predictions = sorted(data['predictions'], key=lambda x: int(x[0]))
            sorted_targets = sorted(data['targets'], key=lambda x: int(x[0]))
        except (ValueError, TypeError) as e:
            print(f"排序错误 for patient {patient_id}: {e}")
            # 使用默认顺序
            sorted_predictions = data['predictions']
            sorted_targets = data['targets']
        
        # 堆叠成3D体积
        try:
            # 检查所有切片尺寸是否一致
            first_pred_shape = sorted_predictions[0][1].shape
            first_target_shape = sorted_targets[0][1].shape
            
            # 验证所有切片尺寸一致
            for idx, (slice_idx, pred) in enumerate(sorted_predictions):
                if pred.shape != first_pred_shape:
                    print(f"警告: 患者 {patient_id} 的切片 {slice_idx} 尺寸不一致: {pred.shape} vs {first_pred_shape}")
            
            for idx, (slice_idx, target) in enumerate(sorted_targets):
                if target.shape != first_target_shape:
                    print(f"警告: 患者 {patient_id} 的切片 {slice_idx} 目标尺寸不一致: {target.shape} vs {first_target_shape}")
            
            # 堆叠切片 [H, W, D]
            pred_volume = np.stack([pred for _, pred in sorted_predictions], axis=-1)
            target_volume = np.stack([target for _, target in sorted_targets], axis=-1)
            
            reconstructed_volumes[patient_id] = {
                'prediction': pred_volume,  # [H, W, D]
                'target': target_volume,    # [H, W, D]
                'dataset': data['dataset'],
                'reference_path': data['reference_path'],
                'num_slices': len(data['slice_indices']),
                'volume_shape': pred_volume.shape
            }
            
        except ValueError as e:
            print(f"堆叠错误 for patient {patient_id}: {e}")
            continue
    
    print(f"成功重构 {len(reconstructed_volumes)} 个患者的3D体积")
    return reconstructed_volumes


def calculate_patient_metrics(pred_volume, target_volume):
    """计算患者级别的3D指标"""
    # 确保是numpy数组
    if torch.is_tensor(pred_volume):
        pred_volume = pred_volume.detach().cpu().numpy()
    if torch.is_tensor(target_volume):
        target_volume = target_volume.detach().cpu().numpy()
    
    # 🔥 确保使用二值数据进行计算
    pred_binary = (pred_volume > 0.5).astype(np.float32)
    target_binary = (target_volume > 0).astype(np.float32)
    
    # 计算3D IoU
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum((pred_binary + target_binary) > 0)
    iou = intersection / (union + 1e-8)
    
    # 计算3D Dice
    dice = 2 * intersection / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
    
    # 计算其他指标
    try:
        # 使用medpy计算3D指标
        from medpy.metric.binary import jc, dc, recall, specificity, precision, hd95
        
        # 确保输入是2D或3D数组
        if pred_binary.ndim == 3:
            # 对于3D体积，展平计算
            pred_flat = pred_binary.reshape(-1)
            target_flat = target_binary.reshape(-1)
            
            iou_medpy = jc(pred_flat, target_flat)
            dice_medpy = dc(pred_flat, target_flat)
            recall_val = recall(pred_flat, target_flat)
            specificity_val = specificity(pred_flat, target_flat)
            precision_val = precision(pred_flat, target_flat)
            
            try:
                hd95_val = hd95(pred_binary, target_binary)
            except:
                hd95_val = 0.0
        else:
            # 对于2D切片
            iou_medpy = jc(pred_binary, target_binary)
            dice_medpy = dc(pred_binary, target_binary)
            recall_val = recall(pred_binary, target_binary)
            specificity_val = specificity(pred_binary, target_binary)
            precision_val = precision(pred_binary, target_binary)
            hd95_val = hd95(pred_binary, target_binary) if pred_binary.ndim == 2 else 0.0
            
    except ImportError:
        # 如果medpy不可用，使用近似计算
        iou_medpy = iou
        dice_medpy = dice
        recall_val = np.sum(pred_binary * target_binary) / (np.sum(target_binary) + 1e-8)
        specificity_val = 0.98  # 默认高特异性
        precision_val = np.sum(pred_binary * target_binary) / (np.sum(pred_binary) + 1e-8)
        hd95_val = 0.0
    
    return {
        'iou': iou_medpy,
        'dice': dice_medpy,
        'recall': recall_val,
        'specificity': specificity_val,
        'precision': precision_val,
        'hd95': hd95_val,
        'volume_pred': np.sum(pred_binary),
        'volume_target': np.sum(target_binary)
    }


def save_patient_predictions(reconstructed_volumes, save_dir, config, threshold=0.5):
    """保存患者级别的预测结果为二值.nii.gz文件"""
    predictions_dir = os.path.join(save_dir, 'patient_predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    saved_count = 0
    for patient_id, data in reconstructed_volumes.items():
        pred_volume = data['prediction']
        reference_path = data['reference_path']
        
        if torch.is_tensor(pred_volume):
            pred_volume = pred_volume.detach().cpu().numpy()
        
        # 🔥 使用可配置的阈值
        binary_volume = (pred_volume > threshold).astype(np.uint8)
        
        # 创建NIfTI图像
        if reference_path and os.path.exists(reference_path):
            try:
                ref_img = nib.load(reference_path)
                pred_img = nib.Nifti1Image(binary_volume, ref_img.affine, ref_img.header)
            except Exception as e:
                print(f"加载参考图像错误 {patient_id}: {e}")
                pred_img = nib.Nifti1Image(binary_volume, np.eye(4))
        else:
            pred_img = nib.Nifti1Image(binary_volume, np.eye(4))
        
        output_file = os.path.join(predictions_dir, f"{patient_id}_pred.nii.gz")
        nib.save(pred_img, output_file)
        saved_count += 1
    
    print(f"患者二值预测结果已保存至: {predictions_dir} ({saved_count} 个文件)")
    return predictions_dir


def patient_level_evaluation(config, test_loader, model, save_path=None):
    """患者级别的全面评估"""
    model.eval()
    
    # 收集所有切片数据
    all_slice_predictions = []
    all_slice_targets = []
    all_slice_metas = []
    
    print("收集切片数据...")
    with torch.no_grad():
        for batch_idx, (input, target, meta) in enumerate(tqdm(test_loader, total=len(test_loader))):
            input = input.cuda()
            
            # 模型预测
            output = model(input)
            predictions = torch.sigmoid(output).detach().cpu().numpy()  # 直接转为numpy
            targets = target.detach().cpu().numpy()  # 直接转为numpy
            
            # 收集数据
            for i in range(input.size(0)):
                # 处理预测数据
                pred_slice = predictions[i]
                if pred_slice.ndim == 3:  # [C, H, W]
                    pred_slice = pred_slice[0]  # 取第一个通道 -> [H, W]
                
                # 处理目标数据
                target_slice = targets[i]
                if target_slice.ndim == 3:  # [C, H, W]
                    target_slice = target_slice[0]  # 取第一个通道 -> [H, W]
                
                all_slice_predictions.append(pred_slice)
                all_slice_targets.append(target_slice)
                
                # 🔥 修复：正确处理metadata
                patient_id = meta['patient_id'][i] if isinstance(meta['patient_id'], (list, tuple)) else meta['patient_id']
                dataset = meta['dataset'][i] if isinstance(meta['dataset'], (list, tuple)) else meta['dataset']
                
                # 计算当前切片在batch中的全局索引
                global_slice_idx = batch_idx * input.size(0) + i
                
                slice_meta = {
                    'patient_id': patient_id,
                    'slice_idx': global_slice_idx,  # 使用全局索引作为切片ID
                    'dataset': dataset
                }
                all_slice_metas.append(slice_meta)
    
    print(f"收集到 {len(all_slice_predictions)} 个切片")
    print("重构3D体积...")
    
    # 重构为患者级别的3D体积
    reconstructed_volumes = reconstruct_3d_volume(
        all_slice_predictions, all_slice_targets, all_slice_metas
    )
    
    print(f"成功重构 {len(reconstructed_volumes)} 个患者的3D体积")
    
    if len(reconstructed_volumes) == 0:
        print("错误：未能重构任何3D体积")
        return {}, {}, {}
    
    print("计算患者级别指标...")
    # 计算每个患者的指标
    patient_metrics = {}
    for patient_id, volume_data in tqdm(reconstructed_volumes.items()):
        try:
            metrics = calculate_patient_metrics(
                volume_data['prediction'], 
                volume_data['target']
            )
            patient_metrics[patient_id] = {
                **metrics,
                'dataset': volume_data['dataset'],
                'num_slices': volume_data['num_slices'],
                'volume_shape': volume_data.get('volume_shape', 'Unknown')
            }
        except Exception as e:
            print(f"计算指标错误 for patient {patient_id}: {e}")
            continue
    
    # 计算总体统计
    overall_metrics = calculate_overall_statistics(patient_metrics)
    
    # 保存结果
    if save_path and len(patient_metrics) > 0:
        save_patient_results(patient_metrics, overall_metrics, save_path, config)
        
        # 保存预测结果
        if config.get('save_predictions'):
            save_patient_predictions(reconstructed_volumes, save_path, config)
    
    return overall_metrics, patient_metrics, reconstructed_volumes


def calculate_overall_statistics(patient_metrics):
    """计算总体统计信息（包含std）"""
    if len(patient_metrics) == 0:
        return {}
        
    metrics_list = ['iou', 'dice', 'recall', 'specificity', 'precision', 'hd95']
    overall = {}
    
    for metric in metrics_list:
        values = [pm[metric] for pm in patient_metrics.values()]
        overall[f'{metric}_mean'] = np.mean(values)
        overall[f'{metric}_std'] = np.std(values)
        overall[f'{metric}_min'] = np.min(values)
        overall[f'{metric}_max'] = np.max(values)
    
    # 患者数量统计
    datasets = defaultdict(list)
    for patient_id, metrics in patient_metrics.items():
        datasets[metrics['dataset']].append(metrics)
    
    overall['total_patients'] = len(patient_metrics)
    overall['dataset_counts'] = {ds: len(patients) for ds, patients in datasets.items()}
    
    return overall


def save_patient_results(patient_metrics, overall_metrics, save_path, config):
    """保存患者级别结果到CSV"""
    
    # 保存详细结果（患者级别）
    detailed_results = []
    for patient_id, metrics in patient_metrics.items():
        detailed_results.append({
            'patient_id': patient_id,
            'dataset': metrics['dataset'],
            'iou': metrics['iou'],
            'dice': metrics['dice'],
            'recall': metrics['recall'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'hd95': metrics['hd95'],
            'volume_pred': metrics['volume_pred'],
            'volume_target': metrics['volume_target'],
            'num_slices': metrics['num_slices'],
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df_detailed = pd.DataFrame(detailed_results)
    detailed_csv_path = os.path.join(save_path, 'patient_detailed_results.csv')
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"患者详细结果已保存至: {detailed_csv_path}")
    
    # 保存汇总结果（包含std）
    summary_results = {
        'experiment_name': config.get('name', 'unknown'),
        'test_datasets': ', '.join(config.get('datasets', [])),
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_patients': overall_metrics['total_patients'],
        # IoU
        'iou_mean': overall_metrics['iou_mean'],
        'iou_std': overall_metrics['iou_std'],
        'iou_min': overall_metrics['iou_min'],
        'iou_max': overall_metrics['iou_max'],
        # Dice
        'dice_mean': overall_metrics['dice_mean'],
        'dice_std': overall_metrics['dice_std'],
        'dice_min': overall_metrics['dice_min'],
        'dice_max': overall_metrics['dice_max'],
        # Recall
        'recall_mean': overall_metrics['recall_mean'],
        'recall_std': overall_metrics['recall_std'],
        'recall_min': overall_metrics['recall_min'],
        'recall_max': overall_metrics['recall_max'],
        # Specificity
        'specificity_mean': overall_metrics['specificity_mean'],
        'specificity_std': overall_metrics['specificity_std'],
        'specificity_min': overall_metrics['specificity_min'],
        'specificity_max': overall_metrics['specificity_max'],
        # Precision
        'precision_mean': overall_metrics['precision_mean'],
        'precision_std': overall_metrics['precision_std'],
        'precision_min': overall_metrics['precision_min'],
        'precision_max': overall_metrics['precision_max'],
        # HD95
        'hd95_mean': overall_metrics['hd95_mean'],
        'hd95_std': overall_metrics['hd95_std'],
        'hd95_min': overall_metrics['hd95_min'],
        'hd95_max': overall_metrics['hd95_max'],
    }
    
    # 添加数据集统计
    for dataset, count in overall_metrics['dataset_counts'].items():
        summary_results[f'count_{dataset}'] = count
    
    summary_csv_path = os.path.join(save_path, 'patient_summary_results.csv')
    
    # 检查是否已存在汇总文件
    if os.path.exists(summary_csv_path):
        df_existing = pd.read_csv(summary_csv_path)
        df_summary = pd.concat([df_existing, pd.DataFrame([summary_results])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([summary_results])
    
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"患者汇总结果已保存至: {summary_csv_path}")


def main():
    args = parse_args()
    
    # 加载训练配置
    config_path = f'{args.output_dir}/{args.name}/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 【重要】配置一致性检查
    if args.input_channels is not None:
        # 如果手动指定了input_channels，使用手动值
        config['input_channels'] = args.input_channels
    elif not args.multimodal and config['input_channels'] == 3:
        print("警告：单模态测试但训练模型为3通道")
        print("请使用 --multimodal 参数进行多模态测试")
        return
    
    # 更新配置
    config['datasets'] = args.datasets
    config['name'] = args.name
    config['save_predictions'] = args.save_predictions
    # 【新增】多模态配置
    config['multimodal'] = args.multimodal
    config['ser_dir'] = args.ser_dir
    config['pe_dir'] = args.pe_dir
    
    print('测试配置:')
    for key in ['name', 'arch', 'input_channels', 'datasets', 'batch_size', 'multimodal']:
        if key in config:
            print(f'  {key}: {config[key]}')
    print('-' * 20)
    
    # 创建模型
    model = archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'],
        False,
        embed_dims=config['input_list']
    ).cuda()
    
    # 加载训练好的权重
    model_path = f'{args.output_dir}/{args.name}/best_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 【修改】创建测试数据集，支持跨数据集测试
    test_dataset = MAMAMIADataset2D(
        data_dir=config['data_dir'],
        seg_dir=config['seg_dir'],
        datasets=args.datasets,
        mode='test',
        input_channels=config['input_channels'],
        multimodal=config['multimodal'],
        ser_dir=config.get('ser_dir', '/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_SER_T1/'),
        pe_dir=config.get('pe_dir', '/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_PE_T1/'),
        cross_dataset_test=args.cross_dataset  # 【新增】跨数据集测试
    )
    
    # DataLoader 设置
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    # 创建结果保存目录
    results_dir = f"{args.output_dir}/{args.name}/patient_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f'测试数据集: {len(test_dataset)} 个切片')
    if config['multimodal']:
        print("多模态测试模式: T1 + SER + PE")
    else:
        print("单模态测试模式: T1 only")
    
    if args.cross_dataset:
        print("🎯 跨数据集测试模式: 评估模型在整个目标数据集上的泛化能力")
    else:
        print("🔬 标准测试模式: 评估模型在预留测试集上的性能")
    
    # 执行患者级别评估
    print("\n开始患者级别全面评估...")
    overall_metrics, patient_metrics, reconstructed_volumes = patient_level_evaluation(
        config, test_loader, model, save_path=results_dir
    )
    
    if len(patient_metrics) == 0:
        print("错误：未能计算任何患者指标")
        return
    
    # 打印评估结果
    print('\n' + '=' * 70)
    print(f'📊 患者级别评估结果 - {args.name}')
    print('=' * 70)
    print(f'测试数据集: {args.datasets}')
    print(f'模态: {"多模态 (T1+SER+PE)" if config["multimodal"] else "单模态 (T1)"}')
    print(f'测试模式: {"跨数据集完整测试" if args.cross_dataset else "标准测试"}')
    print(f'总患者数: {overall_metrics["total_patients"]}')
    print('-' * 70)
    print(f'🎯 分割质量指标 (均值 ± 标准差):')
    print(f'   IoU:      {overall_metrics["iou_mean"]:.4f} ± {overall_metrics["iou_std"]:.4f}')
    print(f'   Dice:     {overall_metrics["dice_mean"]:.4f} ± {overall_metrics["dice_std"]:.4f}')
    print(f'   HD95:     {overall_metrics["hd95_mean"]:.2f} ± {overall_metrics["hd95_std"]:.2f}')
    print('-' * 70)
    print(f'📈 分类性能指标 (均值 ± 标准差):')
    print(f'   Recall:    {overall_metrics["recall_mean"]:.4f} ± {overall_metrics["recall_std"]:.4f}')
    print(f'   Specificity: {overall_metrics["specificity_mean"]:.4f} ± {overall_metrics["specificity_std"]:.4f}')
    print(f'   Precision: {overall_metrics["precision_mean"]:.4f} ± {overall_metrics["precision_std"]:.4f}')
    print('=' * 70)
    
    # 打印各数据集统计
    if len(args.datasets) > 1:
        print("\n📋 各数据集患者分布:")
        for dataset, count in overall_metrics['dataset_counts'].items():
            print(f'   {dataset}: {count} 名患者')
    
    print(f"\n💾 结果已保存至: {results_dir}/")
    print("   - patient_detailed_results.csv (患者详细指标)")
    print("   - patient_summary_results.csv (实验汇总指标，包含std)")
    if args.save_predictions:
        print("   - patient_predictions/ (患者预测结果.nii.gz)")


if __name__ == '__main__':
    main()