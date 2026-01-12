import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

class MAMAMIAFTVAnalyzer:
    def __init__(self, data_path, output_path):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.pe_threshold = 70
        self.ser_min = 0.9
        self.min_neighbor_count = 5
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 存储分析结果
        self.analysis_results = {}
        
        # 设置matplotlib风格
        plt.style.use('default')
        sns.set_palette("viridis")
    
    def find_patient_phase_files(self, patient_folder):
        """
        智能查找患者的所有相位文件，不区分大小写
        """
        patient_path = self.data_path / patient_folder
        
        # 获取所有nii.gz文件，不区分大小写
        all_files = list(patient_path.glob("*.nii.gz"))
        all_files.extend(list(patient_path.glob("*.nii")))
        
        if not all_files:
            self.logger.warning(f"在 {patient_folder} 中未找到图像文件")
            return []
        
        # 提取相位编号并排序
        phase_files = []
        for file_path in all_files:
            # 使用正则表达式提取数字部分
            match = re.search(r'(\d+)\.nii', file_path.name, re.IGNORECASE)
            if match:
                phase_num = int(match.group(1))
                phase_files.append((phase_num, file_path))
        
        # 按相位编号排序
        phase_files.sort(key=lambda x: x[0])
        
        return [file_path for _, file_path in phase_files]
    
    def identify_dataset(self, patient_id):
        """识别患者所属的数据集"""
        patient_id_lower = patient_id.lower()
        if 'duke' in patient_id_lower:
            return "DUKE"
        elif 'ispy2' in patient_id_lower:
            return "ISPY2"
        elif 'ispy1' in patient_id_lower:
            return "ISPY1"
        elif 'nact' in patient_id_lower:
            return "NACT"
        else:
            return "UNKNOWN"
    
    def intelligent_phase_selection(self, phase_files, patient_id):
        """
        基于数据集特性的智能相位选择
        """
        total_phases = len(phase_files)
        
        if total_phases < 3:
            self.logger.error(f"患者 {patient_id} 的相位数量不足: {total_phases}")
            return None, None, None
        
        # 根据患者ID判断数据集
        dataset = self.identify_dataset(patient_id)
        
        self.logger.info(f"患者 {patient_id} ({dataset}): 发现 {total_phases} 个相位")
        
        # 总是选择第一个作为S0 (pre-contrast)
        s0_file = phase_files[0]
        
        if dataset == "ISPY2":
            # ISPY2: 快速采集，6个相位
            if total_phases == 6:
                s1_file = phase_files[1]  # ~2.4分钟
                s2_file = phase_files[4]  # ~6.9分钟 - 理想washout时间
            else:
                s1_file = phase_files[1]
                s2_file = phase_files[-2] if total_phases > 3 else phase_files[-1]
                
        elif dataset == "DUKE":
            # DUKE: 中等速度采集，3-5个相位
            if total_phases == 3:
                # 修复：只有3个相位时，使用第1、2、3个
                s1_file = phase_files[1]
                s2_file = phase_files[2]
            elif total_phases == 4:
                s1_file = phase_files[1]  # ~4.0分钟
                s2_file = phase_files[2]  # ~6.1分钟
            else:  # 5个相位
                s1_file = phase_files[1]  # ~4.0分钟  
                s2_file = phase_files[3]  # ~8.0分钟
                
        elif dataset in ["ISPY1", "NACT"]:
            # 较慢采集，3-4个相位，间隔较长
            if total_phases == 3:
                s1_file = phase_files[1]
                s2_file = phase_files[2]
            else:  # 4个相位
                s1_file = phase_files[1]
                s2_file = phase_files[2]
        else:
            # 默认策略
            s1_file = phase_files[1]
            if total_phases >= 5:
                s2_file = phase_files[3]
            else:
                s2_file = phase_files[-1]
        
        self.logger.info(f"相位选择: S0={s0_file.name}, S1={s1_file.name}, S2={s2_file.name}")
        return s0_file, s1_file, s2_file
    
    def load_phase_images(self, s0_file, s1_file, s2_file):
        """加载三个关键相位的图像"""
        try:
            s0_img = nib.load(s0_file)
            s1_img = nib.load(s1_file)
            s2_img = nib.load(s2_file)
            
            s0_data = s0_img.get_fdata()
            s1_data = s1_img.get_fdata()
            s2_data = s2_img.get_fdata()
            affine = s0_img.affine
            
            # 获取体素体积
            voxel_dims = s0_img.header.get_zooms()
            voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
            
            return s0_data, s1_data, s2_data, affine, voxel_volume
            
        except Exception as e:
            self.logger.error(f"加载图像时出错: {e}")
            return None, None, None, None, None
    
    def calculate_pe_maps(self, s0_data, s1_data, s2_data):
        """计算百分比增强图"""
        s0_nonzero = np.where(s0_data > 1e-6, s0_data, 1e-6)
        
        pe_early = ((s1_data - s0_data) / s0_nonzero) * 100.0
        pe_late = ((s2_data - s0_data) / s0_nonzero) * 100.0
        
        return pe_early, pe_late
    
    def calculate_ser_map(self, pe_early, pe_late):
        """计算信号增强比率图"""
        pe_late_nonzero = np.where(np.abs(pe_late) > 1e-6, pe_late, 1e-6)
        ser_map = pe_early / pe_late_nonzero
        ser_map = np.nan_to_num(ser_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        return ser_map
    
    def apply_connectivity_filter(self, mask, min_neighbors=5):
        """应用3D连通性过滤"""
        structure = np.ones((3, 3, 3))
        labeled_mask, num_features = ndimage.label(mask, structure=structure)
        
        component_sizes = np.bincount(labeled_mask.ravel())
        size_mask = component_sizes >= min_neighbors
        size_mask[0] = False
        
        filtered_mask = size_mask[labeled_mask]
        return filtered_mask
    
    def generate_ftv_maps(self, s0_data, s1_data, s2_data, voxel_volume):
        """生成全图像的FTV图"""
        # 计算PE和SER
        pe_early, pe_late = self.calculate_pe_maps(s0_data, s1_data, s2_data)
        ser_map = self.calculate_ser_map(pe_early, pe_late)
        
        # 创建组织掩膜（排除背景）
        tissue_threshold = np.percentile(s0_data[s0_data > 0], 30)
        tissue_mask = s0_data > tissue_threshold
        
        # 生成FTV_PE和FTV_SER（全图像）
        ftv_pe_mask = (pe_early >= self.pe_threshold) & tissue_mask
        ftv_ser_mask = (pe_early >= self.pe_threshold) & (ser_map >= self.ser_min) & tissue_mask
        
        # 连通性过滤
        ftv_pe_filtered = self.apply_connectivity_filter(ftv_pe_mask, self.min_neighbor_count)
        ftv_ser_filtered = self.apply_connectivity_filter(ftv_ser_mask, self.min_neighbor_count)
        
        # 计算体积
        ftv_pe_volume = np.sum(ftv_pe_filtered) * voxel_volume
        ftv_ser_volume = np.sum(ftv_ser_filtered) * voxel_volume
        
        results = {
            'ftv_pe_mask': ftv_pe_filtered.astype(np.float32),
            'ftv_ser_mask': ftv_ser_filtered.astype(np.float32),
            'pe_early_map': pe_early.astype(np.float32),
            'ser_map': ser_map.astype(np.float32),
            'ftv_pe_volume': ftv_pe_volume,
            'ftv_ser_volume': ftv_ser_volume,
            'tissue_mask': tissue_mask
        }
        
        return results

    def generate_ftv_with_original_intensity(self, patient_id, output_path, ftv_pe_mask, 
                                           ftv_ser_mask, t1_data, affine, voxel_volume):
        """
        生成带有原始信号强度的FTV图像
        FTV区域保留T1时刻的原始信号值，非FTV区域设为0
        """
        try:
            self.logger.info(f"为 {patient_id} 生成带原始信号值的FTV图像")
            
            # 应用掩膜：FTV区域保留原始T1信号值，其他区域设为0
            ftv_pe_t1 = np.where(ftv_pe_mask, t1_data, 0)
            ftv_ser_t1 = np.where(ftv_ser_mask, t1_data, 0)
            
            # 保存新文件
            new_image_types = {
                'FTV_PE_T1': ftv_pe_t1.astype(np.float32),
                'FTV_SER_T1': ftv_ser_t1.astype(np.float32)
            }
            
            for name, data in new_image_types.items():
                img = nib.Nifti1Image(data, affine)
                nib.save(img, output_path / f"{patient_id}_{name}.nii.gz")
            
            # 计算并记录统计信息
            self._calculate_ftv_intensity_stats(patient_id, output_path, 
                                              ftv_pe_t1, ftv_ser_t1, t1_data, voxel_volume)
            
            self.logger.info(f"成功生成带原始信号值的FTV图像 for {patient_id}")
            
        except Exception as e:
            self.logger.error(f"生成带原始信号值的FTV图像时出错: {e}")

    def _calculate_ftv_intensity_stats(self, patient_id, output_path, 
                                     ftv_pe_t1, ftv_ser_t1, t1_data, voxel_volume):
        """
        计算FTV区域内的信号强度统计
        """
        try:
            # 创建FTV区域的布尔掩膜
            ftv_pe_mask = ftv_pe_t1 > 0
            ftv_ser_mask = ftv_ser_t1 > 0
            
            stats = {
                'patient_id': patient_id,
                'ftv_pe_intensity_stats': {
                    'mean_intensity': float(np.mean(ftv_pe_t1[ftv_pe_mask])) if np.sum(ftv_pe_mask) > 0 else 0,
                    'median_intensity': float(np.median(ftv_pe_t1[ftv_pe_mask])) if np.sum(ftv_pe_mask) > 0 else 0,
                    'std_intensity': float(np.std(ftv_pe_t1[ftv_pe_mask])) if np.sum(ftv_pe_mask) > 0 else 0,
                    'max_intensity': float(np.max(ftv_pe_t1[ftv_pe_mask])) if np.sum(ftv_pe_mask) > 0 else 0,
                    'min_intensity': float(np.min(ftv_pe_t1[ftv_pe_mask])) if np.sum(ftv_pe_mask) > 0 else 0,
                    'voxel_count': int(np.sum(ftv_pe_mask)),
                    'volume_mm3': float(np.sum(ftv_pe_mask) * voxel_volume)
                },
                'ftv_ser_intensity_stats': {
                    'mean_intensity': float(np.mean(ftv_ser_t1[ftv_ser_mask])) if np.sum(ftv_ser_mask) > 0 else 0,
                    'median_intensity': float(np.median(ftv_ser_t1[ftv_ser_mask])) if np.sum(ftv_ser_mask) > 0 else 0,
                    'std_intensity': float(np.std(ftv_ser_t1[ftv_ser_mask])) if np.sum(ftv_ser_mask) > 0 else 0,
                    'max_intensity': float(np.max(ftv_ser_t1[ftv_ser_mask])) if np.sum(ftv_ser_mask) > 0 else 0,
                    'min_intensity': float(np.min(ftv_ser_t1[ftv_ser_mask])) if np.sum(ftv_ser_mask) > 0 else 0,
                    'voxel_count': int(np.sum(ftv_ser_mask)),
                    'volume_mm3': float(np.sum(ftv_ser_mask) * voxel_volume)
                }
            }
            
            # 计算对比度比率（如果两个区域都有数据）
            if np.sum(ftv_pe_mask) > 0 and np.sum(ftv_ser_mask) > 0:
                background_mask = ~ftv_pe_mask & ~ftv_ser_mask
                if np.sum(background_mask) > 0:
                    stats['contrast_ratios'] = {
                        'pe_vs_background': float(np.mean(ftv_pe_t1[ftv_pe_mask]) / np.mean(t1_data[background_mask])),
                        'ser_vs_background': float(np.mean(ftv_ser_t1[ftv_ser_mask]) / np.mean(t1_data[background_mask])),
                        'pe_vs_ser': float(np.mean(ftv_pe_t1[ftv_pe_mask]) / np.mean(ftv_ser_t1[ftv_ser_mask]))
                    }
            
            # 保存统计信息
            with open(output_path / f"{patient_id}_FTV_intensity_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"计算FTV强度统计时出错: {e}")

    def create_comprehensive_visualization(self, s0_data, s1_data, s2_data, 
                                         pe_early, ser_map, ftv_pe, ftv_ser, 
                                         tissue_mask, patient_id, voxel_volume):
        """
        创建独立的可视化图表
        """
        patient_output_path = self.output_path / patient_id
        patient_output_path.mkdir(exist_ok=True)
        
        # 创建6个独立的专业图表
        self._create_individual_plots(s0_data, s1_data, s2_data, pe_early, ser_map, 
                                    ftv_pe, ftv_ser, patient_id, patient_output_path, voxel_volume)
        
        # 创建交互式3D可视化
        self._create_3d_visualization(pe_early, ser_map, ftv_ser, patient_id, patient_output_path)
        
        # 保存量化指标
        self._save_quantitative_metrics(s0_data, s1_data, s2_data, pe_early, ser_map, 
                                      ftv_pe, ftv_ser, patient_id, patient_output_path, voxel_volume)

    def _create_individual_plots(self, s0_data, s1_data, s2_data, pe_early, ser_map, 
                               ftv_pe, ftv_ser, patient_id, output_path, voxel_volume):
        """创建6个独立的专业子图"""
        
        # 1. DCE-MRI序列展示
        self._plot_image_sequence_standalone(s0_data, s1_data, s2_data, patient_id, output_path)
        
        # 2. FTV生成流水线
        self._plot_ftv_pipeline_standalone(s0_data, pe_early, ser_map, ftv_pe, ftv_ser, patient_id, output_path)
        
        # 3. 血管动力学分析
        self._plot_kinetic_analysis_standalone(s0_data, s1_data, s2_data, pe_early, ftv_ser, patient_id, output_path)
        
        # 4. 特征空间分析
        self._plot_feature_space_standalone(pe_early, ser_map, ftv_ser, patient_id, output_path)
        
        # 5. 功能肿瘤体积统计
        self._plot_volume_stats_standalone(ftv_pe, ftv_ser, voxel_volume, patient_id, output_path)
        
        # 6. 肿瘤异质性分析
        self._plot_heterogeneity_standalone(pe_early, ser_map, ftv_ser, patient_id, output_path)

    def _plot_image_sequence_standalone(self, s0_data, s1_data, s2_data, patient_id, output_path):
        """独立的DCE-MRI序列可视化"""
        plt.figure(figsize=(15, 5))
        
        slice_idx = s0_data.shape[2] // 2
        images = [s0_data, s1_data, s2_data]
        titles = ['Pre-contrast (S0)', 'Early Post-contrast (S1)', 'Late Post-contrast (S2)']
        cmaps = ['gray', 'hot', 'hot']
        
        for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
            plt.subplot(1, 3, i + 1)
            plt.imshow(img[:, :, slice_idx], cmap=cmap)
            plt.title(title, fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # 添加colorbar对于增强图像
            if i > 0:  # post-contrast images
                plt.colorbar(fraction=0.046, pad=0.04, label='Signal Intensity')
        
        plt.suptitle(f'DCE-MRI Temporal Sequence - {patient_id}', 
                    fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(output_path / f"{patient_id}_01_sequence.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_ftv_pipeline_standalone(self, s0_data, pe_early, ser_map, ftv_pe, ftv_ser, patient_id, output_path):
        """独立的FTV生成流水线可视化"""
        plt.figure(figsize=(16, 10))
        
        slice_idx = s0_data.shape[2] // 2
        
        steps = [
            ('A. Pre-contrast Baseline', s0_data, 'gray', None),
            ('B. Percent Enhancement (PE)', pe_early, 'hot', 'PE (%)'),
            ('C. Signal Enhancement Ratio (SER)', ser_map, 'coolwarm', 'SER'),
            ('D. Tissue Mask', (s0_data > np.percentile(s0_data[s0_data>0], 30)).astype(float), 'binary', None),
            ('E. FTV_PE\n(PE ≥ 70%)', ftv_pe, 'Reds', None),
            ('F. FTV_SER\n(PE ≥ 70% & SER ≥ 0.9)', ftv_ser, 'Purples', None)
        ]
        
        for i, (title, data, cmap, cbar_label) in enumerate(steps):
            plt.subplot(2, 3, i + 1)
            
            if len(data.shape) == 3:
                im = plt.imshow(data[:, :, slice_idx], cmap=cmap)
            else:
                im = plt.imshow(data, cmap=cmap)
            
            plt.title(title, fontsize=11, fontweight='bold', pad=15)
            plt.axis('off')
            
            # 添加colorbar
            if cbar_label:
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label(cbar_label, fontweight='bold')
        
        plt.suptitle(f'FTV Generation Pipeline - {patient_id}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path / f"{patient_id}_02_ftv_pipeline.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_kinetic_analysis_standalone(self, s0_data, s1_data, s2_data, pe_early, ftv_ser, patient_id, output_path):
        """独立的血管动力学分析"""
        plt.figure(figsize=(10, 8))
        
        # 提取肿瘤区域
        tumor_mask = ftv_ser > 0
        if np.sum(tumor_mask) == 0:
            # 如果没有FTV_SER，使用高PE区域作为替代
            tumor_mask = pe_early > self.pe_threshold
        
        if np.sum(tumor_mask) > 0:
            # 计算信号强度
            time_points = ['Pre-contrast', 'Early Post', 'Late Post']
            tumor_intensity = [
                np.mean(s0_data[tumor_mask]),
                np.mean(s1_data[tumor_mask]),
                np.mean(s2_data[tumor_mask])
            ]
            
            # 计算正常组织（排除肿瘤区域）
            normal_mask = ~tumor_mask
            if np.sum(normal_mask) > 0:
                normal_intensity = [
                    np.mean(s0_data[normal_mask]),
                    np.mean(s1_data[normal_mask]),
                    np.mean(s2_data[normal_mask])
                ]
            else:
                normal_intensity = [0, 0, 0]
            
            # 绘制动力学曲线
            plt.plot(time_points, tumor_intensity, 'o-', linewidth=3, 
                    markersize=8, label='Tumor Region', color='#E74C3C')
            plt.plot(time_points, normal_intensity, 'o--', linewidth=2,
                    markersize=6, label='Normal Tissue', color='#3498DB')
            
            # 计算增强百分比
            enhancement_early = ((tumor_intensity[1] - tumor_intensity[0]) / tumor_intensity[0]) * 100
            enhancement_late = ((tumor_intensity[2] - tumor_intensity[0]) / tumor_intensity[0]) * 100
            
            # 添加标注
            plt.text(0.02, 0.98, f'Early Enhancement: {enhancement_early:.1f}%\n'
                                f'Late Enhancement: {enhancement_late:.1f}%',
                    transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
        
        plt.xlabel('Time Point', fontsize=12, fontweight='bold')
        plt.ylabel('Signal Intensity (AU)', fontsize=12, fontweight='bold')
        plt.title(f'DCE-MRI Kinetic Analysis - {patient_id}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"{patient_id}_03_kinetics.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_feature_space_standalone(self, pe_early, ser_map, ftv_ser, patient_id, output_path):
        """独立的特征空间分析"""
        plt.figure(figsize=(10, 8))
        
        tumor_mask = ftv_ser > 0
        if np.sum(tumor_mask) == 0:
            plt.text(0.5, 0.5, 'No FTV_SER Region Detected', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    fontsize=12, fontweight='bold')
        else:
            # 采样数据点
            pe_values = pe_early[tumor_mask]
            ser_values = ser_map[tumor_mask]
            
            if len(pe_values) > 1000:
                indices = np.random.choice(len(pe_values), 1000, replace=False)
                pe_values = pe_values[indices]
                ser_values = ser_values[indices]
            
            # 创建散点图
            scatter = plt.scatter(pe_values, ser_values, c=ser_values, 
                                cmap='viridis', alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            
            # 添加决策边界
            plt.axvline(x=self.pe_threshold, color='red', linestyle='--', linewidth=2,
                       label=f'PE Threshold = {self.pe_threshold}%')
            plt.axhline(y=self.ser_min, color='blue', linestyle='--', linewidth=2,
                       label=f'SER Threshold = {self.ser_min}')
            
            # 标注区域
            plt.text(30, 1.8, 'Low Enhancement\nRegion', fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            plt.text(150, 0.5, 'Persistent\nEnhancement', fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            plt.text(150, 1.8, 'Rapid Washout\n(Malignant)', fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            
            plt.colorbar(scatter, label='SER Value')
        
        plt.xlabel('Percent Enhancement (PE) (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Signal Enhancement Ratio (SER)', fontsize=12, fontweight='bold')
        plt.title(f'Vascular Kinematic Feature Space - {patient_id}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"{patient_id}_04_feature_space.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_volume_stats_standalone(self, ftv_pe, ftv_ser, voxel_volume, patient_id, output_path):
        """独立的功能肿瘤体积统计"""
        plt.figure(figsize=(8, 8))
        
        volumes = {
            'FTV_PE': np.sum(ftv_pe) * voxel_volume,
            'FTV_SER': np.sum(ftv_ser) * voxel_volume
        }
        
        colors = ['#E74C3C', '#9B59B6']
        bars = plt.bar(volumes.keys(), volumes.values(), color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=2)
        
        # 添加数值标签
        for bar, (name, volume) in zip(bars, volumes.items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(volumes.values())*0.01,
                    f'{volume:.1f} mm³', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
            
            # 在柱子上方添加百分比
            if name == 'FTV_SER' and volumes['FTV_PE'] > 0:
                ratio = volumes['FTV_SER'] / volumes['FTV_PE'] * 100
                plt.text(bar.get_x() + bar.get_width()/2., height + max(volumes.values())*0.05,
                        f'({ratio:.1f}% of FTV_PE)', ha='center', va='bottom', 
                        fontsize=10, style='italic')
        
        plt.ylabel('Volume (mm³)', fontsize=12, fontweight='bold')
        plt.title(f'Functional Tumor Volumes - {patient_id}', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 设置y轴范围，为标签留出空间
        plt.ylim(0, max(volumes.values()) * 1.15)
        
        plt.tight_layout()
        plt.savefig(output_path / f"{patient_id}_05_volumes.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_heterogeneity_standalone(self, pe_early, ser_map, ftv_ser, patient_id, output_path):
        """独立的肿瘤异质性分析"""
        plt.figure(figsize=(10, 8))
        
        tumor_mask = ftv_ser > 0
        if np.sum(tumor_mask) == 0:
            plt.text(0.5, 0.5, 'No Tumor Region for Analysis', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    fontsize=12, fontweight='bold')
        else:
            tumor_pe = pe_early[tumor_mask]
            tumor_ser = ser_map[tumor_mask]
            
            # 计算异质性指标
            metrics = {
                'PE Coefficient\nof Variation': np.std(tumor_pe) / np.mean(tumor_pe),
                'SER Coefficient\nof Variation': np.std(tumor_ser) / np.mean(tumor_ser),
                'High SER Proportion\n(SER > 1.0)': np.sum(tumor_ser > 1.0) / len(tumor_ser) * 100,
                'PE Range': (np.max(tumor_pe) - np.min(tumor_pe)) / np.mean(tumor_pe)
            }
            
            colors = ['#3498DB', '#2ECC71', '#F39C12', '#E67E22']
            bars = plt.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.5)
            
            # 添加数值标签
            for bar, (name, value) in zip(bars, metrics.items()):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(metrics.values())*0.01,
                        f'{value:.3f}' if value < 1 else f'{value:.1f}%', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.ylabel('Value', fontsize=12, fontweight='bold')
        plt.title(f'Tumor Heterogeneity Analysis - {patient_id}', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        if np.sum(tumor_mask) > 0:
            plt.ylim(0, max(metrics.values()) * 1.15)
        
        plt.tight_layout()
        plt.savefig(output_path / f"{patient_id}_06_heterogeneity.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_3d_visualization(self, pe_early, ser_map, ftv_ser, patient_id, output_path):
        """创建3D交互式可视化"""
        try:
            import plotly.graph_objects as go
            
            tumor_mask = ftv_ser > 0
            if np.sum(tumor_mask) < 10:  # 太少的点不创建3D图
                return
                
            indices = np.where(tumor_mask)
            if len(indices[0]) > 2000:
                sample_idx = np.random.choice(len(indices[0]), 2000, replace=False)
                x, y, z = indices[0][sample_idx], indices[1][sample_idx], indices[2][sample_idx]
                pe_values = pe_early[tumor_mask][sample_idx]
                ser_values = ser_map[tumor_mask][sample_idx]
            else:
                x, y, z = indices
                pe_values = pe_early[tumor_mask]
                ser_values = ser_map[tumor_mask]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=ser_values,
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="SER")
                ),
                hovertemplate=(
                    "Position: (%{x}, %{y}, %{z})<br>" +
                    "PE: %{customdata[0]:.1f}%<br>" +
                    "SER: %{marker.color:.2f}<extra></extra>"
                ),
                customdata=np.stack([pe_values], axis=-1)
            )])
            
            fig.update_layout(
                title=f"3D Tumor Distribution - {patient_id}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y', 
                    zaxis_title='Z'
                ),
                width=800,
                height=600
            )
            
            fig.write_html(output_path / f"{patient_id}_3d_distribution.html")
            
        except Exception as e:
            self.logger.warning(f"创建3D可视化失败: {e}")

    def _save_quantitative_metrics(self, s0_data, s1_data, s2_data, pe_early, ser_map, 
                                 ftv_pe, ftv_ser, patient_id, output_path, voxel_volume):
        """保存量化指标"""
        tumor_mask = ftv_ser > 0
        
        metrics = {
            'patient_id': patient_id,
            'ftv_pe_volume_mm3': float(np.sum(ftv_pe) * voxel_volume),
            'ftv_ser_volume_mm3': float(np.sum(ftv_ser) * voxel_volume),
            'ftv_ratio': float(np.sum(ftv_ser) / max(np.sum(ftv_pe), 1e-6)),
            'mean_pe_tumor': float(np.mean(pe_early[tumor_mask])) if np.sum(tumor_mask) > 0 else 0,
            'mean_ser_tumor': float(np.mean(ser_map[tumor_mask])) if np.sum(tumor_mask) > 0 else 0,
            'pe_heterogeneity': float(np.std(pe_early[tumor_mask]) / np.mean(pe_early[tumor_mask])) if np.sum(tumor_mask) > 0 else 0,
            'ser_heterogeneity': float(np.std(ser_map[tumor_mask]) / np.mean(ser_map[tumor_mask])) if np.sum(tumor_mask) > 0 else 0,
            'voxel_volume_mm3': float(voxel_volume),
            'pe_threshold': self.pe_threshold,
            'ser_min_threshold': self.ser_min
        }
        
        with open(output_path / f"{patient_id}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存到总体结果
        self.analysis_results[patient_id] = metrics

    def process_patient(self, patient_folder):
        """处理单个患者"""
        self.logger.info(f"处理患者: {patient_folder}")
        
        try:
            # 1. 查找所有相位文件
            phase_files = self.find_patient_phase_files(patient_folder)
            if len(phase_files) < 3:
                self.logger.warning(f"{patient_folder}: 相位数量不足")
                return False
            
            # 2. 智能选择关键相位
            s0_file, s1_file, s2_file = self.intelligent_phase_selection(phase_files, patient_folder)
            if s0_file is None:
                return False
            
            # 3. 加载图像
            s0_data, s1_data, s2_data, affine, voxel_volume = self.load_phase_images(s0_file, s1_file, s2_file)
            if s0_data is None:
                return False
            
            # 4. 生成FTV图
            results = self.generate_ftv_maps(s0_data, s1_data, s2_data, voxel_volume)
            
            # 5. 保存FTV图像
            patient_output_path = self.output_path / patient_folder
            patient_output_path.mkdir(exist_ok=True)
            
            # 保存主要FTV图像
            image_types = {
                'FTV_PE': results['ftv_pe_mask'],
                'FTV_SER': results['ftv_ser_mask'],
                'PE_early': results['pe_early_map'],
                'SER': results['ser_map']
            }
            
            for name, data in image_types.items():
                img = nib.Nifti1Image(data, affine)
                nib.save(img, patient_output_path / f"{patient_folder}_{name}.nii.gz")
            
            # 使用S1数据（T1时刻，早期增强相位）作为原始信号
            self.generate_ftv_with_original_intensity(
                patient_folder, patient_output_path, 
                results['ftv_pe_mask'], results['ftv_ser_mask'], 
                s1_data, affine, voxel_volume
            )
            
            # 6. 创建综合可视化
            self.create_comprehensive_visualization(
                s0_data, s1_data, s2_data, results['pe_early_map'], 
                results['ser_map'], results['ftv_pe_mask'], 
                results['ftv_ser_mask'], results['tissue_mask'],
                patient_folder, voxel_volume
            )
            
            self.logger.info(f"完成 {patient_folder}: "
                           f"FTV_PE={results['ftv_pe_volume']:.1f}mm³, "
                           f"FTV_SER={results['ftv_ser_volume']:.1f}mm³")
            return True
            
        except Exception as e:
            self.logger.error(f"处理患者 {patient_folder} 时出错: {e}")
            return False

    def process_all_patients(self):
        """处理所有患者"""
        patient_folders = [f.name for f in self.data_path.iterdir() if f.is_dir()]
        
        self.logger.info(f"找到 {len(patient_folders)} 个患者文件夹")
        
        success_count = 0
        for patient_folder in tqdm(patient_folders, desc="处理患者"):
            if self.process_patient(patient_folder):
                success_count += 1
        
        # 保存总体统计
        self.save_overall_statistics(success_count, len(patient_folders))
        
        self.logger.info(f"处理完成: {success_count}/{len(patient_folders)} 个患者成功")
        return success_count

    def save_overall_statistics(self, success_count, total_count):
        """保存总体统计信息"""
        stats = {
            'processing_summary': {
                'total_patients': total_count,
                'successfully_processed': success_count,
                'success_rate': success_count / total_count
            },
            'ftv_parameters': {
                'pe_threshold': self.pe_threshold,
                'ser_min_threshold': self.ser_min,
                'min_neighbor_count': self.min_neighbor_count
            },
            'volume_statistics': {
                'ftv_pe_volumes': [v['ftv_pe_volume_mm3'] for v in self.analysis_results.values()],
                'ftv_ser_volumes': [v['ftv_ser_volume_mm3'] for v in self.analysis_results.values()],
                'ftv_ratios': [v['ftv_ratio'] for v in self.analysis_results.values()]
            }
        }
        
        with open(self.output_path / "overall_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 创建汇总图表
        self.create_summary_plots()

    def create_summary_plots(self):
        """创建数据集汇总图表"""
        if not self.analysis_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 体积分布
        pe_volumes = [v['ftv_pe_volume_mm3'] for v in self.analysis_results.values()]
        ser_volumes = [v['ftv_ser_volume_mm3'] for v in self.analysis_results.values()]
        
        axes[0,0].hist(pe_volumes, bins=50, alpha=0.7, label='FTV_PE', color='#E74C3C')
        axes[0,0].hist(ser_volumes, bins=50, alpha=0.7, label='FTV_SER', color='#9B59B6')
        axes[0,0].set_xlabel('Volume (mm³)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('FTV Volume Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. FTV比率分布
        ratios = [v['ftv_ratio'] for v in self.analysis_results.values()]
        axes[0,1].hist(ratios, bins=50, alpha=0.7, color='#3498DB')
        axes[0,1].axvline(x=np.mean(ratios), color='red', linestyle='--', label=f'Mean: {np.mean(ratios):.3f}')
        axes[0,1].set_xlabel('FTV_SER / FTV_PE Ratio')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('FTV Ratio Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 体积散点图
        axes[1,0].scatter(pe_volumes, ser_volumes, alpha=0.6, color='#2ECC71')
        axes[1,0].plot([0, max(pe_volumes)], [0, max(pe_volumes)], 'r--', alpha=0.8)
        axes[1,0].set_xlabel('FTV_PE Volume (mm³)')
        axes[1,0].set_ylabel('FTV_SER Volume (mm³)')
        axes[1,0].set_title('FTV_PE vs FTV_SER Correlation')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 数据集分布
        datasets = {}
        for patient_id in self.analysis_results.keys():
            dataset = self.identify_dataset(patient_id)
            datasets[dataset] = datasets.get(dataset, 0) + 1
        
        axes[1,1].pie(datasets.values(), labels=datasets.keys(), autopct='%1.1f%%')
        axes[1,1].set_title('Dataset Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "dataset_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

# 使用示例
if __name__ == "__main__":
    # 配置路径
    data_path = r""
    output_path = r""
    
    # 创建分析器并运行
    analyzer = MAMAMIAFTVAnalyzer(data_path, output_path)
    success_count = analyzer.process_all_patients()
    
    print(f"分析完成! 成功处理 {success_count} 个患者")

    print(f"结果保存在: {output_path}")
