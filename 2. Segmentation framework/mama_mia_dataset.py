import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from glob import glob
import re
from typing import List, Dict, Optional, Union
from scipy.ndimage import zoom
import random
from torchvision.transforms import functional as F 

class MAMAMIAMultiModalAugmentation:
    """专门为多模态MRI设计的数据增广"""
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, mask):
        """
        image: [3, H, W] - 三通道 [T1, SER, PE]
        mask: [1, H, W] - 分割标签
        """
        # 随机旋转和翻转
        if random.random() < self.p:
            image, mask = self.random_rot_flip(image, mask)
        # 随机小角度旋转
        if random.random() < self.p:
            image, mask = self.random_rotate(image, mask)
        # 随机强度扰动
        if random.random() < self.p:
            image, mask = self.random_intensity_shift(image, mask)
            
        return image, mask
    
    def random_rot_flip(self, image, mask):
        """旋转和翻转 - 所有模态同步"""
        # 随机旋转 (0, 90, 180, 270度)
        k = random.randint(0, 3)
        image = torch.rot90(image, k, [1, 2])  # 所有通道一起旋转
        mask = torch.rot90(mask, k, [1, 2])
        
        # 随机翻转
        if random.random() > 0.5:
            image = torch.flip(image, [1])  # 水平翻转
            mask = torch.flip(mask, [1])
        if random.random() > 0.5:
            image = torch.flip(image, [2])  # 垂直翻转
            mask = torch.flip(mask, [2])
            
        return image, mask
    
    def random_rotate(self, image, mask, angle_range=(-15, 15)):
        """小角度旋转 - 避免信息丢失"""
        angle = random.uniform(angle_range[0], angle_range[1])
        
        # 对每个通道分别旋转（但使用相同的角度）
        rotated_channels = []
        for i in range(image.shape[0]):
            channel_img = image[i].unsqueeze(0)  # [1, H, W]
            rotated_channel = F.rotate(channel_img, angle, interpolation=F.InterpolationMode.BILINEAR)
            rotated_channels.append(rotated_channel)
        
        image = torch.cat(rotated_channels, dim=0)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        
        return image, mask
    
    def random_intensity_shift(self, image, mask):
        """对每个模态分别进行强度扰动"""
        for i in range(image.shape[0]):  # 对每个模态通道
            if random.random() < 0.3:  # 30%概率扰动该模态
                # 小幅度的亮度和对比度变化
                alpha = random.uniform(0.9, 1.1)  # 对比度
                beta = random.uniform(-0.1, 0.1)  # 亮度
                image[i] = alpha * image[i] + beta
                # 确保数值范围合理
                image[i] = torch.clamp(image[i], -3, 3)
                
        return image, mask


# 通用的3D数据加载和预处理
class MAMAMIADataset(Dataset):
    """
    MAMA-MIA 3D MRI分割数据集加载器
    支持DUKE、NACT、ISPY1、ISPY2四个子数据集
    【新增】支持跨数据集完整测试
    """
    
    def __init__(self, 
                 data_dir: str = "",
                 seg_dir: str = "",
                 datasets: List[str] = ["DUKE", "NACT", "ISPY1", "ISPY2"],
                 mode: str = "train",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 input_channels: int = 1,
                 transform=None,
                 seed: int = 42,
                 multimodal: bool = False,
                 ser_dir: str = "",
                 pe_dir: str = "",
                 cross_dataset_test: bool = False):
        """
        Args:
            data_dir: 原始数据路径
            seg_dir: 分割标签路径  
            datasets: 要使用的数据集列表
            mode: 数据集模式
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            input_channels: 输入通道数
            transform: 数据增强
            seed: 随机种子
            multimodal: 是否启用多模态输入
            ser_dir: SER图像路径
            pe_dir: PE图像路径
            cross_dataset_test: 跨数据集测试模式（测试整个数据集）
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.datasets = [d.upper() for d in datasets]
        self.mode = mode
        self.input_channels = input_channels
        self.transform = transform
        self.multimodal = multimodal
        self.ser_dir = ser_dir
        self.pe_dir = pe_dir
        self.cross_dataset_test = cross_dataset_test
        
        # 验证配置
        if self.multimodal and self.input_channels != 3:
            print(f"警告: 多模态模式下输入通道数应为3，但设置为{input_channels}，自动调整为3")
            self.input_channels = 3
        
        # 获取所有患者数据
        self.patient_data = self._load_patient_data()
        
        # 数据集划分
        self.patient_ids = self._split_dataset(list(self.patient_data.keys()), 
                                             train_ratio, val_ratio, seed)
        
        print(f"MAMA-MIA Dataset Info:")
        print(f"  - Total patients: {len(self.patient_data)}")
        print(f"  - Selected datasets: {self.datasets}")
        print(f"  - Mode: {mode}, Patients: {len(self.patient_ids)}")
        print(f"  - Input channels: {self.input_channels}")
        print(f"  - Multi-modal: {self.multimodal}")
        if self.cross_dataset_test:
            print(f"  - Cross-dataset test: 完整数据集测试模式")
    
    def _load_patient_data(self) -> Dict[str, Dict]:
        """加载所有患者的数据路径信息"""
        patient_data = {}
        
        # 遍历所有数据集
        for dataset in self.datasets:
            dataset_pattern = os.path.join(self.data_dir, f"{dataset}_*")
            patient_folders = glob(dataset_pattern)
            
            for patient_folder in patient_folders:
                patient_id = os.path.basename(patient_folder)
                
                # 查找T1时刻图像（不区分大小写）
                t1_files = []
                for file in os.listdir(patient_folder):
                    if file.lower().endswith('_0001.nii.gz'):
                        t1_files.append(os.path.join(patient_folder, file))
                
                if len(t1_files) == 0:
                    print(f"Warning: No T1 image found for {patient_id}")
                    continue
                
                # 取第一个找到的T1文件（通常只有一个）
                t1_path = t1_files[0]
                
                # 【新增】多模态数据加载
                if self.multimodal:
                    # 查找SER图像
                    ser_pattern = os.path.join(self.ser_dir, f"*{patient_id}*_FTV_SER_T1.nii.gz")
                    ser_files = glob(ser_pattern)
                    if len(ser_files) == 0:
                        print(f"Warning: No SER image found for {patient_id}")
                        continue
                    ser_path = ser_files[0]
                    
                    # 查找PE图像  
                    pe_pattern = os.path.join(self.pe_dir, f"*{patient_id}*_FTV_PE_T1.nii.gz")
                    pe_files = glob(pe_pattern)
                    if len(pe_files) == 0:
                        print(f"Warning: No PE image found for {patient_id}")
                        continue
                    pe_path = pe_files[0]
                
                # 查找对应的分割标签
                seg_pattern = os.path.join(self.seg_dir, f"*{patient_id}*.nii.gz")
                seg_files = glob(seg_pattern)
                
                if len(seg_files) == 0:
                    print(f"Warning: No segmentation found for {patient_id}")
                    continue
                
                seg_path = seg_files[0]  # 取第一个匹配的分割文件
                
                if self.multimodal:
                    patient_data[patient_id] = {
                        't1_path': t1_path,
                        'ser_path': ser_path,
                        'pe_path': pe_path,
                        'seg_path': seg_path,
                        'dataset': dataset
                    }
                else:
                    patient_data[patient_id] = {
                        't1_path': t1_path,
                        'seg_path': seg_path,
                        'dataset': dataset
                    }
        
        return patient_data
    
    def _split_dataset(self, all_patients: List[str], train_ratio: float, 
                      val_ratio: float, seed: int) -> List[str]:
        """划分训练集、验证集、测试集"""
        
        if self.cross_dataset_test:
            print(f"跨数据集测试模式：使用整个数据集的 {len(all_patients)} 名患者")
            return all_patients
        
        np.random.seed(seed)
        np.random.shuffle(all_patients)
        
        n_total = len(all_patients)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if self.mode == "train":
            return all_patients[:n_train]
        elif self.mode == "val":
            return all_patients[n_train:n_train + n_val]
        else:  # test
            test_patients = all_patients[n_train + n_val:]
            print(f"标准测试模式：使用 {len(test_patients)} 名患者（总患者数: {n_total}）")
            return test_patients
    
    def _load_nifti(self, file_path: str) -> np.ndarray:
        """加载nifti文件并返回numpy数组"""
        img = nib.load(file_path)
        data = img.get_fdata()
        return data
    
    def _preprocess_data(self, image: np.ndarray, mask: np.ndarray, 
                        ser_image: np.ndarray = None, pe_image: np.ndarray = None) -> tuple:
        """数据预处理"""
        # 确保是3D数据 [H, W, D]
        if image.ndim == 4:  # 如果是4D [H, W, D, C]
            image = image[..., 0]  # 取第一个通道
        
        if self.multimodal:
            if ser_image is None or pe_image is None:
                raise ValueError("多模态模式下需要提供SER和PE图像")
            
            # 确保所有模态数据尺寸一致
            if image.shape != ser_image.shape or image.shape != pe_image.shape:
                print(f"警告: 模态数据尺寸不一致 - T1: {image.shape}, SER: {ser_image.shape}, PE: {pe_image.shape}")
                # 统一调整到T1图像的尺寸
                target_shape = image.shape
                ser_image = self._resize_to_target(ser_image, target_shape)
                pe_image = self._resize_to_target(pe_image, target_shape)
        
        # 统一调整尺寸到 [256, 256]
        target_size = (256, 256)
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            # 调整图像尺寸
            image = self._resize_3d(image, target_size)
            if self.multimodal:
                ser_image = self._resize_3d(ser_image, target_size)
                pe_image = self._resize_3d(pe_image, target_size)
            # 调整mask尺寸（使用最近邻插值保持边缘清晰）
            mask = self._resize_3d(mask, target_size, is_mask=True)
        
        # 归一化
        image = (image - image.mean()) / (image.std() + 1e-8)
        if self.multimodal:
            ser_image = (ser_image - ser_image.mean()) / (ser_image.std() + 1e-8)
            pe_image = (pe_image - pe_image.mean()) / (pe_image.std() + 1e-8)
        
        # 处理mask：二值化
        mask = (mask > 0).astype(np.float32)
        
        if self.multimodal:
            # 堆叠多模态数据 [3, H, W, D]
            multi_modal_image = np.stack([image, ser_image, pe_image], axis=0)
            image_tensor = multi_modal_image
        else:
            # 单模态处理
            if self.input_channels > 1:
                image_tensor = np.stack([image] * self.input_channels, axis=0)
            else:
                image_tensor = image[np.newaxis, ...]  # [1, H, W, D]
        
        mask_tensor = mask[np.newaxis, ...]  # [1, H, W, D]
        
        return image_tensor, mask_tensor
    
    def _resize_to_target(self, volume: np.ndarray, target_shape: tuple) -> np.ndarray:
        """将体积调整到目标形状"""
        zoom_factors = (
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1], 
            target_shape[2] / volume.shape[2]
        )
        return zoom(volume, zoom_factors, order=1)
    
    def _resize_3d(self, volume: np.ndarray, target_size: tuple, is_mask: bool = False) -> np.ndarray:
        # 计算缩放因子
        h, w, d = volume.shape
        target_h, target_w = target_size
        
        zoom_factors = (target_h / h, target_w / w, 1)  # 深度维度保持不变
        
        if is_mask:
            # 对于mask使用最近邻插值
            resized = zoom(volume, zoom_factors, order=0)
        else:
            # 对于图像使用三线性插值
            resized = zoom(volume, zoom_factors, order=1)
        
        return resized

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_info = self.patient_data[patient_id]
        
        # 加载T1图像和分割标签
        image = self._load_nifti(patient_info['t1_path'])
        mask = self._load_nifti(patient_info['seg_path'])
        
        ser_image = None
        pe_image = None
        if self.multimodal:
            ser_image = self._load_nifti(patient_info['ser_path'])
            pe_image = self._load_nifti(patient_info['pe_path'])
        
        # 预处理
        image, mask = self._preprocess_data(image, mask, ser_image, pe_image)
        
        # 转换为torch tensor
        image = torch.FloatTensor(image)  # [C, H, W, D]
        mask = torch.FloatTensor(mask)    # [1, H, W, D]
        
        meta = {
            'patient_id': patient_id, 
            'dataset': patient_info['dataset'],
            'reference_path': patient_info['t1_path']  # 用于保存预测结果时参考
        }
        
        return image, mask, meta

class MAMAMIADataset2D(Dataset):
    """
    2D切片版本的数据集，用于与原有U-KAN模型兼容
    将3D体积切片为2D图像进行训练
    【新增】支持跨数据集完整测试
    【新增】支持平衡采样和数据增广
    """
    
    def __init__(self, 
                 data_dir: str = "",
                 seg_dir: str = "",
                 datasets: List[str] = ["DUKE", "NACT", "ISPY1", "ISPY2"],
                 mode: str = "train",
                 slice_axis: int = 2,  # 切片轴: 0=sagittal, 1=coronal, 2=axial
                 input_channels: int = 1,
                 transform=None,
                 seed: int = 42,
                 multimodal: bool = False,
                 ser_dir: str = "",
                 pe_dir: str = "",
                 cross_dataset_test: bool = False,
                 balanced_sampling: bool = False):
        
        self.slice_axis = slice_axis
        self.multimodal = multimodal
        self.cross_dataset_test = cross_dataset_test
        self.mode = mode
        self.transform = transform  
        self.balanced_sampling = balanced_sampling 
        
        self.original_dataset = MAMAMIADataset(
            data_dir=data_dir,
            seg_dir=seg_dir,
            datasets=datasets,
            mode=mode,
            input_channels=input_channels,
            transform=transform,
            seed=seed,
            multimodal=multimodal,  
            ser_dir=ser_dir,       
            pe_dir=pe_dir,         
            cross_dataset_test=cross_dataset_test 
        )
        
        # 预计算所有切片索引
        self.slice_indices = []
        for patient_idx in range(len(self.original_dataset)):
            patient_id = self.original_dataset.patient_ids[patient_idx]
            patient_info = self.original_dataset.patient_data[patient_id]
            
            # 获取体积维度信息
            image = self.original_dataset._load_nifti(patient_info['t1_path'])
            if image.ndim == 4:
                image = image[..., 0]
            
            n_slices = image.shape[slice_axis]
            
            for slice_idx in range(n_slices):
                self.slice_indices.append((patient_idx, slice_idx))
        
        self.slice_weights = None
        if self.balanced_sampling and self.mode == "train":
            self._compute_slice_weights()
            print(f"平衡采样模式: 已计算{len(self.slice_weights)}个切片的权重")
        
        print(f"2D切片数据集: {len(self.slice_indices)} 个切片")
        if self.cross_dataset_test:
            print("🎯 跨数据集测试模式: 使用整个目标数据集进行泛化能力评估")
    
    def _compute_slice_weights(self):
        """计算每个切片的权重（基于肿瘤面积）"""
        self.slice_weights = []
        
        for (patient_idx, slice_idx) in self.slice_indices:
            patient_id = self.original_dataset.patient_ids[patient_idx]
            patient_info = self.original_dataset.patient_data[patient_id]
            
            # 加载mask体积
            mask_volume = self.original_dataset._load_nifti(patient_info['seg_path'])
            if mask_volume.ndim == 4:
                mask_volume = mask_volume[..., 0]
            
            # 提取该切片的mask
            if self.slice_axis == 0:  # sagittal
                mask_slice = mask_volume[slice_idx, :, :]
            elif self.slice_axis == 1:  # coronal
                mask_slice = mask_volume[:, slice_idx, :]
            else:  # axial (默认)
                mask_slice = mask_volume[:, :, slice_idx]
            
            # 计算肿瘤像素比例
            tumor_ratio = np.sum(mask_slice > 0) / mask_slice.size
            
            # 权重计算：有肿瘤的切片权重更高
            # 基础权重1.0 + 肿瘤比例 × 10
            weight = 1.0 + tumor_ratio * 10.0
            self.slice_weights.append(weight)
    
    def get_weighted_sampler(self):
        """返回加权采样器（用于DataLoader）"""
        if self.slice_weights is None:
            raise ValueError("未启用平衡采样或未计算切片权重")
        
        weights = torch.DoubleTensor(self.slice_weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        return sampler
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        patient_idx, slice_idx = self.slice_indices[idx]
        
        # 获取完整的3D数据
        image_3d, mask_3d, meta = self.original_dataset[patient_idx]
        
        # 沿指定轴切片 [C, H, W, D] -> [C, H, W] 或 [C, H, D] 或 [C, W, D]
        if self.slice_axis == 0:  # sagittal
            image_2d = image_3d[:, slice_idx, :, :]  # [C, W, D]
            mask_2d = mask_3d[:, slice_idx, :, :]    # [1, W, D]
        elif self.slice_axis == 1:  # coronal  
            image_2d = image_3d[:, :, slice_idx, :]  # [C, H, D]
            mask_2d = mask_3d[:, :, slice_idx, :]    # [1, H, D]
        else:  # axial (默认)
            image_2d = image_3d[:, :, :, slice_idx]  # [C, H, W]
            mask_2d = mask_3d[:, :, :, slice_idx]    # [1, H, W]
        
        # 【新增】数据增广（仅在训练模式下且启用了transform）
        if self.mode == "train" and self.transform is not None:
            image_2d, mask_2d = self.transform(image_2d, mask_2d)
        
        # 更新metadata
        meta['slice_idx'] = slice_idx
        meta['slice_axis'] = self.slice_axis
        
        return image_2d, mask_2d, meta


def save_prediction_as_nifti(prediction: np.ndarray, reference_nifti_path: str, 
                           output_path: str, patient_id: str):
    """
    将预测结果保存为nifti格式
    
    Args:
        prediction: 预测的分割结果 [H, W, D]
        reference_nifti_path: 参考nifti文件路径（用于获取头信息）
        output_path: 输出目录
        patient_id: 患者ID
    """
    # 加载参考nifti获取头信息
    ref_img = nib.load(reference_nifti_path)
    
    # 创建新的nifti图像
    pred_img = nib.Nifti1Image(prediction, ref_img.affine, ref_img.header)
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 保存文件
    output_file = os.path.join(output_path, f"{patient_id}_pred.nii.gz")
    nib.save(pred_img, output_file)
    

    print(f"Prediction saved: {output_file}")

