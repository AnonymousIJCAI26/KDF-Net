import os
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BreastCancerVOIExtractor:
    """乳腺癌肿瘤VOI提取器 - 最终版本"""
    
    def __init__(self, 
                 t1_root: str,
                 pk_root: str,
                 gt_root: str,
                 output_root: str):
        """
        初始化VOI提取器
        """
        self.t1_root = Path(t1_root)
        self.pk_root = Path(pk_root)
        self.gt_root = Path(gt_root)
        self.output_root = Path(output_root)
        
        # 创建输出目录
        self.multichannel_dir = self.output_root / "multichannel_volumes"
        self.voi_dir = self.output_root / "tumor_vois"
        self.multichannel_dir.mkdir(parents=True, exist_ok=True)
        self.voi_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取患者列表
        self.patient_ids = self._get_patient_ids()
        logger.info(f"找到 {len(self.patient_ids)} 名患者")
    
    def _get_patient_ids(self) -> List[str]:
        """从T1目录获取患者ID列表"""
        patient_dirs = [d for d in self.t1_root.iterdir() if d.is_dir()]
        patient_ids = [d.name for d in patient_dirs]
        return sorted(patient_ids)
    
    def _find_file_case_insensitive(self, directory: Path, pattern: str) -> Optional[Path]:
        """大小写不敏感地查找文件"""
        pattern_lower = pattern.lower()
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                filename_lower = file_path.name.lower()
                if self._glob_match(filename_lower, pattern_lower):
                    return file_path
        
        # 宽松匹配
        pattern_keywords = pattern_lower.replace('*', '').replace('.', '_').split('_')
        pattern_keywords = [kw for kw in pattern_keywords if kw and len(kw) > 2]
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                filename_lower = file_path.name.lower()
                if all(keyword in filename_lower for keyword in pattern_keywords):
                    return file_path
        
        return None
    
    def _glob_match(self, filename: str, pattern: str) -> bool:
        """简单的glob模式匹配"""
        import re
        pattern_regex = pattern.replace('.', '\\.').replace('*', '.*').replace('?', '.')
        pattern_regex = f'^{pattern_regex}$'
        return re.match(pattern_regex, filename) is not None
    
    def load_image(self, patient_id: str, image_type: str) -> Optional[Tuple[sitk.Image, str]]:
        """加载单个患者的图像"""
        try:
            if image_type == 't1':
                patient_dir = self.t1_root / patient_id
                pattern = "*_0001.nii.gz"
            elif image_type == 'ser':
                patient_dir = self.pk_root / patient_id
                pattern = "*_FTV_SER_T1.nii.gz"
            elif image_type == 'pe':
                patient_dir = self.pk_root / patient_id
                pattern = "*_FTV_PE_T1.nii.gz"
            elif image_type == 'gt':
                patient_dir = self.gt_root
                patterns = [f"{patient_id}*.nii.gz", f"*{patient_id}*.nii.gz"]
                file_path = None
                for p in patterns:
                    file_path = self._find_file_case_insensitive(patient_dir, p)
                    if file_path:
                        break
                if file_path:
                    image = sitk.ReadImage(str(file_path))
                    return image, str(file_path)
                return None, None
            else:
                raise ValueError(f"未知的图像类型: {image_type}")
            
            if not patient_dir.exists():
                logger.warning(f"患者 {patient_id} 的 {image_type.upper()} 目录不存在")
                return None, None
            
            file_path = self._find_file_case_insensitive(patient_dir, pattern)
            
            if not file_path:
                logger.warning(f"未找到患者 {patient_id} 的 {image_type.upper()} 文件")
                return None, None
            
            image = sitk.ReadImage(str(file_path))
            return image, str(file_path)
            
        except Exception as e:
            logger.error(f"加载患者 {patient_id} 的 {image_type.upper()} 图像时出错: {e}")
            return None, None
    
    def normalize_intensity(self, image: sitk.Image, method: str = 'zscore') -> sitk.Image:
        """
        标准化图像强度
        """
        arr = sitk.GetArrayFromImage(image)
        
        if method == 'zscore':
            # Z-score标准化
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            if std_val > 0:
                arr_normalized = (arr - mean_val) / std_val
            else:
                arr_normalized = arr - mean_val
        elif method == 'minmax':
            # 最小-最大归一化到 [0, 1]
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val > min_val:
                arr_normalized = (arr - min_val) / (max_val - min_val)
            else:
                arr_normalized = arr - min_val
        elif method == 'robust':
            # 基于四分位数的鲁棒标准化
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            if iqr > 0:
                arr_normalized = (arr - np.median(arr)) / iqr
            else:
                arr_normalized = arr - np.median(arr)
        else:
            raise ValueError(f"未知的标准化方法: {method}")
        
        normalized_img = sitk.GetImageFromArray(arr_normalized)
        normalized_img.CopyInformation(image)
        
        return normalized_img
    
    def check_image_alignment(self, image1: sitk.Image, image2: sitk.Image, 
                             patient_id: str, img1_name: str, img2_name: str) -> bool:
        """检查两幅图像是否空间对齐"""
        try:
            size1 = image1.GetSize()
            size2 = image2.GetSize()
            
            if size1 != size2:
                logger.warning(f"患者 {patient_id}: {img1_name}({size1}) 和 {img2_name}({size2}) 尺寸不匹配")
                return False
            
            spacing1 = image1.GetSpacing()
            spacing2 = image2.GetSpacing()
            
            if not np.allclose(spacing1, spacing2, rtol=1e-5):
                logger.warning(f"患者 {patient_id}: {img1_name} 和 {img2_name} 像素间距不匹配")
                return False
            
            origin1 = image1.GetOrigin()
            origin2 = image2.GetOrigin()
            
            if not np.allclose(origin1, origin2, rtol=1e-5):
                logger.warning(f"患者 {patient_id}: {img1_name} 和 {img2_name} 原点不匹配")
                return False
            
            direction1 = image1.GetDirection()
            direction2 = image2.GetDirection()
            
            if not np.allclose(direction1, direction2, rtol=1e-5):
                logger.warning(f"患者 {patient_id}: {img1_name} 和 {img2_name} 方向矩阵不匹配")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查图像对齐时出错: {e}")
            return False
    
    def create_multichannel_volume(self, t1_img: sitk.Image, 
                                   ser_img: sitk.Image, 
                                   pe_img: sitk.Image,
                                   patient_id: str) -> Optional[sitk.Image]:
        """
        创建多通道3D体数据
        """
        try:
            logger.info(f"患者 {patient_id}: 创建多通道体积")
            
            # 将所有图像转换为float32
            t1_float = sitk.Cast(t1_img, sitk.sitkFloat32)
            ser_float = sitk.Cast(ser_img, sitk.sitkFloat32)
            pe_float = sitk.Cast(pe_img, sitk.sitkFloat32)
            
            # 对每个通道进行强度标准化
            t1_normalized = self.normalize_intensity(t1_float, method='zscore')
            ser_normalized = self.normalize_intensity(ser_float, method='zscore')
            pe_normalized = self.normalize_intensity(pe_float, method='zscore')
            
            # 使用Compose创建多通道图像
            vector_img = sitk.Compose([t1_normalized, ser_normalized, pe_normalized])
            
            logger.info(f"患者 {patient_id}: 创建多通道体积成功")
            logger.info(f"  输出尺寸: {vector_img.GetSize()}")
            logger.info(f"  通道数: {vector_img.GetNumberOfComponentsPerPixel()}")
            
            return vector_img
            
        except Exception as e:
            logger.error(f"创建多通道体积时出错: {e}")
            traceback.print_exc()
            return None
    
    def extract_voi(self, multichannel_img: sitk.Image, 
                    gt_img: sitk.Image,
                    patient_id: str) -> Optional[sitk.Image]:
        """
        从多通道图像中提取VOI - 使用numpy数组方法
        """
        try:
            logger.info(f"患者 {patient_id}: 提取VOI")
            
            # 将多通道图像转换为numpy数组 (Z, Y, X, C)
            mc_array = sitk.GetArrayFromImage(multichannel_img)
            logger.info(f"  多通道数组形状 (Z,Y,X,C): {mc_array.shape}")
            
            # 获取GT掩码数组 (Z, Y, X)
            gt_mask = sitk.GetArrayFromImage(gt_img)
            
            # 找到肿瘤区域
            tumor_label = 1
            tumor_indices = np.where(gt_mask == tumor_label)
            
            if len(tumor_indices[0]) == 0:
                tumor_indices = np.where(gt_mask > 0)
            
            if len(tumor_indices[0]) == 0:
                logger.error(f"患者 {patient_id}: Ground Truth中没有找到肿瘤区域")
                return None
            
            # 计算边界框
            z_min, z_max = tumor_indices[0].min(), tumor_indices[0].max()
            y_min, y_max = tumor_indices[1].min(), tumor_indices[1].max()
            x_min, x_max = tumor_indices[2].min(), tumor_indices[2].max()
            
            logger.info(f"患者 {patient_id}: 肿瘤原始边界框")
            logger.info(f"  Z轴 (切片): {z_min} 到 {z_max} (共 {z_max - z_min + 1} 层)")
            logger.info(f"  Y轴 (高度): {y_min} 到 {y_max} (共 {y_max - y_min + 1} 像素)")
            logger.info(f"  X轴 (宽度): {x_min} 到 {x_max} (共 {x_max - x_min + 1} 像素)")
            
            # 添加边界余量
            margin = 10
            z_min = max(0, z_min - margin)
            z_max = min(gt_mask.shape[0] - 1, z_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(gt_mask.shape[1] - 1, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(gt_mask.shape[2] - 1, x_max + margin)
            
            logger.info(f"患者 {patient_id}: 带边界的边界框")
            logger.info(f"  Z轴: {z_min} 到 {z_max} (大小: {z_max - z_min + 1})")
            logger.info(f"  Y轴: {y_min} 到 {y_max} (大小: {y_max - y_min + 1})")
            logger.info(f"  X轴: {x_min} 到 {x_max} (大小: {x_max - x_min + 1})")
            
            # 提取VOI区域
            voi_array = mc_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1, :]
            logger.info(f"  VOI数组形状 (Z,Y,X,C): {voi_array.shape}")
            
            # 创建新的多通道图像
            voi_img = sitk.GetImageFromArray(voi_array, isVector=True)
            
            # 设置空间信息
            mc_spacing = multichannel_img.GetSpacing()
            mc_origin = multichannel_img.GetOrigin()
            mc_direction = multichannel_img.GetDirection()
            
            voi_img.SetSpacing(mc_spacing)
            voi_img.SetDirection(mc_direction)
            
            # 计算新原点
            new_origin = (
                mc_origin[0] + x_min * mc_spacing[0],
                mc_origin[1] + y_min * mc_spacing[1],
                mc_origin[2] + z_min * mc_spacing[2]
            )
            voi_img.SetOrigin(new_origin)
            
            logger.info(f"患者 {patient_id}: VOI提取成功")
            logger.info(f"  VOI尺寸 (X,Y,Z): {voi_img.GetSize()}")
            
            return voi_img
            
        except Exception as e:
            logger.error(f"提取VOI时出错: {e}")
            traceback.print_exc()
            return None
    
    def process_patient(self, patient_id: str) -> Dict[str, bool]:
        """处理单个患者"""
        status = {
            't1_loaded': False,
            'ser_loaded': False,
            'pe_loaded': False,
            'gt_loaded': False,
            'alignment_ok': False,
            'multichannel_created': False,
            'voi_extracted': False
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始处理患者: {patient_id}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. 加载所有图像
            t1_img, t1_path = self.load_image(patient_id, 't1')
            if not t1_img:
                logger.error(f"  无法加载T1图像")
                return status
            status['t1_loaded'] = True
            
            ser_img, ser_path = self.load_image(patient_id, 'ser')
            if not ser_img:
                logger.error(f"  无法加载SER图像")
                return status
            status['ser_loaded'] = True
            
            pe_img, pe_path = self.load_image(patient_id, 'pe')
            if not pe_img:
                logger.error(f"  无法加载PE图像")
                return status
            status['pe_loaded'] = True
            
            gt_img, gt_path = self.load_image(patient_id, 'gt')
            if not gt_img:
                logger.error(f"  无法加载Ground Truth图像")
                return status
            status['gt_loaded'] = True
            
            # 2. 检查图像对齐
            alignment_t1_ser = self.check_image_alignment(t1_img, ser_img, patient_id, "T1", "SER")
            alignment_t1_pe = self.check_image_alignment(t1_img, pe_img, patient_id, "T1", "PE")
            alignment_t1_gt = self.check_image_alignment(t1_img, gt_img, patient_id, "T1", "GT")
            
            if alignment_t1_ser and alignment_t1_pe and alignment_t1_gt:
                status['alignment_ok'] = True
                logger.info(f"  所有图像对齐正常")
            else:
                logger.warning(f"  部分图像对齐有问题，将继续处理")
            
            # 3. 创建多通道体积
            multichannel_img = self.create_multichannel_volume(t1_img, ser_img, pe_img, patient_id)
            if not multichannel_img:
                logger.error(f"  无法创建多通道体积")
                return status
            status['multichannel_created'] = True
            
            # 保存多通道体积
            output_path = self.multichannel_dir / f"{patient_id}_multichannel.nii.gz"
            sitk.WriteImage(multichannel_img, str(output_path))
            logger.info(f"  多通道体积已保存: {output_path}")
            
            # 4. 提取VOI
            voi_img = self.extract_voi(multichannel_img, gt_img, patient_id)
            if not voi_img:
                logger.error(f"  无法提取VOI")
                return status
            status['voi_extracted'] = True
            
            # 保存VOI
            output_path = self.voi_dir / f"{patient_id}_voi.nii.gz"
            sitk.WriteImage(voi_img, str(output_path))
            logger.info(f"  VOI已保存: {output_path}")
            
            logger.info(f"患者 {patient_id} 处理完成 ✓")
            
        except Exception as e:
            logger.error(f"处理患者 {patient_id} 时发生异常: {e}")
            traceback.print_exc()
        
        return status
    
    def process_all_patients(self, start_idx: int = 0, end_idx: Optional[int] = None, 
                           batch_size: int = 100) -> Dict[str, Dict[str, bool]]:
        """处理所有患者"""
        results = {}
        
        if end_idx is None or end_idx > len(self.patient_ids):
            end_idx = len(self.patient_ids)
        
        patient_sublist = self.patient_ids[start_idx:end_idx]
        
        logger.info(f"开始处理患者 {start_idx+1} 到 {end_idx}，共 {len(patient_sublist)} 名患者")
        
        success_count = 0
        for i, patient_id in enumerate(patient_sublist, 1):
            logger.info(f"\n处理进度: {i}/{len(patient_sublist)} (总进度: {start_idx+i}/{len(self.patient_ids)})")
            
            status = self.process_patient(patient_id)
            results[patient_id] = status
            
            if all(status.values()):
                success_count += 1
            
            # 每处理batch_size个患者保存一次进度
            if i % batch_size == 0 or i == len(patient_sublist):
                self._save_progress(results, start_idx + i, success_count, len(patient_sublist))
        
        self._print_statistics(results, success_count, len(patient_sublist))
        
        return results
    
    def _save_progress(self, results: Dict, current_idx: int, success_count: int, total: int):
        """保存处理进度"""
        progress_file = self.output_root / "progress_log.txt"
        with open(progress_file, 'w') as f:
            f.write(f"处理进度: {current_idx}/{len(self.patient_ids)}\n")
            f.write(f"本批处理: {total} 名患者\n")
            f.write(f"成功处理: {success_count} ({success_count/total*100:.1f}%)\n")
            f.write(f"失败: {total - success_count}\n\n")
            
            failed_patients = []
            for patient_id, status in results.items():
                if not all(status.values()):
                    failed_steps = [step for step, success in status.items() if not success]
                    failed_patients.append(f"{patient_id}: {', '.join(failed_steps)}")
            
            if failed_patients:
                f.write("失败患者详情:\n")
                for line in failed_patients:
                    f.write(f"{line}\n")
    
    def _print_statistics(self, results: Dict[str, Dict[str, bool]], success_count: int, total_processed: int):
        """打印统计信息"""
        logger.info("\n" + "="*60)
        logger.info("处理完成!")
        logger.info(f"处理患者数: {total_processed}")
        logger.info(f"成功处理: {success_count} ({success_count/total_processed*100:.1f}%)")
        logger.info(f"失败: {total_processed - success_count}")
        
        if total_processed - success_count > 0:
            logger.info("\n失败详情:")
            failure_counts = {}
            for patient_id, status in results.items():
                if not all(status.values()):
                    failed_steps = [step for step, success in status.items() if not success]
                    for step in failed_steps:
                        failure_counts[step] = failure_counts.get(step, 0) + 1
            
            for step, count in failure_counts.items():
                logger.info(f"  {step}: {count} 次失败")
        
        logger.info("="*60)
        
        # 保存最终报告
        self._save_final_report(results, success_count, total_processed)
    
    def _save_final_report(self, results: Dict[str, Dict[str, bool]], success_count: int, total_processed: int):
        """保存最终处理报告"""
        report_file = self.output_root / "processing_report.txt"
        with open(report_file, 'w') as f:
            f.write("乳腺癌肿瘤VOI提取处理报告\n")
            f.write("="*60 + "\n")
            f.write(f"总患者数: {len(self.patient_ids)}\n")
            f.write(f"本次处理: {total_processed}\n")
            f.write(f"成功处理: {success_count} ({success_count/total_processed*100:.1f}%)\n")
            f.write(f"失败: {total_processed - success_count}\n\n")
            
            f.write("成功处理的患者:\n")
            success_patients = [pid for pid, status in results.items() if all(status.values())]
            for i, pid in enumerate(success_patients, 1):
                f.write(f"{i:4d}. {pid}\n")
            
            f.write("\n失败的患者:\n")
            failed_patients = [pid for pid, status in results.items() if not all(status.values())]
            if failed_patients:
                for pid in failed_patients:
                    failed_steps = [step for step, success in results[pid].items() if not success]
                    f.write(f"{pid}: {', '.join(failed_steps)}\n")
            else:
                f.write("无\n")
            
            f.write("\n输出文件位置:\n")
            f.write(f"多通道体积: {self.multichannel_dir}\n")
            f.write(f"肿瘤VOI: {self.voi_dir}\n")

def main():
    """主函数"""
    T1_ROOT = 
    PK_ROOT = 
    GT_ROOT = 
    OUTPUT_ROOT = 
    
    # 创建提取器
    extractor = BreastCancerVOIExtractor(T1_ROOT, PK_ROOT, GT_ROOT, OUTPUT_ROOT)
    
    # 分批处理所有患者（避免内存问题）
    total_patients = len(extractor.patient_ids)
    batch_size = 1506  # 每批处理100个患者
    
    for start_idx in range(0, total_patients, batch_size):
        end_idx = min(start_idx + batch_size, total_patients)
        logger.info(f"\n{'='*80}")
        logger.info(f"处理批次: {start_idx//batch_size + 1}")
        logger.info(f"处理范围: {start_idx+1} 到 {end_idx}")
        logger.info(f"{'='*80}")
        
        results = extractor.process_all_patients(start_idx=start_idx, end_idx=end_idx, batch_size=batch_size)
        
        # 等待用户确认是否继续
        if end_idx < total_patients:
            input(f"已处理 {end_idx}/{total_patients} 名患者。按Enter继续处理下一批...")

if __name__ == "__main__":
    # 确认是否要批量处理
    response = input("是否开始批量处理所有患者？(y/n): ")
    if response.lower() == 'y':
        main()
    else:

        print("请运行 test_single_patient() 函数测试单个患者")
