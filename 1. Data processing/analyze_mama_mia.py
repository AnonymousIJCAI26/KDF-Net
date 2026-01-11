import os
import re
from collections import defaultdict
from datetime import datetime

# 数据集根目录路径
DATA_ROOT = r"D:\MAMA-MIA\data"

# 输出文档路径
OUTPUT_DOC = os.path.join(DATA_ROOT, "MAMA-MIA_Dataset_Analysis_Report.txt")

def extract_dataset_and_patient(folder_name):
    """
    从文件夹名称中提取数据集名称和病人ID。
    文件夹名格式：数据集名称_病人ID
    例如：DUKE_001, ISPY1_123
    """
    # 使用正则表达式匹配前缀和数据集名称
    match = re.match(r'^(DUKE|ISPY1|ISPY2|NACT)_([A-Za-z0-9_-]+)$', folder_name)
    if match:
        dataset_name = match.group(1)
        patient_id = match.group(2)
        return dataset_name, patient_id
    else:
        # 如果未匹配到标准格式，尝试按第一个下划线分割
        parts = folder_name.split('_', 1)
        if len(parts) == 2:
            dataset_name = parts[0]
            patient_id = parts[1]
            # 确保数据集名称在预期列表中
            if dataset_name in {'DUKE', 'ISPY1', 'ISPY2', 'NACT'}:
                return dataset_name, patient_id
        return None, None

def analyze_dataset():
    """
    分析数据集，统计每个数据集的患者数量和图像信息。
    """
    if not os.path.exists(DATA_ROOT):
        print(f"数据目录不存在：{DATA_ROOT}")
        return

    # 用于存储统计结果的字典
    dataset_stats = defaultdict(lambda: {'patient_count': 0, 'patients': {}})
    total_patients = 0
    total_images = 0

    # 遍历数据根目录下的所有文件夹
    for folder_name in os.listdir(DATA_ROOT):
        folder_path = os.path.join(DATA_ROOT, folder_name)
        
        # 只处理文件夹
        if not os.path.isdir(folder_path):
            continue
        
        dataset_name, patient_id = extract_dataset_and_patient(folder_name)
        
        if dataset_name is None or patient_id is None:
            print(f"警告：无法识别文件夹 '{folder_name}' 的格式，已跳过。")
            continue
        
        # 查找该患者文件夹下的所有 .nii.gz 图像文件
        image_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.nii.gz'):
                image_files.append(file)
        
        # 更新统计信息
        dataset_stats[dataset_name]['patient_count'] += 1
        dataset_stats[dataset_name]['patients'][patient_id] = {
            'image_count': len(image_files),
            'images': image_files
        }
        
        total_patients += 1
        total_images += len(image_files)
    
    # 将结果写入文档
    with open(OUTPUT_DOC, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MAMA-MIA 数据集详细分析报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"总患者数：{total_patients}\n")
        f.write(f"总图像数：{total_images}\n\n")
        
        for dataset_name in sorted(dataset_stats.keys()):
            stats = dataset_stats[dataset_name]
            f.write(f"数据集：{dataset_name}\n")
            f.write(f"患者数量：{stats['patient_count']}\n")
            f.write("-" * 40 + "\n")
            
            for patient_id in sorted(stats['patients'].keys()):
                patient_info = stats['patients'][patient_id]
                f.write(f"  患者ID: {patient_id}\n")
                f.write(f"    图像数量: {patient_info['image_count']}\n")
                if patient_info['images']:
                    f.write(f"    图像文件:\n")
                    for img in sorted(patient_info['images']):
                        f.write(f"      - {img}\n")
                else:
                    f.write("    图像文件: 无\n")
                f.write("\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("分析完成。\n")
    
    print(f"分析完成！结果已保存到：{OUTPUT_DOC}")

if __name__ == "__main__":
    analyze_dataset()