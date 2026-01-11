import torch
from torch.utils.data import DataLoader
from mama_mia_dataset import MAMAMIADataset2D, MAMAMIAMultiModalAugmentation  # 【新增】导入增广类

class MAMAMIADataLoader:
    """适配UltraLight VM-UNet的MAMA-MIA数据加载器"""
    
    def __init__(self, config):
        self.config = config
        
    def get_train_loader(self):
        """获取训练数据加载器"""
        print(f"Loading training datasets: {self.config.datasets_list}")
        
        # 【新增】配置数据增广
        transform = None
        if getattr(self.config, 'use_augmentation', True) and self.config.multimodal:
            transform = MAMAMIAMultiModalAugmentation(p=0.5)
            print("启用多模态数据增广")
        
        # 【修改】创建数据集时传递平衡采样参数
        train_dataset = MAMAMIADataset2D(
            data_dir=self.config.data_dir,
            seg_dir=self.config.seg_dir,
            datasets=self.config.datasets_list,
            mode='train',
            input_channels=self.config.input_channels,
            multimodal=self.config.multimodal,
            ser_dir=getattr(self.config, 'ser_dir', ''),
            pe_dir=getattr(self.config, 'pe_dir', ''),
            transform=transform,  # 【新增】传递数据增广
            balanced_sampling=getattr(self.config, 'balanced_sampling', True)  # 【新增】平衡采样
        )
        
        if len(train_dataset) == 0:
            raise ValueError("No training data found! Please check dataset configuration.")
        
        # 【新增】配置采样器
        sampler = None
        shuffle = True
        if getattr(self.config, 'balanced_sampling', True) and self.config.multimodal:
            sampler = train_dataset.get_weighted_sampler()
            shuffle = False  # 采样器控制随机性
            print("启用平衡采样")
        
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,  # 【新增】传递采样器
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def get_val_loader(self):
        """获取验证数据加载器"""
        print(f"Loading validation datasets: {self.config.datasets_list}")
        val_dataset = MAMAMIADataset2D(
            data_dir=self.config.data_dir,
            seg_dir=self.config.seg_dir,
            datasets=self.config.datasets_list,
            mode='val',
            input_channels=self.config.input_channels,
            multimodal=self.config.multimodal,
            ser_dir=getattr(self.config, 'ser_dir', ''),
            pe_dir=getattr(self.config, 'pe_dir', ''),
            balanced_sampling=False  # 【新增】验证集不需要平衡采样
        )
        
        if len(val_dataset) == 0:
            raise ValueError("No validation data found! Please check dataset configuration.")
        
        return DataLoader(
            val_dataset,
            batch_size=256,  # 验证时batch_size=1
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_test_loader(self, test_datasets=None):
        """获取测试数据加载器"""
        if test_datasets is None:
            test_datasets = self.config.datasets_list
            
        print(f"Loading test datasets: {test_datasets}")
        test_dataset = MAMAMIADataset2D(
            data_dir=self.config.data_dir,
            seg_dir=self.config.seg_dir,
            datasets=test_datasets,
            mode='test',
            input_channels=self.config.input_channels,
            multimodal=self.config.multimodal,
            ser_dir=getattr(self.config, 'ser_dir', ''),
            pe_dir=getattr(self.config, 'pe_dir', ''),
            cross_dataset_test=getattr(self.config, 'cross_dataset_test', False),
            balanced_sampling=False  # 【新增】测试集不需要平衡采样
        )
        
        if len(test_dataset) == 0:
            raise ValueError("No test data found! Please check dataset configuration.")
        
        return DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True

        )
