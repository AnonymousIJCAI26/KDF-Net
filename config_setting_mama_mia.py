from datetime import datetime
import os

class MamaMiaConfig:
    """MAMA-MIA数据集上的UltraLight VM-UNet配置"""
    
    # ==================== 基础设置 ====================
    network = 'UltraLight_VM_UNet_MAMA_MIA'
    datasets = 'MAMA_MIA'
    
    # ==================== 数据路径 ====================
    data_dir = '/root/Lty/MAMA-MIA/data/'
    seg_dir = '/root/Lty/MAMA-MIA/segmentations_expert/'
    ser_dir = '/root/Lty/MAMA-MIA/data_FTV_SER_T1/'
    pe_dir = '/root/Lty/MAMA-MIA/data_FTV_PE_T1/'
    
    # ==================== 使用的数据集 ====================
    datasets_list = ['DUKE', 'NACT', 'ISPY1', 'ISPY2']
    
    # ==================== 多模态设置 ====================
    multimodal = False
    input_channels = 1  # 单模态为1，多模态为3
    
    # ==================== 动态模态融合配置 ====================
    enable_fusion = False  # 是否启用动态模态融合
    fusion_verbose = False  # 是否输出融合调试信息
    test_weight_method = 'historical_mean'  # 测试时权重选择方法
    # 可选值:
    # - 'current': 使用当前权重（原始实现）
    # - 'historical_mean': 使用训练历史均值（推荐）
    # - 'historical_median': 使用训练历史中位数
    # - 'last': 使用最后一次训练权重
    
    # ==================== 模型配置 ====================
    model_config = {
        'num_classes': 1,
        'input_channels': 1,  # 根据multimodal自动调整
        'c_list': [8, 16, 24, 32, 48, 64],
        'split_att': 'fc',
        'bridge': True,
    }
    
    # ==================== 训练参数 ====================
    from utils import BceDiceLoss
    criterion = BceDiceLoss()
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    amp = False
    batch_size = 256
    epochs = 400
    
    # 梯度累积步数（用于大batch_size训练）
    gradient_accumulation_steps = 1
    
    # ==================== 工作目录 ====================
    work_dir = f'results/{network}_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    
    # ==================== 日志和保存间隔 ====================
    print_interval = 20
    val_interval = 10
    save_interval = 50
    threshold = 0.5
    
    # ==================== 优化器配置 ====================
    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False
    
    # ==================== 学习率调度器 ====================
    sch = 'CosineAnnealingLR'
    T_max = 50
    eta_min = 1e-6
    last_epoch = -1
    
    # ==================== 数据增强配置 ====================
    use_augmentation = False  # 是否使用数据增强
    augmentation_p = 0.5  # 数据增强概率
    balanced_sampling = False  # 是否使用平衡采样
    
    # ==================== 跨数据集测试配置 ====================
    cross_dataset_test = False  # 是否进行跨数据集测试
    
    def __init__(self, **kwargs):
        """
        初始化配置
        
        Args:
            **kwargs: 覆盖默认配置的参数
        """
        # 应用用户提供的参数
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        
                # ==================== 【新增】模型类型处理 ====================
        # 如果配置中没有model_type，添加默认值
        if not hasattr(self, 'model_type'):
            self.model_type = 'ultralight'
        
        # 打印模型类型信息
        print(f"🎯 Model Type: {self.model_type.upper()}")
        
        # ==================== 自动调整输入通道数 ====================
        if self.multimodal:
            self.input_channels = 3
            self.model_config['input_channels'] = 3
        else:
            self.input_channels = 1
            self.model_config['input_channels'] = 1
            
        # ==================== 动态融合配置验证 ====================
        if self.enable_fusion:
            if not self.multimodal:
                print("⚠️ Warning: Dynamic fusion requires multimodal input. Disabling fusion.")
                self.enable_fusion = False
            else:
                print("✅ Dynamic modal fusion ENABLED")
                print(f"   - Test weight method: {self.test_weight_method}")
                # 验证测试权重方法
                valid_methods = ['current', 'historical_mean', 'historical_median', 'last']
                if self.test_weight_method not in valid_methods:
                    print(f"⚠️ Warning: Invalid test_weight_method: {self.test_weight_method}")
                    print(f"   Valid methods are: {valid_methods}")
                    print(f"   Using default: historical_mean")
                    self.test_weight_method = 'historical_mean'
                
                if self.fusion_verbose:
                    print("   - Verbose mode: ON")
                else:
                    print("   - Verbose mode: OFF")
        else:
            if self.multimodal:
                print("ℹ️  Dynamic modal fusion DISABLED (using direct 3-channel input)")
        
        # ==================== 数据集列表验证 ====================
        # 确保datasets_list是列表
        if isinstance(self.datasets_list, str):
            self.datasets_list = [self.datasets_list]
            
        # 检查数据集名称有效性
        valid_datasets = ['DUKE', 'NACT', 'ISPY1', 'ISPY2']
        invalid_datasets = [d for d in self.datasets_list if d not in valid_datasets]
        if invalid_datasets:
            print(f"⚠️ Warning: Invalid dataset names: {invalid_datasets}")
            print(f"   Valid datasets are: {valid_datasets}")
            # 移除无效数据集
            self.datasets_list = [d for d in self.datasets_list if d in valid_datasets]
        
        # ==================== 数据增强配置验证 ====================
        if self.use_augmentation:
            print(f"✅ Data augmentation ENABLED (p={self.augmentation_p})")
        else:
            print("ℹ️  Data augmentation DISABLED")
            
        if self.balanced_sampling:
            print("✅ Balanced sampling ENABLED")
        else:
            print("ℹ️  Balanced sampling DISABLED")
            
        # ==================== 梯度累积验证 ====================
        if self.gradient_accumulation_steps > 1:
            print(f"✅ Gradient accumulation ENABLED (steps={self.gradient_accumulation_steps})")
            # 调整有效batch size
            effective_batch_size = self.batch_size * self.gradient_accumulation_steps
            print(f"   - Effective batch size: {effective_batch_size}")
        
        # ==================== 跨数据集测试配置 ====================
        if self.cross_dataset_test:
            print("✅ Cross-dataset testing ENABLED")
        
        # ==================== 创建工作目录 ====================
        os.makedirs(self.work_dir, exist_ok=True)
        
        # ==================== 打印最终配置摘要 ====================
        self._print_config_summary()
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print("MAMA-MIA DATASET CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # 数据集信息
        print(f"📁 Dataset Configuration:")
        print(f"   - Dataset: {self.datasets}")
        print(f"   - Sub-datasets: {', '.join(self.datasets_list)}")
        print(f"   - Data directory: {self.data_dir}")
        
        # 模态信息
        print(f"\n🎯 Modal Configuration:")
        print(f"   - Multimodal: {'✅ Yes (T1+SER+PE)' if self.multimodal else '❌ No (T1 only)'}")
        print(f"   - Input channels: {self.input_channels}")
        if self.multimodal:
            print(f"   - SER directory: {self.ser_dir}")
            print(f"   - PE directory: {self.pe_dir}")
        
        # 动态融合信息
        if self.multimodal:
            print(f"\n🔬 Dynamic Fusion Configuration:")
            print(f"   - Enabled: {'✅ YES' if self.enable_fusion else '❌ NO'}")
            if self.enable_fusion:
                print(f"   - Test weight method: {self.test_weight_method}")
                method_desc = {
                    'current': 'Use current model weights',
                    'historical_mean': 'Use mean of training history (recommended)',
                    'historical_median': 'Use median of training history',
                    'last': 'Use last training weights'
                }
                print(f"     ↳ {method_desc.get(self.test_weight_method, 'Unknown method')}")
                print(f"   - Verbose mode: {'✅ ON' if self.fusion_verbose else '❌ OFF'}")
        
        # 模型信息
        print(f"\n🤖 Model Configuration:")
        print(f"   - Network: {self.network}")
        print(f"   - Input size: {self.input_size_h}x{self.input_size_w}")
        print(f"   - Output classes: {self.num_classes}")
        print(f"   - Channel list: {self.model_config['c_list']}")
        print(f"   - Bridge: {'✅ ENABLED' if self.model_config['bridge'] else '❌ DISABLED'}")
        
        # 训练信息
        print(f"\n⚙️  Training Configuration:")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Epochs: {self.epochs}")
        print(f"   - Learning rate: {self.lr}")
        print(f"   - Optimizer: {self.opt}")
        print(f"   - Scheduler: {self.sch}")
        print(f"   - Gradient accumulation: {self.gradient_accumulation_steps}x")
        
        # 数据增强
        print(f"   - Data augmentation: {'✅ ENABLED' if self.use_augmentation else '❌ DISABLED'}")
        if self.use_augmentation:
            print(f"   - Augmentation probability: {self.augmentation_p}")
        print(f"   - Balanced sampling: {'✅ ENABLED' if self.balanced_sampling else '❌ DISABLED'}")
        
        # 其他设置
        print(f"\n🔧 Other Settings:")
        print(f"   - Random seed: {self.seed}")
        print(f"   - Number of workers: {self.num_workers}")
        print(f"   - Mixed precision: {'✅ ENABLED' if self.amp else '❌ DISABLED'}")
        print(f"   - Validation interval: every {self.val_interval} epochs")
        print(f"   - Checkpoint save interval: every {self.save_interval} epochs")
        print(f"   - Segmentation threshold: {self.threshold}")
        
        # 跨数据集测试
        if self.cross_dataset_test:
            print(f"   - Cross-dataset test: ✅ ENABLED")
        
        # 工作目录
        print(f"\n📂 Output Directory:")
        print(f"   - {self.work_dir}")
        
        print("=" * 60 + "\n")
    
    def to_dict(self):
        """将配置转换为字典（用于保存）"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # 处理特殊类型
                if key == 'criterion' and hasattr(value, '__class__'):
                    config_dict[key] = value.__class__.__name__
                else:
                    config_dict[key] = value
        return config_dict
    
    def save(self, path=None):
        """保存配置到文件"""
        if path is None:
            path = os.path.join(self.work_dir, "config.json")
        
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"📄 Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path):
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # 处理特殊字段
        if 'criterion' in config_dict and isinstance(config_dict['criterion'], str):
            if config_dict['criterion'] == 'BceDiceLoss':
                from utils import BceDiceLoss
                config_dict['criterion'] = BceDiceLoss()
        
        # 创建配置实例
        config = cls()
        
        # 应用保存的配置
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"📂 Configuration loaded from: {path}")
        return config
    
    def get_fusion_config_info(self):
        """获取融合配置的详细信息"""
        if not self.multimodal or not self.enable_fusion:
            return None
        
        info = {
            'enabled': True,
            'test_weight_method': self.test_weight_method,
            'verbose': self.fusion_verbose,
            'description': {
                'current': '使用当前模型权重（可能不稳定）',
                'historical_mean': '使用训练历史均值（推荐，更稳健）',
                'historical_median': '使用训练历史中位数（抗异常值）',
                'last': '使用最后一次训练权重'
            }
        }
        
        return info


# ==================== 配置验证函数 ====================
def validate_config(config):
    """
    验证配置的有效性
    
    Args:
        config: MamaMiaConfig实例
        
    Returns:
        bool: 配置是否有效
        str: 错误信息（如果无效）
    """
    errors = []
    
    # 检查必要目录是否存在
    required_dirs = [
        (config.data_dir, "Data directory"),
        (config.seg_dir, "Segmentation directory"),
    ]
    
    if config.multimodal:
        required_dirs.extend([
            (config.ser_dir, "SER directory"),
            (config.pe_dir, "PE directory"),
        ])
    
    for dir_path, dir_name in required_dirs:
        if not os.path.exists(dir_path):
            errors.append(f"{dir_name} does not exist: {dir_path}")
    
    # 检查数据集列表
    if not config.datasets_list:
        errors.append("No datasets specified in datasets_list")
    
    # 检查模型配置
    if config.model_config['input_channels'] not in [1, 3]:
        errors.append(f"Invalid input_channels: {config.model_config['input_channels']}. Must be 1 or 3.")
    
    # 检查动态融合配置
    if config.enable_fusion and not config.multimodal:
        errors.append("Dynamic fusion requires multimodal input (multimodal=True)")
    
    # 检查测试权重方法
    valid_weight_methods = ['current', 'historical_mean', 'historical_median', 'last']
    if config.test_weight_method not in valid_weight_methods:
        errors.append(f"Invalid test_weight_method: {config.test_weight_method}. Must be one of {valid_weight_methods}")
    
    # 检查学习率
    if config.lr <= 0:
        errors.append(f"Invalid learning rate: {config.lr}. Must be > 0.")
    
    # 检查batch size
    if config.batch_size <= 0:
        errors.append(f"Invalid batch size: {config.batch_size}. Must be > 0.")
    
    # 检查梯度累积步数
    if config.gradient_accumulation_steps <= 0:
        errors.append(f"Invalid gradient accumulation steps: {config.gradient_accumulation_steps}. Must be > 0.")
    
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False, "\n".join(errors)
    
    print("✅ Configuration validation passed")
    return True, None


# ==================== 配置创建辅助函数 ====================
def create_fusion_config(name="fusion_experiment", multimodal=True, enable_fusion=True, 
                        test_weight_method='historical_mean', **kwargs):
    """
    创建启用动态融合的配置
    
    Args:
        name: 实验名称
        multimodal: 是否使用多模态
        enable_fusion: 是否启用动态融合
        test_weight_method: 测试时权重选择方法
        **kwargs: 其他配置参数
        
    Returns:
        MamaMiaConfig实例
    """
    base_kwargs = {
        'network': f'Enhanced_{name}',
        'multimodal': multimodal,
        'enable_fusion': enable_fusion,
        'test_weight_method': test_weight_method,
        'fusion_verbose': kwargs.get('fusion_verbose', False),
        'datasets_list': kwargs.get('datasets_list', ['DUKE', 'NACT', 'ISPY1', 'ISPY2']),
    }
    
    # 合并用户提供的参数
    base_kwargs.update(kwargs)
    
    return MamaMiaConfig(**base_kwargs)


def create_baseline_config(name="baseline_experiment", multimodal=True, **kwargs):
    """
    创建基线配置（禁用动态融合）
    
    Args:
        name: 实验名称
        multimodal: 是否使用多模态
        **kwargs: 其他配置参数
        
    Returns:
        MamaMiaConfig实例
    """
    base_kwargs = {
        'network': f'Baseline_{name}',
        'multimodal': multimodal,
        'enable_fusion': False,
        'fusion_verbose': False,
        'datasets_list': kwargs.get('datasets_list', ['DUKE', 'NACT', 'ISPY1', 'ISPY2']),
    }
    
    # 合并用户提供的参数
    base_kwargs.update(kwargs)
    
    return MamaMiaConfig(**base_kwargs)


def create_comparison_configs(name_prefix="experiment", multimodal=True, 
                             test_methods=['current', 'historical_mean', 'historical_median', 'last'], **kwargs):
    """
    创建多个配置用于比较不同的测试权重方法
    
    Args:
        name_prefix: 实验名称前缀
        multimodal: 是否使用多模态
        test_methods: 要测试的方法列表
        **kwargs: 其他配置参数
        
    Returns:
        dict: 方法名到配置的映射
    """
    configs = {}
    
    for method in test_methods:
        config_name = f"{name_prefix}_{method}"
        configs[method] = create_fusion_config(
            name=config_name,
            multimodal=multimodal,
            enable_fusion=True,
            test_weight_method=method,
            **kwargs
        )
    
    return configs


# ==================== 示例用法 ====================
if __name__ == "__main__":
    print("Testing MamaMiaConfig class...\n")
    
    # 示例1：创建使用历史均值的融合配置
    print("Example 1: Configuration with historical mean fusion")
    config_fusion = create_fusion_config(
        name="test_historical_mean",
        multimodal=True,
        enable_fusion=True,
        test_weight_method='historical_mean',
        fusion_verbose=True,
        batch_size=256,
        epochs=50
    )
    
    # 验证配置
    is_valid, error_msg = validate_config(config_fusion)
    
    if is_valid:
        print("\n✅ Fusion configuration is valid")
        
        # 显示融合配置信息
        fusion_info = config_fusion.get_fusion_config_info()
        if fusion_info:
            print(f"\n🔬 Fusion Configuration Details:")
            print(f"   - Method: {fusion_info['test_weight_method']}")
            print(f"   - Description: {fusion_info['description'][fusion_info['test_weight_method']]}")
        
        config_fusion.save()
    else:
        print(f"\n❌ Fusion configuration is invalid: {error_msg}")
    
    print("\n" + "-" * 60 + "\n")
    
    # 示例2：创建比较不同方法的配置
    print("Example 2: Creating comparison configurations for different test methods")
    comparison_configs = create_comparison_configs(
        name_prefix="comparison",
        multimodal=True,
        test_methods=['current', 'historical_mean', 'historical_median'],
        batch_size=16,
        epochs=100
    )
    
    for method, config in comparison_configs.items():
        print(f"\n📋 {method.upper()} configuration:")
        print(f"   - Network: {config.network}")
        print(f"   - Test method: {config.test_weight_method}")
    
    print("\n" + "-" * 60 + "\n")
    
    # 示例3：创建基线配置
    print("Example 3: Baseline configuration (no fusion)")
    config_baseline = create_baseline_config(
        name="test_baseline",
        multimodal=True,
        batch_size=256,
        epochs=200
    )
    
    # 验证配置
    is_valid, error_msg = validate_config(config_baseline)
    
    if is_valid:
        print("\n✅ Baseline configuration is valid")
        config_baseline.save()
    else:
        print(f"\n❌ Baseline configuration is invalid: {error_msg}")
    
    print("\n✨ Configuration test completed successfully!")