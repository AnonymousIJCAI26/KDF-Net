"""
模型包初始化文件
"""
from .UltraLight_VM_UNet import UltraLight_VM_UNet
from .create_baseline_model import create_baseline_model, prepare_baseline_model_for_testing

# 尝试导入baseline模型
try:
    from .baseline_unet import Baseline_UNet
    from .baseline_attention_unet import Baseline_Attention_UNet
    from .baseline_unet_plusplus import Baseline_UNetPlusPlus
    from .baseline_deeplabv3 import Baseline_DeeplabV3
    from .baseline_swin_unet import Baseline_Swin_UNet
    from .baseline_nnunet import Baseline_nnUNet
    from .baseline_transunet import Baseline_TransUNet
    from .baseline_fcn import Baseline_FCN
    BASELINE_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some baseline models not available: {e}")
    BASELINE_MODELS_AVAILABLE = False

__all__ = [
    'UltraLight_VM_UNet',
    'create_baseline_model',
    'prepare_baseline_model_for_testing',
    'BASELINE_MODELS_AVAILABLE'
]

# 动态添加可用的baseline模型
if BASELINE_MODELS_AVAILABLE:
    __all__.extend([
        'Baseline_UNet',
        'Baseline_Attention_UNet',
        'Baseline_UNetPlusPlus',
        'Baseline_DeeplabV3',
        'Baseline_Swin_UNet',
        'Baseline_nnUNet',
        'Baseline_TransUNet',
        'Baseline_FCN'
    ])