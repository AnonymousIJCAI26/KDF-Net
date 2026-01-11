import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import argparse
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from config_setting_mama_mia import MamaMiaConfig
from mama_mia_loader import MAMAMIADataLoader
from engine import train_one_epoch, val_one_epoch

# ==================== 【新增导入】 ====================
# 导入增强版模型
try:
    from models.ultralight_vm_unet_enhanced import create_ultralight_model
    USE_ENHANCED_MODEL = True
    print("✅ Enhanced model module found")
except ImportError:
    # 如果增强版模型不存在，使用原始模型（向后兼容）
    from models.UltraLight_VM_UNet import UltraLight_VM_UNet
    USE_ENHANCED_MODEL = False
    print("⚠️ Enhanced model module not found, using original model")
# ==================== 【新增结束】 ====================

import warnings
warnings.filterwarnings("ignore")

def print_memory_usage():
    """打印内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {gpu_memory:.2f}GB")

def parse_args():
    parser = argparse.ArgumentParser(description='UltraLight VM-UNet Training for MAMA-MIA')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--multimodal', action='store_true', help='Use multimodal input (T1+SER+PE)')
    parser.add_argument('--datasets', nargs='+', required=True, 
                       help='Datasets to use for training, e.g., DUKE NACT ISPY1 ISPY2')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--input_channels', type=int, default=1, help='Input channels')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/data/', help='Data directory')
    parser.add_argument('--seg_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/segmentations_expert/', help='Segmentation directory')
    parser.add_argument('--ser_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_SER_T1/', help='SER directory')
    parser.add_argument('--pe_dir', type=str, default='/root/autodl-tmp/Lty/MAMA_MIA/data_FTV_PE_T1/', help='PE directory')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--skip_flops', action='store_true', help='Skip FLOPs calculation')
    
    # 【原有参数】
    parser.add_argument('--balanced_sampling', action='store_true', help='Use balanced sampling for class imbalance')
    parser.add_argument('--data_augmentation', action='store_true', help='Use data augmentation during training')
    parser.add_argument('--augmentation_p', type=float, default=0.5, help='Probability for data augmentation')
    
    # ==================== 【新增参数】 ====================
    # 动态融合参数
    parser.add_argument('--enable_fusion', action='store_true', 
                       help='Enable dynamic modal fusion (requires multimodal)')
    parser.add_argument('--fusion_verbose', action='store_true',
                       help='Enable verbose output for fusion module')
    parser.add_argument('--test_weight_method', type=str, default='historical_mean',
                       choices=['current', 'historical_mean', 'historical_median', 'last'],
                       help='Test weight selection method for dynamic fusion')
    # ==================== 【新增结束】 ====================
    
    return parser.parse_args()

def main():
    args = parse_args()
    
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
    
    print("=== Configuration Summary ===")
    print(f"Experiment Name: {args.name}")
    print(f"Multimodal: {args.multimodal}")
    if args.multimodal:
        print("✓ Using T1 + SER + PE multimodal input")
    else:
        print("✓ Using T1 only single modal input")
    
    # ==================== 【新增】显示融合配置 ====================
    if args.multimodal:
        if args.enable_fusion:
            print("🎯 Dynamic Modal Fusion: ✅ ENABLED")
            if args.fusion_verbose:
                print("   - Verbose mode: ✅ ON")
        else:
            print("🎯 Dynamic Modal Fusion: ❌ DISABLED (direct 3-channel input)")
    # ==================== 【新增结束】 ====================
    
    print(f"Datasets: {args.datasets}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Input Channels: {args.input_channels}")
    print(f"Data Workers: {args.num_workers}")
    print(f"Data Directory: {args.data_dir}")
    # 【原有】显示平衡采样和数据增广配置
    print(f"Balanced Sampling: {args.balanced_sampling}")
    print(f"Data Augmentation: {args.data_augmentation}")
    if args.data_augmentation:
        print(f"Augmentation Probability: {args.augmentation_p}")
    print("=============================")
    
    # 创建配置
    config = MamaMiaConfig(
        multimodal=args.multimodal,
        datasets_list=args.datasets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        input_channels=args.input_channels,
        data_dir=args.data_dir,
        seg_dir=args.seg_dir,
        ser_dir=args.ser_dir,
        pe_dir=args.pe_dir,
        num_workers=args.num_workers,
        # ==================== 【新增】传递融合参数 ====================
        enable_fusion=args.enable_fusion,
        fusion_verbose=args.fusion_verbose,
        test_weight_method=args.test_weight_method,
        # ==================== 【新增结束】 ====================
        balanced_sampling=args.balanced_sampling,
        use_augmentation=args.data_augmentation,
        augmentation_p=args.augmentation_p
    )
    
    # 【原有】设置平衡采样和数据增广参数
    config.balanced_sampling = args.balanced_sampling
    config.use_augmentation = args.data_augmentation
    config.augmentation_p = args.augmentation_p
    
    # 【原有】设置完整的随机种子
    print('#----------Setting random seed for reproducibility----------#')
    set_seed(config.seed)
    
    # 设置工作目录
    config.work_dir = f'results/{args.name}'
    config.network = args.name

    print('#----------Creating logger----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    data_loader = MAMAMIADataLoader(config)
    
    try:
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        test_loader = data_loader.get_test_loader()
        
        # 【原有】为DataLoader设置随机种子
        train_loader = seed_data_loader(train_loader, config.seed)
        val_loader = seed_data_loader(val_loader, config.seed)
        test_loader = seed_data_loader(test_loader, config.seed)
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please check:")
        print("1. Dataset names are correct: DUKE, NACT, ISPY1, ISPY2")
        print("2. Data directories exist and contain the required files")
        print("3. For multimodal, ensure SER and PE directories contain the required files")
        return

    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    if len(train_loader.dataset) == 0:
        print("ERROR: No training samples found! Please check your dataset configuration.")
        return

    print('#----------Preparing Models----------#')
    
    # ==================== 【新增】模型创建逻辑 ====================
    if USE_ENHANCED_MODEL:
        # 使用增强版模型（支持动态融合）
        model = create_ultralight_model(
            config,
            enable_fusion=config.enable_fusion,
            fusion_verbose=config.fusion_verbose,
            test_weight_method=config.test_weight_method 
        )
        model_type = "Enhanced UltraLight VM-UNet"
    else:
        # 使用原始模型（向后兼容）
        model = UltraLight_VM_UNet(
            num_classes=config.model_config['num_classes'],
            input_channels=config.model_config['input_channels'],
            c_list=config.model_config['c_list'],
            split_att=config.model_config['split_att'],
            bridge=config.model_config['bridge'],
        )
        model_type = "Original UltraLight VM-UNet"
        if config.enable_fusion:
            print("⚠️ Warning: Fusion requested but enhanced model not available")
            print("   Using original model without fusion")
    # ==================== 【新增结束】 ====================
    
    # 【原有】确保模型权重初始化也是确定的
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    print("\n=== Model Information ===")
    print(f"Model Type: {model_type}")
    if USE_ENHANCED_MODEL and hasattr(model, 'fusion_enabled'):
        print(f"Dynamic Fusion: {'✅ Enabled' if model.fusion_enabled else '❌ Disabled'}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    # 【原有】计算FLOPs（可选，跳过可能导致错误）
    if not args.skip_flops:
        try:
            from thop import profile
            # 先将模型移到GPU，然后计算FLOPs
            model_temp = model.cuda()
            dummy_input = torch.randn(1, config.input_channels, 256, 256).cuda()
            flops, params = profile(model_temp, inputs=(dummy_input,), verbose=False)
            print(f"FLOPs: {flops / 1e9:.2f} G")
            print(f"Params: {params / 1e6:.2f} M")
            logger.info(f'Model FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M')
            # 清理临时模型
            del model_temp, dummy_input
            torch.cuda.empty_cache()
        except ImportError:
            print("thop not installed, skipping FLOPs calculation")
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            print("Skipping FLOPs calculation...")
    else:
        print("Skipping FLOPs calculation as requested")
    
    # 正式将模型移到GPU
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler() if config.amp else None

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    best_val_loss = float('inf')
    
    # 记录训练开始时间
    import time
    start_time = time.time()

    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        print(f'#----------Resume Model from {args.resume}----------#')
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        best_val_loss = min_loss

        log_info = f'resuming model from {args.resume}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    print('\n#----------Training Started----------#')
    print(f"Total epochs: {config.epochs}")
    print(f"Training samples per epoch: {len(train_loader)}")
    print(f"Validation interval: every {config.val_interval} epochs")
    print(f"Checkpoint saving: every {config.save_interval} epochs")
    
    # 【原有】显示平衡采样和数据增广状态
    if config.balanced_sampling:
        print("✓ Balanced Sampling: ENABLED")
    else:
        print("✗ Balanced Sampling: DISABLED")
    if config.use_augmentation:
        print(f"✓ Data Augmentation: ENABLED (p={config.augmentation_p})")
    else:
        print("✗ Data Augmentation: DISABLED")
    
    # ==================== 【新增】显示融合状态 ====================
    if USE_ENHANCED_MODEL and hasattr(model.module, 'fusion_enabled'):
        if model.module.fusion_enabled:
            print("🎯 Dynamic Fusion: ✅ ENABLED")
            print(f"   - Test weight method: {model.module.test_weight_method}")
            print(f"   - Verbose mode: {'✅ ON' if config.fusion_verbose else '❌ OFF'}")
        else:
            print("🎯 Dynamic Fusion: ❌ DISABLED")
    # ==================== 【新增结束】 ====================
    
    print_memory_usage()
    
    # ==================== 【新增】融合分析保存目录 ====================
    if config.enable_fusion and USE_ENHANCED_MODEL:
        fusion_analysis_dir = os.path.join(config.work_dir, "fusion_analysis")
        os.makedirs(fusion_analysis_dir, exist_ok=True)
        print(f"📊 Fusion analysis will be saved to: {fusion_analysis_dir}")
    # ==================== 【新增结束】 ====================

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

        print(f'\n=== Epoch {epoch}/{config.epochs} ===')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 【原有】显示epoch开始前的显存状态
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f'GPU Memory before training: {memory_before:.2f}GB')
        
        # 训练一个epoch
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        # 【原有】训练后显示显存变化
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**3
            print(f'GPU Memory after training: {memory_after:.2f}GB')
        
        # 每个epoch都进行验证
        print(f'\n--- Validation Epoch {epoch} ---')
        val_loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )
        
        # ==================== 【新增】融合分析（每10个epoch） ====================
        if config.enable_fusion and USE_ENHANCED_MODEL and epoch % 10 == 0:
            try:
                if hasattr(model.module, 'analyze_fusion'):
                    analysis = model.module.analyze_fusion()
                    if analysis and analysis.get("status") == "success":
                        print(f"\n🔍 Fusion Analysis Epoch {epoch}:")
                        weights = analysis["modal_weights"]
                        print(f"  T1 weight: {weights['T1_mean']:.3f} ± {weights['T1_std']:.3f}")
                        print(f"  SER weight: {weights['SER_mean']:.3f} ± {weights['SER_std']:.3f}")
                        print(f"  PE weight: {weights['PE_mean']:.3f} ± {weights['PE_std']:.3f}")
                        
                        # 保存阶段性分析
                        if epoch % 50 == 0:
                            epoch_fusion_dir = os.path.join(fusion_analysis_dir, f"epoch_{epoch}")
                            model.module.visualize_fusion(epoch_fusion_dir)
            except Exception as e:
                print(f"⚠️ Fusion analysis failed: {e}")
        # ==================== 【新增结束】 ====================
        
        # 【原有】epoch结束后强制清理
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        torch.cuda.empty_cache()
        gc.collect()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            min_epoch = epoch
            # 使用清理后的state_dict保存模型
            torch.save(clean_state_dict(model.module.state_dict()), os.path.join(checkpoint_dir, 'best.pth'))
            print(f'>>> 🎯 New best model saved! Epoch: {epoch}, Val Loss: {val_loss:.4f}')

        # 定期保存检查点
        if epoch % config.save_interval == 0 or epoch == config.epochs:
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': best_val_loss,
                    'min_epoch': min_epoch,
                    'loss': val_loss,
                    'model_state_dict': clean_state_dict(model.module.state_dict()),  # 使用清理后的state_dict
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))
            print(f'>>> 💾 Checkpoint saved at epoch {epoch}')

        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        print_memory_usage()

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f'\n=== Training Completed ===')
    print(f'Total training time: {total_time:.2f}s ({total_time/60:.2f}min)')
    print(f'Best validation loss: {best_val_loss:.4f} at epoch {min_epoch}')

    # 重命名最佳模型文件（与原始逻辑一致）
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_model_path):
        new_best_model_path = os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{best_val_loss:.4f}.pth')
        os.rename(best_model_path, new_best_model_path)
        print(f'>>> 📁 Best model renamed to: {new_best_model_path}')

    # ==================== 【新增】训练后生成完整融合分析报告 ====================
    if config.enable_fusion and USE_ENHANCED_MODEL:
        print("\n📊 Generating final fusion analysis report...")
        try:
            final_fusion_dir = os.path.join(config.work_dir, "final_fusion_analysis")
            model.module.visualize_fusion(final_fusion_dir)
            print(f"✅ Final fusion analysis saved to: {final_fusion_dir}")
        except Exception as e:
            print(f"⚠️ Final fusion analysis failed: {e}")
    # ==================== 【新增结束】 ====================

    print("\n🎉 Training completed successfully!")
    print(f"Best model saved as: {new_best_model_path}")
    
    # 【原有】在提示信息中包含新参数
    multimodal_flag = "--multimodal" if args.multimodal else ""
    balanced_flag = "--balanced_sampling" if args.balanced_sampling else ""
    aug_flag = "--data_augmentation" if args.data_augmentation else ""
    
    # ==================== 【新增】在提示中包含融合参数 ====================
    fusion_flag = "--enable_fusion" if args.enable_fusion else ""
    fusion_verbose_flag = "--fusion_verbose" if args.fusion_verbose else ""
    test_method_flag = f"--test_weight_method {args.test_weight_method}" if args.enable_fusion else ""
    # ==================== 【新增结束】 ====================
    
    print(f"\n📋 Testing command:")
    print(f"python test_mama_mia_ultralight_advanced.py \\")
    print(f"  --name {args.name} \\")
    print(f"  --datasets {' '.join(args.datasets)} \\")
    print(f"  {multimodal_flag} \\")
    print(f"  {balanced_flag} \\")
    print(f"  {aug_flag} \\")
    # ==================== 【新增】添加融合参数到测试命令 ====================
    if args.enable_fusion:
        print(f"  {fusion_flag} \\")
        print(f"  {test_method_flag} \\")
        if args.fusion_verbose:
            print(f"  {fusion_verbose_flag} \\")
        print(f"  --analyze_fusion  # 可选：生成融合分析报告")
    # ==================== 【新增结束】 ====================

if __name__ == '__main__':
    main()